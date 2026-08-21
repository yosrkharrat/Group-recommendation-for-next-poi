"""
Stage 0 for the LBSN2Vec++ social-network track: raw global dump -> house-format CSVs.

Reproduces the `preprocessing-global-fsq` Kaggle notebook (notebooks/archive/) as a script --
same NYC bounding box, same activity filters, same counts -- with two deliberate fixes:

1. **friendship_old and friendship_new are kept SEPARATE.** The notebook concatenated them
   into one edge list, which erases the before/after-period distinction: the "new" snapshot
   postdates the check-in data, and (new - old) is exactly the eval set of the paper's
   friendship-prediction task. Training on it is future leakage. Here `friendship_old` is the
   only file a KG may consume; `friendship_new_only` is exported for evaluation.
2. **The outputs are the house schema**, so the EXISTING pipeline runs unchanged on this
   dataset: `build_groups.py --dataset LBSN_NYC` (stage 1 -- real timestamps, so real
   co-location groups), then `build_kg_lbsn.py --csv-dir` (stage 2, adds FRIEND_OF), then
   `train_roth.py --dataset LBSN_NYC` (stage 3).

Inputs
------
The 2.68 GB `dataset_WWW2019` zip (the notebook's gdown file id 1PNk3zY8NjLcDiAbzjABzY5FiPAFHq6T8),
read in streaming chunks -- nothing is extracted to disk. Files used inside it:

    raw_POIs.txt                        venue id, lat, lon, category NAME, country code
    dataset_WWW_Checkins_anonymized.txt user id, venue id, UTC time, timezone offset (minutes)
    dataset_WWW_friendship_old.txt      user, user   (before Apr 2012)
    dataset_WWW_friendship_new.txt      user, user   (after Jan 2014)

Recipe (identical to the notebook)
----------------------------------
    NYC bbox 40.49..40.92 / -74.27..-73.65  ->  ~102,687 POIs, ~288,188 check-ins
    venues >= 10 visits, then users >= 30   ->  159,304 check-ins, 1,665 users, 6,103 POIs
    friendships filtered to BOTH endpoints in the final user set: 1,507 old / 2,278 new

Outputs (--out-dir, default ./data/lbsn)
----------------------------------------
    train/val/test_<DS>.csv      user_id, venue_id, venue_category_id, venue_category_name,
                                 latitude, longitude, timezone_offset, utc_time, poi_idx
                                 -- the exact TSMC-style layout build_groups.load_checkins reads;
                                 per-user chronological 70/10/20 via resplit_per_user
    poi_metadata_<DS>.csv        venue_id, name, category, locality, region, description,
                                 poi_idx, latitude, longitude
                                 -- `category` is the FULL 2014-Foursquare taxonomy path
                                 ("Nightlife Spot>Bar>Sports Bar") via --taxonomy, restoring the
                                 SUBCATEGORY_OF spine; names missing from the tree stay depth-1.
                                 `locality` is a ~1km grid cell (the raw dump has no localities),
                                 `region` = NYC -- same containment role, different granularity.
    friendship_old_<DS>.csv      u1, u2 -- compact user_id space, the ONLY file a KG may use
    friendship_new_only_<DS>.csv u1, u2 -- (new - old), the friendship-prediction eval set
    users_<DS>.csv               user_id -> the raw anonymised id, for joins back to the dump
    lbsn_manifest.json           every count + config

Id spaces: `user_id` = rank of the raw anonymised id (numeric ascending), `poi_idx` = rank of
the venue hex id (ascending). Both are 0-based, contiguous, and recorded in the exports.

Usage
-----
    python prepare_lbsn_csvs.py --zip /path/to/lsbn2vec_global.zip --out-dir ./data/lbsn
    python prepare_lbsn_csvs.py --self-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import resplit_per_user

SEP = ">"
DS_DEFAULT = "LBSN_NYC"

POI_FILE = "dataset_WWW2019/raw_POIs.txt"
CHECKIN_FILE = "dataset_WWW2019/dataset_WWW_Checkins_anonymized.txt"
FRIEND_OLD_FILE = "dataset_WWW2019/dataset_WWW_friendship_old.txt"
FRIEND_NEW_FILE = "dataset_WWW2019/dataset_WWW_friendship_new.txt"

POI_COLS = ["venue_id", "latitude", "longitude", "venue_category_name", "country_code"]
CHECKIN_COLS = ["user_id", "venue_id", "utc_time", "timezone_offset"]

# the notebook's cell-17 config, verbatim
BBOX = dict(lat_min=40.49, lat_max=40.92, lon_min=-74.27, lon_max=-73.65)
MIN_VENUE_VISITS = 10
MIN_USER_VISITS = 30


# --------------------------------------------------------------------------
# streaming extraction from the zip
# --------------------------------------------------------------------------

def scan_pois(zpath, bbox, chunksize=500_000):
    frames = []
    with zipfile.ZipFile(zpath) as z, z.open(POI_FILE) as f:
        for chunk in pd.read_csv(f, sep="\t", header=None, names=POI_COLS,
                                 chunksize=chunksize,
                                 dtype={"venue_id": str, "country_code": str}):
            lat = pd.to_numeric(chunk["latitude"], errors="coerce")
            lon = pd.to_numeric(chunk["longitude"], errors="coerce")
            m = lat.between(bbox["lat_min"], bbox["lat_max"]) & \
                lon.between(bbox["lon_min"], bbox["lon_max"])
            if m.any():
                frames.append(chunk.loc[m])
    pois = pd.concat(frames, ignore_index=True).drop_duplicates("venue_id")
    print(f"POIs in bbox: {len(pois):,}")
    return pois


def scan_checkins(zpath, venue_ids, chunksize=500_000):
    frames = []
    with zipfile.ZipFile(zpath) as z, z.open(CHECKIN_FILE) as f:
        for i, chunk in enumerate(pd.read_csv(f, sep="\t", header=None, names=CHECKIN_COLS,
                                              chunksize=chunksize,
                                              dtype={"user_id": str, "venue_id": str})):
            m = chunk["venue_id"].isin(venue_ids)
            if m.any():
                frames.append(chunk.loc[m])
            if i % 20 == 0:
                print(f"  check-in chunk {i:,}  kept so far "
                      f"{sum(len(x) for x in frames):,}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    print(f"check-ins at bbox venues: {len(df):,}")
    return df


def scan_friendships(zpath, member, chunksize=500_000):
    out = {}
    with zipfile.ZipFile(zpath) as z:
        for key, name in (("old", FRIEND_OLD_FILE), ("new", FRIEND_NEW_FILE)):
            frames = []
            with z.open(name) as f:
                for chunk in pd.read_csv(f, sep="\t", header=None, names=["u1", "u2"],
                                         chunksize=chunksize, dtype=str):
                    m = chunk["u1"].isin(member) & chunk["u2"].isin(member)
                    if m.any():
                        frames.append(chunk.loc[m])
            out[key] = pd.concat(frames, ignore_index=True) if frames else \
                pd.DataFrame(columns=["u1", "u2"])
            print(f"friendship_{key}: {len(out[key]):,} edges with both endpoints kept")
    return out["old"], out["new"]


# --------------------------------------------------------------------------
# the notebook's filters, then house-format assembly
# --------------------------------------------------------------------------

def activity_filter(df, min_venue=MIN_VENUE_VISITS, min_user=MIN_USER_VISITS):
    """Venue floor first, then user floor -- one pass each, exactly as the notebook did."""
    vc = df["venue_id"].value_counts()
    df = df[df["venue_id"].isin(vc[vc >= min_venue].index)]
    uc = df["user_id"].value_counts()
    df = df[df["user_id"].isin(uc[uc >= min_user].index)].copy()
    print(f"after filters (venue>={min_venue}, user>={min_user}): {len(df):,} check-ins, "
          f"{df.user_id.nunique():,} users, {df.venue_id.nunique():,} venues")
    return df


def taxonomy_path(name, taxonomy):
    if taxonomy and name in taxonomy:
        return sorted(taxonomy[name])[0]
    return name        # not in the 2014 tree: keep as a depth-1 node


def assemble(df, pois, friends_old, friends_new, taxonomy, cell_deg=0.01):
    """Compact ids, split, and shape everything into the house schema."""
    # ids: numeric-ascending rank for users, hex-ascending rank for venues
    raw_users = sorted(df["user_id"].unique(), key=lambda s: (len(s), s))
    uid_of = {u: i for i, u in enumerate(raw_users)}
    raw_venues = sorted(df["venue_id"].unique())
    pid_of = {v: i for i, v in enumerate(raw_venues)}

    pois = pois.set_index("venue_id")
    meta = pd.DataFrame({"venue_id": raw_venues})
    meta["poi_idx"] = np.arange(len(meta))
    meta["latitude"] = [float(pois.at[v, "latitude"]) for v in raw_venues]
    meta["longitude"] = [float(pois.at[v, "longitude"]) for v in raw_venues]
    meta["venue_category_name"] = [str(pois.at[v, "venue_category_name"]) for v in raw_venues]
    meta["category"] = [taxonomy_path(n, taxonomy) for n in meta["venue_category_name"]]
    meta["name"] = ""
    meta["description"] = ""
    meta["locality"] = [f"{np.floor(a / cell_deg) * cell_deg:.4f},"
                        f"{np.floor(o / cell_deg) * cell_deg:.4f}"
                        for a, o in zip(meta["latitude"], meta["longitude"])]
    meta["region"] = "NYC"

    ck = df.copy()
    ck["user_id"] = ck["user_id"].map(uid_of)
    ck["poi_idx"] = ck["venue_id"].map(pid_of)
    ck["utc_time"] = pd.to_datetime(ck["utc_time"], utc=True,
                                    format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    n_bad = int(ck["utc_time"].isna().sum())
    assert n_bad == 0, f"{n_bad} unparseable timestamps"
    ck = ck.merge(meta[["poi_idx", "venue_category_name", "latitude", "longitude"]],
                  on="poi_idx", how="left")
    ck["venue_category_id"] = ""      # the raw dump has category names only
    ck = resplit_per_user(ck, 0.70, 0.10)

    def remap_edges(fr):
        fr = fr[fr["u1"].isin(uid_of) & fr["u2"].isin(uid_of)]
        e = pd.DataFrame({"u1": fr["u1"].map(uid_of), "u2": fr["u2"].map(uid_of)})
        e = e[e.u1 != e.u2]
        return pd.DataFrame(np.sort(e.to_numpy(), axis=1),
                            columns=["u1", "u2"]).drop_duplicates().reset_index(drop=True)

    old = remap_edges(friends_old)
    new = remap_edges(friends_new)
    old_set = set(map(tuple, old.to_numpy()))
    new_only = new[[tuple(r) not in old_set for r in new.to_numpy()]].reset_index(drop=True)

    users = pd.DataFrame({"user_id": range(len(raw_users)), "raw_id": raw_users})
    return ck, meta, old, new_only, users


def write_outputs(out_dir, ds, ck, meta, old, new_only, users, manifest):
    os.makedirs(out_dir, exist_ok=True)
    split_cols = ["user_id", "venue_id", "venue_category_id", "venue_category_name",
                  "latitude", "longitude", "timezone_offset", "utc_time", "poi_idx"]
    for split in ("train", "val", "test"):
        part = ck[ck["split"] == split]
        part[split_cols].to_csv(os.path.join(out_dir, f"{split}_{ds}.csv"), index=False)
        print(f"  {split}_{ds}.csv  {len(part):,} rows")
    meta_cols = ["venue_id", "name", "category", "locality", "region", "description",
                 "poi_idx", "latitude", "longitude"]
    meta[meta_cols].to_csv(os.path.join(out_dir, f"poi_metadata_{ds}.csv"), index=False)
    old.to_csv(os.path.join(out_dir, f"friendship_old_{ds}.csv"), index=False)
    new_only.to_csv(os.path.join(out_dir, f"friendship_new_only_{ds}.csv"), index=False)
    users.to_csv(os.path.join(out_dir, f"users_{ds}.csv"), index=False)
    with open(os.path.join(out_dir, "lbsn_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  poi_metadata_{ds}.csv  {len(meta):,} POIs")
    print(f"  friendship_old_{ds}.csv  {len(old):,} edges   "
          f"friendship_new_only_{ds}.csv  {len(new_only):,} eval pairs")


def run(a):
    taxonomy = json.load(open(a.taxonomy)) if a.taxonomy and os.path.exists(a.taxonomy) \
        else None
    if taxonomy is None:
        print("WARNING: no taxonomy file -- categories will be flat depth-1 names")

    pois = scan_pois(a.zip, BBOX)
    df = scan_checkins(a.zip, set(pois["venue_id"]))
    df = activity_filter(df)
    friends_old, friends_new = scan_friendships(a.zip, set(df["user_id"]))

    ck, meta, old, new_only, users = assemble(df, pois, friends_old, friends_new, taxonomy,
                                              a.cell_deg)
    depth = Counter(len([p for p in c.split(SEP) if p.strip()]) for c in meta["category"])
    n_flat = sum(v for k, v in depth.items() if k == 1)
    print(f"taxonomy depths over POIs: {dict(sorted(depth.items()))} "
          f"({n_flat:,} POIs at depth 1)")

    manifest = dict(config=vars(a), bbox=BBOX, min_venue_visits=MIN_VENUE_VISITS,
                    min_user_visits=MIN_USER_VISITS,
                    n_checkins=len(ck), n_users=int(ck.user_id.nunique()),
                    n_pois=len(meta), n_friend_old=len(old),
                    n_friend_new_only=len(new_only),
                    split_sizes=ck["split"].value_counts().to_dict(),
                    category_depths={str(k): v for k, v in sorted(depth.items())},
                    notebook_reference_counts=dict(checkins=159304, users=1665, pois=6103,
                                                   friends_old=1507, friends_new=2278))
    write_outputs(a.out_dir, a.dataset, ck, meta, old, new_only, users, manifest)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def _self_check():
    import tempfile
    print("SELF-CHECK on a synthetic zip fixture")
    pois = [  # 3 in-bbox NYC, 1 Boston, 1 in-bbox but too few visits
        ("v_aaa", 40.70, -73.95, "Wine Bar", "US"),
        ("v_bbb", 40.71, -73.96, "Sports Bar", "US"),
        ("v_ccc", 40.75, -73.87, "Coffee Shop", "US"),
        ("v_bos", 42.35, -71.08, "Coffee Shop", "US"),
        ("v_thin", 40.72, -73.94, "Museum", "US"),
    ]
    t0 = pd.Timestamp("2012-04-03 18:00:00+00:00")
    rows = []
    for i in range(40):   # users 10 and 20 alternate between two venues; 99 is sub-threshold
        u, v = ("10", "v_aaa") if i % 2 == 0 else ("20", "v_bbb")
        rows.append((u, v, (t0 + pd.Timedelta(hours=i)).strftime("%a %b %d %H:%M:%S +0000 %Y"),
                     -240))
    for i in range(30):
        rows.append(("30", "v_ccc",
                     (t0 + pd.Timedelta(hours=i)).strftime("%a %b %d %H:%M:%S +0000 %Y"), -240))
    rows.append(("99", "v_thin",
                 t0.strftime("%a %b %d %H:%M:%S +0000 %Y"), -240))   # dies with its venue
    rows.append(("10", "v_bos",
                 t0.strftime("%a %b %d %H:%M:%S +0000 %Y"), -240))   # out of bbox, dropped
    old = [("10", "20"), ("10", "99")]                # (10,99): 99 filtered out -> edge dropped
    new = [("10", "20"), ("20", "30"), ("20", "20")]  # (20,30) is new-only; self-loop dropped

    with tempfile.TemporaryDirectory() as td:
        zpath = os.path.join(td, "fixture.zip")
        with zipfile.ZipFile(zpath, "w") as z:
            z.writestr(POI_FILE, "\n".join("\t".join(map(str, r)) for r in pois))
            z.writestr(CHECKIN_FILE, "\n".join("\t".join(map(str, r)) for r in rows))
            z.writestr(FRIEND_OLD_FILE, "\n".join("\t".join(r) for r in old))
            z.writestr(FRIEND_NEW_FILE, "\n".join("\t".join(r) for r in new))

        taxonomy = {"Wine Bar": ["Nightlife Spot>Bar>Wine Bar"],
                    "Sports Bar": ["Nightlife Spot>Bar>Sports Bar"],
                    "Coffee Shop": ["Food>Coffee Shop"]}
        p = scan_pois(zpath, BBOX)
        df = scan_checkins(zpath, set(p["venue_id"]))
        df = activity_filter(df, 10, 20)   # fixture users have 20-30 visits
        fo, fn = scan_friendships(zpath, set(df["user_id"]))
        ck, meta, fold, fnew_only, users = assemble(df, p, fo, fn, taxonomy)

        out = os.path.join(td, "out")
        write_outputs(out, "TEST", ck, meta, fold, fnew_only, users, {"source": "self_check"})
        tr = pd.read_csv(os.path.join(out, "train_TEST.csv"))
        me = pd.read_csv(os.path.join(out, "poi_metadata_TEST.csv"))

    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    print()
    res = [
        ok("bbox filter drops the Boston POI", "v_bos" not in set(p.venue_id)),
        ok("out-of-bbox check-in dropped", not (df.venue_id == "v_bos").any()),
        ok("activity filters: thin venue and sub-threshold user gone",
           sorted(meta.venue_id) == ["v_aaa", "v_bbb", "v_ccc"]
           and 3 not in set(ck.user_id)),
        ok("user_id and poi_idx are 0-based contiguous ranks",
           sorted(ck.user_id.unique()) == [0, 1, 2]
           and sorted(meta.poi_idx) == [0, 1, 2]),
        ok("friendship edge to a filtered-out user is dropped",
           len(fold) == 1 and (fold.iloc[0].u1, fold.iloc[0].u2) == (0, 1)),
        ok("new-only = new minus old, self-loops dropped",
           len(fnew_only) == 1 and (fnew_only.iloc[0].u1, fnew_only.iloc[0].u2) == (1, 2)),
        ok("category is the full taxonomy path",
           set(meta.category) == {"Nightlife Spot>Bar>Wine Bar",
                                  "Nightlife Spot>Bar>Sports Bar", "Food>Coffee Shop"}),
        ok("split files carry the house columns",
           list(tr.columns) == ["user_id", "venue_id", "venue_category_id",
                                "venue_category_name", "latitude", "longitude",
                                "timezone_offset", "utc_time", "poi_idx"]),
        ok("per-user 70/10/20: train fraction correct for a 40-visit user",
           len(ck[(ck.user_id == 0) & (ck.split == "train")]) == 14
           and len(ck[ck.user_id == 0]) == 20),
        ok("metadata locality is a grid cell, region NYC",
           me.locality.str.contains(",").all() and (me.region == "NYC").all()),
    ]
    return all(res)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default="./lsbn2vec_global.zip",
                   help="the dataset_WWW2019 zip (the notebook's gdown download)")
    p.add_argument("--out-dir", default="./data/lbsn")
    p.add_argument("--dataset", default=DS_DEFAULT)
    p.add_argument("--taxonomy", default="./data/lbsn/fsq_category_paths_2014.json")
    p.add_argument("--cell-deg", type=float, default=0.01)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        sys.exit(0 if _self_check() else 1)
    run(a)


if __name__ == "__main__":
    main()
