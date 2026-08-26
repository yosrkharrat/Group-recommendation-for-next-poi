"""
Stage 0 for the LLMGPR 3-city track: the `llmgpr_final_*.csv` exports (from
`llmgpr-boundaries-and-region.ipynb`, see `LLMGPR_TRACK.md`) -> house-format CSVs.

Why this exists: `build_groups.py` / `build_kg.py` / `build_kg_lbsn.py` / `train_roth.py` were
written against the TSMC2014-NYC layout (`train/val/test_NYC.csv` + `poi_metadata_NYC.csv`) and,
for the social-graph track, against `prepare_lbsn_csvs.py`'s output. This script does the same
job for LLMGPR (CIKM'25): reuse the EXISTING pipeline unchanged, on a bigger (3-city, ~423k
check-in) dataset with a REAL friendship graph -- matching how LLMGPR itself builds groups (a
social edge, not an inferred affinity tie) -- by reshaping the recovered check-ins into the same
schema `prepare_lbsn_csvs.py` established. `LLMGPR_TRACK.md`'s own protocol (its literal group
rule, GBSR denoising, 500-candidate leave-one-out eval) is a SEPARATE, unstarted track; this
feeds the KCGRS-style occasional/established/random + FRIEND_OF pipeline instead.

Three gaps versus prepare_lbsn_csvs.py, each handled rather than worked around silently:

1. **No single input carries lat/lon for every check-in venue.** `llmgpr_final_catalogue.csv` is
   a *10 km-radius statistic catalogue* (the paper's #POIs reading), not the visited-venue set --
   only ~43% of the ~14.4k venues actually checked into fall inside it. Venues outside it get
   NaN lat/lon here. `build_kg.py::add_proximity` already drops non-finite coordinates before
   its k-NN pass, so `IS_NEAR_TO` is simply built over the covered 43% and every other relation
   (taxonomy, LOCATED_IN, the whole group and social layers) is unaffected. Coverage is
   asserted and reported, not hidden.
2. **`category` is a flat 2014-Foursquare category NAME** (e.g. "Jazz Club"), not a taxonomy
   path -- same source family as the LBSN track, so the already-committed
   `data/lbsn/fsq_category_paths_2014.json` maps it to a full path exactly as
   `prepare_lbsn_csvs.py` does; measured 97.3% row coverage, the rest fall back to a flat
   depth-1 node.
3. **The friendship export (`llmgpr_final_friendships.csv`) is tab-separated, raw user ids, one
   row per directed edge, tagged by snapshot** (`source` in {old, new, new,old}). `old` predates
   the check-in period and is the only snapshot safe to feed into group construction or the KG;
   `new - old` is a friendship-prediction eval set and must never enter either -- the exact
   leakage rule `prepare_lbsn_csvs.py` already enforces for the LBSN track, applied here too.

Outputs (--out-dir, default ./data/llmgpr)
-------------------------------------------
    train/val/test_<DS>.csv     user_id, venue_id, venue_category_id, venue_category_name,
                                latitude, longitude, timezone_offset, utc_time, poi_idx
                                -- per-user chronological 70/10/20 via build_groups.resplit_per_user
    poi_metadata_<DS>.csv       venue_id, name, category, locality, region, description,
                                poi_idx, latitude, longitude
                                -- locality = city (New York/Chicago/Los Angeles), region = state
                                (NY/IL/CA), giving LOCATED_IN a real 2-level hierarchy across cities
    friendship_old_<DS>.csv     u1, u2 -- compact user_id space, the ONLY file group construction
                                (--group-source social) or the KG (build_kg_lbsn.py) may use
    friendship_new_only_<DS>.csv  u1, u2 -- (new - old), the friendship-prediction eval set
    users_<DS>.csv               user_id -> the raw numeric id, for joins back to the source files
    llmgpr_prep_manifest.json    every count, the lat/lon and taxonomy coverage measurements

Id spaces: `user_id` = rank of the raw numeric user id (ascending), `poi_idx` = rank of the
venue hex id (ascending). Both 0-based, contiguous.

Usage
-----
    python prepare_llmgpr_csvs.py --checkins ./data/llmgpr/llmgpr_final_checkins.csv \
        --catalogue ./data/llmgpr/llmgpr_final_catalogue.csv \
        --friendships ./data/llmgpr/llmgpr_final_friendships.csv --out-dir ./data/llmgpr
    python prepare_llmgpr_csvs.py --self-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import resplit_per_user

SEP = ">"
DS_DEFAULT = "LLMGPR"

STATE_OF = {"New York": "NY", "Chicago": "IL", "Los Angeles": "CA"}


def taxonomy_path(name, taxonomy):
    if taxonomy and name in taxonomy:
        return sorted(taxonomy[name])[0]
    return name        # not in the 2014 tree: keep as a depth-1 node


def remap_friend_edges(friends, uid_of):
    """u1,u2 (0-based, matching train_<DS>.csv) -- self-loops dropped, both directions of a pair
    collapsed to one row. Edges to a user outside the check-in population (the friendship file
    may cover more people than checked in) are dropped."""
    u1 = friends["user_id"].map(uid_of)
    u2 = friends["friend_id"].map(uid_of)
    keep = u1.notna() & u2.notna() & (u1 != u2)
    e = pd.DataFrame({"u1": u1[keep].astype(int), "u2": u2[keep].astype(int)})
    e = pd.DataFrame(np.sort(e.to_numpy(), axis=1), columns=["u1", "u2"])
    return e.drop_duplicates().reset_index(drop=True)


def assemble(ck, cat, taxonomy, friendships=None):
    """Compact ids, split, and shape everything into the house schema."""
    raw_users = sorted(ck["user_id"].unique())
    uid_of = {u: i for i, u in enumerate(raw_users)}
    raw_venues = sorted(ck["venue_id"].unique())
    pid_of = {v: i for i, v in enumerate(raw_venues)}

    venue_cat = ck.drop_duplicates("venue_id").set_index("venue_id")[["category", "city"]]
    latlon = cat.drop_duplicates("venue_id").set_index("venue_id")[["lat", "lon"]]

    meta = pd.DataFrame({"venue_id": raw_venues})
    meta["poi_idx"] = np.arange(len(meta))
    meta["venue_category_name"] = venue_cat.loc[raw_venues, "category"].to_numpy()
    meta["category"] = [taxonomy_path(n, taxonomy) for n in meta["venue_category_name"]]
    meta["locality"] = venue_cat.loc[raw_venues, "city"].to_numpy()
    meta["region"] = [STATE_OF.get(c, "Unknown") for c in meta["locality"]]
    ll = latlon.reindex(raw_venues)
    meta["latitude"] = ll["lat"].to_numpy()
    meta["longitude"] = ll["lon"].to_numpy()
    meta["name"] = ""
    meta["description"] = ""

    df = ck.copy()
    df["user_id"] = df["user_id"].map(uid_of)
    df["poi_idx"] = df["venue_id"].map(pid_of)
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True,
                                    format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    n_bad = int(df["utc_time"].isna().sum())
    assert n_bad == 0, f"{n_bad} unparseable timestamps"
    df = df.rename(columns={"tz": "timezone_offset", "category": "venue_category_name"})
    df["venue_category_id"] = ""
    df = df.merge(meta[["poi_idx", "latitude", "longitude"]], on="poi_idx", how="left")
    df = resplit_per_user(df, 0.70, 0.10)

    users = pd.DataFrame({"user_id": range(len(raw_users)), "raw_id": raw_users})

    friend_old, friend_new_only = pd.DataFrame(columns=["u1", "u2"]), \
        pd.DataFrame(columns=["u1", "u2"])
    if friendships is not None:
        # "old" predates the check-in period -> safe for the KG/group construction.
        # "new" postdates it; (new - old) is a friendship-prediction EVAL set, not a training
        # input -- same leakage rule prepare_lbsn_csvs.py already enforces for the LBSN track.
        is_old = friendships["source"].str.contains("old")
        is_new = friendships["source"].str.contains("new")
        friend_old = remap_friend_edges(friendships[is_old], uid_of)
        new_all = remap_friend_edges(friendships[is_new], uid_of)
        old_set = set(map(tuple, friend_old.to_numpy()))
        friend_new_only = new_all[[tuple(r) not in old_set for r in new_all.to_numpy()]] \
            .reset_index(drop=True)

    return df, meta, users, friend_old, friend_new_only


def write_outputs(out_dir, ds, df, meta, users, friend_old, friend_new_only, manifest):
    os.makedirs(out_dir, exist_ok=True)
    split_cols = ["user_id", "venue_id", "venue_category_id", "venue_category_name",
                  "latitude", "longitude", "timezone_offset", "utc_time", "poi_idx"]
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        part[split_cols].to_csv(os.path.join(out_dir, f"{split}_{ds}.csv"), index=False)
        print(f"  {split}_{ds}.csv  {len(part):,} rows")
    meta_cols = ["venue_id", "name", "category", "locality", "region", "description",
                 "poi_idx", "latitude", "longitude"]
    meta[meta_cols].to_csv(os.path.join(out_dir, f"poi_metadata_{ds}.csv"), index=False)
    users.to_csv(os.path.join(out_dir, f"users_{ds}.csv"), index=False)
    print(f"  poi_metadata_{ds}.csv  {len(meta):,} POIs")
    if len(friend_old) or len(friend_new_only):
        friend_old.to_csv(os.path.join(out_dir, f"friendship_old_{ds}.csv"), index=False)
        friend_new_only.to_csv(os.path.join(out_dir, f"friendship_new_only_{ds}.csv"), index=False)
        print(f"  friendship_old_{ds}.csv  {len(friend_old):,} edges   "
              f"friendship_new_only_{ds}.csv  {len(friend_new_only):,} eval pairs")
    with open(os.path.join(out_dir, "llmgpr_prep_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def run(a):
    taxonomy = json.load(open(a.taxonomy, encoding="utf-8")) \
        if a.taxonomy and os.path.exists(a.taxonomy) else None
    if taxonomy is None:
        print("WARNING: no taxonomy file -- categories will be flat depth-1 names")

    ck = pd.read_csv(a.checkins, dtype={"user_id": "int64", "venue_id": str})
    cat = pd.read_csv(a.catalogue, dtype={"venue_id": str})
    print(f"check-ins: {len(ck):,} rows, {ck.user_id.nunique():,} users, "
          f"{ck.venue_id.nunique():,} venues, cities {sorted(ck.city.unique())}")

    latlon_cov = ck["venue_id"].drop_duplicates().isin(set(cat["venue_id"])).mean()
    print(f"lat/lon coverage over checkin venues (via catalogue join): {latlon_cov:.1%}")

    friendships = None
    if a.friendships and os.path.exists(a.friendships):
        friendships = pd.read_csv(a.friendships, sep="\t",
                                  dtype={"user_id": "int64", "friend_id": "int64"})
        both_in = friendships["user_id"].isin(set(ck.user_id)) & \
            friendships["friend_id"].isin(set(ck.user_id))
        print(f"friendships: {len(friendships):,} rows, {both_in.mean():.1%} with both "
              f"endpoints in the check-in population, source={friendships.source.value_counts().to_dict()}")

    df, meta, users, friend_old, friend_new_only = assemble(ck, cat, taxonomy, friendships)

    depth = Counter(len([p for p in c.split(SEP) if p.strip()]) for c in meta["category"])
    n_flat = sum(v for k, v in depth.items() if k == 1)
    tax_cov = 1.0 - (meta["venue_category_name"].map(
        lambda n: n not in (taxonomy or {})).sum() / max(len(meta), 1))
    print(f"taxonomy depths over POIs: {dict(sorted(depth.items()))} "
          f"({n_flat:,} POIs at depth 1); mapped {tax_cov:.1%} of POIs to a >1-depth path")

    manifest = dict(config=vars(a),
                    n_checkins=len(df), n_users=int(df.user_id.nunique()), n_pois=len(meta),
                    split_sizes=df["split"].value_counts().to_dict(),
                    city_counts=ck["city"].value_counts().to_dict(),
                    latlon_coverage=round(float(latlon_cov), 4),
                    taxonomy_poi_coverage=round(float(tax_cov), 4),
                    category_depths={str(k): v for k, v in sorted(depth.items())},
                    n_friendship_old=len(friend_old), n_friendship_new_only=len(friend_new_only))
    write_outputs(a.out_dir, a.dataset, df, meta, users, friend_old, friend_new_only, manifest)


def _self_check():
    import tempfile
    print("SELF-CHECK on a synthetic fixture")
    t0 = pd.Timestamp("2012-04-03 18:00:00+00:00")

    def fmt(ts):
        return ts.strftime("%a %b %d %H:%M:%S +0000 %Y")

    rows = []
    for i in range(40):   # two users alternate between two NYC venues -> in catalogue
        u, v, city = (10, "v_aaa", "New York") if i % 2 == 0 else (20, "v_bbb", "New York")
        rows.append((u, v, fmt(t0 + pd.Timedelta(hours=i)), -240, city, "Wine Bar"))
    for i in range(25):   # one user at a Chicago venue that is OUTSIDE the catalogue (no lat/lon)
        rows.append((30, "v_ccc_nocat", fmt(t0 + pd.Timedelta(hours=i)), -300, "Chicago",
                    "Coffee Shop"))
    ck = pd.DataFrame(rows, columns=["user_id", "venue_id", "utc_time", "tz", "city", "category"])

    cat = pd.DataFrame([
        dict(venue_id="v_aaa", lat=40.73, lon=-74.00, category="Wine Bar", city="New York", km=1.0),
        dict(venue_id="v_bbb", lat=40.75, lon=-73.98, category="Wine Bar", city="New York", km=2.0),
        # v_ccc_nocat deliberately absent -> must surface as NaN lat/lon, not a crash
    ])
    taxonomy = {"Wine Bar": ["Nightlife Spot>Bar>Wine Bar"]}   # "Coffee Shop" deliberately absent

    # raw user ids 10, 20, 30 sort to compact ids 0, 1, 2
    friend = pd.DataFrame([
        dict(user_id=10, friend_id=20, source="old"),          # safe: predates check-ins -> KG
        dict(user_id=10, friend_id=30, source="new"),          # (new - old): eval-only, held out
        dict(user_id=10, friend_id=999, source="old"),         # 999 never checked in -> dropped
    ])

    with tempfile.TemporaryDirectory() as td:
        ckp = os.path.join(td, "ck.csv"); catp = os.path.join(td, "cat.csv")
        frp = os.path.join(td, "fr.csv")
        ck.to_csv(ckp, index=False); cat.to_csv(catp, index=False)
        friend.to_csv(frp, sep="\t", index=False)

        class A: pass
        a = A()
        a.checkins, a.catalogue, a.out_dir, a.dataset = ckp, catp, os.path.join(td, "out"), "TEST"
        a.taxonomy = os.path.join(td, "tax.json")
        a.friendships = frp
        json.dump(taxonomy, open(a.taxonomy, "w"))

        run(a)
        tr = pd.read_csv(os.path.join(a.out_dir, "train_TEST.csv"))
        va = pd.read_csv(os.path.join(a.out_dir, "val_TEST.csv"))
        te = pd.read_csv(os.path.join(a.out_dir, "test_TEST.csv"))
        me = pd.read_csv(os.path.join(a.out_dir, "poi_metadata_TEST.csv"))
        fo = pd.read_csv(os.path.join(a.out_dir, "friendship_old_TEST.csv"))
        fn = pd.read_csv(os.path.join(a.out_dir, "friendship_new_only_TEST.csv"))
        all_ck = pd.concat([tr, va, te])

    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    print()
    res = [
        ok("all check-in rows preserved", len(all_ck) == len(ck)),
        ok("user_id and poi_idx are 0-based contiguous ranks",
           sorted(all_ck.user_id.unique()) == [0, 1, 2]
           and sorted(me.poi_idx) == [0, 1, 2]),
        ok("catalogue-covered venue has real lat/lon",
           me.loc[me.venue_id == "v_aaa", "latitude"].iloc[0] == 40.73),
        ok("catalogue-uncovered venue gets NaN lat/lon, not a crash",
           me.loc[me.venue_id == "v_ccc_nocat", "latitude"].isna().all()),
        ok("mapped category becomes the full taxonomy path",
           me.loc[me.venue_id == "v_aaa", "category"].iloc[0] == "Nightlife Spot>Bar>Wine Bar"),
        ok("unmapped category falls back to a flat depth-1 node",
           me.loc[me.venue_id == "v_ccc_nocat", "category"].iloc[0] == "Coffee Shop"),
        ok("locality is the city, region is the state",
           me.loc[me.venue_id == "v_aaa", "locality"].iloc[0] == "New York"
           and me.loc[me.venue_id == "v_aaa", "region"].iloc[0] == "NY"
           and me.loc[me.venue_id == "v_ccc_nocat", "region"].iloc[0] == "IL"),
        ok("split files carry the house columns",
           list(tr.columns) == ["user_id", "venue_id", "venue_category_id",
                                "venue_category_name", "latitude", "longitude",
                                "timezone_offset", "utc_time", "poi_idx"]),
        ok("no timestamp parse failures",
           tr["utc_time"].notna().all() and va["utc_time"].notna().all()
           and te["utc_time"].notna().all()),
        ok("friendship_old remapped to compact ids, edge to a non-checkin user dropped",
           len(fo) == 1 and (fo.iloc[0].u1, fo.iloc[0].u2) == (0, 1)),
        ok("friendship_new_only = new minus old",
           len(fn) == 1 and (fn.iloc[0].u1, fn.iloc[0].u2) == (0, 2)),
    ]
    return all(res)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkins", default="./data/llmgpr/llmgpr_final_checkins.csv")
    p.add_argument("--catalogue", default="./data/llmgpr/llmgpr_final_catalogue.csv")
    p.add_argument("--friendships", default="./data/llmgpr/llmgpr_final_friendships.csv",
                   help="tab-separated user_id, friend_id, source ('old'/'new'/'new,old'); "
                        "omit or point at a missing path to skip the social layer entirely")
    p.add_argument("--out-dir", default="./data/llmgpr")
    p.add_argument("--dataset", default=DS_DEFAULT)
    p.add_argument("--taxonomy", default="./data/lbsn/fsq_category_paths_2014.json")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        sys.exit(0 if _self_check() else 1)
    run(a)


if __name__ == "__main__":
    main()
