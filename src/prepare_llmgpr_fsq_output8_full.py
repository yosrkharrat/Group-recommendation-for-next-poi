"""
Foursquare for group matching, on the PIPELINE's user set (Kaggle `output8`) with all their
three-city check-ins.

Why (GROUP_RULE_SWEEP.md §2.8): our two Foursquare builds have different users (3,036 shared).
`output42` (7,849 users, the 0.978 Table-1 fit) has friendship for only 28% of users, so it can
never reach LLMGPR's group numbers. `output8` (6,889 users, 0.938 fit, the pipeline's set) has
71%. The pipeline kept only check-ins within 10 km of the city centres (423k). This rebuilds the
same 6,889 users with every check-in inside the three city bounding boxes.

Inputs (Kaggle `yosrkharrat/output8` outputs, downloaded to data/llmgpr_full/output8/):
    dataset_WWW2019/dataset_WWW_Checkins_anonymized.txt   user, venue, utc time, tz offset
    dataset_WWW2019/raw_POIs.txt                           venue, lat, lon, category, country
    dataset_WWW2019/dataset_WWW_friendship_old.txt         global friendship_old
    llmgpr_final_users.csv                                 the 6,889 users
plus data/llmgpr/friendship_new_only_LLMGPR.csv (pipeline ids = rank of raw id) for `new`.

    python src/prepare_llmgpr_fsq_output8_full.py [--scope whole]
Writes data/llmgpr_full/house8[_whole]/{train,val,test}_LLMGPR.csv, friendship_old_LLMGPR.csv,
friendship_new_only_LLMGPR.csv, id_map.csv.
"""

import argparse
import os

import pandas as pd

SRC = "./data/llmgpr_full/output8"
RAW = os.path.join(SRC, "dataset_WWW2019")
OUT = "./data/llmgpr_full/house8"
BBOX = {"New York": (-74.3, -73.6, 40.4, 41.0), "Chicago": (-88.0, -87.5, 41.6, 42.1),
        "Los Angeles": (-118.7, -117.6, 33.6, 34.4)}          # LLMGPR_TRACK.md, do not widen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["bbox", "whole"], default="bbox",
                    help="bbox: check-ins inside the three city boxes; whole: every check-in of "
                         "these users, anywhere (LLMGPR's 1.21M is closer to whole-history volume)")
    a = ap.parse_args()
    out_dir = OUT if a.scope == "bbox" else OUT + "_whole"
    users = pd.read_csv(os.path.join(SRC, "llmgpr_final_users.csv"))["user_id"].astype("int64")
    raw_ids = sorted(users.unique())
    uid = {u: i for i, u in enumerate(raw_ids)}          # same rank map as prepare_llmgpr_csvs
    keep = set(raw_ids)

    parts = []
    for ch in pd.read_csv(os.path.join(RAW, "dataset_WWW_Checkins_anonymized.txt"), sep="\t",
                          header=None, names=["user", "venue", "utc", "tz"], chunksize=2_000_000,
                          dtype={"user": "int64", "venue": str, "utc": str, "tz": "int64"}):
        parts.append(ch[ch["user"].isin(keep)])
    ck = pd.concat(parts, ignore_index=True)
    venues = set(ck["venue"])
    print(f"check-ins of the {len(keep):,} users, whole history: {len(ck):,}", flush=True)

    vp = []
    for ch in pd.read_csv(os.path.join(RAW, "raw_POIs.txt"), sep="\t", header=None,
                          names=["venue", "lat", "lon", "cat", "cc"], chunksize=2_000_000,
                          dtype={"venue": str, "lat": float, "lon": float, "cat": str, "cc": str}):
        vp.append(ch[ch["venue"].isin(venues)])
    poi = pd.concat(vp, ignore_index=True).drop_duplicates("venue")
    poi["city"] = None
    for c, (lo1, lo2, la1, la2) in BBOX.items():
        m = poi.lon.between(lo1, lo2) & poi.lat.between(la1, la2)
        poi.loc[m, "city"] = c
    if a.scope == "bbox":
        poi = poi[poi["city"].notna()]
    ck = ck.merge(poi, on="venue", how="inner" if a.scope == "bbox" else "left")
    print(f"scope={a.scope}: {len(ck):,} check-ins, {ck.user.nunique():,} users, "
          f"{ck.venue.nunique():,} venues", flush=True)

    t = pd.to_datetime(ck["utc"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    pid = {v: i for i, v in enumerate(sorted(ck["venue"].unique()))}
    out = pd.DataFrame({"user_id": ck["user"].map(uid).astype("int64"), "venue_id": ck["venue"],
                        "venue_category_id": "", "venue_category_name": ck["cat"],
                        "latitude": ck["lat"], "longitude": ck["lon"],
                        "timezone_offset": ck["tz"],
                        "utc_time": t.dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                        "poi_idx": ck["venue"].map(pid).astype("int64")})

    g = pd.read_csv(os.path.join(RAW, "dataset_WWW_friendship_old.txt"), sep="\t", header=None,
                    names=["u", "v"], dtype="int64")
    g = g[g.u.isin(keep) & g.v.isin(keep) & (g.u != g.v)]
    old = pd.DataFrame({"u1": g[["u", "v"]].min(axis=1).map(uid),
                        "u2": g[["u", "v"]].max(axis=1).map(uid)}).drop_duplicates()
    # consistency check: the pipeline's own old file must be this graph in the same id space
    pipe_old = pd.read_csv("./data/llmgpr/friendship_old_LLMGPR.csv")
    a = set(map(tuple, old.to_numpy()))
    b = set(map(tuple, pd.DataFrame({"u1": pipe_old.min(axis=1), "u2": pipe_old.max(axis=1)}).to_numpy()))
    print(f"friendship_old: {len(a):,} edges here vs {len(b):,} in the pipeline; "
          f"shared {len(a & b):,}  -> id map {'OK' if len(a & b) >= 0.99 * len(b) else 'MISMATCH'}")
    new_only = pd.read_csv("./data/llmgpr/friendship_new_only_LLMGPR.csv")

    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "train_LLMGPR.csv"), index=False)
    out.head(0).to_csv(os.path.join(out_dir, "val_LLMGPR.csv"), index=False)
    out.head(0).to_csv(os.path.join(out_dir, "test_LLMGPR.csv"), index=False)
    old.to_csv(os.path.join(out_dir, "friendship_old_LLMGPR.csv"), index=False)
    new_only.to_csv(os.path.join(out_dir, "friendship_new_only_LLMGPR.csv"), index=False)
    pd.DataFrame({"user_id": range(len(raw_ids)), "raw_id": raw_ids}).to_csv(
        os.path.join(out_dir, "id_map.csv"), index=False)
    print(f"wrote {out_dir}: {len(out):,} check-ins, {out.user_id.nunique():,} users; "
          f"old {len(old):,} + new-only {len(new_only):,} friendship edges")


if __name__ == "__main__":
    main()
