"""
House CSVs for the FULL Foursquare LLMGPR dataset, for group-rule matching.

The pipeline's Foursquare arm (data/llmgpr) holds 6,889 users / 423k check-ins, about a third of
LLMGPR's 1.21M, which caps every group count (GROUP_RULE_SWEEP.md §2.7). This converts the Kaggle
run `yosrkharrat/output42` (llmgpr-threshold-sweep) into the column layout `build_groups.
load_checkins` reads:

    llmgpr_final_checkins.parquet        7,849 users / 1,111,794 check-ins (Yang WWW'19 §5 raw,
                                         NY+CHI+LA; the Tp=3 build: 93% of output6's 1,191,781,
                                         whose outputs are no longer on Kaggle)
    llmgpr_final_friendship_old.parquet  3,555 edges (friendship_old, original user ids)

Download (Kaggle token in KAGGLE_API_TOKEN):
    kaggle kernels output yosrkharrat/output42 -p data/llmgpr_full/output42
Run:
    python src/prepare_llmgpr_full_csvs.py
Writes data/llmgpr_full/house/{train,val,test}_LLMGPR.csv (all rows in train; val/test empty, since
group matching uses every check-in) + friendship_old_LLMGPR.csv + id_map.csv.
"""

import os

import pandas as pd

SRC = "./data/llmgpr_full/output42"
OUT = "./data/llmgpr_full/house"


def main():
    ck = pd.read_parquet(os.path.join(SRC, "llmgpr_final_checkins.parquet"))
    fr = pd.read_parquet(os.path.join(SRC, "llmgpr_final_friendship_old.parquet"))
    users = sorted(ck["user_id"].astype(str).unique(), key=lambda x: int(x))
    uid = {u: i for i, u in enumerate(users)}
    venues = sorted(ck["venue_id"].unique())
    pid = {v: i for i, v in enumerate(venues)}

    t = pd.to_datetime(ck["utc_time"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    out = pd.DataFrame({
        "user_id": ck["user_id"].astype(str).map(uid).astype("int64"),
        "venue_id": ck["venue_id"],
        "venue_category_id": "",
        "venue_category_name": ck["category"],
        "latitude": float("nan"), "longitude": float("nan"),
        "timezone_offset": ck["tz_offset"].astype("int64"),
        "utc_time": t.dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "poi_idx": ck["venue_id"].map(pid).astype("int64"),
    })
    assert out["utc_time"].notna().all()

    f = pd.DataFrame({"u1": fr["u"].astype(str).map(uid), "u2": fr["v"].astype(str).map(uid)})
    dropped = int(f.isna().any(axis=1).sum())
    f = f.dropna().astype("int64")
    f = f[f.u1 != f.u2]
    f = pd.DataFrame({"u1": f[["u1", "u2"]].min(axis=1), "u2": f[["u1", "u2"]].max(axis=1)}).drop_duplicates()

    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "train_LLMGPR.csv"), index=False)
    out.head(0).to_csv(os.path.join(OUT, "val_LLMGPR.csv"), index=False)
    out.head(0).to_csv(os.path.join(OUT, "test_LLMGPR.csv"), index=False)
    f.to_csv(os.path.join(OUT, "friendship_old_LLMGPR.csv"), index=False)
    pd.DataFrame({"user_id": range(len(users)), "raw_id": users}).to_csv(
        os.path.join(OUT, "id_map.csv"), index=False)
    print(f"check-ins {len(out):,} | users {len(users):,} | POIs {len(venues):,} | friendship "
          f"{len(f):,} edges over {len(set(f.u1) | set(f.u2)):,} users "
          f"({dropped} edges dropped: endpoint outside the user set)")


if __name__ == "__main__":
    main()
