"""
Extend the Foursquare (LLMGPR) POI catalogue so it covers the three-city reproduction groups.

Why (2026-11-19, option B chosen by the user)
---------------------------------------------
The pipeline's catalogue (data/llmgpr/poi_metadata_LLMGPR.csv, 14,402 POIs) holds the venues the
users checked into within 10 km of the three city centres. The LLMGPR-matched groups use more
check-ins than that. On whole histories only 46% of their test targets were in the catalogue (75%
of the missing venues were outside NY/LA/Chicago), so the model could not even output most answers.
The groups are now rebuilt on three-city check-ins (data/groups_llmgpr_match/LLMGPR_3CITY,
`build_llmgpr_match_groups.py --datasets LLMGPR_3CITY`). This script then:

1. **Appends every venue those groups use that the catalogue lacks** (913 on the 2026-11-19
   build). New ids are contiguous after the old ones, so every existing poi_idx, split CSV and
   embedding row keeps its meaning.
2. **Fills coordinates for the existing POIs that had none.** Only 43% had any: the old
   catalogue took them from a 10 km statistic file. The raw Foursquare venue file (via
   data/llmgpr_full/house8_whole) has them for 100% of catalogue venues, and where both exist they
   agree exactly (max |diff| = 0). Coordinates are static venue metadata, with no split dependence,
   so this cannot leak. It lets IS_NEAR_TO and LLMGPR's nearest-candidate protocol cover every POI.

New rows follow `prepare_llmgpr_csvs.py`'s conventions exactly:
- `category` is the full 2014 taxonomy path (`taxonomy_path`);
- `locality` / `region` is the city / state of the city box the venue lies in;
- `name` / `description` are empty, as for every existing row.

Inputs never change: the train/val/test split CSVs (the new venues have no individual check-ins
in the 10 km pipeline data; they enter the model through the KG and the group sequences).

    python src/extend_llmgpr_catalogue.py            # rewrites data/llmgpr/poi_metadata_LLMGPR.csv
    python src/extend_llmgpr_catalogue.py --dry-run  # report only

Writes data/llmgpr/catalogue_extension_LLMGPR.json (counts, coverage, inputs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_llmgpr_csvs import STATE_OF, taxonomy_path  # noqa: E402

BBOX = {"New York": (-74.3, -73.6, 40.4, 41.0), "Chicago": (-88.0, -87.5, 41.6, 42.1),
        "Los Angeles": (-118.7, -117.6, 33.6, 34.4)}     # = prepare_llmgpr_fsq_output8_full.BBOX


def city_of(lat, lon):
    for city, (x0, x1, y0, y1) in BBOX.items():
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            return city
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="./data/llmgpr/poi_metadata_LLMGPR.csv")
    ap.add_argument("--venues", default="./data/llmgpr_full/house8_whole/train_LLMGPR.csv",
                    help="check-in table with venue_id, venue_category_name, latitude, longitude")
    ap.add_argument("--groups", default="./data/groups_llmgpr_match/LLMGPR_3CITY/group_checkins.csv")
    ap.add_argument("--taxonomy", default="./data/lbsn/fsq_category_paths_2014.json")
    ap.add_argument("--report", default="./data/llmgpr/catalogue_extension_LLMGPR.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    meta = pd.read_csv(a.meta)
    n_old = len(meta)
    assert (meta["poi_idx"].to_numpy() == np.arange(n_old)).all(), "catalogue poi_idx not 0..N-1"
    assert "__extended__" not in meta.columns
    known = set(meta["venue_id"])
    ven = (pd.read_csv(a.venues, usecols=["venue_id", "venue_category_name", "latitude", "longitude"])
           .drop_duplicates("venue_id").set_index("venue_id"))
    with open(a.taxonomy) as f:
        taxonomy = json.load(f)

    g = pd.read_csv(a.groups)
    g = g[g["split"] != "masked"]                      # masked rows are dropped from training anyway
    need = sorted(set(g["venue_id"]) - known)
    miss_src = [v for v in need if v not in ven.index]
    assert not miss_src, f"{len(miss_src)} group venues have no venue record"

    # 1. new rows
    new = ven.loc[need].reset_index()
    new["city"] = [city_of(la, lo) for la, lo in zip(new["latitude"], new["longitude"])]
    assert new["city"].notna().all(), "a three-city group venue lies outside the three city boxes"
    new_rows = pd.DataFrame(dict(
        venue_id=new["venue_id"], name=np.nan,
        category=[taxonomy_path(c, taxonomy) for c in new["venue_category_name"]],
        locality=new["city"], region=new["city"].map(STATE_OF), description=np.nan,
        poi_idx=np.arange(n_old, n_old + len(new)),          # contiguous after the old ids
        latitude=new["latitude"].to_numpy(), longitude=new["longitude"].to_numpy()))
    tax_hit = float(np.mean([c in taxonomy for c in new["venue_category_name"]])) if len(new) else 1.0

    # 2. fill the old rows' missing coordinates
    had = meta["latitude"].notna()
    src = ven.reindex(meta["venue_id"])
    both = had.to_numpy() & src["latitude"].notna().to_numpy()
    max_diff = float(np.nanmax(np.abs(meta.loc[both, "latitude"].to_numpy()
                                      - src["latitude"].to_numpy()[both]))) if both.any() else 0.0
    assert max_diff < 1e-6, f"existing coordinates disagree with the venue file ({max_diff})"
    filled = meta.copy()
    filled["latitude"] = filled["latitude"].fillna(pd.Series(src["latitude"].to_numpy(), index=meta.index))
    filled["longitude"] = filled["longitude"].fillna(pd.Series(src["longitude"].to_numpy(), index=meta.index))

    out = pd.concat([filled, new_rows[filled.columns]], ignore_index=True)
    assert (out["poi_idx"].to_numpy() == np.arange(len(out))).all()
    assert out["venue_id"].is_unique
    gi = g["venue_id"].isin(set(out["venue_id"]))
    assert gi.all(), "a group venue is still missing"

    test = g[g["split"] == "test"]
    rep = dict(
        inputs=dict(meta=a.meta, venues=a.venues, groups=a.groups, taxonomy=a.taxonomy),
        old_pois=n_old, added_pois=len(new_rows), new_pois=len(out),
        added_by_city=new_rows["locality"].value_counts().to_dict(),
        added_category_in_taxonomy=round(tax_hit, 4),
        coords_before=round(float(had.mean()), 4), coords_after=round(float(out["latitude"].notna().mean()), 4),
        coords_filled_for_old_pois=int(out["latitude"].notna().sum() - had.sum() - len(new_rows)),
        group_test_targets_in_catalogue_before=round(float(test["venue_id"].isin(known).mean()), 4),
        group_test_targets_in_catalogue_after=1.0,
        note="new poi_idx = old N + rank of venue_id among the added venues; old ids unchanged")
    print(json.dumps(rep, indent=1))
    if a.dry_run:
        return
    out.to_csv(a.meta, index=False)
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"wrote {a.meta} ({len(out):,} POIs) + {a.report}")


if __name__ == "__main__":
    main()
