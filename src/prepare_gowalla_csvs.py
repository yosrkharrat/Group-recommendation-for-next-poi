"""
Stage 0 for the LLMGPR Gowalla track: `data/gowalla/gowalla_final_*` (from
`llmgpr-gowalla-recovery.ipynb`, see `LLMGPR_GOWALLA.md`) -> house-format CSVs.

The Gowalla twin of `prepare_llmgpr_csvs.py`: same output schema, so
`build_groups.py --group-source social`, `denoise_social_gbsr.py`, `build_kg_lbsn.py` and
`train_roth.py` run unchanged with `--dataset GOWALLA`. Three Gowalla-specific decisions,
each handled rather than papered over:

1. **The taxonomy is Gowalla's own**, not 2014-Foursquare, so `fsq_category_paths_2014.json`
   does not apply. `gowalla_category_structure.json` (7 mains / 134 depth-2 / 128 depth-3) is
   flattened into the same `{name: ["Main>Sub>Leaf", ...]}` format and ALSO written to
   `--out-dir/gowalla_category_paths.json` for `build_kg_lbsn.py --taxonomy`. Ambiguous names
   (3 of 269) resolve to the lexicographically first path, same rule as the FSQ track. Spots
   whose category is a legacy id absent from the structure keep their flat name as a depth-1
   node; spots with no category at all become "Unknown".

2. **The friendship graph is ONE crawl snapshot** — Gowalla has no `friendship_old`/`new`
   split, so the before/after leakage rule from the FSQ track cannot be applied. All edges are
   written to `friendship_old_GOWALLA.csv` (the filename every downstream stage hardcodes);
   `friendship_new_only_GOWALLA.csv` is written EMPTY so nothing can silently treat crawl-era
   edges as a held-out prediction set. Any temporal claim about these edges is unsupported —
   documented in `LLMGPR_GOWALLA.md`.

3. **Timestamps are UTC (ISO-8601 from the crawl); Gowalla ships no per-check-in timezone.**
   `timezone_offset` is filled with the city's STANDARD-time offset in minutes
   (NY -300, Chicago -360, LA -480), no DST. Only prompt formatting downstream reads it; every
   structural stage (splits, buckets, hour-of-week) uses `utc_time`.

Outputs (--out-dir, default ./data/gowalla)
--------------------------------------------
    train/val/test_GOWALLA.csv     user_id, venue_id, venue_category_id, venue_category_name,
                                   latitude, longitude, timezone_offset, utc_time, poi_idx
    poi_metadata_GOWALLA.csv       venue_id, name, category (taxonomy path), locality, region,
                                   description, poi_idx, latitude, longitude
    users_GOWALLA.csv              user_id -> raw crawl user id
    friendship_old_GOWALLA.csv     u1, u2 (0-based user_id, undirected, deduped)
    friendship_new_only_GOWALLA.csv  empty (see 2 above)
    gowalla_category_paths.json    {category name: [taxonomy paths]}
    gowalla_prep_manifest.json     every count and coverage measurement

Usage
-----
    python src/prepare_gowalla_csvs.py
    python src/prepare_gowalla_csvs.py --self-check
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
DS = "GOWALLA"
STATE_OF = {"New York": "NY", "Chicago": "IL", "Los Angeles": "CA"}
TZ_OFFSET_MIN = {"New York": -300, "Chicago": -360, "Los Angeles": -480}   # standard time


def flatten_structure(structure):
    """gowalla_category_structure.json -> {name: sorted [SEP-joined paths]}."""
    paths = {}

    def walk(nodes, prefix):
        for nd in nodes:
            name = (nd.get("name") or "").strip()
            if not name:
                continue
            path = prefix + [name]
            paths.setdefault(name, set()).add(SEP.join(path))
            walk(nd.get("spot_categories") or [], path)

    walk(structure.get("spot_categories") or [], [])
    return {k: sorted(v) for k, v in paths.items()}


def taxonomy_path(name, taxonomy):
    name = (name or "").strip()
    if not name:
        return "Unknown"
    if name in taxonomy:
        return taxonomy[name][0]
    return name                       # legacy category absent from the structure: depth-1 node


def remap_friend_edges(fr, uid_of):
    """Directed crawl rows -> undirected, deduped, compact-id edges; drops self-loops and
    edges touching users outside the check-in population."""
    u1 = fr["userid1"].map(uid_of)
    u2 = fr["userid2"].map(uid_of)
    keep = u1.notna() & u2.notna() & (u1 != u2)
    e = pd.DataFrame({"u1": u1[keep].astype(int), "u2": u2[keep].astype(int)})
    e = pd.DataFrame(np.sort(e.to_numpy(), axis=1), columns=["u1", "u2"])
    return e.drop_duplicates().sort_values(["u1", "u2"]).reset_index(drop=True)


def assemble(ck, cat, taxonomy, friends):
    raw_users = sorted(ck["userid"].unique())
    uid_of = {u: i for i, u in enumerate(raw_users)}
    raw_venues = sorted(ck["placeid"].unique())
    pid_of = {v: i for i, v in enumerate(raw_venues)}

    catv = cat.drop_duplicates("venue_id").set_index("venue_id")
    ll = catv[["lat", "lng"]].reindex(raw_venues)
    latlon_cov = float(ll["lat"].notna().mean())

    venue_row = ck.drop_duplicates("placeid").set_index("placeid")
    meta = pd.DataFrame({"venue_id": raw_venues})
    meta["poi_idx"] = np.arange(len(meta))
    meta["venue_category_name"] = venue_row.loc[raw_venues, "category"].fillna("").to_numpy()
    meta["venue_category_id"] = venue_row.loc[raw_venues, "category_id"].to_numpy()
    meta["category"] = [taxonomy_path(n, taxonomy) for n in meta["venue_category_name"]]
    meta["locality"] = venue_row.loc[raw_venues, "city"].to_numpy()
    meta["region"] = [STATE_OF.get(c, "Unknown") for c in meta["locality"]]
    meta["latitude"] = ll["lat"].to_numpy()
    meta["longitude"] = ll["lng"].to_numpy()
    meta["name"] = ""
    meta["description"] = ""

    df = ck.copy()
    df["user_id"] = df["userid"].map(uid_of)
    df["poi_idx"] = df["placeid"].map(pid_of)
    df["venue_id"] = df["placeid"]
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True, errors="coerce")
    n_bad = int(df["utc_time"].isna().sum())
    assert n_bad == 0, f"{n_bad} unparseable timestamps"
    df["timezone_offset"] = df["city"].map(TZ_OFFSET_MIN).astype("int64")
    df = df.rename(columns={"category": "venue_category_name"})
    df["venue_category_id"] = df["category_id"].fillna("").astype(str)
    df = df.merge(meta[["poi_idx", "latitude", "longitude"]], on="poi_idx", how="left")
    df = resplit_per_user(df, 0.70, 0.10)

    users = pd.DataFrame({"user_id": range(len(raw_users)), "raw_id": raw_users})
    friend_old = remap_friend_edges(friends, uid_of)
    return df, meta, users, friend_old, latlon_cov


def run(a):
    structure = json.load(open(a.structure, encoding="utf-8"))
    taxonomy = flatten_structure(structure)

    ck = pd.read_csv(a.checkins, dtype={"userid": "int64", "placeid": "int64",
                                        "category_id": str})
    cat = pd.read_csv(a.catalogue, dtype={"venue_id": "int64"})
    friends = pd.read_csv(a.friendships, dtype={"userid1": "int64", "userid2": "int64"})
    print(f"check-ins: {len(ck):,} rows, {ck.userid.nunique():,} users, "
          f"{ck.placeid.nunique():,} venues, cities {sorted(ck.city.unique())}")
    print(f"friendships: {len(friends):,} directed crawl rows "
          f"(single snapshot -- see module docstring)")

    df, meta, users, friend_old, latlon_cov = assemble(ck, cat, taxonomy, friends)
    assert latlon_cov > 0.999, f"catalogue must cover the R=15 venues (got {latlon_cov:.1%})"

    depth = Counter(len([p for p in c.split(SEP) if p.strip()]) for c in meta["category"])
    in_tree = float(np.mean([n in taxonomy for n in meta["venue_category_name"] if n]))
    print(f"lat/lon coverage {latlon_cov:.1%} | taxonomy depth {dict(sorted(depth.items()))} "
          f"| named categories resolving into the tree: {in_tree:.1%}")

    os.makedirs(a.out_dir, exist_ok=True)
    split_cols = ["user_id", "venue_id", "venue_category_id", "venue_category_name",
                  "latitude", "longitude", "timezone_offset", "utc_time", "poi_idx"]
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        part[split_cols].to_csv(os.path.join(a.out_dir, f"{split}_{DS}.csv"), index=False)
        print(f"  {split}_{DS}.csv  {len(part):,} rows")
    meta_cols = ["venue_id", "name", "category", "locality", "region", "description",
                 "poi_idx", "latitude", "longitude"]
    meta[meta_cols].to_csv(os.path.join(a.out_dir, f"poi_metadata_{DS}.csv"), index=False)
    users.to_csv(os.path.join(a.out_dir, f"users_{DS}.csv"), index=False)
    friend_old.to_csv(os.path.join(a.out_dir, f"friendship_old_{DS}.csv"), index=False)
    pd.DataFrame(columns=["u1", "u2"]).to_csv(
        os.path.join(a.out_dir, f"friendship_new_only_{DS}.csv"), index=False)
    with open(os.path.join(a.out_dir, "gowalla_category_paths.json"), "w") as f:
        json.dump(taxonomy, f, indent=1, sort_keys=True)
    print(f"  poi_metadata_{DS}.csv  {len(meta):,} POIs | friendship_old_{DS}.csv  "
          f"{len(friend_old):,} undirected edges | friendship_new_only empty (single snapshot)")

    manifest = dict(
        dataset=DS,
        source="LLMGPR_GOWALLA.md adopted build (R=15 collection, day-dedup, no user filter)",
        n_checkins=int(len(df)), n_users=int(len(users)), n_pois=int(len(meta)),
        n_friend_edges_undirected=int(len(friend_old)),
        friendship_semantics="single crawl snapshot; 'old' filename kept for pipeline compat; "
                             "no before/after split exists for Gowalla",
        timezone_offset="city standard time, minutes, no DST (UTC times are authoritative)",
        latlon_coverage=latlon_cov,
        taxonomy_depth_histogram={int(k): int(v) for k, v in depth.items()},
        named_category_tree_coverage=in_tree,
        splits={s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
    )
    with open(os.path.join(a.out_dir, "gowalla_prep_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("  gowalla_prep_manifest.json written")


def self_check():
    print("SELF-CHECK on a synthetic fixture")
    taxonomy = {"Coffee Shop": ["Food>Coffee Shop"], "BBQ": ["Food>BBQ"]}
    assert taxonomy_path("Coffee Shop", taxonomy) == "Food>Coffee Shop"
    assert taxonomy_path("Mystery Legacy", taxonomy) == "Mystery Legacy"
    assert taxonomy_path("", taxonomy) == "Unknown"
    assert taxonomy_path(None, taxonomy) == "Unknown"

    structure = {"spot_categories": [
        {"name": "Food", "spot_categories": [
            {"name": "BBQ"}, {"name": "Coffee Shop", "spot_categories": [{"name": "Espresso"}]}]}]}
    t = flatten_structure(structure)
    assert t["Espresso"] == ["Food>Coffee Shop>Espresso"], t
    assert t["Food"] == ["Food"]

    fr = pd.DataFrame({"userid1": [10, 20, 10, 30, 40], "userid2": [20, 10, 10, 40, 30]})
    uid_of = {10: 0, 20: 1, 30: 2}          # 40 not in the check-in population
    e = remap_friend_edges(fr, uid_of)
    assert e.to_numpy().tolist() == [[0, 1]], e   # both directions collapse; self-loop + 30-40 drop

    ck = pd.DataFrame({
        "userid": [10, 10, 10, 20],
        "placeid": [5, 6, 5, 6],
        "utc_time": ["2010-01-01T10:00:00Z", "2010-01-02T10:00:00Z",
                     "2010-01-03T10:00:00Z", "2010-01-01T09:00:00Z"],
        "city": ["New York"] * 4,
        "category_id": ["1", "2", "1", "2"],
        "category": ["Coffee Shop", "BBQ", "Coffee Shop", "BBQ"],
        "km": [1.0, 2.0, 1.0, 2.0]})
    cat = pd.DataFrame({"venue_id": [5, 6], "lat": [40.7, 40.8], "lng": [-74.0, -73.9]})
    df, meta, users, friend_old, cov = assemble(ck, cat, taxonomy,
                                                pd.DataFrame({"userid1": [10], "userid2": [20]}))
    assert cov == 1.0
    assert list(meta["category"]) == ["Food>Coffee Shop", "Food>BBQ"]
    assert set(df["timezone_offset"]) == {-300}
    u10 = df[df["user_id"] == 0].sort_values("utc_time")
    # resplit_per_user: n=3, ceil(3*0.7)=3 -> all train (tiny users hold no val/test rows)
    assert list(u10["split"]) == ["train", "train", "train"], list(u10["split"])
    u20 = df[df["user_id"] == 1]
    assert list(u20["split"]) == ["train"]
    print("self-check OK")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkins", default="./data/gowalla/gowalla_final_checkins.csv.gz")
    p.add_argument("--catalogue", default="./data/gowalla/gowalla_final_catalogue.csv")
    p.add_argument("--friendships", default="./data/gowalla/gowalla_final_friendships.csv")
    p.add_argument("--structure", default="./data/gowalla/gowalla_category_structure.json")
    p.add_argument("--out-dir", default="./data/gowalla")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        sys.exit(0 if self_check() else 1)
    run(a)


if __name__ == "__main__":
    main()
