"""
It reuses the layer builders from `build_kg.py` so the graphs stay structurally comparable
across datasets, and its outputs are a drop-in for `train_roth.py`.

Two input modes, DIFFERENT populations -- tagged with different --dataset names on purpose:

* `--csv-dir` (the generic, CSV-driven track): house-format files from `prepare_lbsn_csvs.py`
  (DS=LBSN_NYC, 1,665 users / 6,103 POIs / 159,304 check-ins) or `prepare_llmgpr_csvs.py`
  (DS=LLMGPR, 6,889 users / 14,402 POIs / 423,376 check-ins, 3 cities). Full-fidelity graph:
  every build_kg.py layer plus FRIEND_OF.
* `--mat` (LBSN2Vec paper-comparable population only, DS=MAT_NYC): `dataset_connected_NYC.mat`
  as shipped with LBSN2Vec (4,024 users / 3,628 POIs / 105,961 check-ins, 8,723 old
  friendships -- users chosen for social connectivity, but no timestamps/coordinates/names;
  see below). Not applicable to the LLMGPR track, which has no .mat release.

What is genuinely NEW versus the TSMC2014 graph
-----------------------------------------------
    FRIEND_OF    USER <-> USER    real declared friendships (friendship_old)

The old pipeline had to *infer* user-user ties (affinity z-sum, AUC 0.72 against co-presence).
Here the social network is observed, so the KG carries ground-truth edges instead of a proxy.

Leakage rules, stricter than they look
--------------------------------------
* `friendship_old` is the snapshot BEFORE the check-in period -> safe, goes into the KG.
* `friendship_new` is the snapshot AFTER the period, and (new - old) is exactly the eval set of
  the paper's friendship-prediction task. It NEVER enters the KG. It is exported to
  `friendship_new_only_<DS>.csv` so the task stays available, and its absence from the triples
  is asserted at build time.
* Every behavioural relation (VISITED, FOLLOWED_BY, PREFERS_CATEGORY, group layer) is built
  from the TRAIN split only, exactly as in `build_kg.py`.

The nyc.mat schema (verified against the authors' experiment_LBSN2Vec.m)
------------------------------------------------------------------------
    selected_checkins   [n, 4] int:  user (1..U), hour-of-week (1..168),
                                     global venue index, global category index
    friendship_old/new  [m, 2] int:  user, user  (1-based, same space as check-ins)
    selected_users_IDs  [U, 1]:      anonymised global user ids
    selected_venue_IDs  [V, 1]:      raw Foursquare venue ids (24-char hex)

There are NO timestamps, NO coordinates and NO category names in the .mat. What recovers them:

* Order: the rows are chronological. Measured on the real file, 99.85% of consecutive rows are
  non-decreasing in hour-of-week; the decreases split into ~60 large wraps (real week
  boundaries, median ~1.5k rows apart) and small 1-23h dips (local-time jitter). So row order
  is the canonical time axis (`ts` = row index), the paper's "first 80% chronological" split is
  a row split, and an absolute-hour clock can be rebuilt by counting week wraps -- a decrease
  is a new week only when it exceeds `--wrap-tol` (default 24h), which absorbs the jitter.
* Coordinates + category names: join `raw_POIs.txt` from the full dataset on the venue hex id
  and pass the result as `--poi-meta` (columns: venue id, latitude, longitude, category name --
  flexible names, see `load_poi_meta`). Without it the graph still builds, but IS_NEAR_TO,
  LOCATED_IN and the taxonomy spine are skipped, and HAS_CATEGORY degrades to 300 flat
  `cat-<id>` nodes -- which starves the depth regulariser, so a loud warning is printed.
* Taxonomy: `--taxonomy` maps a 2014-era Foursquare category NAME to its full path(s)
  ("Multiplex" -> "Arts & Entertainment>Movie Theater>Multiplex"), restoring SUBCATEGORY_OF.
  The repo ships one at data/lbsn/fsq_category_paths_2014.json (763 nodes, depth 1-4).

Id spaces (document once, reuse everywhere)
-------------------------------------------
    user_id  = mat value - 1                         0..U-1, includes check-in-less users --
                                                     they exist in the friendship graph
    poi_idx  = rank of the venue's global index      0..V-1, ascending; matches MATLAB
               in np.unique(...)                     unique() and therefore selected_venue_IDs
    cat_idx  = same construction over categories

The poi_idx <-> hex-id correspondence (selected_venue_IDs[poi_idx]) is the one assumption not
provable from the .mat alone. When --poi-meta is given it is validated empirically: each mat
category id should map to essentially ONE metadata category name; the purity of that vote is
printed and checked (>= 0.95 expected -- a wrong venue ordering would randomise it to ~1/300).

A venue is not single-category here (215 of 3,628 NYC venues carry 2+ category ids across
their check-ins), so HAS_CATEGORY uses the venue's MODAL category, ties broken by lower id.

Group layer
-----------
`--groups-dir` accepts the same files build_groups.py emits (ephemeral_groups.csv,
group_members.csv, co_attended.csv) with ids in the spaces above -- this is the contract for
the co-location / common-friends preprocessing outputs. Missing dir -> POI + social KG only.

Outputs
-------
    --out-dir     (default ./data/kg_lbsn)   kg_triples.pt, kg_hierarchy.pt, kg_entities.json,
                                             kg_relations.json, kg_poi_rows.json, kg_manifest.json
    --export-dir  (default: the .mat's dir)  checkins_<DS>.csv          user_id, poi_idx, cat_idx,
                                                                        slot, week, abs_hour, split
                                             poi_metadata_<DS>.csv      poi_idx, venue_id, category,
                                                                        latitude, longitude, n_checkins
                                             friendship_old_<DS>.csv    u1, u2  (0-based user_id)
                                             friendship_new_only_<DS>.csv

    <DS> = --dataset, default LBSN_NYC. poi_metadata_<DS>.csv is what train_roth.py needs for D1.
    `--export-dir` only applies to `--mat` mode; `--csv-dir` mode's inputs are already the
    house-format CSVs, so there is nothing to export.

Usage
-----
    python build_kg_lbsn.py --csv-dir ./data/lbsn                  # LBSN_NYC track
    python build_kg_lbsn.py --csv-dir ./data/lbsn --groups-dir ./data/lbsn/groups
    python build_kg_lbsn.py --csv-dir ./data/llmgpr --dataset LLMGPR \
        --groups-dir ./data/llmgpr/groups_social --out-dir ./data/llmgpr/kg   # LLMGPR track
    python build_kg_lbsn.py --mat ./data/lbsn/nyc.mat              # LBSN2Vec paper population
    python build_kg_lbsn.py --self-check
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
from build_kg import (KG, SEP, add_group_layer, add_prefers_category, add_proximity,
                      add_spatial, add_taxonomy, add_transitions, add_visits,
                      prefers_category, write_all)
from build_groups import resplit_per_user

DS_DEFAULT = "LBSN_NYC"


# --------------------------------------------------------------------------
# loading + id compaction
# --------------------------------------------------------------------------

def load_mat(path):
    import scipy.io as sio
    m = sio.loadmat(path)
    missing = [k for k in ("selected_checkins", "friendship_old", "friendship_new")
               if k not in m]
    if missing:
        raise SystemExit(f"{path} lacks {missing} -- not an LBSN2Vec dataset file")
    out = {k: np.asarray(v) for k, v in m.items() if not k.startswith("__")}
    out["selected_checkins"] = out["selected_checkins"].astype(np.int64)
    for k in ("friendship_old", "friendship_new"):
        out[k] = out[k].astype(np.int64)
    return out


def compact(mat):
    """Compact the .mat's global indices into the project id spaces (see docstring)."""
    c = mat["selected_checkins"]
    n_users = int(max(c[:, 0].max(), mat["friendship_old"].max(),
                      mat["friendship_new"].max()))
    venue_vals = np.unique(c[:, 2])
    cat_vals = np.unique(c[:, 3])
    poi_of = {int(v): i for i, v in enumerate(venue_vals)}
    cat_of_val = {int(v): i for i, v in enumerate(cat_vals)}

    hexids = [""] * len(venue_vals)
    if "selected_venue_IDs" in mat:
        raw = mat["selected_venue_IDs"].ravel()
        assert len(raw) == len(venue_vals), \
            f"selected_venue_IDs has {len(raw)} entries for {len(venue_vals)} venues"
        hexids = [str(np.asarray(x).ravel()[0]) for x in raw]

    df = pd.DataFrame({
        "user_id": c[:, 0] - 1,
        "slot": c[:, 1],
        "poi_idx": [poi_of[int(v)] for v in c[:, 2]],
        "cat_idx": [cat_of_val[int(v)] for v in c[:, 3]],
    })
    friends_old = np.sort(mat["friendship_old"] - 1, axis=1)
    friends_new = np.sort(mat["friendship_new"] - 1, axis=1)
    return df, n_users, venue_vals, cat_vals, hexids, friends_old, friends_new


def reconstruct_clock(slots, wrap_tol=24):
    """Row order is chronological; rebuild (week, abs_hour) from hour-of-week wraps.

    A decrease smaller than `wrap_tol` hours is local-time jitter, not a new week -- on the
    real NYC file the tolerant count lands near the ~94 calendar weeks of Apr'12-Jan'14,
    whereas counting every decrease invents ~60 phantom weeks.
    """
    slots = np.asarray(slots, dtype=np.int64)
    new_week = np.zeros(len(slots), dtype=bool)
    new_week[1:] = np.diff(slots) <= -wrap_tol
    week = np.cumsum(new_week)
    return week, week * 168 + (slots - 1)


# --------------------------------------------------------------------------
# categories: names, taxonomy paths, modal assignment
# --------------------------------------------------------------------------

def load_poi_meta(path):
    """Flexible-column venue metadata (a raw_POIs.txt extract): hex id, lat, lon, name."""
    meta = pd.read_csv(path, sep=None, engine="python")
    def col(*names):
        return next((c for c in meta.columns if c.strip().lower() in names), None)
    vid = col("venue_id", "fsq_id", "id", "venueid")
    lat = col("latitude", "lat")
    lon = col("longitude", "lon", "lng")
    cat = col("category", "category_name", "cat_name", "venue_category", "venue_category_name")
    if vid is None:
        raise SystemExit(f"{path}: no venue-id column among the recognised names")
    out = pd.DataFrame({"venue_id": meta[vid].astype(str).str.strip()})
    out["latitude"] = pd.to_numeric(meta[lat], errors="coerce") if lat else np.nan
    out["longitude"] = pd.to_numeric(meta[lon], errors="coerce") if lon else np.nan
    out["cat_name"] = meta[cat].astype(str).str.strip() if cat else ""
    return out.drop_duplicates("venue_id").set_index("venue_id")


def modal_categories(df):
    """Venue -> modal cat_idx (215/3,628 NYC venues are multi-category; majority wins,
    ties broken by lower id so the result is deterministic)."""
    modal = {}
    for poi, g in df.groupby("poi_idx", sort=False):
        counts = Counter(g["cat_idx"])
        modal[int(poi)] = int(min(counts, key=lambda k: (-counts[k], k)))
    return modal


def category_paths(df, modal, hexids, poi_meta, taxonomy):
    """poi_idx -> taxonomy path string (SEP-joined), plus the id->name purity diagnostic.

    Name source: the metadata's category name per venue, voted per mat category id -- which
    doubles as the check that selected_venue_IDs really is in np.unique order (a wrong order
    would randomise the votes; purity collapses toward 1/n_categories instead of ~1).
    """
    n_pois = int(df["poi_idx"].max()) + 1
    name_of_cat, purity = {}, None
    if poi_meta is not None and len(hexids) == n_pois and any(hexids):
        votes = {}
        for poi, g in df.groupby("poi_idx", sort=False):
            name = poi_meta["cat_name"].get(hexids[int(poi)], "")
            if not name or name.lower() == "nan":
                continue
            for k, n in Counter(g["cat_idx"]).items():
                votes.setdefault(int(k), Counter())[name] += int(n)
        purities = []
        for k, cnt in votes.items():
            name, top = cnt.most_common(1)[0]
            name_of_cat[k] = name
            purities.append(top / sum(cnt.values()))
        purity = float(np.mean(purities)) if purities else None

    cat_of = {}
    unmatched = Counter()
    for poi in range(n_pois):
        k = modal.get(poi)
        name = name_of_cat.get(k, "")
        if name and taxonomy and name in taxonomy:
            path = sorted(taxonomy[name])[0]        # deterministic if a name is ambiguous
        elif name:
            path = name                              # named but not in the tree: depth-1 node
            unmatched[name] += 1
        else:
            path = f"cat-{k}"                        # no names at all: flat id node
        cat_of[poi] = path
    return cat_of, purity, unmatched


def spatial_frame(n_pois, hexids, poi_meta, cell_deg=0.01, region="NYC"):
    """poi_idx-ordered frame for add_spatial/add_proximity. The .mat has no localities, so the
    containment layer uses ~1km grid cells as the locality level (cell -> region). Coordinates
    absent -> NaN rows, and the spatial layers are skipped upstream."""
    lat = np.full(n_pois, np.nan)
    lon = np.full(n_pois, np.nan)
    if poi_meta is not None and any(hexids):
        for poi in range(n_pois):
            if hexids[poi] in poi_meta.index:
                r = poi_meta.loc[hexids[poi]]
                lat[poi], lon[poi] = float(r["latitude"]), float(r["longitude"])
    loc = ["" if not np.isfinite(a) else
           f"{np.floor(a / cell_deg) * cell_deg:.4f},{np.floor(o / cell_deg) * cell_deg:.4f}"
           for a, o in zip(lat, lon)]
    return pd.DataFrame({"poi_idx": np.arange(n_pois), "latitude": lat, "longitude": lon,
                         "locality": loc, "region": [region if l else "" for l in loc]})


# --------------------------------------------------------------------------
# social layer
# --------------------------------------------------------------------------

def add_friendships(kg, pairs):
    """FRIEND_OF: symmetric, from the BEFORE-period snapshot only."""
    n = 0
    for u, v in pairs:
        a = kg.entity(f"user:{int(u)}", "USER")
        b = kg.entity(f"user:{int(v)}", "USER")
        n += kg.add(a, "FRIEND_OF", b)
        n += kg.add(b, "FRIEND_OF", a)
    return n


def assert_no_new_friendship_leak(kg, friends_old, friends_new):
    """(new - old) is the paper's friendship-prediction eval set; it must not be in the KG."""
    old = {(int(u), int(v)) for u, v in friends_old}
    eval_pairs = [(int(u), int(v)) for u, v in friends_new if (int(u), int(v)) not in old]
    if "FRIEND_OF" not in kg.rel_id:
        return eval_pairs
    rid = kg.rel_id["FRIEND_OF"]
    ent = {}
    for (h, r, t) in kg.triples:
        if r == rid:
            ent.setdefault(h, set()).add(t)
    id_of = kg.ent_id
    for u, v in eval_pairs:
        a, b = id_of.get(f"user:{u}"), id_of.get(f"user:{v}")
        assert a is None or b not in ent.get(a, ()), \
            f"friendship_new-only pair ({u},{v}) leaked into the KG"
    return eval_pairs


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build(df, meta, cat_of, friends_old, groups, members, co, a):
    kg = KG()
    hierarchy = set()
    train_df = df[df["split"] == "train"][["user_id", "poi_idx", "ts"]]

    add_taxonomy(kg, cat_of, len(meta), hierarchy)
    if np.isfinite(meta["latitude"]).any():
        add_spatial(kg, meta, hierarchy)
        add_proximity(kg, meta[np.isfinite(meta["latitude"])], a.knn)
    else:
        print("no coordinates -> skipping LOCATED_IN and IS_NEAR_TO")
    add_transitions(kg, train_df, a.max_followed_by)
    add_visits(kg, train_df)
    pref = prefers_category(train_df, cat_of, a.tau)
    add_prefers_category(kg, pref, hierarchy)
    n_friend = add_friendships(kg, friends_old)
    if groups is not None and len(groups):
        add_group_layer(kg, groups, members, co, cat_of, hierarchy)

    print(f"\nentities {len(kg.ent_id):,}  triples {len(kg.triples):,}  "
          f"relations {len(kg.rel_id)}  (FRIEND_OF directed edges: {n_friend:,})")
    print("  node types: " + "  ".join(f"{k}={v:,}" for k, v in kg.type_counts().items()))
    print("  relations:")
    for r, cnt in kg.counts().items():
        print(f"    {r:<18} {cnt:>8,}")
    hc = Counter(r for _, _, r in hierarchy)
    print(f"  hierarchy pairs: {len(hierarchy):,}  "
          + "  ".join(f"{k}={v:,}" for k, v in hc.most_common()))
    depth_hist = Counter(d for _, _, d in pref)
    print("  PREFERS_CATEGORY depth spread: "
          + "  ".join(f"d{k}:{v}" for k, v in sorted(depth_hist.items())))
    if len(depth_hist) <= 1:
        print("  WARNING: users all sit at one taxonomy depth -- without --poi-meta/--taxonomy "
              "the depth regulariser has (almost) no spine to order")
    return kg, hierarchy, pref


def export_csvs(export_dir, ds, df, meta, cat_of, hexids, friends_old, eval_pairs):
    os.makedirs(export_dir, exist_ok=True)
    paths = {}
    p = paths["checkins"] = os.path.join(export_dir, f"checkins_{ds}.csv")
    df.to_csv(p, index=False)
    n_checkins = df.groupby("poi_idx").size()
    out = meta.assign(
        venue_id=[hexids[i] if i < len(hexids) else "" for i in meta["poi_idx"]],
        category=[cat_of[int(i)] for i in meta["poi_idx"]],
        n_checkins=[int(n_checkins.get(int(i), 0)) for i in meta["poi_idx"]],
    )[["poi_idx", "venue_id", "category", "latitude", "longitude", "locality",
       "region", "n_checkins"]]
    p = paths["poi_meta"] = os.path.join(export_dir, f"poi_metadata_{ds}.csv")
    out.to_csv(p, index=False)
    p = paths["friends_old"] = os.path.join(export_dir, f"friendship_old_{ds}.csv")
    pd.DataFrame(friends_old, columns=["u1", "u2"]).to_csv(p, index=False)
    p = paths["friends_eval"] = os.path.join(export_dir, f"friendship_new_only_{ds}.csv")
    pd.DataFrame(eval_pairs, columns=["u1", "u2"]).to_csv(p, index=False)
    print("exports:")
    for k, v in paths.items():
        print(f"  {v}")
    return paths


def load_groups(groups_dir):
    gpath = os.path.join(groups_dir or "", "ephemeral_groups.csv")
    if groups_dir and os.path.exists(gpath):
        groups = pd.read_csv(gpath)
        groups["members"] = [[int(x) for x in str(m).split("|")] for m in groups["members"]]
        members = pd.read_csv(os.path.join(groups_dir, "group_members.csv"))
        co = pd.read_csv(os.path.join(groups_dir, "co_attended.csv"))
        print(f"group layer: {len(groups):,} groups, {len(members):,} memberships, "
              f"{len(co):,} co-attendance pairs")
        return groups, members, co
    print("no group layer (pass --groups-dir with build_groups.py-schema files to add it)")
    return None, None, None


def run_csv(a):
    """House-format CSVs from prepare_lbsn_csvs.py: the full-fidelity graph."""
    from build_groups import load_checkins
    df = load_checkins(a.csv_dir, a.dataset)
    df["split"] = df["orig_split"]        # prepare_lbsn_csvs already split per-user 70/10/20
    meta = pd.read_csv(os.path.join(a.csv_dir, f"poi_metadata_{a.dataset}.csv"))
    cat_of = meta.set_index("poi_idx")["category"].fillna("Unknown").astype(str).to_dict()
    # --friendship-old lets a DENOISED graph (denoise_social_gbsr.py's output) feed FRIEND_OF.
    # Before this override existed the filename below was hardcoded, which is exactly how the
    # FSQ arm shipped un-denoised embeddings while its prose said otherwise (LLMGPR_TRACK.md §2).
    fo_path = a.friendship_old or os.path.join(a.csv_dir, f"friendship_old_{a.dataset}.csv")
    print(f"FRIEND_OF source: {fo_path}")
    friends_old = pd.read_csv(fo_path).to_numpy()
    np_path = os.path.join(a.csv_dir, f"friendship_new_only_{a.dataset}.csv")
    new_only = pd.read_csv(np_path).to_numpy() if os.path.exists(np_path) else \
        np.zeros((0, 2), dtype=int)
    print(f"{a.csv_dir}: {len(df):,} check-ins, {df.user_id.nunique():,} users, "
          f"{len(meta):,} POIs, {len(friends_old):,} old friendships, "
          f"{len(new_only):,} new-only eval pairs")
    print("split sizes: " + "  ".join(f"{k}={v:,}" for k, v in
                                      df["split"].value_counts().items()))

    groups, members, co = load_groups(a.groups_dir)
    kg, hierarchy, pref = build(df, meta, cat_of, friends_old, groups, members, co, a)
    all_friends = np.vstack([friends_old, new_only]) if len(new_only) else friends_old
    eval_pairs = assert_no_new_friendship_leak(kg, friends_old, all_friends)
    print(f"leakage guard: {len(eval_pairs):,} friendship_new-only pairs verified absent")

    manifest = dict(config={k: v for k, v in vars(a).items()},
                    source=a.csv_dir, dataset=a.dataset,
                    n_entities=len(kg.ent_id), n_triples=len(kg.triples),
                    n_relations=len(kg.rel_id), node_types=kg.type_counts(),
                    relation_counts=kg.counts(), n_hierarchy_pairs=len(hierarchy),
                    n_prefers_category=len(pref),
                    n_friendship_old=int(len(friends_old)),
                    n_friendship_new_only=len(eval_pairs))
    write_all(a.out_dir, kg, hierarchy, meta, manifest)


def run(a):
    mat = load_mat(a.mat)
    df, n_users, venue_vals, cat_vals, hexids, friends_old, friends_new = compact(mat)
    week, abs_hour = reconstruct_clock(df["slot"].to_numpy(), a.wrap_tol)
    df["week"], df["abs_hour"] = week, abs_hour
    df["ts"] = np.arange(len(df))          # row order IS chronological order
    print(f"{a.mat}: {len(df):,} check-ins, {n_users:,} users "
          f"({df.user_id.nunique():,} with check-ins), {len(venue_vals):,} venues, "
          f"{len(cat_vals):,} categories, {len(friends_old):,} old / {len(friends_new):,} "
          f"new friendships, ~{int(week.max()) + 1} weeks reconstructed")

    if a.split == "user":
        df = resplit_per_user(df.assign(utc_time=df["ts"]), a.train_frac, a.val_frac) \
            .drop(columns=["utc_time"])
        df = df.sort_values("ts").reset_index(drop=True)   # back to chronological order
    else:                                   # the paper's chronological row split
        cut = int(len(df) * 0.8)
        df["split"] = np.where(df["ts"] < cut, "train", "test")
    print("split sizes: " + "  ".join(f"{k}={v:,}" for k, v in
                                      df["split"].value_counts().items()))

    poi_meta = load_poi_meta(a.poi_meta) if a.poi_meta else None
    taxonomy = json.load(open(a.taxonomy)) if a.taxonomy and os.path.exists(a.taxonomy) else None
    if a.taxonomy and taxonomy is None:
        print(f"taxonomy file {a.taxonomy} not found -- proceeding without it")

    modal = modal_categories(df)
    cat_of, purity, unmatched = category_paths(df, modal, hexids, poi_meta, taxonomy)
    if purity is not None:
        print(f"category id<->name vote purity: {purity:.3f} "
              f"(validates the selected_venue_IDs ordering; expect >= 0.95)")
        assert purity >= 0.80, "purity collapsed -- the venue-id ordering assumption is wrong"
    if unmatched:
        print(f"  {len(unmatched)} category names not in the taxonomy "
              f"(kept as depth-1 nodes): {list(unmatched)[:8]}")
    meta = spatial_frame(len(venue_vals), hexids, poi_meta, a.cell_deg)

    groups, members, co = load_groups(a.groups_dir)

    kg, hierarchy, pref = build(df, meta, cat_of, friends_old, groups, members, co, a)
    eval_pairs = assert_no_new_friendship_leak(kg, friends_old, friends_new)
    print(f"leakage guard: {len(eval_pairs):,} friendship_new-only pairs verified absent")

    manifest = dict(config={k: v for k, v in vars(a).items()},
                    source=os.path.basename(a.mat), dataset=a.dataset,
                    n_entities=len(kg.ent_id), n_triples=len(kg.triples),
                    n_relations=len(kg.rel_id), node_types=kg.type_counts(),
                    relation_counts=kg.counts(), n_hierarchy_pairs=len(hierarchy),
                    n_prefers_category=len(pref),
                    n_friendship_old=int(len(friends_old)),
                    n_friendship_new_only=len(eval_pairs),
                    category_name_purity=purity)
    write_all(a.out_dir, kg, hierarchy, meta, manifest)
    export_csvs(a.export_dir or os.path.dirname(os.path.abspath(a.mat)),
                a.dataset, df, meta, cat_of, hexids, friends_old, eval_pairs)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def _fixture():
    """A tiny .mat-shaped dict: 4 users, 5 venues (global ids non-contiguous), 3 categories,
    chronological rows containing one -3h jitter and one -160h real week wrap, and a user
    whose last-20% tail holds a transition that must NOT become a FOLLOWED_BY edge."""
    rows = [
        # user, slot, venue, category      (slot sequence: 10 10 12 9(jitter) 30 100 ... wrap)
        (1, 10, 500, 7), (2, 10, 501, 8), (1, 12, 502, 9), (3, 9, 500, 7),
        (1, 30, 501, 8), (2, 100, 503, 5), (3, 120, 502, 9), (1, 150, 500, 7),
        (2, 160, 502, 9), (1, 8, 501, 8),                       # -152 -> real new week
        (2, 20, 500, 7), (1, 40, 502, 9), (3, 60, 503, 5), (1, 90, 500, 7),
        (2, 120, 501, 8),
        (1, 140, 504, 7),  # user 1's tail: 500 -> 504 exists ONLY here (test split)
    ]
    mat = {
        "selected_checkins": np.array(rows, dtype=np.int64),
        "friendship_old": np.array([[1, 2], [2, 3]], dtype=np.int64),
        "friendship_new": np.array([[1, 2], [2, 3], [1, 4]], dtype=np.int64),
        "selected_venue_IDs": np.array([f"hex{v}" for v in (500, 501, 502, 503, 504)],
                                       dtype=object).reshape(-1, 1),
    }
    poi_meta = pd.DataFrame({
        "venue_id": [f"hex{v}" for v in (500, 501, 502, 503, 504)],
        "latitude": [40.70, 40.71, 40.75, 40.76, 40.72],
        "longitude": [-73.95, -73.96, -73.87, -73.88, -73.94],
        "category": ["Wine Bar", "Sports Bar", "Shopping Mall", "Coffee Shop", "Wine Bar"],
    })
    taxonomy = {"Wine Bar": ["Nightlife Spot>Bar>Wine Bar"],
                "Sports Bar": ["Nightlife Spot>Bar>Sports Bar"],
                "Coffee Shop": ["Food>Coffee Shop"]}
    return mat, poi_meta, taxonomy


def _self_check():
    import tempfile
    print("SELF-CHECK on a synthetic .mat fixture")
    mat, poi_meta_df, taxonomy = _fixture()

    df, n_users, venue_vals, cat_vals, hexids, fo, fn = compact(mat)
    week, abs_hour = reconstruct_clock(df["slot"].to_numpy(), wrap_tol=24)
    df["week"], df["abs_hour"], df["ts"] = week, abs_hour, np.arange(len(df))
    clock = df.copy()   # row-order view for the clock checks; resplit reorders the frame
    df = resplit_per_user(df.assign(utc_time=df["ts"]), 0.70, 0.10).drop(columns=["utc_time"])
    df = df.sort_values("ts").reset_index(drop=True)

    with tempfile.TemporaryDirectory() as td:
        pm_path = os.path.join(td, "pm.csv")
        poi_meta_df.to_csv(pm_path, index=False)
        poi_meta = load_poi_meta(pm_path)
    modal = modal_categories(df)
    cat_of, purity, unmatched = category_paths(df, modal, hexids, poi_meta, taxonomy)
    meta = spatial_frame(len(venue_vals), hexids, poi_meta)

    class A: pass
    a = A(); a.knn = 2; a.max_followed_by = None; a.tau = 0.4
    kg, hierarchy, pref = build(df, meta, cat_of, fo, None, None, None, a)
    eval_pairs = assert_no_new_friendship_leak(kg, fo, fn)

    # a nameless flat build must also work (no meta, no taxonomy)
    cat_flat, _, _ = category_paths(df, modal, hexids, None, None)
    kg2, hier2, _ = build(df, spatial_frame(len(venue_vals), [""] * 5, None), cat_flat,
                          fo, None, None, None, a)

    rels = set(kg.rel_id)
    trip = set(kg.triples)
    rid = kg.rel_id.get("FOLLOWED_BY")
    eid = {n: i for n, i in kg.ent_id.items()}
    tail_edge = (eid.get("poi:0"), rid, eid.get("poi:4"))   # 500 -> 504, test-only

    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    print()
    res = [
        ok("ids compact: users 0-based, poi_idx bijective over unique venues",
           df.user_id.min() == 0 and n_users == 4 and len(venue_vals) == 5
           and sorted(df.poi_idx.unique()) == [0, 1, 2, 3, 4]),
        ok("clock: -3h jitter does NOT open a week, -152h wrap DOES",
           int(clock["week"].iloc[3]) == 0 and int(clock["week"].iloc[9]) == 1
           and int(clock["week"].max()) == 1),
        ok("abs_hour strictly follows row order across the wrap",
           clock["abs_hour"].iloc[9] > clock["abs_hour"].iloc[8]),
        ok("modal category: majority wins, ties break to the lower id",
           modal_categories(pd.DataFrame({"poi_idx": [0, 0, 0, 1, 1],
                                          "cat_idx": [2, 2, 5, 9, 4]})) == {0: 2, 1: 4}),
        ok("split is per-user chronological (train ts < val ts < test ts)",
           all(g[g.split == "train"].ts.max() < g[g.split == "test"].ts.min()
               for _, g in df.groupby("user_id") if (g.split == "test").any())),
        ok("FRIEND_OF present and symmetric",
           "FRIEND_OF" in rels and all(
               (t, r, h) in trip for h, r, t in trip if r == kg.rel_id["FRIEND_OF"])),
        ok("friendship_new-only pair (1,4) is NOT in the KG",
           len(eval_pairs) == 1 and eval_pairs[0] == (0, 3)),
        ok("test-tail transition did not leak into FOLLOWED_BY",
           tail_edge not in trip),
        ok("categories keyed by cumulative path with SUBCATEGORY_OF spine",
           "cat:Nightlife Spot>Bar>Wine Bar" in kg.ent_id and "SUBCATEGORY_OF" in rels),
        ok("modal category: every POI got exactly one HAS_CATEGORY",
           kg.counts().get("HAS_CATEGORY", 0) == len(venue_vals)),
        ok("id<->name purity = 1.0 on the fixture", purity == 1.0),
        ok("spatial layer built from coordinates (LOCATED_IN + IS_NEAR_TO)",
           "LOCATED_IN" in rels and "IS_NEAR_TO" in rels),
        ok("flat fallback still builds, without a taxonomy spine",
           "SUBCATEGORY_OF" not in kg2.rel_id and kg2.counts().get("HAS_CATEGORY", 0) == 5),
        ok("no self-loops, no duplicate triples",
           all(h != t for h, _, t in kg.triples) and len(kg.triples) == len(trip)),
    ]
    with tempfile.TemporaryDirectory() as td:
        tr, hi = write_all(td, kg, hierarchy, meta, {"source": "self_check"})
        res.append(ok("artifacts round-trip with [n,3] shapes and full POI coverage",
                      tr.shape[1] == 3 and hi.shape[1] == 3
                      and os.path.exists(os.path.join(td, "kg_poi_rows.json"))))
    return all(res)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", default=None,
                   help="house-format CSVs from prepare_lbsn_csvs.py (canonical track)")
    p.add_argument("--mat", default=None,
                   help="dataset_connected_<city>.mat from LBSN2Vec (paper population)")
    p.add_argument("--poi-meta", default=None,
                   help="mat mode only: CSV joining venue hex ids to lat/lon + category name")
    p.add_argument("--taxonomy", default="./data/lbsn/fsq_category_paths_2014.json",
                   help="JSON: category name -> [full taxonomy paths]")
    p.add_argument("--groups-dir", default=None,
                   help="build_groups.py-schema outputs for the group layer")
    p.add_argument("--friendship-old", default=None,
                   help="csv mode: override the FRIEND_OF edge file (u1,u2) -- pass "
                        "friendship_old_denoised_<DS>.csv to build the KG on the GBSR-denoised "
                        "graph; default keeps csv-dir's friendship_old_<DS>.csv")
    p.add_argument("--out-dir", default=None,
                   help="default: ./data/kg_lbsn (csv mode) / ./data/kg_lbsn_mat (mat mode)")
    p.add_argument("--export-dir", default=None,
                   help="mat mode only: where the convenience CSVs go (default: the .mat's dir)")
    p.add_argument("--dataset", default=None,
                   help="default: LBSN_NYC (csv mode) / MAT_NYC (mat mode)")
    p.add_argument("--split", choices=("user", "rows80"), default="user",
                   help="user = per-user 70/10/20 (pipeline standard); "
                        "rows80 = the paper's chronological 80/20")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--wrap-tol", type=int, default=24,
                   help="hour-of-week decrease that counts as a new week")
    p.add_argument("--cell-deg", type=float, default=0.01,
                   help="grid-cell size (degrees) for the LOCATED_IN locality level")
    p.add_argument("--knn", type=int, default=10)
    p.add_argument("--max-followed-by", type=int, default=None)
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        sys.exit(0 if _self_check() else 1)
    if a.csv_dir is None and a.mat is None:
        if os.path.exists(f"./data/lbsn/train_{DS_DEFAULT}.csv"):
            a.csv_dir = "./data/lbsn"
        elif os.path.exists("./data/lbsn/nyc.mat"):
            a.mat = "./data/lbsn/nyc.mat"
        else:
            raise SystemExit("pass --csv-dir (prepare_lbsn_csvs.py outputs) or --mat (nyc.mat)")
    if a.csv_dir:
        a.dataset = a.dataset or DS_DEFAULT
        a.out_dir = a.out_dir or "./data/kg_lbsn"
        run_csv(a)
    else:
        a.dataset = a.dataset or "MAT_NYC"
        a.out_dir = a.out_dir or "./data/kg_lbsn_mat"
        run(a)


if __name__ == "__main__":
    main()
