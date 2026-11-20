"""Build the heterogeneous user graph exported by build_social_graph.py.

Social ties remain in their original CSV. This export adds observed group membership,
category preferences, local-time activity and geographic habits; it does not train
embeddings. All behavioral input must be the builder's guarded TRAIN check-ins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd

TRIPLE_COLS = ["head", "relation", "tail", "weight", "count", "level"]


def build_user_graph(train, meta, social_edges, groups, users, cell_deg=0.01):
    """Return named triples, typed nodes, observed outings and construction statistics.

    MEMBER_OF uses one node per retained co-presence outing, including single
    occurrences. Existing window/size/subset rules are supplied by the caller.
    Social ties are expanded to both directions; the four new relations are directed.
    Category weights sum to one per user PER DEPTH. Time and area weights sum to
    one per user over the check-ins with a valid value for that relation.
    """
    if not math.isfinite(cell_deg) or cell_deg <= 0:
        raise ValueError("cell_deg must be finite and positive")
    if not set(train["orig_split"].unique()) <= {"train"}:
        raise ValueError("user graph requires train-only check-ins")
    if meta["poi_idx"].duplicated().any():
        raise ValueError("duplicate poi_idx in metadata")
    cat_col = next((c for c in ("category_path", "category", "categories") if c in meta), None)
    if cat_col is None:
        raise ValueError("metadata has no category column")
    metadata = meta.set_index("poi_idx")
    nodes, triples, outings = {}, [], []

    def node(key, kind, **attrs):
        value = dict(entity=key, type=kind, attributes=json.dumps(attrs, sort_keys=True))
        if key in nodes and nodes[key] != value:
            raise ValueError(f"conflicting metadata for {key}")
        nodes[key] = value
        return key

    def user(u):
        return node(f"user:{int(u)}", "USER", user_id=int(u))

    def edge(h, r, t, weight=1.0, count=1, level=0):
        triples.append((h, r, t, float(weight), int(count), int(level)))

    for u in sorted(set(map(int, users))):
        user(u)
    for e in social_edges.itertuples(index=False):
        h, t = user(e.u1), user(e.u2)
        if e.relation not in ("FRIEND_OF", "CO_PRESENT"):
            raise ValueError(f"unexpected social relation {e.relation}")
        count = int(e.n_meetings) if e.relation == "CO_PRESENT" else 1
        edge(h, e.relation, t, e.weight, count)
        edge(t, e.relation, h, e.weight, count)

    if not groups.empty:
        for g in groups.itertuples(index=False):
            members = sorted(set(map(int, g.members)))
            identity = json.dumps([str(g.venue_id), int(g.start_ts), int(g.end_ts), members])
            key = "group:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            node(key, "GROUP", venue_id=str(g.venue_id), start_ts=int(g.start_ts),
                 end_ts=int(g.end_ts), members=members)
            outings.append(dict(group=key, venue_id=str(g.venue_id), poi_idx=int(g.poi_idx),
                                start_ts=int(g.start_ts), end_ts=int(g.end_ts),
                                size=len(members), members="|".join(map(str, members))))
            for u in members:
                edge(user(u), "MEMBER_OF", key)

    # Resolve static attributes from the same POI catalogue as the POI KG.
    categories, areas = {}, {}
    for p, row in metadata.iterrows():
        value = row[cat_col]
        parts = [] if pd.isna(value) else [s.strip() for s in str(value).split(">") if s.strip()]
        if parts and parts[0].lower() not in ("unknown", "nan", "none"):
            categories[int(p)] = [">".join(parts[:d]) for d in range(1, len(parts) + 1)]
        lat, lon = row.get("latitude", np.nan), row.get("longitude", np.nan)
        if pd.notna(lat) and pd.notna(lon):
            lat, lon = float(lat), float(lon)
            if math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                ilat, ilon = math.floor(lat / cell_deg), math.floor(lon / cell_deg)
                key = f"area:{cell_deg:g}:{ilat}:{ilon}"
                areas[int(p)] = node(key, "AREA", cell_deg=cell_deg, lat_cell=ilat, lon_cell=ilon)

    prefs, activity, geography = Counter(), Counter(), Counter()
    cat_total, time_total, area_total = Counter(), Counter(), Counter()
    missing_cat = missing_area = missing_time = utc_fallback = 0
    timestamps = pd.to_datetime(train["utc_time"], utc=True, errors="coerce")
    offsets = pd.to_numeric(train.get("timezone_offset", pd.Series(np.nan, index=train.index)),
                            errors="coerce")
    utc_fallback = int(offsets.isna().sum())
    local = timestamps + pd.to_timedelta(offsets.fillna(0), unit="m")
    for u, p, ts in zip(train["user_id"], train["poi_idx"], local):
        u, p = int(u), int(p)
        user(u)
        paths = categories.get(p, [])
        if not paths:
            missing_cat += 1
        for depth, path in enumerate(paths, 1):
            prefs[u, depth, path] += 1
            cat_total[u, depth] += 1
        if pd.isna(ts):
            missing_time += 1
        else:
            period = ("morning" if 6 <= ts.hour < 12 else "afternoon" if 12 <= ts.hour < 18
                      else "evening" if 18 <= ts.hour < 24 else "night")
            label = ("weekend" if ts.dayofweek >= 5 else "weekday") + ":" + period
            activity[u, label] += 1
            time_total[u] += 1
        if p in areas:
            geography[u, areas[p]] += 1
            area_total[u] += 1
        else:
            missing_area += 1

    for (u, depth, path), n in sorted(prefs.items()):
        key = node("cat:" + path, "CATEGORY", path=path, depth=depth)
        edge(user(u), "PREFERS_CATEGORY", key, n / cat_total[u, depth], n, depth)
    for (u, label), n in sorted(activity.items()):
        key = node("time:" + label, "TIME_BIN", label=label)
        edge(user(u), "ACTIVE_DURING", key, n / time_total[u], n)
    for (u, key), n in sorted(geography.items()):
        edge(user(u), "FREQUENTS_AREA", key, n / area_total[u], n)

    edges = pd.DataFrame(triples, columns=TRIPLE_COLS).sort_values(
        ["relation", "head", "tail"], kind="mergesort").reset_index(drop=True)
    # Do not export unused metadata-only nodes.
    used = set(edges["head"]) | set(edges["tail"]) | {f"user:{int(u)}" for u in users}
    entities = pd.DataFrame([nodes[k] for k in sorted(used)], columns=["entity", "type", "attributes"])
    entities.insert(0, "id", np.arange(len(entities)))
    group_frame = pd.DataFrame(outings, columns=["group", "venue_id", "poi_idx", "start_ts", "end_ts", "size", "members"])
    assert not edges.duplicated(["head", "relation", "tail"]).any(), "duplicate user triple"
    assert np.isfinite(edges["weight"]).all() and (edges["weight"] > 0).all()
    for relation in ("PREFERS_CATEGORY", "ACTIVE_DURING", "FREQUENTS_AREA"):
        sub = edges[edges["relation"] == relation]
        keys = ["head", "level"] if relation == "PREFERS_CATEGORY" else ["head"]
        assert np.allclose(sub.groupby(keys)["weight"].sum(), 1.0), relation
    stats = dict(n_entities=len(entities), n_triples=len(edges), n_groups=len(group_frame),
                 node_types=entities["type"].value_counts().to_dict(),
                 relation_counts=edges["relation"].value_counts().to_dict(),
                 checkins_used=len(train), missing_category_checkins=missing_cat,
                 missing_area_checkins=missing_area, missing_time_checkins=missing_time,
                 timezone_utc_fallback_checkins=utc_fallback)
    return edges, entities, group_frame, stats


def export_user_graph(train, meta, social_edges, groups, users, out_dir, dataset,
                      cell_deg=0.01, provenance=None):
    edges, nodes, outings, stats = build_user_graph(train, meta, social_edges, groups, users, cell_deg)
    os.makedirs(out_dir, exist_ok=True)
    files = dict(triples=f"user_graph_triples_{dataset}.csv", entities=f"user_graph_entities_{dataset}.csv",
                 groups=f"user_graph_groups_{dataset}.csv", relations=f"user_graph_relations_{dataset}.json")
    edges.to_csv(os.path.join(out_dir, files["triples"]), index=False)
    nodes.to_csv(os.path.join(out_dir, files["entities"]), index=False)
    outings.to_csv(os.path.join(out_dir, files["groups"]), index=False)
    relations = {r: i for i, r in enumerate(sorted(stats["relation_counts"]))}
    with open(os.path.join(out_dir, files["relations"]), "w", encoding="utf-8") as f:
        json.dump(dict(relation_to_id=relations, symmetric_relations=["FRIEND_OF", "CO_PRESENT"]), f, indent=2)
    manifest = dict(dataset=dataset, script="src/build_user_graph.py", files=files, counts=stats,
                    config=dict(area_cell_deg=cell_deg, membership="one node per observed co-presence outing; single occurrences included",
                                category_weight="visit share normalized separately at each taxonomy depth",
                                time_weight="visit share in local weekday/weekend x morning/afternoon/evening/night",
                                time_boundaries="night [00,06), morning [06,12), afternoon [12,18), evening [18,24)",
                                area_weight="visit share among check-ins with valid catalogue coordinates",
                                coordinates="floor(latitude / cell_deg), floor(longitude / cell_deg); angular grid, not equal-area",
                                social_edges="both directions; original weights", new_relations="directed"),
                    provenance=provenance or {},
                    audit=dict(train_only=True, unique_triples=True, positive_finite_weights=True,
                               preference_activity_area_weights_normalized=True),
                    embeddings="not trained; current user-token consumer still uses the user-user social CSV")
    with open(os.path.join(out_dir, f"user_graph_manifest_{dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=int)
    return manifest


def self_check():
    meta = pd.DataFrame(dict(poi_idx=[0, 1, 2], category=["Food>Cafe", "Food>Restaurant", "Unknown"],
                             latitude=[40.71, 40.72, np.nan], longitude=[-74.01, -74.02, np.nan]))
    # UTC Saturday 01:00 at UTC-5 is Friday evening; do not use UTC's weekday.
    tr = pd.DataFrame(dict(user_id=[0, 0, 1], poi_idx=[0, 1, 2],
                           utc_time=["2026-11-21T01:00:00Z"] * 3, timezone_offset=[-300] * 3,
                           orig_split=["train"] * 3))
    social = pd.DataFrame([dict(u1=0, u2=1, relation="FRIEND_OF", weight=1., n_meetings=0)])
    groups = pd.DataFrame([dict(venue_id="venue-a", poi_idx=0, start_ts=1, end_ts=2, members=[0, 1])])
    e, n, g, st = build_user_graph(tr, meta, social, groups, [0, 1, 2])
    assert len(e[e.relation == "MEMBER_OF"]) == 2 and len(g) == 1
    assert len(e[e.relation == "FRIEND_OF"]) == 2
    pref = e[e.relation == "PREFERS_CATEGORY"].set_index("tail")
    assert pref.loc["cat:Food", "weight"] == 1 and pref.loc["cat:Food>Cafe", "weight"] == .5
    assert set(e.loc[e.relation == "ACTIVE_DURING", "tail"]) == {"time:weekday:evening"}
    assert st["missing_area_checkins"] == 1 and st["missing_category_checkins"] == 1
    assert "user:2" in set(n.entity) and not (e["head"] == "user:2").any()
    shuffled = build_user_graph(tr.iloc[::-1], meta.iloc[::-1], social, groups, [2, 1, 0])
    pd.testing.assert_frame_equal(e, shuffled[0])
    pd.testing.assert_frame_equal(n, shuffled[1])
    empty_groups = groups.iloc[:0]
    assert build_user_graph(tr, meta, social.iloc[:0], empty_groups, [0, 1])[2].empty
    bad = tr.copy(); bad.loc[0, "orig_split"] = "test"
    try:
        build_user_graph(bad, meta, social, groups, [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("held-out data accepted")
    print("user graph self-check passed: one-off membership, symmetry, category weights, local time, missing metadata, isolated users, determinism, train-only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        parser.error("build via src/build_social_graph.py; or pass --self-check")
