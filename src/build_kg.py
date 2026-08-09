"""
Build the merged POI + group knowledge graph for hyperbolic (RotH) training.

"Keep the POI KG" is the operative instruction: the POI structural layer -- category taxonomy,
spatial containment, proximity, sequence -- is what carries the hierarchy that hyperbolic space
is supposed to exploit. The group layer is added ON TOP of it, not instead of it. The earlier
`group-kg.ipynb` graph dropped the taxonomy entirely (flat one-node-per-category-string, no
SUBCATEGORY_OF) and with it the only hierarchy RotH could have learned a radius from.

Layers
------
POI structural (rebuilt here from poi_metadata_NYC.csv + the TRAIN split, reproducing Stage 1):

    HAS_CATEGORY      POI      -> CATEGORY (leaf)          one per POI
    SUBCATEGORY_OF    CATEGORY -> CATEGORY (parent)        the taxonomy spine
    LOCATED_IN        POI      -> LOCALITY -> REGION       spatial containment
    IS_NEAR_TO        POI      -> POI                      k-NN by haversine, symmetric
    FOLLOWED_BY       POI      -> POI                      TRAIN-only check-in transitions

Group layer (from build_groups.py outputs):

    VISITED           USER     -> POI                      TRAIN-only, aggregated
    PREFERS_CATEGORY  USER     -> CATEGORY                 the user's deepest justified node
    MEMBER_OF         USER     -> GROUP
    OCCURRED_AT       GROUP    -> POI
    CO_ATTENDED       USER     -> USER                     symmetric, IDF-weighted
    GROUP_PREFERS     GROUP    -> CATEGORY                 the group's consensus taxonomy node

PREFERS_CATEGORY and GROUP_PREFERS matter more than they look: they are what place users and
groups INSIDE the taxonomy, so a hyperbolic model can give them a meaningful depth. Attaching
every user to every leaf they ever touched would instead put all users at the same depth and
destroy exactly the signal we want, so each user is attached to the deepest node their behaviour
justifies (dominant child must hold >= `tau` of their visits).

Leakage: every behavioural relation (VISITED, FOLLOWED_BY, CO_ATTENDED, MEMBER_OF, OCCURRED_AT,
PREFERS_CATEGORY, GROUP_PREFERS) is built from the TRAIN split alone. Only the structural POI
relations use the full metadata, which contains no split-dependent information.

Outputs (--out-dir)
-------------------
    kg_triples.pt          LongTensor [n, 3]  (head_id, rel_id, tail_id)
    kg_entities.json       entity name -> {id, type}
    kg_relations.json      relation name -> id, plus counts
    kg_hierarchy.pt        LongTensor [m, 3]  (parent_id, child_id, relation_id) -- the depth
                           regulariser needs the relation so it can weight each hierarchy type
                           equally instead of letting HAS_CATEGORY drown out SUBCATEGORY_OF
    kg_poi_rows.json       poi_idx -> entity_id, so embeddings can be extracted in poi_idx order
    kg_manifest.json

Usage
-----
    python build_kg.py --data-dir ./data --groups-dir ./data/groups --out-dir ./data/kg
    python build_kg.py --self-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

SEP = ">"

# Relations whose (head, tail) is a child -> parent step in a containment hierarchy. These are
# the edges the depth regulariser pushes apart radially, and the ones that get type-restricted
# negatives (the Stage-2 audit found only 7.0-7.4% of uniform negatives were even the right
# node type for these, so the model learned type separation instead of hierarchy).
HIERARCHY_RELATIONS = ("HAS_CATEGORY", "SUBCATEGORY_OF", "LOCATED_IN", "PREFERS_CATEGORY",
                       "GROUP_PREFERS")


class KG:
    """Entity/relation vocabularies plus the triple list, built incrementally."""

    def __init__(self):
        self.ent_id, self.ent_type = {}, {}
        self.rel_id = {}
        self.triples = []
        self._seen = set()

    def entity(self, name, node_type):
        if name not in self.ent_id:
            self.ent_id[name] = len(self.ent_id)
            self.ent_type[name] = node_type
        return self.ent_id[name]

    def add(self, h, r, t):
        """h/t are entity ids. Silently drops self-loops and duplicates."""
        if h == t:
            return False
        if r not in self.rel_id:
            self.rel_id[r] = len(self.rel_id)
        key = (h, self.rel_id[r], t)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.triples.append(key)
        return True

    def counts(self):
        inv = {v: k for k, v in self.rel_id.items()}
        c = Counter(inv[r] for _, r, _ in self.triples)
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))

    def type_counts(self):
        return dict(sorted(Counter(self.ent_type.values()).items()))


# --------------------------------------------------------------------------
# POI structural layer
# --------------------------------------------------------------------------

def cat_path(cat_of, poi):
    return [p.strip() for p in str(cat_of.get(int(poi), "Unknown")).split(SEP) if p.strip()]


def add_taxonomy(kg, cat_of, n_pois, hierarchy):
    """HAS_CATEGORY (POI -> leaf) + SUBCATEGORY_OF (child -> parent), keyed by cumulative path.

    Keying category nodes by the CUMULATIVE path, not the leaf name, is what makes the taxonomy
    a tree: "Bar" under "Dining and Drinking" and "Bar" under "Travel" are different nodes.
    """
    for poi in range(n_pois):
        parts = cat_path(cat_of, poi)
        if not parts:
            continue
        p_id = kg.entity(f"poi:{poi}", "POI")
        nodes = [SEP.join(parts[:d]) for d in range(1, len(parts) + 1)]
        leaf = kg.entity(f"cat:{nodes[-1]}", "CATEGORY")
        kg.add(p_id, "HAS_CATEGORY", leaf)
        hierarchy.add((leaf, p_id, "HAS_CATEGORY"))   # POI sits deeper than its leaf category
        for d in range(len(nodes) - 1, 0, -1):
            child = kg.entity(f"cat:{nodes[d]}", "CATEGORY")
            parent = kg.entity(f"cat:{nodes[d - 1]}", "CATEGORY")
            kg.add(child, "SUBCATEGORY_OF", parent)
            hierarchy.add((parent, child, "SUBCATEGORY_OF"))   # child sits further out


def add_spatial(kg, meta, hierarchy):
    """LOCATED_IN: POI -> locality -> region."""
    for r in meta.itertuples(index=False):
        poi = kg.entity(f"poi:{int(r.poi_idx)}", "POI")
        loc = str(getattr(r, "locality", "") or "").strip()
        reg = str(getattr(r, "region", "") or "").strip()
        if loc and loc.lower() != "nan":
            l_id = kg.entity(f"loc:{loc}", "LOCALITY")
            kg.add(poi, "LOCATED_IN", l_id)
            hierarchy.add((l_id, poi, "LOCATED_IN"))
            if reg and reg.lower() != "nan":
                r_id = kg.entity(f"reg:{reg}", "REGION")
                kg.add(l_id, "LOCATED_IN", r_id)
                hierarchy.add((r_id, l_id, "LOCATED_IN"))


def add_proximity(kg, meta, k=10):
    """IS_NEAR_TO: k nearest POIs by haversine, symmetric."""
    lat = np.radians(meta["latitude"].to_numpy(dtype=float))
    lon = np.radians(meta["longitude"].to_numpy(dtype=float))
    idx = meta["poi_idx"].to_numpy(dtype=int)
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon, idx = lat[ok], lon[ok], idx[ok]

    # equirectangular approximation is fine for ranking neighbours inside one city
    x = np.cos(lat) * lon
    y = lat
    P = np.stack([x, y], 1)
    n = 0
    step = 512
    for s in range(0, len(P), step):
        blk = P[s:s + step]
        d = ((blk[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d[:, s:s + len(blk)], np.inf)
        nn = np.argpartition(d, k, axis=1)[:, :k]
        for i, row in enumerate(nn):
            a = kg.entity(f"poi:{int(idx[s + i])}", "POI")
            for j in row:
                b = kg.entity(f"poi:{int(idx[j])}", "POI")
                n += kg.add(a, "IS_NEAR_TO", b)
                n += kg.add(b, "IS_NEAR_TO", a)
    return n


def add_transitions(kg, train_df, max_edges=None):
    """FOLLOWED_BY: consecutive check-ins by the same user, TRAIN split only."""
    cnt = Counter()
    for _, g in train_df.groupby("user_id", sort=False):
        seq = g.sort_values("ts", kind="mergesort")["poi_idx"].to_numpy(dtype=int)
        for a, b in zip(seq, seq[1:]):
            if a != b:
                cnt[(int(a), int(b))] += 1
    items = cnt.most_common(max_edges) if max_edges else cnt.items()
    n = 0
    for (a, b), _ in items:
        n += kg.add(kg.entity(f"poi:{a}", "POI"), "FOLLOWED_BY", kg.entity(f"poi:{b}", "POI"))
    return n


# --------------------------------------------------------------------------
# group layer
# --------------------------------------------------------------------------

def add_visits(kg, train_df):
    n = 0
    for (u, p), _ in train_df.groupby(["user_id", "poi_idx"], sort=False):
        n += kg.add(kg.entity(f"user:{int(u)}", "USER"), "VISITED",
                    kg.entity(f"poi:{int(p)}", "POI"))
    return n


def prefers_category(train_df, cat_of, tau=0.4, max_branches=2):
    """Attach each user to the deepest taxonomy node their behaviour justifies.

    Mass from every visit flows to all ancestors of its category path; from each top-level node
    we descend while the dominant child still holds >= tau of the user's visits. A specialist
    reaches a leaf, a generalist stops at level 1 -- which is precisely the spread of depths that
    gives users distinct radii in hyperbolic space.
    """
    rows = []
    for uid, g in train_df.groupby("user_id", sort=False):
        paths = [tuple(cat_path(cat_of, p)) for p in g["poi_idx"]]
        paths = [p for p in paths if p]
        total = len(paths)
        if not total:
            continue
        mass = Counter()
        for p in paths:
            for d in range(1, len(p) + 1):
                mass[p[:d]] += 1
        roots = sorted({p[:1] for p in paths}, key=lambda nd: -mass[nd])[:max_branches]
        for rank, root in enumerate(roots):
            if rank > 0 and mass[root] / total < tau:
                continue
            node = root
            while True:
                rows.append((int(uid), SEP.join(node), len(node)))
                children = {p[:len(node) + 1] for p in paths
                            if len(p) > len(node) and p[:len(node)] == node}
                if not children:
                    break
                best = max(children, key=lambda c: mass[c])
                if mass[best] / total < tau:
                    break
                node = best
    return rows


def add_prefers_category(kg, rows, hierarchy):
    n = 0
    for uid, path, _depth in rows:
        u = kg.entity(f"user:{uid}", "USER")
        c = kg.entity(f"cat:{path}", "CATEGORY")
        if kg.add(u, "PREFERS_CATEGORY", c):
            hierarchy.add((c, u, "PREFERS_CATEGORY"))   # user is a leaf below its category
            n += 1
    return n


def add_group_layer(kg, groups, members, co, cat_of, hierarchy):
    """GROUP nodes + MEMBER_OF / OCCURRED_AT / CO_ATTENDED / GROUP_PREFERS."""
    stats = Counter()
    poi_of = {}
    for r in groups.itertuples(index=False):
        gid = kg.entity(f"group:{int(r.group_id)}", "GROUP")
        poi = kg.entity(f"poi:{int(r.poi_idx)}", "POI")
        poi_of[int(r.group_id)] = int(r.poi_idx)
        stats["OCCURRED_AT"] += kg.add(gid, "OCCURRED_AT", poi)
        # the group's consensus taxonomy node = the category of the venue it met at, one level up
        parts = cat_path(cat_of, int(r.poi_idx))
        if len(parts) >= 1:
            node = SEP.join(parts[:max(1, len(parts) - 1)])
            c = kg.entity(f"cat:{node}", "CATEGORY")
            if kg.add(gid, "GROUP_PREFERS", c):
                hierarchy.add((c, gid, "GROUP_PREFERS"))
                stats["GROUP_PREFERS"] += 1

    for r in members.itertuples(index=False):
        u = kg.entity(f"user:{int(r.user_id)}", "USER")
        g = kg.entity(f"group:{int(r.group_id)}", "GROUP")
        stats["MEMBER_OF"] += kg.add(u, "MEMBER_OF", g)

    for r in co.itertuples(index=False):
        a = kg.entity(f"user:{int(r.u1)}", "USER")
        b = kg.entity(f"user:{int(r.u2)}", "USER")
        stats["CO_ATTENDED"] += kg.add(a, "CO_ATTENDED", b)
        stats["CO_ATTENDED"] += kg.add(b, "CO_ATTENDED", a)
    return stats


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build(meta, train_df, groups, members, co, cat_of, a):
    kg = KG()
    hierarchy = set()

    add_taxonomy(kg, cat_of, len(meta), hierarchy)
    add_spatial(kg, meta, hierarchy)
    n_near = add_proximity(kg, meta, a.knn)
    n_follow = add_transitions(kg, train_df, a.max_followed_by)
    n_visit = add_visits(kg, train_df)
    pref = prefers_category(train_df, cat_of, a.tau)
    n_pref = add_prefers_category(kg, pref, hierarchy)
    gstats = add_group_layer(kg, groups, members, co, cat_of, hierarchy)

    print(f"\nentities {len(kg.ent_id):,}  triples {len(kg.triples):,}  "
          f"relations {len(kg.rel_id)}")
    print("  node types: " + "  ".join(f"{k}={v:,}" for k, v in kg.type_counts().items()))
    print("  relations:")
    for r, c in kg.counts().items():
        flag = "  <- hierarchy" if r in HIERARCHY_RELATIONS else ""
        print(f"    {r:<18} {c:>8,}{flag}")
    hc = Counter(r for _, _, r in hierarchy)
    print(f"  hierarchy (parent, child) pairs for the depth regulariser: {len(hierarchy):,}  "
          + "  ".join(f"{k}={v:,}" for k, v in hc.most_common()))
    depth_hist = Counter(d for _, _, d in pref)
    print(f"  PREFERS_CATEGORY depth spread (users get distinct radii from this): "
          + "  ".join(f"d{k}:{v}" for k, v in sorted(depth_hist.items())))
    return kg, hierarchy, pref


def write_all(out_dir, kg, hierarchy, meta, manifest):
    os.makedirs(out_dir, exist_ok=True)
    triples = torch.tensor(kg.triples, dtype=torch.long)
    torch.save(triples, os.path.join(out_dir, "kg_triples.pt"))

    # [m, 3] = (parent_id, child_id, relation_id). The relation matters: SUBCATEGORY_OF is only
    # 417 of ~14k pairs, and in a flat mean over pairs it contributes 2.9% of the depth loss --
    # not enough to order the taxonomy chain, which is exactly what D1 measures. train_roth.py
    # averages the depth loss PER RELATION so each hierarchy type carries equal weight.
    hier = torch.tensor([(p, c, kg.rel_id[r]) for p, c, r in sorted(hierarchy)],
                        dtype=torch.long) if hierarchy else torch.zeros((0, 3), dtype=torch.long)
    torch.save(hier, os.path.join(out_dir, "kg_hierarchy.pt"))

    with open(os.path.join(out_dir, "kg_entities.json"), "w") as f:
        json.dump({name: {"id": i, "type": kg.ent_type[name]}
                   for name, i in kg.ent_id.items()}, f)
    with open(os.path.join(out_dir, "kg_relations.json"), "w") as f:
        json.dump({"relation_to_id": kg.rel_id, "counts": kg.counts(),
                   "hierarchy_relations": list(HIERARCHY_RELATIONS)}, f, indent=2)

    # poi_idx -> entity id, so poi_hyperbolic_embs.npy can be written in poi_idx order
    poi_rows = {}
    for poi in meta["poi_idx"].astype(int):
        name = f"poi:{int(poi)}"
        if name in kg.ent_id:
            poi_rows[int(poi)] = kg.ent_id[name]
    with open(os.path.join(out_dir, "kg_poi_rows.json"), "w") as f:
        json.dump(poi_rows, f)
    assert len(poi_rows) == len(meta), \
        f"only {len(poi_rows)} of {len(meta)} POIs made it into the graph"

    with open(os.path.join(out_dir, "kg_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote to {out_dir}:")
    for n in ("kg_triples.pt", "kg_hierarchy.pt", "kg_entities.json", "kg_relations.json",
              "kg_poi_rows.json", "kg_manifest.json"):
        print(f"  {n}")
    return triples, hier


def run(a):
    meta = pd.read_csv(os.path.join(a.data_dir, f"poi_metadata_{a.dataset}.csv"))
    cat_col = next((c for c in ("category_path", "category") if c in meta.columns), None)
    cat_of = meta.set_index("poi_idx")[cat_col].fillna("Unknown").astype(str).to_dict()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_groups import load_checkins, resplit_per_user
    df = load_checkins(a.data_dir, a.dataset)
    df = resplit_per_user(df, a.train_frac, a.val_frac) if a.resplit else \
        df.assign(split=df["orig_split"])
    train_df = df[df["split"] == "train"]
    print(f"POIs={len(meta):,}  train check-ins={len(train_df):,}  users={train_df.user_id.nunique()}")

    gpath = os.path.join(a.groups_dir, "ephemeral_groups.csv")
    if os.path.exists(gpath):
        groups = pd.read_csv(gpath)
        groups["members"] = [[int(x) for x in str(m).split("|")] for m in groups["members"]]
        members = pd.read_csv(os.path.join(a.groups_dir, "group_members.csv"))
        co = pd.read_csv(os.path.join(a.groups_dir, "co_attended.csv"))
        print(f"group layer: {len(groups):,} groups, {len(members):,} memberships, "
              f"{len(co):,} co-attendance pairs")
    else:
        print(f"no group layer found at {gpath} -- building the POI KG only")
        groups = members = co = pd.DataFrame()

    kg, hierarchy, pref = build(meta, train_df, groups, members, co, cat_of, a)
    manifest = dict(config=vars(a), n_entities=len(kg.ent_id), n_triples=len(kg.triples),
                    n_relations=len(kg.rel_id), node_types=kg.type_counts(),
                    relation_counts=kg.counts(), n_hierarchy_pairs=len(hierarchy),
                    n_prefers_category=len(pref))
    write_all(a.out_dir, kg, hierarchy, meta, manifest)


def _self_check():
    import tempfile
    print("SELF-CHECK on a synthetic KG fixture")
    meta = pd.DataFrame([
        dict(poi_idx=0, category="Dining and Drinking>Bar>Wine Bar", locality="Brooklyn",
             region="NY", latitude=40.70, longitude=-73.95),
        dict(poi_idx=1, category="Dining and Drinking>Bar>Sports Bar", locality="Brooklyn",
             region="NY", latitude=40.71, longitude=-73.96),
        dict(poi_idx=2, category="Retail>Shopping Mall", locality="Queens",
             region="NY", latitude=40.75, longitude=-73.87),
        dict(poi_idx=3, category="Dining and Drinking>Cafe", locality="Queens",
             region="NY", latitude=40.76, longitude=-73.88),
    ])
    cat_of = meta.set_index("poi_idx")["category"].to_dict()
    train = pd.DataFrame([dict(user_id=1, poi_idx=p, ts=i * 100)
                          for i, p in enumerate([0, 1, 0, 1, 3])] +
                         [dict(user_id=2, poi_idx=p, ts=i * 100)
                          for i, p in enumerate([2, 3, 2, 3])])
    groups = pd.DataFrame([dict(group_id=0, poi_idx=0, members=[1, 2])])
    members = pd.DataFrame([dict(group_id=0, user_id=1), dict(group_id=0, user_id=2)])
    co = pd.DataFrame([dict(u1=1, u2=2, n_groups=1, n_venues=1, weight_idf=0.5)])

    class A: pass
    a = A(); a.knn = 2; a.max_followed_by = None; a.tau = 0.4
    kg, hierarchy, pref = build(meta, train, groups, members, co, cat_of, a)

    rels = set(kg.rel_id)
    cats = {n for n, t in kg.ent_type.items() if t == "CATEGORY"}
    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    print()
    res = [
        ok("taxonomy is keyed by cumulative path (Bar under D&D, not bare 'Bar')",
           "cat:Dining and Drinking>Bar" in cats and "cat:Bar" not in cats),
        ok("SUBCATEGORY_OF present -- the hierarchy spine survived",
           "SUBCATEGORY_OF" in rels),
        ok("every POI has a leaf category",
           kg.counts().get("HAS_CATEGORY", 0) == len(meta)),
        ok("spatial containment POI->locality->region", "LOCATED_IN" in rels),
        ok("group layer merged onto the POI KG, not replacing it",
           {"MEMBER_OF", "OCCURRED_AT", "CO_ATTENDED", "GROUP_PREFERS"} <= rels),
        ok("users are placed in the taxonomy (PREFERS_CATEGORY)",
           "PREFERS_CATEGORY" in rels and len(pref) > 0),
        ok("hierarchy pairs collected for the depth regulariser", len(hierarchy) > 0),
        ok("no self-loops", all(h != t for h, _, t in kg.triples)),
        ok("no duplicate triples", len(kg.triples) == len(set(kg.triples))),
        ok("IS_NEAR_TO is symmetric",
           all((t, kg.rel_id["IS_NEAR_TO"], h) in set(kg.triples)
               for h, r, t in kg.triples if r == kg.rel_id["IS_NEAR_TO"])),
    ]
    with tempfile.TemporaryDirectory() as td:
        tr, hi = write_all(td, kg, hierarchy, meta, {"source": "self_check"})
        res.append(ok("triples/hierarchy tensors saved with the right shape",
                      tr.shape[1] == 3 and hi.shape[1] == 3))
        res.append(ok("hierarchy rows carry a valid relation id",
                      hi.numel() > 0 and int(hi[:, 2].max()) < len(kg.rel_id)))
    return all(res)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--groups-dir", default="./data/groups")
    p.add_argument("--out-dir", default="./data/kg")
    p.add_argument("--dataset", default="NYC")
    p.add_argument("--knn", type=int, default=10, help="IS_NEAR_TO neighbours per POI")
    p.add_argument("--max-followed-by", type=int, default=None,
                   help="cap FOLLOWED_BY to the N most frequent transitions")
    p.add_argument("--tau", type=float, default=0.4,
                   help="PREFERS_CATEGORY concentration threshold")
    p.add_argument("--resplit", action="store_true", default=True)
    p.add_argument("--no-resplit", dest="resplit", action="store_false")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        sys.exit(0 if _self_check() else 1)
    run(a)


if __name__ == "__main__":
    main()
