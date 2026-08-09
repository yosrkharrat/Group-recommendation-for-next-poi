"""
Build POI-POI knowledge-graph triples in this project's own `poi_idx` numbering, so the
curvature-aware alignment stage in `stage6b_run2_server.ipynb` has real (head, relation, tail)
triples for its triple-preservation loss term.

Source: **this project's own knowledge graph** (`kg_NYC_v3.gpickle`, built by Stage 1/1v3), whose
POI nodes are already keyed `poi:{poi_idx}` in the same 0..N_POI-1 space as
`poi_metadata_NYC.csv`. No crosswalk is needed -- the original version of this script mapped a
*reference* project's own poi ids through `venue_id` into ours; reading our KG directly removes
that dependency and any chance of an id-space mismatch.

Outputs (identical format to the crosswalk version, so §6b of the notebook is unchanged):
    <out-dir>/poi_poi_triples_NYC.pt     torch.LongTensor [n_triples, 3] = (h_idx, rel_id, t_idx)
    <out-dir>/poi_relation_vocab_NYC.json {relation_to_id, counts, provenance}

Which relations are kept
------------------------
Every edge whose head *and* tail are both POI nodes. In KG v3 that is `IS_NEAR_TO` (spatial
k-NN), `FOLLOWED_BY` (train-split check-in transitions), `CO_VISITED_WITH` and `SAME_BRAND_AS`.
POI->Category / POI->Locality / POI->Station edges are skipped here: they are not POI-POI, and
the hierarchy they encode already reaches the alignment through the radius term.

All four native relations are *flat* (proximity, sequence, brand), which leaves the TransE term
with no hierarchical signal of its own. `--derive taxonomy` adds one back: POI pairs sharing the
first k levels of their category path become `SAME_TAXONOMY_L{k}`, bucketed by k exactly the way
the reference script bucketed `externalSemanticSimilar:{conf}` by confidence decile. Off by
default -- turn it on if the triple loss looks like it is only learning geography.

Leakage
-------
`FOLLOWED_BY` was built from the train split alone at KG-construction time (Stage 1). Any other
behavioural relation you add later must respect the same rule; `--exclude` is there for that.

Usage
-----
    python build_poi_poi_triples.py --kg data/kg_NYC_v3.gpickle --meta data/poi_metadata_NYC.csv
    python build_poi_poi_triples.py --derive taxonomy --max-per-relation 40000
    python build_poi_poi_triples.py --self-check
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

# Relations that encode user behaviour and must therefore be train-split-only. Listed so the
# script can say so out loud rather than relying on the reader to remember.
BEHAVIOURAL = {"FOLLOWED_BY", "CO_VISITED_WITH", "VISITED", "VISITED_AT_HOUR", "VISITED_ON"}

_POI_NODE = re.compile(r"^poi[:_]?(\d+)$", re.IGNORECASE)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_graph(path: Path):
    """Load a NetworkX graph from .gpickle / .pickle / .graphml."""
    import networkx as nx

    suffix = path.suffix.lower()
    if suffix in (".gpickle", ".pickle", ".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if suffix == ".graphml":
        return nx.read_graphml(path)
    raise SystemExit(f"unsupported graph format {suffix!r} (want .gpickle/.pickle/.graphml)")


def extract_from_kg_dir(kg_dir: Path, verbose=True):
    """Read the tensor-format KG produced by `group/build_kg.py`.

    That stage emits kg_triples.pt (LongTensor [n,3] over entity ids) plus kg_entities.json, not
    a NetworkX object -- so this is the path to use when the KG came from build_kg.py rather than
    from the original Stage-1 gpickle. Returns (h_poi_idx, relation_name, t_poi_idx) triples.
    """
    triples = torch.load(kg_dir / "kg_triples.pt")
    with open(kg_dir / "kg_entities.json") as f:
        ents = json.load(f)
    with open(kg_dir / "kg_relations.json") as f:
        rels = json.load(f)

    name_of_rel = {v: k for k, v in rels["relation_to_id"].items()}
    # entity id -> poi_idx, for POI entities only
    poi_of_id = {}
    for name, rec in ents.items():
        if rec["type"] == "POI":
            m = _POI_NODE.match(name)
            if m:
                poi_of_id[rec["id"]] = int(m.group(1))

    kept, skipped = [], Counter()
    for h, r, t in triples.tolist():
        hu, tv = poi_of_id.get(h), poi_of_id.get(t)
        rel = name_of_rel[r]
        if hu is None or tv is None:
            skipped[rel] += 1
            continue
        if hu != tv:
            kept.append((hu, rel, tv))

    if verbose:
        print(f"KG dir: {kg_dir}")
        print(f"  entities {len(ents):,}  triples {len(triples):,}  "
              f"POI entities {len(poi_of_id):,}")
        if skipped:
            print("  skipped (not POI->POI): " +
                  ", ".join(f"{k}={v:,}" for k, v in skipped.most_common(8)))
    # dedupe, preserving order
    seen, out = set(), []
    for tr in kept:
        if tr not in seen:
            seen.add(tr)
            out.append(tr)
    return out


def poi_index_of(node, attrs: dict):
    """Return the integer poi_idx for a POI node, or None if the node is not a POI.

    Handles the `poi:{idx}` convention used by Stage 1 / group-kg, a `node_type == "POI"`
    attribute carrying an explicit `poi_idx`, and bare integer node ids.
    """
    ntype = str(attrs.get("node_type", attrs.get("type", ""))).upper()
    if ntype and ntype != "POI":
        return None

    if "poi_idx" in attrs:
        try:
            return int(attrs["poi_idx"])
        except (TypeError, ValueError):
            pass

    m = _POI_NODE.match(str(node))
    if m:
        return int(m.group(1))
    if ntype == "POI" and isinstance(node, int):
        return int(node)
    return None


def relation_of(u, v, key, attrs: dict) -> str:
    """Relation name, preferring the explicit attribute over the MultiDiGraph edge key."""
    rel = attrs.get("relation") or attrs.get("rel") or attrs.get("label") or attrs.get("type")
    if rel is None and isinstance(key, str):
        rel = key
    return str(rel) if rel is not None else "UNKNOWN"


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def extract_poi_poi(G, exclude=(), verbose=True):
    """(h_idx, relation_name, t_idx) for every edge with POI on both ends. Deduplicated."""
    idx_of, n_nodes_by_type = {}, Counter()
    for node, attrs in G.nodes(data=True):
        n_nodes_by_type[str(attrs.get("node_type", attrs.get("type", "?"))).upper() or "?"] += 1
        pi = poi_index_of(node, attrs)
        if pi is not None:
            idx_of[node] = pi

    if verbose:
        print(f"nodes: {G.number_of_nodes():,}  edges: {G.number_of_edges():,}")
        print("  node types: " + ", ".join(f"{k}={v}" for k, v in sorted(n_nodes_by_type.items())))
        print(f"  POI nodes resolved to a poi_idx: {len(idx_of):,}")

    edges = G.edges(keys=True, data=True) if G.is_multigraph() else (
        (u, v, None, d) for u, v, d in G.edges(data=True))

    kept, seen = [], set()
    skipped_non_poi = Counter()
    excluded = set(exclude)
    for u, v, key, attrs in edges:
        rel = relation_of(u, v, key, attrs)
        hu, tv = idx_of.get(u), idx_of.get(v)
        if hu is None or tv is None:
            skipped_non_poi[rel] += 1
            continue
        if rel in excluded or hu == tv:
            continue
        if (hu, rel, tv) in seen:
            continue
        seen.add((hu, rel, tv))
        kept.append((hu, rel, tv))

    if verbose and skipped_non_poi:
        top = ", ".join(f"{r}={c:,}" for r, c in skipped_non_poi.most_common(8))
        print(f"  skipped (not POI->POI): {top}")
    return kept


def derive_taxonomy(cat_of: dict, n_pois: int, sep=">", min_level=2,
                    max_partners=8, seed=42, verbose=True):
    """POI pairs sharing the first k>=min_level levels of their category path.

    Bucketed by k, mirroring the reference script's confidence-decile buckets: sharing three
    taxonomy levels is a much stronger statement than sharing one, and a single flat
    `SAME_CATEGORY` relation would throw that ordering away. Each POI is linked to at most
    `max_partners` others per bucket (sampled with a fixed seed) so a popular category cannot
    dominate the triple set the way IS_NEAR_TO already dominates by volume.
    """
    rng = random.Random(seed)
    paths = {}
    for i in range(n_pois):
        parts = [p.strip() for p in str(cat_of.get(i, "")).split(sep) if p.strip()]
        if parts:
            paths[i] = parts

    max_depth = max((len(p) for p in paths.values()), default=0)
    out = []
    for k in range(min_level, max_depth + 1):
        buckets = defaultdict(list)
        for i, parts in paths.items():
            if len(parts) >= k:
                buckets[tuple(parts[:k])].append(i)
        rel = f"SAME_TAXONOMY_L{k}"
        n_before = len(out)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for i in members:
                others = [m for m in members if m != i]
                rng.shuffle(others)
                for j in others[:max_partners]:
                    out.append((i, rel, j))
        if verbose:
            print(f"  derived {rel}: {len(out) - n_before:,} triples "
                  f"over {sum(1 for m in buckets.values() if len(m) > 1):,} shared prefixes")
    return out


def cap_per_relation(triples, max_per_relation, seed=42, verbose=True):
    """Downsample over-represented relations. IS_NEAR_TO alone is ~63k edges in KG v3, ~8x
    SAME_BRAND_AS; the Stage-2 audit already showed that letting one relation dominate teaches
    the model that relation instead of the structure."""
    if not max_per_relation:
        return triples
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for t in triples:
        by_rel[t[1]].append(t)
    out = []
    for rel, group in sorted(by_rel.items()):
        if len(group) > max_per_relation:
            if verbose:
                print(f"  capped {rel}: {len(group):,} -> {max_per_relation:,}")
            group = rng.sample(group, max_per_relation)
        out.extend(group)
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build(triples, n_pois):
    """(h, rel_name, t) list -> (LongTensor[n,3], relation_to_id, counts)."""
    relation_to_id, counts = {}, Counter()
    rows = []
    for h, rel, t in triples:
        if rel not in relation_to_id:
            relation_to_id[rel] = len(relation_to_id)
        counts[rel] += 1
        rows.append((int(h), relation_to_id[rel], int(t)))

    if not rows:
        raise SystemExit("no POI-POI triples found -- check --kg and the node naming convention")

    tensor = torch.tensor(rows, dtype=torch.long)
    assert tensor[:, [0, 2]].min().item() >= 0
    assert tensor[:, [0, 2]].max().item() < n_pois, (
        f"triple references poi_idx {tensor[:, [0, 2]].max().item()} but metadata has {n_pois} POIs")
    return tensor, relation_to_id, dict(counts)


def write_outputs(out_dir: Path, dataset, tensor, relation_to_id, counts, provenance):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_triples = out_dir / f"poi_poi_triples_{dataset}.pt"
    out_relvocab = out_dir / f"poi_relation_vocab_{dataset}.json"

    torch.save(tensor, out_triples)
    with open(out_relvocab, "w") as f:
        json.dump({"relation_to_id": relation_to_id, "counts": counts,
                   "provenance": provenance}, f, indent=2)

    print(f"\nPOI-POI triples: {len(tensor):,}   relations: {len(relation_to_id)}")
    for name, rid in sorted(relation_to_id.items(), key=lambda kv: kv[1]):
        print(f"  [{rid:2d}] {name:24s} n={counts.get(name, 0):,}")
    print(f"Saved: {out_triples}")
    print(f"Saved: {out_relvocab}")
    return out_triples, out_relvocab


def run(a):
    meta_path = Path(a.meta)
    if not meta_path.exists():
        raise SystemExit(f"missing required input: {meta_path}")

    meta = pd.read_csv(meta_path)
    n_pois = len(meta)
    cat_col = next((c for c in ("category_path", "category", "categories")
                    if c in meta.columns), None)
    cat_of = (meta.set_index("poi_idx")[cat_col].fillna("").astype(str).to_dict()
              if cat_col else {})

    if a.kg_dir:
        kg_path = Path(a.kg_dir)
        if not (kg_path / "kg_triples.pt").exists():
            raise SystemExit(f"{kg_path} has no kg_triples.pt -- did group/build_kg.py run?")
        native = extract_from_kg_dir(kg_path)
        if a.exclude:
            native = [t for t in native if t[1] not in set(a.exclude)]
    else:
        kg_path = Path(a.kg)
        if not kg_path.exists():
            raise SystemExit(f"missing required input: {kg_path}")
        G = load_graph(kg_path)
        native = extract_poi_poi(G, exclude=a.exclude)

    found_behavioural = sorted({r for _, r, _ in native} & BEHAVIOURAL)
    if found_behavioural:
        print(f"  note: {', '.join(found_behavioural)} are behavioural relations -- they must "
              f"have been built from the TRAIN split only (Stage 1 does this for FOLLOWED_BY)")

    triples = list(native)
    if a.derive == "taxonomy":
        if not cat_of:
            raise SystemExit(f"--derive taxonomy needs a category column in {meta_path}")
        triples += derive_taxonomy(cat_of, n_pois, min_level=a.taxonomy_min_level,
                                   max_partners=a.taxonomy_max_partners, seed=a.seed)

    triples = cap_per_relation(triples, a.max_per_relation, seed=a.seed)
    tensor, relation_to_id, counts = build(triples, n_pois)

    provenance = {
        "source": "build_kg_dir" if a.kg_dir else "own_kg",
        "kg_file": str(kg_path.resolve()),
        "metadata_file": str(meta_path.resolve()),
        "n_pois": n_pois,
        "n_native_poi_poi_triples": len(native),
        "n_after_derive_and_cap": len(triples),
        "derived": a.derive,
        "excluded_relations": list(a.exclude),
        "max_per_relation": a.max_per_relation,
        "seed": a.seed,
        "behavioural_relations_present": found_behavioural,
    }
    return write_outputs(Path(a.out_dir), a.dataset, tensor, relation_to_id, counts, provenance)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def _self_check():
    """Synthetic KG with the same shape as v3: POI/Category/Locality nodes, POI-POI and
    POI->other edges. Confirms only POI-POI survives and that ids stay in poi_idx space."""
    import tempfile

    import networkx as nx

    print("SELF-CHECK on a synthetic KG")
    G = nx.MultiDiGraph()
    n_pois = 12
    for i in range(n_pois):
        G.add_node(f"poi:{i}", node_type="POI", poi_idx=i)
    for c in ("Dining and Drinking", "Dining and Drinking > Bar", "Retail"):
        G.add_node(f"cat:{c}", node_type="CATEGORY")
    G.add_node("loc:Brooklyn", node_type="LOCALITY")

    for i in range(n_pois):                                     # POI -> non-POI (must be dropped)
        G.add_edge(f"poi:{i}", "cat:Retail" if i % 2 else "cat:Dining and Drinking > Bar",
                   key="HAS_CATEGORY", relation="HAS_CATEGORY")
        G.add_edge(f"poi:{i}", "loc:Brooklyn", key="LOCATED_IN", relation="LOCATED_IN")
    G.add_edge("cat:Dining and Drinking > Bar", "cat:Dining and Drinking",
               key="SUBCATEGORY_OF", relation="SUBCATEGORY_OF")

    for i in range(n_pois):                                     # POI -> POI (must survive)
        for j in ((i + 1) % n_pois, (i + 2) % n_pois):
            G.add_edge(f"poi:{i}", f"poi:{j}", key="IS_NEAR_TO", relation="IS_NEAR_TO")
    for i in range(0, n_pois - 1, 2):
        G.add_edge(f"poi:{i}", f"poi:{i+1}", key="FOLLOWED_BY", relation="FOLLOWED_BY")
    G.add_edge("poi:0", "poi:0", key="IS_NEAR_TO", relation="IS_NEAR_TO")   # self-loop, dropped
    G.add_edge("poi:1", "poi:2", key="IS_NEAR_TO", relation="IS_NEAR_TO")   # duplicate, dropped

    native = extract_poi_poi(G, exclude=())
    rels = {r for _, r, _ in native}
    tensor, relation_to_id, counts = build(native, n_pois)

    cat_of = {i: ("Retail > Shopping Mall" if i % 2 else "Dining and Drinking > Bar > Wine Bar")
              for i in range(n_pois)}
    derived = derive_taxonomy(cat_of, n_pois, min_level=2, max_partners=3, seed=0, verbose=False)
    capped = cap_per_relation([("x", "IS_NEAR_TO", "y")] * 100, 10, seed=0, verbose=False)

    with tempfile.TemporaryDirectory() as td:
        write_outputs(Path(td), "SELFCHECK", tensor, relation_to_id, counts, {"source": "self_check"})
        reloaded = torch.load(Path(td) / "poi_poi_triples_SELFCHECK.pt")

    ok = lambda name, cond: (print(f"  {'PASS' if cond else 'FAIL'}  {name}"), cond)[1]
    print()
    results = [
        ok("only POI->POI relations kept", rels == {"IS_NEAR_TO", "FOLLOWED_BY"}),
        ok("HAS_CATEGORY / LOCATED_IN / SUBCATEGORY_OF dropped",
           not rels & {"HAS_CATEGORY", "LOCATED_IN", "SUBCATEGORY_OF"}),
        ok("self-loops dropped", all(h != t for h, _, t in native)),
        ok("duplicate (h,r,t) dropped", len(native) == len({(h, r, t) for h, r, t in native})),
        ok("ids stay inside poi_idx range", int(tensor[:, [0, 2]].max()) < n_pois),
        ok("relation ids are contiguous from 0",
           sorted(relation_to_id.values()) == list(range(len(relation_to_id)))),
        ok("derived taxonomy buckets by shared depth",
           {r for _, r, _ in derived} == {"SAME_TAXONOMY_L2", "SAME_TAXONOMY_L3"}),
        ok("per-relation cap applied", len(capped) == 10),
        ok("saved tensor round-trips", torch.equal(reloaded, tensor)),
    ]
    return all(results)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kg", default="data/kg_NYC_v3.gpickle",
                   help="a NetworkX KG (.gpickle/.graphml) with POI nodes keyed poi:{poi_idx}")
    p.add_argument("--kg-dir", default=None,
                   help="directory written by group/build_kg.py (kg_triples.pt + "
                        "kg_entities.json + kg_relations.json). Takes precedence over --kg.")
    p.add_argument("--meta", default="data/poi_metadata_NYC.csv")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--dataset", default="NYC")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="relation names to drop, e.g. --exclude CO_VISITED_WITH")
    p.add_argument("--derive", choices=["none", "taxonomy"], default="none",
                   help="add SAME_TAXONOMY_L{k} triples so the TransE term sees hierarchy")
    p.add_argument("--taxonomy-min-level", type=int, default=2)
    p.add_argument("--taxonomy-max-partners", type=int, default=8)
    p.add_argument("--max-per-relation", type=int, default=None,
                   help="downsample any relation above this count (IS_NEAR_TO is ~63k in v3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()

    if a.self_check:
        sys.exit(0 if _self_check() else 1)
    run(a)


if __name__ == "__main__":
    main()
