"""
Does "connected on the social network" use the FULL friendship graph?

Our pipeline's friendship file keeps only edges among the dataset's 31,667 Gowalla users. The crawl
LLMGPR cites (Liu et al. CIKM'13, figshare 22126586) has a friendship graph over all ~319k users,
so two co-present users with no direct tie may still be connected through users outside the
dataset. If LLMGPR's "connected" used that graph, groups would be larger. Their Gowalla reports
2.95 users per group, and every reading on the in-dataset graph gives about 2.06.

Co-presence is fixed at the reading that matches their group count and check-ins: same POI,
same calendar minute. The friendship rules, all connected components of the co-present set:
    direct       an edge in the in-dataset graph (reference; = group_rule_sweep bucket1 hops1_cc)
    full_2hop    direct, or sharing a friend in the FULL graph (that friend may be anyone)
    full_comp    in the same connected component of the FULL graph (any path length)

    python src/group_fullgraph_probe.py
Output: data/group_rule_sweep/fullgraph/GOWALLA_fullgraph.csv
"""

import os
import sys
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import load_checkins, resplit_per_user  # noqa: E402
from group_rule_sweep import LLMGPR, bucket_candidates, summarise  # noqa: E402

DATA, RAW = "./data/gowalla", "./data/gowalla_raw/gowalla_friendship.csv"


def main():
    t0 = time.time()
    df = resplit_per_user(load_checkins(DATA, "GOWALLA"))
    raw_of = pd.read_csv(os.path.join(DATA, "users_GOWALLA.csv")).set_index("user_id")["raw_id"].to_dict()

    fr = pd.read_csv(RAW, dtype={"userid1": "int64", "userid2": "int64"})
    fr = fr[fr.userid1 != fr.userid2]
    nodes, inv = np.unique(np.concatenate([fr.userid1.values, fr.userid2.values]), return_inverse=True)
    m = len(fr)
    a, b = inv[:m], inv[m:]
    n = len(nodes)
    A = coo_matrix((np.ones(m, dtype=np.int8), (a, b)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).tocsr()
    n_comp, comp = connected_components(A, directed=False)
    idx_of = {int(r): i for i, r in enumerate(nodes)}
    print(f"full graph: {n:,} users, {A.nnz // 2:,} undirected edges, {n_comp:,} components; "
          f"largest {np.bincount(comp).max():,}  ({time.time() - t0:.0f}s)", flush=True)

    def nbrs(u):                       # neighbour set of a dataset user, as full-graph indices
        i = idx_of.get(raw_of[u])
        return set(A.indices[A.indptr[i]:A.indptr[i + 1]].tolist()) if i is not None else set()

    nb_cache = {}

    def N(u):
        if u not in nb_cache:
            nb_cache[u] = nbrs(u)
        return nb_cache[u]

    def ix(u):
        return idx_of.get(raw_of[u])

    cands = bucket_candidates(df, 1)
    target = LLMGPR["GOWALLA"]
    rows = []
    for rule in ("direct", "full_2hop", "full_comp"):
        events, pois = [], []
        for poi, _, users in cands:
            users = sorted(users)
            sub = nx.Graph()
            sub.add_nodes_from(users)
            for i, u in enumerate(users):
                iu = ix(u)
                if iu is None:
                    continue
                for v in users[i + 1:]:
                    iv = ix(v)
                    if iv is None:
                        continue
                    if rule == "full_comp":
                        ok = comp[iu] == comp[iv]
                    else:
                        ok = iv in N(u) or (rule == "full_2hop" and bool(N(u) & N(v)))
                    if ok:
                        sub.add_edge(u, v)
            for c in nx.connected_components(sub):
                if 2 <= len(c) <= 8:
                    events.append(frozenset(c))
                    pois.append(poi)
        for s in summarise(events, [1, 2, 3], pois):
            s.update(rule=rule, copresence="bucket1", scope="all",
                     groups_vs_llmgpr=round(s["groups"] / target["groups"], 3),
                     ck_vs_llmgpr=round(s["group_checkins"] / target["group_checkins"], 3))
            rows.append(s)
        print(f"  {rule:<10s} done ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)[["rule", "copresence", "scope", "min_events", "groups", "group_checkins",
                              "ck_per_group", "ck_per_group_median", "users_per_group", "share_pairs",
                              "users_in_groups", "groups_vs_llmgpr", "ck_vs_llmgpr"]]
    os.makedirs("./data/group_rule_sweep/fullgraph", exist_ok=True)
    path = "./data/group_rule_sweep/fullgraph/GOWALLA_fullgraph.csv"
    out.to_csv(path, index=False)
    print(f"LLMGPR Gowalla: {target}")
    print(out.to_string(index=False))
    print(f"wrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
