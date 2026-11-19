"""
Group-rule sweep: how many groups does each reading of LLMGPR's rule produce?

LLMGPR (CIKM'25, §4.1, via CubeRec [3]) defines a group as "a set of users who are connected on
the social network [who] visit the same venue at the same time". Three things in that sentence are
never pinned down, by LLMGPR or by CubeRec:

  * "same time"  -> the co-presence window
  * "connected"  -> direct friends, or anything reachable through the friendship graph
  * what a *group* is -> one co-presence event, or a member-set that recurs

The pipeline (`build_groups.py --group-source social`) currently answers 180-min floor buckets /
<=5 friendship hops / connected components / no recurrence filter. Those values were ported from
`group-recommendation-using-fsq-section3.ipynb` (720 min there) by commit aa6679d; no sweep chose
them. This script measures every combination against LLMGPR's Table 1 so the choice is made from
numbers.

Grid
----
    copresence   bucket60 | bucket180   floor(local time / W) per venue  (the pipeline's method)
                 rolling60              anchor windows [t, t+60], maximal sets  (build_ephemeral_groups)
    friend rule  hops5_cc   connected components, edges = reachable within 5 hops  (current pipeline)
                 hops1_cc   connected components, edges = direct friendship
                 clique     maximal cliques of direct friendship: EVERY pair are friends
    recurrence   a member-set is a group if it has >= r co-presence events, r in {1, 2, 3}
    scope        all    every check-in (comparable to LLMGPR's Table 1, which is dataset-wide)
                 train  the per-user 70% train split the pipeline mines groups from

Group check-ins = events of the kept member-sets; #users per group = mean member-set size.

Sanity anchor: scope=train / bucket180 / hops5_cc / r=1 must reproduce the committed
`n_ephemeral_groups` (FSQ 4,804, Gowalla 36,621). The script asserts it.

Usage
-----
    python src/group_rule_sweep.py --self-check
    python src/group_rule_sweep.py --dataset LLMGPR  --data-dir ./data/llmgpr
    python src/group_rule_sweep.py --dataset GOWALLA --data-dir ./data/gowalla
Outputs: data/group_rule_sweep/{DATASET}_sweep.csv + {DATASET}_sweep.json
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import (_drop_nested, load_checkins, load_friend_graph,  # noqa: E402
                          resplit_per_user)

# LLMGPR Table 1. Their Foursquare is the full 3-city dataset (1.21M check-ins); the pipeline's
# FSQ arm (data/llmgpr) is a ~423k-check-in subset, so FSQ counts here undershoot by construction.
LLMGPR = {
    "LLMGPR": dict(users=7507, groups=1715, group_checkins=12594, ck_per_group=7.34,
                   users_per_group=3.72, user_checkins=1214631),
    "GOWALLA": dict(users=31751, groups=2186, group_checkins=7738, ck_per_group=3.54,
                    users_per_group=2.95, user_checkins=862502),
}
COMMITTED_PIPELINE_EVENTS = {"LLMGPR": 4804, "GOWALLA": 36621}

FRIEND_FILE = {"LLMGPR": "friendship_old_LLMGPR.csv", "GOWALLA": "friendship_old_GOWALLA.csv"}


# --------------------------------------------------------------------------
# co-presence candidates: (poi, t0, frozenset(users)) with >= 2 distinct users
# --------------------------------------------------------------------------

def bucket_candidates(df, window_min):
    """Identical membership to build_groups.venue_time_bucket_clusters, vectorised pre-filter."""
    tz = df["timezone_offset"].to_numpy() if "timezone_offset" in df.columns else np.zeros(len(df))
    ts_local = df["ts"].to_numpy() + tz
    bucket = np.floor(ts_local / window_min).astype(np.int64) * window_min
    tmp = pd.DataFrame({"poi": df["poi_idx"].to_numpy(), "b": bucket,
                        "u": df["user_id"].to_numpy(), "ts": df["ts"].to_numpy()})
    nu = tmp.drop_duplicates(["poi", "b", "u"]).groupby(["poi", "b"]).size()
    keep = nu[nu >= 2].index
    tmp = tmp.set_index(["poi", "b"]).loc[keep].reset_index()
    out = []
    for (poi, _), g in tmp.groupby(["poi", "b"], sort=False):
        out.append((int(poi), int(g["ts"].min()), frozenset(g["u"].tolist())))
    return out


def rolling_candidates(df, window_min, max_size):
    """Anchor windows [t, t+W] per venue, maximal sets only -- build_ephemeral_groups' method.
    Sets above max_size are dropped BEFORE nesting, exactly as there."""
    out = []
    for poi, block in df.groupby("poi_idx", sort=False):
        block = block.sort_values("ts", kind="mergesort")
        ts, users = block["ts"].to_numpy(), block["user_id"].to_numpy()
        n, cand, j = len(block), [], 0
        for i in range(n):
            j = max(j, i)
            while j + 1 < n and ts[j + 1] - ts[i] <= window_min:
                j += 1
            m = frozenset(users[i:j + 1].tolist())
            if 2 <= len(m) <= max_size:
                cand.append((m, int(ts[i]), int(ts[j])))
        for m, t0, _ in _drop_nested(cand):
            out.append((int(poi), t0, m))
    return out


# --------------------------------------------------------------------------
# friendship rules applied to one co-present set
# --------------------------------------------------------------------------

class FriendRules:
    def __init__(self, G, max_hops=5):
        self.G, self.max_hops, self._reach = G, max_hops, {}

    def _within(self, u):
        if u not in self._reach:
            self._reach[u] = nx.single_source_shortest_path_length(self.G, u, cutoff=self.max_hops)
        return self._reach[u]

    def groups(self, users, rule):
        present = [u for u in users if self.G.has_node(u)]
        if len(present) < 2:
            return []
        sub = nx.Graph()
        sub.add_nodes_from(present)
        if rule == "hops5_cc":
            for i, u in enumerate(present):
                r = self._within(u)
                for v in present[i + 1:]:
                    if v in r:
                        sub.add_edge(u, v)
        else:
            sub.add_edges_from(self.G.subgraph(present).edges())
        if rule == "clique":
            parts = [c for c in nx.find_cliques(sub) if len(c) >= 2]
        else:
            parts = [c for c in nx.connected_components(sub) if len(c) >= 2]
        return [frozenset(c) for c in parts]


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def summarise(events, recurrence, pois=None):
    """events: list of frozenset member-sets (one per group check-in); pois: the venue of each
    event, parallel to `events`. `group_pois` counts DISTINCT venues per kept set -- the reading
    under which a group's "check-ins" are the places it went, not how often it went there."""
    per_set = Counter(events)
    venues = {}
    if pois is not None:
        for e, p in zip(events, pois):
            venues.setdefault(e, set()).add(p)
    rows = []
    for r in recurrence:
        kept = {s: c for s, c in per_set.items() if c >= r}
        n_sets, n_ck = len(kept), sum(kept.values())
        users = set().union(*kept) if kept else set()
        row = dict(
            min_events=r, groups=n_sets, group_checkins=n_ck,
            ck_per_group=round(n_ck / n_sets, 3) if n_sets else float("nan"),
            ck_per_group_median=float(np.median(list(kept.values()))) if n_sets else float("nan"),
            users_per_group=round(float(np.mean([len(s) for s in kept])), 3) if n_sets else float("nan"),
            share_pairs=round(float(np.mean([len(s) == 2 for s in kept])), 3) if n_sets else float("nan"),
            users_in_groups=len(users))
        if pois is not None:
            gp = sum(len(venues[s]) for s in kept)
            row.update(group_pois=gp, pois_per_group=round(gp / n_sets, 3) if n_sets else float("nan"))
        rows.append(row)
    return rows


def ego_events(df, G, window_min, max_size):
    """GroupIM's rule (SIGIR'20 §5.1), the only paper in LLMGPR's citation chain that states one:
    "check-ins at the same POI within 15 minutes by an individual and her subset of friends in the
    social network together constitutes a single group interaction". Ego-centric: every member is
    a direct friend of the ANCHOR, not necessarily of each other.

    For each check-in (u, p, t): group = {u} + friends of u with a check-in at p in [t-W, t+W].
    The same outing is seen from several anchors (and from consecutive check-ins of one anchor),
    so identical (member-set, POI) events whose anchor times chain within W are merged into one.
    Returns (events, pois) like the other paths."""
    raw = []
    for poi, block in df.groupby("poi_idx", sort=False):
        if len(block) < 2:
            continue
        block = block.sort_values("ts", kind="mergesort")
        ts, users = block["ts"].to_numpy(), block["user_id"].to_numpy()
        n, lo = len(block), 0
        for i in range(n):
            u = users[i]
            if not G.has_node(u):
                continue
            while ts[i] - ts[lo] > window_min:
                lo += 1
            hi = i
            while hi + 1 < n and ts[hi + 1] - ts[i] <= window_min:
                hi += 1
            nb = G[u]
            fr = {int(users[j]) for j in range(lo, hi + 1) if users[j] != u and users[j] in nb}
            if fr:
                g = frozenset(fr | {int(u)})
                if len(g) <= max_size:
                    raw.append((g, int(poi), int(ts[i])))
    raw.sort(key=lambda r: (r[1], hash(r[0]), r[2]))
    events, pois, last = [], [], {}
    for g, p, t in raw:
        k = (g, p)
        if k in last and t - last[k] <= window_min:
            last[k] = t                    # same outing, chained
            continue
        last[k] = t
        events.append(g)
        pois.append(p)
    return events, pois


def _min_size(events, pois, k):
    """Keep only events whose member-set has >= k users (k=2 keeps everything)."""
    if k <= 2:
        return events, pois
    keep = [i for i, g in enumerate(events) if len(g) >= k]
    return [events[i] for i in keep], [pois[i] for i in keep]


def sweep(df_scopes, G, copresence, rules, recurrence, max_size, dataset, min_size=2):
    fr = FriendRules(G)
    target = LLMGPR.get(dataset, {})
    rows = []
    for scope, sdf in df_scopes.items():
        for cp in copresence:
            t0 = time.time()
            if cp.startswith("ego"):
                events, pois = ego_events(sdf, G, int(cp[len("ego"):]), max_size)
                events, pois = _min_size(events, pois, min_size)
                for s in summarise(events, recurrence, pois):
                    s.update(scope=scope, copresence=cp, friend_rule="ego_anchor",
                             candidates=len(events))
                    if target:
                        s["groups_vs_llmgpr"] = round(s["groups"] / target["groups"], 3)
                        s["ck_vs_llmgpr"] = round(s["group_checkins"] / target["group_checkins"], 3)
                    rows.append(s)
                print(f"  [{scope:5s}] {cp:<10s} {len(events):>8,} ego events  "
                      f"({time.time() - t0:5.1f}s)", flush=True)
                continue
            if cp.startswith("bucket"):
                cands = bucket_candidates(sdf, int(cp[len("bucket"):]))
            else:
                cands = rolling_candidates(sdf, int(cp[len("rolling"):]), max_size)
            for rule in rules:
                events, pois = [], []
                for poi, _, users in cands:
                    for g in fr.groups(sorted(users), rule):
                        if 2 <= len(g) <= max_size:
                            events.append(g)
                            pois.append(poi)
                events, pois = _min_size(events, pois, min_size)
                for s in summarise(events, recurrence, pois):
                    s.update(scope=scope, copresence=cp, friend_rule=rule,
                             candidates=len(cands))
                    if target:
                        s["groups_vs_llmgpr"] = round(s["groups"] / target["groups"], 3)
                        s["ck_vs_llmgpr"] = round(s["group_checkins"] / target["group_checkins"], 3)
                    rows.append(s)
            print(f"  [{scope:5s}] {cp:<10s} {len(cands):>8,} co-present sets  "
                  f"({time.time() - t0:5.1f}s)", flush=True)
    cols = ["scope", "copresence", "friend_rule", "min_events", "candidates", "groups",
            "group_checkins", "ck_per_group", "ck_per_group_median", "group_pois", "pois_per_group",
            "users_per_group", "share_pairs", "users_in_groups", "groups_vs_llmgpr", "ck_vs_llmgpr"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]]


def self_check():
    """Planted fixture: one venue, friends a-b-c in a chain (a~b, b~c), d a stranger 2 hops from
    a via e. Checks the three friend rules and the recurrence count."""
    G = nx.Graph([(1, 2), (2, 3), (1, 5), (5, 4)])
    fr = FriendRules(G)
    users = [1, 2, 3, 4]
    cc5 = fr.groups(users, "hops5_cc")
    cc1 = fr.groups(users, "hops1_cc")
    cl = fr.groups(users, "clique")
    assert cc5 == [frozenset({1, 2, 3, 4})], cc5            # 4 reachable from 1 via 5
    assert cc1 == [frozenset({1, 2, 3})], cc1               # 4 has no direct friend present
    assert sorted(map(sorted, cl)) == [[1, 2], [2, 3]], cl  # 1,3 not friends -> no triangle
    ev = [frozenset({1, 2})] * 3 + [frozenset({2, 3})] * 2 + [frozenset({1, 3})]
    s = {r["min_events"]: r for r in summarise(ev, [1, 2, 3])}
    assert (s[1]["groups"], s[1]["group_checkins"]) == (3, 6)
    assert (s[2]["groups"], s[2]["group_checkins"]) == (2, 5)
    assert (s[3]["groups"], s[3]["group_checkins"]) == (1, 3)
    s = {r["min_events"]: r for r in summarise(ev, [2], pois=[7, 7, 8, 9, 9, 9])}
    assert s[2]["group_pois"] == 3, s       # {1,2} went to venues 7,8; {2,3} only to 9
    # bucket vs rolling on a toy venue: 10:00 a, 10:50 b, 11:10 c (60-min windows)
    base = pd.Timestamp("2012-01-01 10:00", tz="UTC")
    toy = pd.DataFrame(dict(user_id=[1, 2, 3], poi_idx=0, timezone_offset=0,
                            utc_time=[base, base + pd.Timedelta("50min"), base + pd.Timedelta("70min")]))
    toy["ts"] = toy["utc_time"].dt.as_unit("ns").astype("int64") // (60 * 10 ** 9)
    b = sorted(sorted(m) for _, _, m in bucket_candidates(toy, 60))
    r = sorted(sorted(m) for _, _, m in rolling_candidates(toy, 60, 8))
    assert b == [[1, 2]], b                  # 11:10 falls in the next floor bucket
    assert r == [[1, 2], [2, 3]], r          # rolling sees both overlaps
    # GroupIM ego rule: anchor 2 is friends with 1 and 3, who are not friends with each other.
    # 10:00 u1, 10:05 u2, 10:10 u3, all at venue 0; W=15.
    Gs = nx.Graph([(1, 2), (2, 3)])
    ego = pd.DataFrame(dict(user_id=[1, 2, 3], poi_idx=0, ts=[600, 605, 610]))
    ev, _ = ego_events(ego, Gs, 15, 8)
    assert sorted(map(sorted, ev)) == [[1, 2], [1, 2, 3], [2, 3]], ev   # star from anchor 2
    ev2, _ = ego_events(pd.DataFrame(dict(user_id=[1, 2, 1, 2], poi_idx=0,
                                          ts=[600, 601, 610, 611])), Gs, 15, 8)
    assert len(ev2) == 1, ev2                 # one outing, seen 4 times -> merged
    print("self-check passed: friend rules, recurrence counting, bucket vs rolling")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["LLMGPR", "GOWALLA"])
    p.add_argument("--data-dir")
    p.add_argument("--out-dir", default="./data/group_rule_sweep")
    p.add_argument("--copresence", nargs="+", default=["bucket180", "bucket60", "rolling60"])
    p.add_argument("--rules", nargs="+", default=["hops5_cc", "hops1_cc", "clique"])
    p.add_argument("--recurrence", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--scopes", nargs="+", default=["train", "all"])
    p.add_argument("--max-size", type=int, default=8, help="matches the pipeline's --max-size")
    p.add_argument("--min-size", type=int, default=2,
                   help="drop groups with fewer members (3 = no pairs)")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check()
        return
    if not (a.dataset and a.data_dir):
        p.error("--dataset and --data-dir are required")

    t0 = time.time()
    df = load_checkins(a.data_dir, a.dataset)
    df = resplit_per_user(df)                     # pipeline default (--resplit, 70/10/20)
    G = load_friend_graph([os.path.join(a.data_dir, FRIEND_FILE[a.dataset])])
    print(f"{a.dataset}: {len(df):,} check-ins, {df.user_id.nunique():,} users; friendship "
          f"{G.number_of_nodes():,} users / {G.number_of_edges():,} edges")
    scopes = {"train": df[df["split"] == "train"], "all": df}
    res = sweep({s: scopes[s] for s in a.scopes}, G, a.copresence, a.rules, a.recurrence,
                a.max_size, a.dataset, a.min_size)

    anchor = res[(res.scope == "train") & (res.copresence == "bucket180")
                 & (res.friend_rule == "hops5_cc") & (res.min_events == 1)]
    if len(anchor):
        got, want = int(anchor["group_checkins"].iloc[0]), COMMITTED_PIPELINE_EVENTS[a.dataset]
        assert got == want, f"pipeline anchor mismatch: {got} events vs committed {want}"
        print(f"  anchor OK: train/bucket180/hops5_cc reproduces the committed {want:,} events")

    os.makedirs(a.out_dir, exist_ok=True)
    csv = os.path.join(a.out_dir, f"{a.dataset}_sweep.csv")
    res.to_csv(csv, index=False)
    meta = dict(dataset=a.dataset, data_dir=a.data_dir, n_checkins=int(len(df)),
                n_users=int(df.user_id.nunique()), friendship_edges=G.number_of_edges(),
                friendship_users=G.number_of_nodes(), max_size=a.max_size, min_size=a.min_size,
                llmgpr_table1=LLMGPR[a.dataset], runtime_s=round(time.time() - t0, 1),
                rows=res.to_dict(orient="records"))
    with open(os.path.join(a.out_dir, f"{a.dataset}_sweep.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(res.to_string(index=False))
    print(f"wrote {csv}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
