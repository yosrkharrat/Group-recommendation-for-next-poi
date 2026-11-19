"""
Group-IDENTITY probe: what is "a group" in LLMGPR's Table 1?

`group_rule_sweep.py` showed that once a group is an exact member-set that recurs, 88-100% of
groups are pairs (mean size 2.0-2.1) under every window and friendship rule, including GroupIM's
"individual + her friends within 15 minutes". LLMGPR reports 3.72 (Foursquare), 4.37 (Weeplace)
and 2.95 (Gowalla). So the mismatch is not the co-presence rule; it is what counts as ONE group.
This probe holds the co-presence rule fixed (direct friends, co-present within W minutes at the
same POI) and varies the group identity:

  A  member-set   a group is an exact set of users; its check-ins are the events of that set
                  (the sweep's reading, for reference)
  B  anchor       a group is one ANCHOR user (GroupIM's "an individual and her subset of
                  friends"); members = anchor + every friend ever co-present with them; check-ins =
                  the anchor's co-present check-ins
  C  circle       a group is a connected component of the graph "friends who were co-present at
                  least k times"; check-ins = events where >= 2 of its members are co-present

Each identity is reported with groups filtered to >= 1/2/3 check-ins, and with the friendship
graph as `old` (our default, avoids leakage) or `old+new` (Foursquare only; Gowalla has one
snapshot).

    python src/group_identity_probe.py --dataset LLMGPR  --data-dir ./data/llmgpr
    python src/group_identity_probe.py --dataset GOWALLA --data-dir ./data/gowalla
Output: data/group_rule_sweep/identity/{DATASET}_identity.csv
"""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import load_checkins, load_friend_graph, resplit_per_user  # noqa: E402
from group_rule_sweep import LLMGPR  # noqa: E402

FRIENDS = {"LLMGPR": {"old": ["friendship_old_LLMGPR.csv"],
                      "old+new": ["friendship_old_LLMGPR.csv", "friendship_new_only_LLMGPR.csv"]},
           "GOWALLA": {"old": ["friendship_old_GOWALLA.csv"]}}


def copresent_events(df, G, window_min):
    """Events = (poi, anchor_time, frozenset of users) where users are the anchor plus its
    friends with a check-in at the same POI within +-W. Duplicate views of one outing (same set,
    same POI, anchor times chaining within W) are merged. Also returns, per event, the anchor."""
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
                raw.append((int(u), frozenset(fr | {int(u)}), int(poi), int(ts[i])))
    raw.sort(key=lambda r: (r[2], r[0], hash(r[1]), r[3]))
    out, last = [], {}
    for a, g, p, t in raw:
        k = (a, g, p)
        if k in last and t - last[k] <= window_min:
            last[k] = t
            continue
        last[k] = t
        out.append((a, g, p, t))
    return out


def stats(groups, ck, recurrence, tag, target):
    """groups: {gid: set(members)}, ck: {gid: n_checkins}."""
    rows = []
    for r in recurrence:
        keep = [g for g in groups if ck[g] >= r and len(groups[g]) >= 2]
        n = len(keep)
        tot = sum(ck[g] for g in keep)
        sizes = [len(groups[g]) for g in keep]
        rows.append(dict(identity=tag, min_checkins=r, groups=n, group_checkins=tot,
                         ck_per_group=round(tot / n, 3) if n else np.nan,
                         ck_median=float(np.median([ck[g] for g in keep])) if n else np.nan,
                         users_per_group=round(float(np.mean(sizes)), 3) if n else np.nan,
                         size_median=float(np.median(sizes)) if n else np.nan,
                         max_size=max(sizes) if n else 0,
                         users_in_groups=len(set().union(*[groups[g] for g in keep])) if n else 0,
                         groups_vs_llmgpr=round(n / target["groups"], 3),
                         ck_vs_llmgpr=round(tot / target["group_checkins"], 3)))
    return rows


def identities(events, recurrence, target, circle_k, window=0):
    rows = []
    # A: exact member-set. Each member of a k-person outing is an anchor once; views of the same
    # set at the same POI whose times chain within `window` are ONE outing. (Fixed 2026-11-19:
    # keying on the anchor's minute counted every anchor's view, inflating A/C for window > 1 min.)
    ms_ck, ms, last = Counter(), {}, {}
    for a, g, p, t in sorted(events, key=lambda e: (e[2], hash(e[1]), e[3])):
        k = (g, p)
        if k in last and t - last[k] <= window:
            last[k] = t
            continue
        last[k] = t
        ms_ck[g] += 1
        ms[g] = g
    rows += stats(ms, ms_ck, recurrence, "A_member_set", target)

    # B: anchor identity
    an_mem, an_ck = defaultdict(set), Counter()
    for a, g, p, t in events:
        an_mem[a] |= g
        an_ck[a] += 1
    rows += stats(an_mem, an_ck, recurrence, "B_anchor", target)

    # C: circles = components of the co-present-friends graph (pair co-present >= k times)
    pair = Counter()
    for g in ms:
        mem = sorted(g)
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                pair[(mem[i], mem[j])] += ms_ck[g]
    for k in circle_k:
        H = nx.Graph([e for e, c in pair.items() if c >= k])
        comp_of, comps = {}, {}
        for cid, comp in enumerate(nx.connected_components(H)):
            comps[cid] = set(comp)
            for u in comp:
                comp_of[u] = cid
        c_ck = Counter()
        for g in ms:
            cids = Counter(comp_of[u] for u in g if u in comp_of)
            for cid, m in cids.items():
                if m >= 2:
                    c_ck[cid] += ms_ck[g]
        rows += stats(comps, c_ck, recurrence, f"C_circle_k{k}", target)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["LLMGPR", "GOWALLA"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--windows", nargs="+", type=int, default=[15, 60])
    p.add_argument("--recurrence", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--circle-k", nargs="+", type=int, default=[1, 2])
    p.add_argument("--out-dir", default="./data/group_rule_sweep/identity")
    p.add_argument("--seconds", action="store_true",
                   help="interpret --windows in SECONDS (ts at 1-s resolution); window 0 = identical "
                        "timestamp, the most literal reading of 'the same time'")
    a = p.parse_args()
    t0 = time.time()
    df = resplit_per_user(load_checkins(a.data_dir, a.dataset))   # all check-ins used below
    if a.seconds:
        df["ts"] = df["utc_time"].dt.as_unit("ns").astype("int64") // 10 ** 9
    target = LLMGPR[a.dataset]
    rows = []
    for fname, files in FRIENDS[a.dataset].items():
        G = load_friend_graph([os.path.join(a.data_dir, f) for f in files])
        for w in a.windows:
            ev = copresent_events(df, G, w)
            for r in identities(ev, a.recurrence, target, a.circle_k, w):
                r.update(friendship=fname, window=w, friend_edges=G.number_of_edges(),
                         window_unit="s" if a.seconds else "min")
                rows.append(r)
            print(f"  friendship={fname:<8s} ({G.number_of_edges():,} edges) W={w:>3} min: "
                  f"{len(ev):,} anchor views  ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    cols = ["friendship", "window", "window_unit", "identity", "min_checkins", "groups", "group_checkins",
            "ck_per_group", "ck_median", "users_per_group", "size_median", "max_size",
            "users_in_groups", "groups_vs_llmgpr", "ck_vs_llmgpr", "friend_edges"]
    out = out[cols]
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"{a.dataset}_identity.csv")
    out.to_csv(path, index=False)
    print(f"LLMGPR target: {target}")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(out.drop(columns=["friend_edges"]).to_string(index=False))
    print(f"wrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
