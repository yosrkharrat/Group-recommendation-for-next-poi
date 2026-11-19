"""
Closest reading of LLMGPR's group rule, per dataset.

LLMGPR never states its window, its group identity or any filter (GROUP_RULE_SWEEP.md Part 2),
and §2.6 there suggests it built groups differently per dataset. Decision 2026-11-19: use, for
each dataset, the reading that comes closest to its Table 1 group statistics, and state it.

Score: match = mean over the four group statistics (#groups, #group check-ins, #check-ins per
group, #users per group) of min(ours/theirs, theirs/ours). 1.0 = exact; this is the same family
of score as the 0.978 Table-1 recovery in LLMGPR_TRACK.md.

Grid (all check-ins, direct friendship; co-presence = anchor + friends at the same POI within
+-W, GroupIM-style, deduplicated per outing):
    window      0 s (identical second) .. 1 day, plus "same calendar minute"
    identity    A member-set | B anchor | C circle (k=1, k=2)   -- see group_identity_probe.py
    min check-ins per group   1, 2, 3
    min group size            2, 3

    python src/group_match_search.py --dataset GOWALLA  --data-dir ./data/gowalla
    python src/group_match_search.py --dataset WEEPLACE --data-dir <dir with *_WEEPLACE.csv>
    python src/group_match_search.py --dataset LLMGPR   --data-dir ./data/llmgpr
Output: data/group_rule_sweep/match/{DATASET}_match.csv (all configs, ranked)
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
from build_groups import load_checkins, load_friend_graph  # noqa: E402
from group_identity_probe import copresent_events  # noqa: E402

TARGETS = {
    "LLMGPR":   dict(groups=1715, group_checkins=12594, ck_per_group=7.34, users_per_group=3.72),
    "WEEPLACE": dict(groups=923, group_checkins=11974, ck_per_group=12.97, users_per_group=4.37),
    "GOWALLA":  dict(groups=2186, group_checkins=7738, ck_per_group=3.54, users_per_group=2.95),
}
STATS = ["groups", "group_checkins", "ck_per_group", "users_per_group"]
# Table 1, individual columns. #POIs is LLMGPR's region CATALOGUE (LLMGPR_TRACK.md §1.4), so it
# is compared with --catalogue-pois, not with the visited-venue count.
TARGETS_IND = {
    "LLMGPR":   dict(users=7507, pois=80962, user_checkins=1214631, ck_per_user=162.80),
    "WEEPLACE": dict(users=4560, pois=44194, user_checkins=623654, ck_per_user=136.77),
    "GOWALLA":  dict(users=31751, pois=81123, user_checkins=862502, ck_per_user=27.16),
}
IND = ["users", "pois", "user_checkins", "ck_per_user"]
WINDOWS_S = [0, 30, 60, 120, 300, 900, 1800, 3600, 10800, 86400]


def match(row, target):
    r = [min(row[k] / target[k], target[k] / row[k]) if row[k] and row[k] == row[k] else 0.0
         for k in STATS]
    return float(np.mean(r)), r


def tally(groups, ck, min_ck, min_size):
    keep = [g for g in groups if ck[g] >= min_ck and len(groups[g]) >= min_size]
    n = len(keep)
    if not n:
        return dict(groups=0, group_checkins=0, ck_per_group=np.nan, users_per_group=np.nan,
                    users_in_groups=0)
    tot = sum(ck[g] for g in keep)
    return dict(groups=n, group_checkins=tot, ck_per_group=tot / n,
                users_per_group=float(np.mean([len(groups[g]) for g in keep])),
                users_in_groups=len(set().union(*[groups[g] for g in keep])))


def member_set_outings(events, window_s):
    """One outing per (member-set, POI) chain. Every member of a mutually-friendly group is an
    anchor once, so a k-person outing arrives as up to k views with different anchor times; views
    of the same set at the same POI whose times chain within the window are ONE group check-in.
    (Fixed 2026-11-19: an earlier version keyed on the anchor's minute and counted each anchor's
    view separately, inflating A/C check-ins by up to the group size for windows > 1 min.)
    Returns [(members, poi, t_first)]."""
    out, last = [], {}
    for a, g, p, t in sorted(events, key=lambda e: (e[2], hash(e[1]), e[3])):
        k = (g, p)
        if k in last and t - last[k] <= window_s:
            last[k] = t
            continue
        last[k] = t
        out.append((g, p, t))
    return out


def identity_tables(events, window_s):
    """events: (anchor, frozenset members, poi, t). Returns {identity: (groups, ck)}."""
    out = {}
    ms_ck, ms = Counter(), {}
    for g, p, t in member_set_outings(events, window_s):   # A: exact member-set
        ms_ck[g] += 1
        ms[g] = g
    out["A_member_set"] = (ms, ms_ck)
    an_mem, an_ck = defaultdict(set), Counter()      # B: anchor + all co-present friends
    for a, g, p, t in events:
        an_mem[a] |= g
        an_ck[a] += 1
    out["B_anchor"] = (an_mem, an_ck)
    pair = Counter()                                 # C: circles of co-present friends
    for g in ms:
        mem = sorted(g)
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                pair[(mem[i], mem[j])] += ms_ck[g]
    for k in (1, 2):
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
        out[f"C_circle_k{k}"] = (comps, c_ck)
    return out


def same_minute_events(df, G):
    """Same POI, same calendar minute (UTC), direct friends: the reading that fit Gowalla in
    GROUP_RULE_SWEEP.md §2.2 (c). Shaped like copresent_events' output."""
    minute = df["utc_time"].dt.floor("min").astype("int64") // 10 ** 9
    tmp = pd.DataFrame({"p": df["poi_idx"].to_numpy(), "m": minute.to_numpy(),
                        "u": df["user_id"].to_numpy()}).drop_duplicates()
    sizes = tmp.groupby(["p", "m"]).size()
    tmp = tmp.set_index(["p", "m"]).loc[sizes[sizes >= 2].index].reset_index()
    ev = []
    for (p, m), g in tmp.groupby(["p", "m"], sort=False):
        users = [int(u) for u in g["u"] if G.has_node(int(u))]
        for u in users:
            fr = {v for v in users if v != u and G.has_edge(u, v)}
            if fr:
                ev.append((u, frozenset(fr | {u}), int(p), int(m)))
    seen, out = set(), []
    for a, g, p, m in ev:                       # one view per anchor per (set, poi, minute)
        if (a, g, p, m) not in seen:
            seen.add((a, g, p, m))
            out.append((a, g, p, m))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(TARGETS))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--friend-files", nargs="+", default=None,
                    help="default: friendship_old_<DATASET>.csv in --data-dir")
    ap.add_argument("--windows", nargs="+", type=int, default=WINDOWS_S, help="seconds")
    ap.add_argument("--out-dir", default="./data/group_rule_sweep/match")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--catalogue-pois", type=int, default=None,
                    help="POI catalogue size for the #POIs column (default: visited venues)")
    ap.add_argument("--label", default="", help="data-variant label stored in every row")
    a = ap.parse_args()
    t0 = time.time()
    target = TARGETS[a.dataset]
    df = load_checkins(a.data_dir, a.dataset)
    df["ts"] = df["utc_time"].dt.as_unit("ns").astype("int64") // 10 ** 9        # seconds
    files = a.friend_files or [os.path.join(a.data_dir, f"friendship_old_{a.dataset}.csv")]
    G = load_friend_graph(files)
    print(f"{a.dataset}: {len(df):,} check-ins, {df.user_id.nunique():,} users; friendship "
          f"{G.number_of_nodes():,} users / {G.number_of_edges():,} edges", flush=True)

    ind = dict(users=int(df.user_id.nunique()),
               pois=int(a.catalogue_pois or df.poi_idx.nunique()),
               user_checkins=int(len(df)))
    ind["ck_per_user"] = ind["user_checkins"] / ind["users"]
    ti = TARGETS_IND[a.dataset]
    ind_r = {k: min(ind[k] / ti[k], ti[k] / ind[k]) for k in IND}
    print("individual columns vs Table 1: " + ", ".join(
        f"{k} {ind[k]:,.2f} ({ind[k] / ti[k]:.3f}x)" for k in IND), flush=True)

    rows = []
    settings = [("same_minute", None)] + [(f"{w}s", w) for w in a.windows]
    for label, w in settings:
        ev = same_minute_events(df, G) if w is None else copresent_events(df, G, w)
        for ident, (groups, ck) in identity_tables(ev, 0 if w is None else w).items():
            for min_ck in (1, 2, 3):
                for min_size in (2, 3):
                    r = tally(groups, ck, min_ck, min_size)
                    score, parts = match(r, target)
                    m8 = (sum(parts) + sum(ind_r.values())) / 8
                    rows.append(dict(variant=a.label, window=label, identity=ident,
                                     min_checkins=min_ck, min_size=min_size, match=round(score, 4),
                                     match8=round(m8, 4),
                                     **{k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in r.items()},
                                     **{f"r_{k}": round(x, 3) for k, x in zip(STATS, parts)},
                                     **{f"r_{k}": round(v, 3) for k, v in ind_r.items()}))
        print(f"  {label:>12}: {len(ev):,} anchor views  ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows).sort_values("match", ascending=False).reset_index(drop=True)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"{a.dataset}_match.csv")
    out.to_csv(path, index=False)
    print(f"\nLLMGPR {a.dataset}: {target}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.head(a.top).to_string(index=False))
    print(f"wrote {path}  ({len(out)} configs, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
