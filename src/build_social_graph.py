"""
Weighted user-user social graph (Foursquare / LLMGPR first).

What it is, and what it is not
------------------------------
A weighted ADJACENCY between users, kept as separate relations. It is NOT a second embedding:
stage 5 places each member on the POI ball (`_member_signature`, the gyromidpoint of their top
POIs), and a separately trained user-user KGE would live in a different space that the
gyromidpoint consensus cannot mix with POIs. The graph changes how members are COMBINED (member
weights in the consensus, one-hop smoothing of a member's signature towards their ties). It does
not add new user coordinates. Decisions are in HANDOFF_SOCIAL_GRAPH_BUILD.md §3-4, agreed 2026-11-19.

Relations (undirected, stored once with u1 < u2)
------------------------------------------------
    FRIEND_OF    friendship_old only. weight 1. `friendship_new_only` postdates the check-ins and is
                 the friendship-prediction eval set: it never enters (asserted).
    CO_PRESENT   same venue within `--window` minutes (`build_groups.build_ephemeral_groups`: bounded
                 anchor windows, no single-linkage chaining, groups of 2..`--max-size`), with NO
                 friendship requirement, from TRAIN-split check-ins only, pairs met
                 >= `--min-meetings` times (`tie_pool`'s default: on the raw graph most pairs met
                 once, which is what two strangers at the same bar look like).
                 weight = `weight_idf` (Σ 1/log(1+distinct visitors of the venue): rare venues count
                 more) or `weight_count` (log1p(meetings)), picked on validation data (below).
    SIMILAR_TO   left out of v1: affinity comes from the same check-ins as the member signatures,
                 so it is partly circular. Ablation later.

Why train-split edges cannot leak into a member's val/test events
-----------------------------------------------------------------
The splits are per-user chronological (70/10/20), not one global time cut. A CO_PRESENT meeting is
two check-ins at the same time, each in its owner's TRAIN split, so it precedes every val/test
check-in of BOTH endpoints. Edges incident to a user therefore only use evidence from before that
user's held-out period.

The LLMGPR reproduction set (data/groups_llmgpr_match/LLMGPR) is built from the whole history
(house8_whole) with old+new friendship, so its held-out outings can also sit in our train split.
Every val/test/masked outing of that set is removed from the train check-ins before co-presence is
computed: the outing's members' check-ins at that venue inside that set's manifest window.
The default is per-set (three-city +/-1440 min, whole-history +/-60 min); --guard may widen it. Matched on
venue_id: that set's poi_idx is a different id space; user_id is shared (checked on 5,085 joined
rows: 5,083 agree).

Choosing the CO_PRESENT weight on data, not taste
-------------------------------------------------
Labels: pairs co-present in the VALIDATION period (same rule, val-split check-ins of both users).
Score: each weighting, 0 for non-edges. Mann-Whitney AUC over ALL user pairs, with exact average
ranks for ties (≈ all pairs tie at 0, which argsort-based AUC gets wrong), plus the AUC restricted
to CO_PRESENT edges. Both weightings share the same support, so they can only differ in how they rank
the edges, and the all-pairs AUC hides that (on LLMGPR idf and count agree there to 4 decimals).
`--co-weight auto` therefore keeps the higher WITHIN-EDGE AUC (ties go to idf). The all-pairs AUC is
reported as a coverage measure (and to compare --min-meetings 1 vs 2).

Measured on LLMGPR (432 validation co-present pairs):
    within-edge AUC  idf 0.830  count 0.771   (min 2 meetings)   -> idf
    all-pairs AUC    min1 0.566 (covers 57)  min2 0.537 (covers 32)  FRIEND_OF 0.532 (covers 28)

Outputs (--out-dir)
-------------------
    social_edges_<DS>.csv           u1, u2, relation, weight, weight_idf, weight_count, n_meetings, n_venues
                                    (n_meetings / n_venues count ALL train meetings, also those
                                    below the threshold, so a FRIEND_OF row shows how often the
                                    friends actually met). Dataset-suffixed because stage 5's
                                    find() resolves files by exact name anywhere under data/.
    social_graph_manifest_<DS>.json inputs (+ sha256), every config value, counts, sanity stats, weighting AUCs,
                                    audit results, git head

Usage
-----
    python src/build_social_graph.py                 # LLMGPR defaults
    python src/build_social_graph.py --self-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import build_co_attended, build_ephemeral_groups, load_checkins, venue_rarity  # noqa: E402
from group_outing_guard import (drop_outing_checkins, file_sha256, group_guard_policy, holdout_outings)  # noqa: E402

COLS = ["u1", "u2", "relation", "weight", "weight_idf", "weight_count", "n_meetings", "n_venues"]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def load_pairs(path):
    """Undirected pair set {(min, max)} from a `u1,u2` CSV; self-pairs dropped."""
    e = pd.read_csv(path, dtype={"u1": int, "u2": int})
    return {(min(a, b), max(a, b)) for a, b in zip(e["u1"], e["u2"]) if a != b}


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------

def co_presence(df, window_min, max_size):
    """All co-present pairs in `df` (any number of meetings): u1 < u2, n_groups, n_venues,
    weight_idf. Rarity is measured on the same frame, so train edges never see val visitors."""
    groups = build_ephemeral_groups(df, window_min=window_min, min_size=2, max_size=max_size)
    co = build_co_attended(groups, venue_rarity(df))
    if co.empty:
        return pd.DataFrame(columns=["u1", "u2", "n_groups", "n_venues", "weight_idf"]), groups
    co[["u1", "u2"]] = np.sort(co[["u1", "u2"]].to_numpy(), axis=1)   # members are sorted; be sure
    return co, groups


def build_edges(friends_old, co, min_meetings, co_weight):
    stats = co.set_index(["u1", "u2"])[["n_groups", "n_venues", "weight_idf"]]
    fr = pd.DataFrame(sorted(friends_old), columns=["u1", "u2"])
    fr = fr.join(stats, on=["u1", "u2"]).fillna({"n_groups": 0, "n_venues": 0, "weight_idf": 0.0})
    fr = fr.assign(relation="FRIEND_OF", weight=1.0)

    cp = co[co["n_groups"] >= min_meetings].copy().assign(relation="CO_PRESENT")
    cp["weight"] = cp["weight_idf"] if co_weight == "idf" else np.log1p(cp["n_groups"])

    edges = pd.concat([fr, cp], ignore_index=True)
    edges["weight_count"] = np.log1p(edges["n_groups"])
    edges = edges.rename(columns={"n_groups": "n_meetings"})
    edges[["u1", "u2", "n_meetings", "n_venues"]] = edges[["u1", "u2", "n_meetings", "n_venues"]].astype(int)
    edges[["weight", "weight_idf", "weight_count"]] = edges[["weight", "weight_idf", "weight_count"]].round(6)
    return edges[COLS].sort_values(["relation", "u1", "u2"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# weighting selection
# --------------------------------------------------------------------------

def pair_auc(scores, positives, n_pairs):
    """Mann-Whitney AUC over a universe of `n_pairs` pairs where only the pairs in `scores`
    ({pair: s > 0}) are non-zero, with exact average ranks for ties."""
    P = len(positives)
    if P == 0 or P == n_pairs:
        return float("nan")
    s = pd.Series(scores, dtype=float)
    Z = n_pairs - len(s)                                   # pairs at score 0
    r = s.rank(method="average") + Z                       # every non-zero score outranks the zeros
    pos_in = [p for p in positives if p in scores]
    rank_sum = float(r.loc[pos_in].sum()) if pos_in else 0.0
    rank_sum += (P - len(pos_in)) * (Z + 1) / 2.0          # positives tied at 0
    return (rank_sum - P * (P + 1) / 2.0) / (P * (n_pairs - P))


def edge_auc(scores, positives):
    """AUC restricted to the scored pairs (the edges themselves)."""
    keys = list(scores)
    pos = {k for k in keys if k in positives}
    if not pos or len(pos) == len(keys):
        return float("nan")
    return pair_auc(scores, pos, len(keys))


def select_weighting(co, friends_old, val_df, window_min, max_size, min_meetings, n_users):
    val_co, _ = co_presence(val_df, window_min, max_size)
    Y = set(zip(val_co["u1"], val_co["u2"]))
    N = n_users * (n_users - 1) // 2
    out = {"label": f"pairs co-present in the validation split (±{window_min} min, val check-ins of both users)",
           "n_users": n_users, "n_pairs": N, "n_positive": len(Y), "candidates": {}}
    for mm in sorted({1, min_meetings}):
        sub = co[co["n_groups"] >= mm]
        keys = list(zip(sub["u1"], sub["u2"]))
        for name, vals in (("idf", sub["weight_idf"]), ("count", np.log1p(sub["n_groups"]))):
            sc = dict(zip(keys, vals.astype(float)))
            out["candidates"][f"{name}_min{mm}"] = dict(
                n_edges=len(sc), positives_covered=sum(p in sc for p in Y),
                auc_all_pairs=round(pair_auc(sc, Y, N), 5), auc_within_edges=round(edge_auc(sc, Y), 5))
    fsc = {p: 1.0 for p in friends_old}
    out["reference_friend_of"] = dict(n_edges=len(fsc), positives_covered=sum(p in fsc for p in Y),
                                      auc_all_pairs=round(pair_auc(fsc, Y, N), 5))
    # Both weightings have the same support, so they differ only in how they rank the edges. The
    # all-pairs AUC hides that: the ~23.7M zero-score pairs swamp it, and on LLMGPR idf and count
    # agree there to 4 decimals. Choose on the within-edge AUC; the all-pairs AUC measures coverage.
    a_idf = out["candidates"][f"idf_min{min_meetings}"]["auc_within_edges"]
    a_cnt = out["candidates"][f"count_min{min_meetings}"]["auc_within_edges"]
    out["chosen_by"] = "auc_within_edges"
    out["chosen"] = "count" if a_cnt > a_idf else "idf"
    return out


# --------------------------------------------------------------------------
# audit + sanity statistics
# --------------------------------------------------------------------------

def audit(edges, train_used, friends_old, friends_new, outings, window_min, max_size):
    res = {}
    assert set(train_used["orig_split"].unique()) <= {"train"}, "non-train check-in in co-presence input"
    res["checkins_used_all_train"] = True

    assert (edges["u1"] < edges["u2"]).all(), "edge not stored as u1 < u2 (or a self-loop)"
    assert not edges.duplicated(["u1", "u2", "relation"]).any(), "duplicate edge"
    fo = edges[edges["relation"] == "FRIEND_OF"]
    fpairs = set(zip(fo["u1"], fo["u2"]))
    assert fpairs <= friends_old, "FRIEND_OF pair not in friendship_old"
    leak = fpairs & (friends_new - friends_old)
    assert not leak, f"{len(leak)} friendship_new_only pairs in FRIEND_OF"
    res["friend_of_subset_of_old"] = True
    res["new_only_pairs_in_friend_of"] = 0
    cp = set(zip(*[edges.loc[edges["relation"] == "CO_PRESENT", c] for c in ("u1", "u2")]))
    res["co_present_pairs_that_are_new_only_friends"] = len(cp & (friends_new - friends_old))
    res["note_new_only"] = ("allowed: those CO_PRESENT edges come from train check-ins, not from the "
                            "friendship file. Co-presence predicting a later friendship is expected.")

    # no held-out reproduction outing survives as a train co-presence of >= 2 of its members
    n_bad = 0
    if outings is not None and not outings.empty:
        _, groups = co_presence(train_used, window_min, max_size)
        if not groups.empty:
            byv = {v: x for v, x in groups.groupby("venue_id")}
            for (_, venue, ts), o in outings.groupby(["outing", "venue_id", "ev_ts"]):
                x = byv.get(venue)
                if x is None:
                    continue
                mem = set(o["user_id"])
                near = x[(x["end_ts"] >= ts - window_min) & (x["start_ts"] <= ts + window_min)]
                if any(len(mem & set(m)) >= 2 for m in near["members"]):
                    n_bad += 1
        assert n_bad == 0, f"{n_bad} held-out reproduction outings still co-present in train"
    res["heldout_repro_outings_left"] = n_bad
    return res


def _q(s):
    return {k: round(float(v), 4) for k, v in
            zip(("min", "p25", "p50", "p75", "p90", "p99", "max"),
                np.quantile(s, [0, .25, .5, .75, .9, .99, 1]))} if len(s) else {}


def sanity_stats(edges, users_train, member_sets):
    st = {}
    for rel, e in edges.groupby("relation"):
        deg = pd.concat([e["u1"], e["u2"]]).value_counts()
        st[rel] = dict(edges=len(e), users=int(deg.size),
                       share_of_train_users=round(deg.size / len(users_train), 4),
                       degree=_q(deg.to_numpy()), weight=_q(e["weight"].to_numpy()),
                       n_meetings=_q(e["n_meetings"].to_numpy()))
    f = set(zip(*[edges.loc[edges["relation"] == "FRIEND_OF", c] for c in ("u1", "u2")]))
    c = set(zip(*[edges.loc[edges["relation"] == "CO_PRESENT", c] for c in ("u1", "u2")]))
    st["overlap"] = dict(friend_and_copresent=len(f & c), copresent_only=len(c - f), friend_only=len(f - c),
                         union=len(f | c),
                         friends_who_met_at_least_once=int((edges.loc[edges["relation"] == "FRIEND_OF",
                                                                      "n_meetings"] > 0).sum()))
    allu = set(edges["u1"]) | set(edges["u2"])
    st["isolated_train_users"] = len(set(users_train) - allu)

    # coverage of group members: does a member have >= 1 edge, and >= 1 edge INSIDE their group?
    adj = {}
    for a, b in f | c:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    st["group_member_coverage"] = {}
    for name, groups in member_sets.items():
        n_m = n_any = n_in = n_groups = n_conn = 0
        for mem in groups:
            mem = set(mem)
            n_groups += 1
            inside = [len(adj.get(u, set()) & (mem - {u})) for u in mem]
            n_m += len(mem)
            n_any += sum(1 for u in mem if adj.get(u))
            n_in += sum(1 for k in inside if k > 0)
            n_conn += all(k > 0 for k in inside)
        st["group_member_coverage"][name] = dict(
            groups=n_groups, members=n_m,
            member_has_any_edge=round(n_any / max(n_m, 1), 4),
            member_has_edge_inside_group=round(n_in / max(n_m, 1), 4),
            groups_every_member_tied_inside=round(n_conn / max(n_groups, 1), 4))
    return st


def read_member_sets(path):
    if not os.path.exists(path):
        return None
    gm = pd.read_csv(path)
    return [list(v) for _, v in gm.groupby("group_id")["user_id"]]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def run(a):
    t0 = time.time()
    ds = a.dataset
    f_old = os.path.join(a.data_dir, f"friendship_old_{ds}.csv")
    f_new = os.path.join(a.data_dir, f"friendship_new_only_{ds}.csv")
    df = load_checkins(a.data_dir, ds)
    train = df[df["orig_split"] == "train"]
    val = df[df["orig_split"] == "val"]
    friends_old, friends_new = load_pairs(f_old), load_pairs(f_new)
    users_train = sorted(train["user_id"].unique())

    a.repro_dir = [d for d in (a.repro_dir or []) if d]
    outings = holdout_outings(a.repro_dir) if a.repro_dir else None
    train_used, n_dropped = drop_outing_checkins(train, outings, a.guard)
    assert drop_outing_checkins(train_used, outings, a.guard)[1] == 0
    guard_policy = group_guard_policy(a.repro_dir, a.guard)
    print(f"train check-ins {len(train):,}  dropped as held-out reproduction outings {n_dropped:,}")

    co, _ = co_presence(train_used, a.window, a.max_size)
    print(f"co-present pairs (>=1 meeting) {len(co):,}  (>= {a.min_meetings}: "
          f"{int((co['n_groups'] >= a.min_meetings).sum()):,})")

    sel = select_weighting(co, friends_old, val, a.window, a.max_size, a.min_meetings, len(users_train))
    co_weight = sel["chosen"] if a.co_weight == "auto" else a.co_weight
    print(f"weighting on validation co-presence ({sel['n_positive']:,} positive pairs):")
    for k, v in sel["candidates"].items():
        print(f"  {k:<11} edges {v['n_edges']:>6,}  covers {v['positives_covered']:>4}  "
              f"AUC all {v['auc_all_pairs']:.4f}  within edges {v['auc_within_edges']:.4f}")
    rf = sel["reference_friend_of"]
    print(f"  FRIEND_OF   edges {rf['n_edges']:>6,}  covers {rf['positives_covered']:>4}  AUC all {rf['auc_all_pairs']:.4f}")
    print(f"  -> CO_PRESENT weight = {co_weight}" + (" (auto)" if a.co_weight == "auto" else " (forced)"))

    edges = build_edges(friends_old, co, a.min_meetings, co_weight)
    aud = audit(edges, train_used, friends_old, friends_new, outings, a.window, a.max_size)

    member_sets = {}
    for name, p in ([(f"repro_{os.path.basename(d.rstrip('/'))}", os.path.join(d, "group_members.csv"))
                     for d in a.repro_dir]
                    + [("groups_social_180min_5hop", os.path.join(a.data_dir, "groups_social", "group_members.csv"))]):
        ms = read_member_sets(p)
        if ms is not None:
            member_sets[name] = ms
    stats = sanity_stats(edges, users_train, member_sets)

    os.makedirs(a.out_dir, exist_ok=True)
    edges.to_csv(os.path.join(a.out_dir, f"social_edges_{ds}.csv"), index=False)
    inputs = {p: sha256(p) for p in (f_old, f_new, *(os.path.join(a.data_dir, f"{s}_{ds}.csv")
                                                    for s in ("train", "val", "test")))}
    for d in a.repro_dir:
        p = os.path.join(d, "group_checkins.csv")
        inputs[p] = sha256(p)
    manifest = dict(
        dataset=ds, built=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), git_head=git_head(),
        script="src/build_social_graph.py", undirected=True, storage="one row per pair and relation, u1 < u2",
        social_edges_sha256=file_sha256(os.path.join(a.out_dir, f"social_edges_{ds}.csv")),
        config=dict(window_min=a.window, max_group_size=a.max_size, min_meetings=a.min_meetings,
                    co_weight=co_weight, co_weight_arg=a.co_weight, guard_min=a.guard,
                    heldout_repro_splits=["val", "test", "masked"], friendship="old only",
                    guard_policy=guard_policy),
        inputs=inputs,
        counts=dict(train_checkins=len(train), dropped_repro_outing_checkins=n_dropped,
                    train_checkins_used=len(train_used), train_users=len(users_train),
                    copresent_pairs_any=len(co), edges=len(edges),
                    by_relation=edges["relation"].value_counts().to_dict()),
        weighting_selection=sel, audit=aud, stats=stats, seconds=round(time.time() - t0, 1))
    with open(os.path.join(a.out_dir, f"social_graph_manifest_{ds}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=int)

    print(f"\nedges {len(edges):,}  " + "  ".join(f"{k}={v:,}" for k, v in manifest["counts"]["by_relation"].items()))
    for rel in ("FRIEND_OF", "CO_PRESENT"):
        s = stats.get(rel, {})
        print(f"  {rel:<10} users {s.get('users', 0):,} ({s.get('share_of_train_users', 0):.1%} of train users)  "
              f"degree p50 {s.get('degree', {}).get('p50')}  p99 {s.get('degree', {}).get('p99')}")
    print(f"  overlap {stats['overlap']}")
    print(f"  isolated train users {stats['isolated_train_users']:,}")
    for k, v in stats["group_member_coverage"].items():
        print(f"  coverage [{k}] {v}")
    print(f"  audit {json.dumps({k: v for k, v in aud.items() if k != 'note_new_only'})}")
    print(f"wrote {a.out_dir}/social_edges_{ds}.csv + social_graph_manifest_{ds}.json  ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------
# self-check (synthetic fixture, no data needed)
# --------------------------------------------------------------------------

def _brute_auc(scores, positives, universe):
    pos = [scores.get(p, 0.0) for p in universe if p in positives]
    neg = [scores.get(p, 0.0) for p in universe if p not in positives]
    tot = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
    return tot / (len(pos) * len(neg))


def _self_check():
    import itertools
    import random
    rng = random.Random(0)
    users = list(range(12))
    universe = list(itertools.combinations(users, 2))
    for _ in range(50):
        sc = {p: float(rng.choice([1, 2, 2, 3, 0.5])) for p in rng.sample(universe, 20)}
        pos = set(rng.sample(universe, 9))
        assert abs(pair_auc(sc, pos, len(universe)) - _brute_auc(sc, pos, universe)) < 1e-12
        if 0 < len(pos & set(sc)) < len(sc):
            assert abs(edge_auc(sc, pos) - _brute_auc(sc, pos & set(sc), list(sc))) < 1e-12
    print("pair_auc / edge_auc == brute force on 50 random cases")

    # fixture: users 0..5, venues A..D (poi 0..3), epoch minutes
    rows = []

    def ck(u, poi, minute, split="train"):
        rows.append(dict(user_id=u, poi_idx=poi, venue_id="ABCD"[poi], ts=minute, orig_split=split))

    for k in range(3):                       # 0 & 1 meet 3 times, at two venues
        ck(0, k % 2, 1000 * k)
        ck(1, k % 2, 1000 * k + 10)
    ck(2, 2, 5000); ck(3, 2, 5030)           # 2 & 3 meet once  -> below min_meetings=2
    ck(4, 3, 7000); ck(5, 3, 7020)           # 4 & 5 meet twice, but the 2nd is a held-out outing
    ck(4, 3, 9000); ck(5, 3, 9005)
    ck(2, 2, 20000, "val"); ck(3, 2, 20010, "val")   # validation co-presence: 2 & 3
    ck(0, 0, 20000, "val"); ck(1, 1, 20010, "val")   # different venues: no label
    df = pd.DataFrame(rows)
    train = df[df["orig_split"] == "train"]
    outings = pd.DataFrame(dict(outing=[0, 0], group_id=[9, 9], split=["test", "test"], venue_id=["D", "D"],
                                ev_ts=[9002, 9002], user_id=[4, 5]))
    tr_used, n_drop = drop_outing_checkins(train, outings, guard_min=120)
    assert n_drop == 2, n_drop
    co, _ = co_presence(tr_used, 60, 8)
    got = {(r.u1, r.u2): r.n_groups for r in co.itertuples()}
    assert got == {(0, 1): 3, (2, 3): 1, (4, 5): 1}, got
    # without the guard, 4 & 5 would have met twice
    assert dict(((r.u1, r.u2), r.n_groups) for r in co_presence(train, 60, 8)[0].itertuples())[(4, 5)] == 2

    friends_old = {(0, 1), (3, 5)}
    friends_new = {(0, 1), (3, 5), (2, 4)}           # (2, 4) is new-only
    edges = build_edges(friends_old, co, min_meetings=2, co_weight="idf")
    rel = {(r.u1, r.u2, r.relation): r for r in edges.itertuples()}
    assert set(rel) == {(0, 1, "FRIEND_OF"), (3, 5, "FRIEND_OF"), (0, 1, "CO_PRESENT")}, set(rel)
    assert rel[(0, 1, "FRIEND_OF")].n_meetings == 3 and rel[(3, 5, "FRIEND_OF")].n_meetings == 0
    assert rel[(0, 1, "CO_PRESENT")].n_venues == 2
    assert abs(rel[(0, 1, "CO_PRESENT")].weight_count - math.log1p(3)) < 1e-6
    aud = audit(edges, tr_used, friends_old, friends_new, outings, 60, 8)
    assert aud["heldout_repro_outings_left"] == 0

    try:                                              # the audit must catch a surviving outing
        audit(edges, train, friends_old, friends_new, outings, 60, 8)
        raise AssertionError("audit missed a held-out outing")
    except AssertionError as e:
        assert "still co-present" in str(e), e
    bad = pd.concat([edges, pd.DataFrame([dict(u1=2, u2=4, relation="FRIEND_OF", weight=1.0, weight_idf=0.0,
                                               weight_count=0.0, n_meetings=0, n_venues=0)])])
    try:
        audit(bad, tr_used, friends_old, friends_new, outings, 60, 8)
        raise AssertionError("audit missed a new-only friendship")
    except AssertionError as e:
        assert "not in friendship_old" in str(e), e
    try:
        audit(edges, df, friends_old, friends_new, None, 60, 8)
        raise AssertionError("audit missed a val check-in")
    except AssertionError as e:
        assert "non-train" in str(e), e

    sel = select_weighting(co, friends_old, df[df["orig_split"] == "val"], 60, 8, 2, 6)
    assert sel["n_positive"] == 1 and sel["candidates"]["idf_min1"]["positives_covered"] == 1
    assert sel["candidates"]["idf_min2"]["positives_covered"] == 0
    print("self-check passed: guard, thresholds, FRIEND_OF leak, train-only and held-out-outing audits, selection")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default="./data/llmgpr")
    p.add_argument("--dataset", default="LLMGPR")
    p.add_argument("--repro-dir", nargs="*",
                   default=["./data/groups_llmgpr_match/LLMGPR_3CITY", "./data/groups_llmgpr_match/LLMGPR"],
                   help="reproduction group set(s) whose held-out outings are removed from train "
                        "(none given = no guard). Default: the three-city set the comparison evaluates "
                        "on, plus the whole-history set kept for the record.")
    p.add_argument("--out-dir", default="./data/social_graph/LLMGPR")
    p.add_argument("--window", type=int, default=60, help="co-presence window, minutes")
    p.add_argument("--max-size", type=int, default=8, help="larger co-present sets are crowds, not ties")
    p.add_argument("--min-meetings", type=int, default=2)
    p.add_argument("--co-weight", choices=("auto", "idf", "count"), default="auto")
    p.add_argument("--guard", type=float, default=None,
                   help="default: each set's manifest window; an override may only widen it")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        _self_check()
        return
    run(a)


if __name__ == "__main__":
    main()
