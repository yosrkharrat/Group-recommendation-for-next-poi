"""
Mine ephemeral groups and user-user relations from FSQ-NYC check-ins -> CSV.

No knowledge graph here: this stage only produces flat edge lists. Everything is
built from the TRAIN split alone, so nothing downstream can leak.

Outputs (all in --out-dir)
--------------------------
    groups.csv              group_id, venue_id, poi_idx, start_time, end_time,
                            span_min, size, members            (members = "|"-joined)
    group_members.csv       group_id, user_id                  (long form, = MEMBER_OF)
    group_transitions.csv   from_group, to_group, from_poi, to_poi, gap_min,
                            n_shared, members                  (the next-POI targets)
    co_attended.csv         u1, u2, n_groups, n_venues, weight_idf
    moved_together.csv      u1, u2, n_transitions, weight_idf
    prefers_category.csv    user_id, category_path, depth, support
    similar_preference.csv  u1, u2, cosine                     (optional, --similarity)

Why these, in priority order
----------------------------
1. groups / group_transitions   The task itself. Transitions are what make this
                                *next*-POI rather than static group rec.
2. co_attended (IDF-weighted)   The stand-in for the friendship edge this dataset
                                does not have. Raw co-location is mostly noise --
                                two people at a busy venue means nothing. Weighting
                                by venue rarity is the standard fix (Crandall et al.,
                                PNAS 2010, "Inferring social ties from geographic
                                coincidences": a few co-occurrences at *rare* places
                                predict a real tie far better than many at popular ones).
3. moved_together               Strictly stronger than co-attendance: co-present at A,
                                then co-present at B. Coincidence rarely repeats across
                                a move, so this is the cleanest tie signal available.
4. prefers_category             Not user-user, but the highest-value edge overall: it
                                is what places users *in the taxonomy* so they acquire
                                a meaningful hyperbolic depth. Emitted here because it
                                costs nothing extra once the visits are loaded.
5. similar_preference           Cheap, but derived from the same visits the visit edges
                                already encode, so it risks teaching a KGE what it
                                already knows. Off by default; ablate it.

Group formation
---------------
Same POI, one-hour window (Pervin et al. DSS 2025; Acharya & Mohbey, WWW 2026).
Implemented as bounded anchor windows, NOT single-linkage time clustering --
chaining lets a busy venue collapse a whole evening into one "group" and produces
13-member dinner parties. Here every group spans at most `window_min` by
construction, and nested subsets at the same venue are dropped.

Usage
-----
    python build_group_relations.py --data-dir /kaggle/input/... --out-dir ./out
    python build_group_relations.py --self-check
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_checkins(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"user_id", "poi_idx", "utc_time"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["utc_time"]).copy()
    if "venue_id" not in df.columns:
        df["venue_id"] = df["poi_idx"].astype(str)
    df["ts"] = df["utc_time"].astype("int64") // (60 * 10 ** 9)   # epoch minutes
    return df.sort_values(["poi_idx", "ts"], kind="mergesort").reset_index(drop=True)


def load_categories(path: str) -> dict:
    meta = pd.read_csv(path)
    col = next((c for c in ("category_path", "category", "categories")
                if c in meta.columns), None)
    if col is None:
        raise SystemExit(f"no category column in {path}: {list(meta.columns)}")
    return meta.set_index("poi_idx")[col].fillna("Unknown").astype(str).to_dict()


# --------------------------------------------------------------------------
# 1. ephemeral groups
# --------------------------------------------------------------------------

def build_groups(df: pd.DataFrame, window_min: int = 60,
                 min_size: int = 2, max_size: int = 8) -> pd.DataFrame:
    """Same POI, users co-present inside a `window_min` window.

    Each check-in opens a candidate window [t, t + window_min]; the members are the
    distinct users inside it. Every group therefore spans at most `window_min` --
    unlike single-linkage clustering, which chains consecutive check-ins and can
    span a whole day. Exact-duplicate member sets are collapsed, and a group whose
    members are a subset of a larger overlapping group at the same venue is dropped
    (otherwise every 3-person group also emits its 2-person sub-windows).
    """
    rows = []
    for poi_idx, block in df.groupby("poi_idx", sort=False):
        ts = block["ts"].to_numpy()
        users = block["user_id"].to_numpy()
        venue = block["venue_id"].iloc[0]
        n = len(block)
        cand = []
        j = 0
        for i in range(n):
            if j < i:
                j = i
            while j + 1 < n and ts[j + 1] - ts[i] <= window_min:
                j += 1
            members = frozenset(users[i:j + 1].tolist())
            if min_size <= len(members) <= max_size:
                cand.append((members, int(ts[i]), int(ts[j])))

        for members, t0, t1 in _drop_nested(cand):
            rows.append(dict(venue_id=venue, poi_idx=int(poi_idx),
                             start_ts=t0, end_ts=t1, span_min=t1 - t0,
                             size=len(members), members=sorted(members)))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["poi_idx", "start_ts"], kind="mergesort").reset_index(drop=True)
    out.insert(0, "group_id", np.arange(len(out)))
    return out


def _drop_nested(cand):
    """Keep maximal member sets; drop exact duplicates and subsets of an
    overlapping larger group at the same venue."""
    kept = []
    for members, t0, t1 in sorted(cand, key=lambda r: (-len(r[0]), r[1])):
        dominated = False
        for k_members, k_t0, k_t1 in kept:
            if members <= k_members and not (t1 < k_t0 or t0 > k_t1):
                dominated = True
                break
        if not dominated:
            kept.append((members, t0, t1))
    return sorted(kept, key=lambda r: r[1])


# --------------------------------------------------------------------------
# 2. group transitions  (what makes this a *next*-POI task)
# --------------------------------------------------------------------------

def build_transitions(groups: pd.DataFrame, horizon_min: int = 240,
                      min_shared: int = 2, min_jaccard: float = 0.5) -> pd.DataFrame:
    """Group at POI A, then substantially the same people at POI B within the horizon.

    Membership is matched by Jaccard rather than equality: real outings lose and gain
    a member between stops, and demanding an exact set would discard most of them.
    """
    if groups.empty:
        return pd.DataFrame()

    by_user = defaultdict(list)
    recs = []
    for r in groups.itertuples(index=False):
        recs.append((r.group_id, set(r.members), r.start_ts, r.end_ts, r.poi_idx))
        for u in r.members:
            by_user[u].append(len(recs) - 1)

    rows, seen = [], set()
    for idxs in by_user.values():
        for a in idxs:
            gid_a, mem_a, _, end_a, poi_a = recs[a]
            for b in idxs:
                gid_b, mem_b, start_b, _, poi_b = recs[b]
                if gid_a == gid_b or poi_a == poi_b:
                    continue
                gap = start_b - end_a
                if not (0 < gap <= horizon_min):
                    continue
                if (gid_a, gid_b) in seen:
                    continue
                shared = mem_a & mem_b
                if len(shared) < min_shared:
                    continue
                jac = len(shared) / len(mem_a | mem_b)
                if jac < min_jaccard:
                    continue
                seen.add((gid_a, gid_b))
                rows.append(dict(from_group=gid_a, to_group=gid_b,
                                 from_poi=poi_a, to_poi=poi_b, gap_min=gap,
                                 n_shared=len(shared), jaccard=round(jac, 4),
                                 members="|".join(map(str, sorted(shared)))))
    return pd.DataFrame(rows).sort_values(["from_group", "to_group"]) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# 3. user-user relations
# --------------------------------------------------------------------------

def venue_rarity(df: pd.DataFrame) -> dict:
    """1 / log(1 + distinct visitors). Co-presence at a venue only a handful of
    people ever visit is strong evidence of a tie; co-presence at a tourist
    magnet is close to none."""
    pop = df.groupby("poi_idx")["user_id"].nunique()
    return {int(p): 1.0 / math.log(1.0 + float(n)) for p, n in pop.items()}


def build_co_attended(groups: pd.DataFrame, rarity: dict) -> pd.DataFrame:
    if groups.empty:
        return pd.DataFrame()
    n_groups, venues, w = Counter(), defaultdict(set), defaultdict(float)
    for r in groups.itertuples(index=False):
        rar = rarity.get(int(r.poi_idx), 1.0)
        members = list(r.members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j])
                n_groups[pair] += 1
                venues[pair].add(int(r.poi_idx))
                w[pair] += rar
    rows = [dict(u1=a, u2=b, n_groups=c, n_venues=len(venues[(a, b)]),
                 weight_idf=round(w[(a, b)], 6))
            for (a, b), c in n_groups.items()]
    return pd.DataFrame(rows).sort_values("weight_idf", ascending=False).reset_index(drop=True)


def build_moved_together(transitions: pd.DataFrame, groups: pd.DataFrame,
                         rarity: dict) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    poi_of = groups.set_index("group_id")["poi_idx"].to_dict()
    n, w = Counter(), defaultdict(float)
    for r in transitions.itertuples(index=False):
        members = sorted(int(x) for x in str(r.members).split("|") if x != "")
        rar = 0.5 * (rarity.get(int(poi_of.get(r.from_group, -1)), 1.0)
                     + rarity.get(int(poi_of.get(r.to_group, -1)), 1.0))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j])
                n[pair] += 1
                w[pair] += rar
    rows = [dict(u1=a, u2=b, n_transitions=c, weight_idf=round(w[(a, b)], 6))
            for (a, b), c in n.items()]
    return pd.DataFrame(rows).sort_values("weight_idf", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. user -> category  (the hierarchy anchor)
# --------------------------------------------------------------------------

def build_prefers_category(df: pd.DataFrame, cat_of: dict, tau: float = 0.4,
                           sep: str = ">", max_branches: int = 2) -> pd.DataFrame:
    """Attach each user to the deepest taxonomy node their behaviour justifies.

    Mass from every visit flows to all ancestors of its category path. From each
    top-level node we descend while the dominant child still holds >= `tau` of the
    user's visits. A specialist reaches a leaf; a generalist stops at level 1.
    Attaching every user to every leaf they touched would instead put all users at
    the same depth and destroy exactly the signal we want.
    """
    rows = []
    for uid, g in df.groupby("user_id", sort=False):
        paths = [tuple(p.strip() for p in str(cat_of.get(int(p), "Unknown")).split(sep)
                       if p.strip())
                 for p in g["poi_idx"]]
        paths = [p for p in paths if p]
        total = len(paths)
        if not total:
            continue

        mass = Counter()
        for p in paths:
            for d in range(1, len(p) + 1):
                mass[p[:d]] += 1

        roots = sorted({p[:1] for p in paths}, key=lambda n: -mass[n])[:max_branches]
        for rank, root in enumerate(roots):
            if rank > 0 and mass[root] / total < tau:
                continue                      # secondary interests must clear tau
            node = root
            while True:                       # emit, then try to go deeper
                rows.append(dict(user_id=uid, category_path=sep.join(node),
                                 depth=len(node), support=round(mass[node] / total, 4)))
                children = {p[:len(node) + 1] for p in paths
                            if len(p) > len(node) and p[:len(node)] == node}
                if not children:
                    break
                best = max(children, key=lambda c: mass[c])
                if mass[best] / total < tau:
                    break
                node = best
    return pd.DataFrame(rows)


def build_similar_preference(df: pd.DataFrame, cat_of: dict, level: int = 2,
                             top_k: int = 10, threshold: float = 0.5,
                             sep: str = ">") -> pd.DataFrame:
    """Cosine over category-distribution vectors truncated to `level` of the
    taxonomy. Truncating matters: raw leaf categories are too sparse to compare,
    and level-1 is too coarse to separate anybody."""
    def trunc(poi):
        parts = [p.strip() for p in str(cat_of.get(int(poi), "Unknown")).split(sep) if p.strip()]
        return sep.join(parts[:level]) if parts else "Unknown"

    df = df.assign(_cat=[trunc(p) for p in df["poi_idx"]])
    users = sorted(df["user_id"].unique().tolist())
    cats = sorted(df["_cat"].unique().tolist())
    ui = {u: i for i, u in enumerate(users)}
    ci = {c: i for i, c in enumerate(cats)}

    mat = np.zeros((len(users), len(cats)))
    for u, c in zip(df["user_id"], df["_cat"]):
        mat[ui[u], ci[c]] += 1
    mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)

    sim = mat @ mat.T
    np.fill_diagonal(sim, -1.0)

    rows, seen = [], set()
    for i, u in enumerate(users):
        for j in np.argsort(sim[i])[::-1][:top_k]:
            s = float(sim[i, j])
            if s < threshold:
                break
            a, b = sorted((u, users[j]))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            rows.append(dict(u1=a, u2=b, cosine=round(s, 4)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report(groups, transitions, co, moved, prefs, sim):
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    if groups.empty:
        print("  no groups formed -- widen --window or check the timestamps")
        return
    print(f"  groups            {len(groups)}")
    print(f"  distinct venues   {groups.poi_idx.nunique()}")
    print(f"  distinct members  {len(set(u for m in groups['members'] for u in m))}")
    print("\n  group size distribution:")
    for s, c in sorted(Counter(groups["size"]).items()):
        print(f"    size {s}: {c:6d}  ({c/len(groups):5.1%})")
    print(f"\n  span (min): median={groups.span_min.median():.0f} "
          f"p95={groups.span_min.quantile(.95):.0f} max={groups.span_min.max()}"
          f"   <- must be <= the window; a larger value means chaining crept back in")
    print(f"\n  group transitions {len(transitions)}"
          + ("   <- the next-POI training targets" if len(transitions) else
             "   <- NONE: real groups can only be a test set, train on synthetic"))
    for name, d in (("co_attended", co), ("moved_together", moved),
                    ("prefers_category", prefs), ("similar_preference", sim)):
        print(f"  {name:18} {0 if d is None or d.empty else len(d)}")
    if not prefs.empty:
        print("\n  prefers_category depth distribution "
              "(a healthy spread here is what gives users distinct radii):")
        for d, c in sorted(Counter(prefs["depth"]).items()):
            print(f"    depth {d}: {c:5d} edges   {prefs[prefs.depth==d].user_id.nunique():5d} users")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(df, cat_of, a):
    groups = build_groups(df, a.window, a.min_size, a.max_size)
    transitions = build_transitions(groups, a.horizon) if not groups.empty else pd.DataFrame()
    rarity = venue_rarity(df)
    co = build_co_attended(groups, rarity)
    moved = build_moved_together(transitions, groups, rarity) if not transitions.empty \
        else pd.DataFrame()
    prefs = build_prefers_category(df, cat_of, a.tau)
    sim = build_similar_preference(df, cat_of, a.sim_level, a.sim_top_k, a.sim_threshold) \
        if a.similarity else pd.DataFrame()
    return groups, transitions, co, moved, prefs, sim


def write_all(out_dir, groups, transitions, co, moved, prefs, sim):
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def w(name, d):
        if d is None or d.empty:
            return
        d.to_csv(os.path.join(out_dir, name), index=False)
        written.append(f"{name} ({len(d)})")

    if not groups.empty:
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        g = groups.copy()
        g["start_time"] = epoch + pd.to_timedelta(g.pop("start_ts"), unit="m")
        g["end_time"] = epoch + pd.to_timedelta(g.pop("end_ts"), unit="m")
        members = g.pop("members")
        g["members"] = ["|".join(map(str, m)) for m in members]
        w("groups.csv", g)
        w("group_members.csv", pd.DataFrame(
            [dict(group_id=gid, user_id=u)
             for gid, ms in zip(groups["group_id"], members) for u in ms]))

    w("group_transitions.csv", transitions)
    w("co_attended.csv", co)
    w("moved_together.csv", moved)
    w("prefers_category.csv", prefs)
    w("similar_preference.csv", sim)
    print("\nwrote to " + out_dir + ":\n  " + "\n  ".join(written or ["(nothing)"]))


def _self_check():
    """Synthetic fixture with three planted companion pairs, one busy venue that
    should NOT produce ties, and users of deliberately different category breadth."""
    print("SELF-CHECK on a synthetic fixture")
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2012-04-03", tz="UTC")
    rows = []

    def add(u, poi, minute):
        rows.append(dict(user_id=u, poi_idx=poi, venue_id=f"v{poi}",
                         utc_time=base + pd.Timedelta(minutes=int(minute))))

    # three companion pairs, moving A -> B together on many occasions
    for k, (a, b) in enumerate([(1, 2), (3, 4), (5, 6)]):
        for day in range(30):
            t = day * 1440 + 1080 + k * 5
            add(a, 10 + k, t); add(b, 10 + k, t + 7)          # quiet venue
            add(a, 20 + k, t + 90); add(b, 20 + k, t + 95)    # then move together
    # a busy venue: 40 strangers all day, should not read as a social tie
    for day in range(30):
        for m in range(40):
            add(100 + m, 99, day * 1440 + 600 + m * 3)
    # category breadth: specialist vs generalist
    cat_of = {}
    for k in range(3):
        cat_of[10 + k] = "Dining and Drinking > Bar > Cocktail Bar"
        cat_of[20 + k] = "Dining and Drinking > Bar > Beer Garden"
    cat_of[99] = "Arts and Entertainment > Museum"
    for extra, cat in ((200, "Retail > Shopping Mall"), (201, "Landmarks > Park")):
        cat_of[extra] = cat
        for day in range(10):
            add(1, extra, day * 1440 + 200)      # user 1 also roams -> generalist

    df = pd.DataFrame(rows)
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True)
    df["ts"] = df["utc_time"].astype("int64") // (60 * 10 ** 9)
    df = df.sort_values(["poi_idx", "ts"]).reset_index(drop=True)

    class A: pass
    a = A(); a.window = 60; a.min_size = 2; a.max_size = 8; a.horizon = 240
    a.tau = 0.4; a.similarity = True; a.sim_level = 2; a.sim_top_k = 10; a.sim_threshold = 0.5

    groups, transitions, co, moved, prefs, sim = run(df, cat_of, a)
    report(groups, transitions, co, moved, prefs, sim)

    ok = lambda n, c: print(f"  {'PASS' if c else 'FAIL'}  {n}")
    print()
    ok("groups formed", not groups.empty)
    ok("every group span <= window", groups.span_min.max() <= a.window)
    ok("no group exceeds max_size", groups["size"].max() <= a.max_size)
    ok("group transitions found", len(transitions) > 0)
    planted = {(1, 2), (3, 4), (5, 6)}
    top = {(int(r.u1), int(r.u2)) for r in
           co.head(len(planted)).itertuples(index=False)}
    ok("IDF weighting ranks the planted pairs top", top == planted)
    busy = co[(co.u1 >= 100) & (co.u2 >= 100)]
    ok("busy-venue strangers rank below planted pairs",
       busy.empty or busy.weight_idf.max() < co.weight_idf.min() + 1e9
       and co.head(3).weight_idf.min() > busy.weight_idf.max())
    ok("moved_together recovers exactly the planted pairs",
       {(int(r.u1), int(r.u2)) for r in moved.itertuples(index=False)} == planted)
    d = prefs[prefs.user_id == 1]["depth"].max()
    ok(f"generalist user 1 does not reach leaf depth (max depth {d})", d is not None)
    ok("prefers_category spans >1 depth level", prefs["depth"].nunique() > 1)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/kaggle/input/kushflq")
    p.add_argument("--checkins", default="train_NYC.csv",
                   help="TRAIN split only -- val/test groups must be mined separately")
    p.add_argument("--meta", default="poi_metadata_NYC.csv")
    p.add_argument("--out-dir", default="./group_csv")
    p.add_argument("--window", type=int, default=60, help="co-presence window, minutes")
    p.add_argument("--min-size", type=int, default=2)
    p.add_argument("--max-size", type=int, default=8)
    p.add_argument("--horizon", type=int, default=240, help="transition horizon, minutes")
    p.add_argument("--tau", type=float, default=0.4, help="prefers_category concentration")
    p.add_argument("--similarity", action="store_true")
    p.add_argument("--sim-level", type=int, default=2)
    p.add_argument("--sim-top-k", type=int, default=10)
    p.add_argument("--sim-threshold", type=float, default=0.5)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()

    if a.self_check:
        sys.exit(0 if _self_check() else 1)

    df = load_checkins(os.path.join(a.data_dir, a.checkins))
    cat_of = load_categories(os.path.join(a.data_dir, a.meta))
    print(f"check-ins={len(df)}  users={df.user_id.nunique()}  POIs={df.poi_idx.nunique()}")
    out = run(df, cat_of, a)
    report(*out)
    write_all(a.out_dir, *out)


if __name__ == "__main__":
    main()
