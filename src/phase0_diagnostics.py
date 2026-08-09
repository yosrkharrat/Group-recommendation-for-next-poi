"""
Phase-0 go/no-go diagnostics for the group extension. CPU only, ~1 minute.

Run these BEFORE committing GPU time. Each answers a question that, if answered
badly, invalidates a different part of the proposed design:

  D1  Radial hierarchy.  Does `poi_hyperbolic_embs.npy` actually place general
      categories near the origin and specific ones near the boundary? The whole
      "consensus of a divided group is automatically more generic" mechanism
      rests on this. If Spearman(depth, radius) is not clearly positive, the
      RotH objective did not produce a radial hierarchy and must be fixed
      (add a depth-ranking regulariser) before anything is built on top.

  D2  Consensus semantics.  When two POIs come from different branches of the
      taxonomy, does their gyromidpoint move toward their common ancestor --
      and does the "depth drop" grow with how far apart the branches are?
      This is the claim of the paper's core contribution, tested directly on
      the trained embeddings, with no model involved.

  D3  Group mining feasibility.  How many real co-visit groups (and how many
      group *transitions*, i.e. group next-POI targets) does FSQ-NYC actually
      yield at various time windows? Decides whether the evaluation can use
      real groups (T1) or must fall back to synthetic ones (T2).

Usage
-----
    python phase0_diagnostics.py --data-dir /kaggle/input/kushflq
    python phase0_diagnostics.py --self-check      # synthetic fixture, no data
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# numpy versions of the ball primitives (keeps this script torch-free)
# --------------------------------------------------------------------------

BALL_EPS = 1e-5


def np_project(x, c=1.0):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-15)
    m = (1.0 - BALL_EPS) / np.sqrt(c)
    return np.where(n > m, x / n * m, x)


def np_radius(x, c=1.0):
    n = np.linalg.norm(x, axis=-1)
    n = np.clip(n, 0.0, (1.0 - BALL_EPS) / np.sqrt(c))
    return 2.0 / np.sqrt(c) * np.arctanh(np.sqrt(c) * n)


def np_gyromidpoint(x, weights=None, c=1.0):
    """Einstein midpoint via the Klein model. x: (..., k, d)."""
    x = np_project(np.asarray(x, dtype=np.float64), c)
    if weights is None:
        weights = np.ones(x.shape[:-1])
    x2 = (x * x).sum(-1, keepdims=True)
    xk = 2.0 * x / (1.0 + c * x2)
    xk2 = (xk * xk).sum(-1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - c * xk2, 1e-15))
    wg = (weights * gamma)[..., None]
    mk = (wg * xk).sum(-2) / np.maximum(wg.sum(-2), 1e-15)
    mk2 = (mk * mk).sum(-1, keepdims=True)
    return np_project(mk / (1.0 + np.sqrt(np.maximum(1.0 - c * mk2, 1e-15))), c)


def spearman(a, b):
    """Rank correlation without scipy."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------
# D1 -- radial hierarchy
# --------------------------------------------------------------------------

def d1_radial_hierarchy(embs, cat_paths, c=1.0, sep=">"):
    """embs: (N, d) on the ball. cat_paths: list[str] full taxonomy path per POI."""
    print("\n" + "=" * 74)
    print("D1  Radial hierarchy: does depth in the taxonomy predict hyperbolic radius?")
    print("=" * 74)

    depth = np.array([len([p for p in str(s).split(sep) if p.strip()])
                      for s in cat_paths], dtype=np.float64)
    r = np_radius(np_project(embs, c), c)

    keep = np.isfinite(r) & (depth > 0)
    rho = spearman(depth[keep], r[keep])

    print(f"  POIs: {keep.sum()}   embedding norms: "
          f"min={np.linalg.norm(embs,axis=-1).min():.4f} "
          f"med={np.median(np.linalg.norm(embs,axis=-1)):.4f} "
          f"max={np.linalg.norm(embs,axis=-1).max():.4f}")
    print(f"  radius:  min={r[keep].min():.3f}  med={np.median(r[keep]):.3f}  "
          f"max={r[keep].max():.3f}")
    print(f"\n  Spearman(taxonomy depth, hyperbolic radius) = {rho:+.4f}")
    print("\n  mean radius by depth:")
    for d in sorted(set(depth[keep].astype(int))):
        m = keep & (depth == d)
        if m.sum():
            print(f"    depth {d}:  n={m.sum():5d}   mean radius = {r[m].mean():.4f}"
                  f"   (sd {r[m].std():.4f})")

    verdict = ("STRONG  - the radial hierarchy is present; the consensus mechanism has a substrate"
               if rho > 0.30 else
               "WEAK    - present but noisy; consider a depth-ranking regulariser in RotH"
               if rho > 0.10 else
               "ABSENT  - STOP. RotH did not learn a radial hierarchy. Fix this first "
               "(see the proposal, R1) or the group-consensus claim has no basis.")
    print(f"\n  VERDICT: {verdict}")
    return dict(spearman=rho, depth=depth, radius=r)


# --------------------------------------------------------------------------
# D2 -- consensus semantics
# --------------------------------------------------------------------------

def _lca_depth(a, b, sep=">"):
    pa = [p.strip() for p in str(a).split(sep) if p.strip()]
    pb = [p.strip() for p in str(b).split(sep) if p.strip()]
    n = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        n += 1
    return n, len(pa), len(pb)


def d2_consensus_semantics(embs, cat_paths, c=1.0, n_pairs=120000, seed=42, sep=">"):
    """Does the gyromidpoint of two POIs generalise more when they are further
    apart in the taxonomy?

    IMPORTANT -- the pooled correlation is confounded and must not be the
    headline number. Two POIs that are both deep in the taxonomy start at a
    large radius, so their midpoint drops further in absolute terms no matter
    how related they are. On a synthetic hierarchy where the effect is present
    by construction, pooling reports rho = +0.10 while the equal-depth strata
    report up to +0.67. This function therefore reports a depth-STRATIFIED
    statistic as the verdict, and prints the pooled one only for contrast.
    """
    print("\n" + "=" * 74)
    print("D2  Consensus semantics: does divergence push the consensus toward the root?")
    print("=" * 74)

    rng = np.random.default_rng(seed)
    N = len(embs)
    x = np_project(embs, c)
    r = np_radius(x, c)

    i = rng.integers(0, N, n_pairs)
    j = rng.integers(0, N, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    mid = np_gyromidpoint(np.stack([x[i], x[j]], axis=1), c=c)
    r_mid = np_radius(mid, c)
    base = 0.5 * (r[i] + r[j])
    depth_drop = base - r_mid
    rel_drop = depth_drop / np.maximum(base, 1e-9)

    shared, da, db = zip(*[_lca_depth(cat_paths[a], cat_paths[b], sep) for a, b in zip(i, j)])
    shared = np.array(shared, dtype=np.float64)
    da, db = np.array(da), np.array(db)
    # normalised divergence in [0, 1]: 0 = one path is a prefix of the other,
    # 1 = the two paths already disagree at the root
    denom = np.maximum(np.minimum(da, db), 1)
    divergence = 1.0 - shared / denom

    pooled = spearman(divergence, depth_drop)
    print(f"  pairs: {len(i)}")
    print(f"  mean depth_drop = {depth_drop.mean():+.4f}   "
          f"mean relative drop = {rel_drop.mean():.1%}   "
          f"positive for {(depth_drop > 0).mean():.1%} of pairs")
    print(f"\n  pooled Spearman(divergence, depth_drop) = {pooled:+.4f}   "
          f"<-- CONFOUNDED by member depth, do not report this")

    print("\n  depth-stratified (the number that means something):")
    print(f"    {'depths':>10} {'n':>7} {'rho':>8}   mean depth_drop by shared-prefix length")
    rhos, ws, eq_rhos, eq_ws = [], [], [], []
    for a in range(1, 8):
        for b in range(a, 8):
            m = ((da == a) & (db == b)) | ((da == b) & (db == a))
            if m.sum() < 300 or len(set(divergence[m])) < 2:
                continue
            rho = spearman(divergence[m], depth_drop[m])
            rhos.append(rho); ws.append(m.sum())
            if a == b:
                eq_rhos.append(rho); eq_ws.append(m.sum())
            cells = "  ".join(
                f"sh{int(s)}:{depth_drop[m & (shared == s)].mean():.3f}"
                for s in sorted(set(shared[m])) if (m & (shared == s)).sum() > 30)
            print(f"    {f'({a},{b})':>10} {m.sum():>7} {rho:>+8.4f}   {cells}")

    strat = float(np.average(rhos, weights=ws)) if rhos else 0.0
    eq = float(np.average(eq_rhos, weights=eq_ws)) if eq_rhos else 0.0
    print(f"\n  weighted mean within-stratum rho          = {strat:+.4f}")
    print(f"  weighted mean over EQUAL-depth strata     = {eq:+.4f}   <-- headline")

    verdict = ("STRONG  - consensus generalises with disagreement, exactly as designed"
               if eq > 0.15 and depth_drop.mean() > 0 else
               "WEAK    - the drop is positive but barely tracks taxonomy; report it "
               "honestly and lean on D1"
               if depth_drop.mean() > 0 else
               "ABSENT  - the midpoint does not generalise; re-check D1 first")
    print(f"\n  VERDICT: {verdict}")
    return dict(spearman=eq, pooled=pooled, stratified=strat,
                depth_drop=depth_drop, rel_drop=rel_drop, divergence=divergence)


# --------------------------------------------------------------------------
# D3 -- group mining feasibility
# --------------------------------------------------------------------------

def d3_group_feasibility(df, windows=(15, 30, 60, 120), min_shared=2,
                         max_group=5, horizon_min=240):
    """Count implicit co-visit groups and, crucially, group *transitions*.

    A group transition is what makes this a *next*-POI task: members co-present
    at POI A at time t, then co-present at some POI B within `horizon_min`.
    Those B's are the supervision signal. If this count is too small, real
    groups can only be a test set, not a training set.
    """
    print("\n" + "=" * 74)
    print("D3  Group mining feasibility on the check-in data")
    print("=" * 74)

    need = {"user_id", "poi_idx", "utc_time"}
    if not need.issubset(df.columns):
        print(f"  SKIP - need columns {need}, got {list(df.columns)[:12]}")
        return {}

    d = df[["user_id", "poi_idx", "utc_time", "split"]].copy() if "split" in df.columns \
        else df[["user_id", "poi_idx", "utc_time"]].copy()
    d["ts"] = _to_epoch_minutes(d["utc_time"])
    d = d.sort_values(["poi_idx", "ts"], kind="mergesort").reset_index(drop=True)

    print(f"  check-ins={len(d)}  users={d.user_id.nunique()}  POIs={d.poi_idx.nunique()}")
    print(f"\n  {'window':>8} {'co-visit':>10} {'pairs':>8} {'groups':>8} "
          f"{'g>=3':>7} {'transitions':>12}")
    print("  " + "-" * 60)

    out = {}
    for w in windows:
        events = _covisit_events(d, w, max_group)
        pairs = set()
        gsize = Counter()
        for _, members, _ in events:
            gsize[len(members)] += 1
            for a, b in itertools.combinations(sorted(members), 2):
                pairs.add((a, b))
        trans = _group_transitions(events, horizon_min)
        n_ge3 = sum(v for k, v in gsize.items() if k >= 3)
        print(f"  {w:>6}m {len(events):>10} {len(pairs):>8} {sum(gsize.values()):>8} "
              f"{n_ge3:>7} {len(trans):>12}")
        out[w] = dict(events=len(events), pairs=len(pairs),
                      groups=sum(gsize.values()), ge3=n_ge3, transitions=len(trans),
                      size_hist=dict(gsize))

    best = max(out.values(), key=lambda v: v["transitions"]) if out else {}
    n = best.get("transitions", 0)
    verdict = ("T1 VIABLE      - enough real group transitions to train and test on"
               if n >= 5000 else
               "T1 TEST-ONLY   - too few to train on; train on synthetic groups (T2), "
               "report real groups as a held-out test set"
               if n >= 300 else
               "T1 NOT VIABLE  - fall back to synthetic groups (T2) and add a dataset "
               "with real social ties (T3: Gowalla / Yelp)")
    print(f"\n  VERDICT: {verdict}  (best window yields {n} group transitions)")
    return out


def _to_epoch_minutes(s):
    import pandas as pd
    return pd.to_datetime(s, errors="coerce", utc=True).astype("int64") // (60 * 10 ** 9)


def _covisit_events(d, window_min, max_group):
    """Sliding window over each POI's timeline -> (poi, frozenset(users), t)."""
    events = []
    poi = d["poi_idx"].to_numpy()
    usr = d["user_id"].to_numpy()
    ts = d["ts"].to_numpy()
    start = 0
    for end in range(1, len(d) + 1):
        if end == len(d) or poi[end] != poi[start]:
            _scan_poi_block(poi, usr, ts, start, end, window_min, max_group, events)
            start = end
    return events


def _scan_poi_block(poi, usr, ts, lo, hi, window_min, max_group, events):
    left = lo
    seen = set()
    for right in range(lo, hi):
        while ts[right] - ts[left] > window_min:
            left += 1
        members = frozenset(usr[left:right + 1].tolist())
        if 2 <= len(members) <= max_group and (poi[right], members) not in seen:
            seen.add((poi[right], members))
            events.append((int(poi[right]), members, int(ts[left])))


def _group_transitions(events, horizon_min):
    """(group, poi_A, t_A) -> (poi_B, t_B) where the same member set is co-present
    again within the horizon. These are the group next-POI training targets."""
    by_group = defaultdict(list)
    for p, members, t in events:
        by_group[members].append((t, p))
    trans = []
    for members, seq in by_group.items():
        seq.sort()
        for (t1, p1), (t2, p2) in zip(seq, seq[1:]):
            if p1 != p2 and 0 < t2 - t1 <= horizon_min:
                trans.append((members, p1, p2, t1, t2))
    return trans


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _load(data_dir, emb_file, meta_file):
    import pandas as pd
    embs = np.load(os.path.join(data_dir, emb_file)).astype(np.float64)
    meta = pd.read_csv(os.path.join(data_dir, meta_file))
    cat_col = next((c for c in ("category_path", "category", "categories", "cat")
                    if c in meta.columns), None)
    if cat_col is None:
        raise SystemExit(f"no category column in {meta_file}: {list(meta.columns)}")
    if "poi_idx" in meta.columns:
        meta = meta.sort_values("poi_idx")
    paths = meta[cat_col].astype(str).tolist()[: len(embs)]
    if len(paths) < len(embs):
        paths += ["Venue"] * (len(embs) - len(paths))
    checkins = []
    for f in ("train_NYC.csv", "val_NYC.csv", "test_NYC.csv"):
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            c = pd.read_csv(p)
            c["split"] = f.split("_")[0]
            checkins.append(c)
    df = pd.concat(checkins, ignore_index=True) if checkins else None
    return embs, paths, df


def _self_check():
    """Synthetic fixture: exercises every code path without the real data, and
    confirms D1/D2 detect a hierarchy that is there by construction."""
    print("SELF-CHECK on a synthetic ball hierarchy (no project data needed)")
    rng = np.random.default_rng(0)
    d, N = 64, 2000

    # A genuine nested hierarchy: each taxonomy node owns a direction that is a
    # perturbation of its parent's, so siblings point similarly and cousins
    # diverge. Depth sets the radius. This is the structure D1 and D2 are meant
    # to detect -- if they cannot find it here, they cannot find it anywhere.
    def unit(v):
        return v / np.linalg.norm(v)

    dirs = {(): unit(rng.normal(size=d))}

    def direction(node):
        if node not in dirs:
            dirs[node] = unit(direction(node[:-1]) + 0.8 * rng.normal(size=d))
        return dirs[node]

    embs, paths = [], []
    for _ in range(N):
        depth = int(rng.integers(1, 5))
        node = tuple(int(rng.integers(0, 3)) for _ in range(depth))
        r = 0.30 + 0.14 * depth        # deeper -> nearer the boundary, by construction
        v = unit(direction(node) + 0.15 * rng.normal(size=d))
        embs.append(v * r)
        paths.append(" > ".join(f"L{k}_{node[k]}" for k in range(depth)))
    embs = np.array(embs)

    r1 = d1_radial_hierarchy(embs, paths)
    r2 = d2_consensus_semantics(embs, paths, n_pairs=120000)

    import pandas as pd
    n = 4000
    fake = pd.DataFrame(dict(
        user_id=rng.integers(0, 60, n),
        poi_idx=rng.integers(0, 40, n),
        utc_time=pd.to_datetime("2012-04-03", utc=True)
                 + pd.to_timedelta(np.sort(rng.integers(0, 60 * 24 * 200, n)), unit="m"),
    ))
    r3 = d3_group_feasibility(fake)

    print("\n" + "=" * 74)
    ok1 = r1["spearman"] > 0.30
    ok2 = r2["depth_drop"].mean() > 0
    ok2b = r2["spearman"] > 0.15
    ok3 = isinstance(r3, dict) and len(r3) == 4
    print(f"  {'PASS' if ok1 else 'FAIL'}  D1 recovers a hierarchy that is present by construction")
    print(f"  {'PASS' if ok2 else 'FAIL'}  D2 measures a positive depth drop")
    print(f"  {'PASS' if ok2b else 'FAIL'}  D2 ranks taxonomic divergence against that drop")
    print(f"  {'PASS' if ok3 else 'FAIL'}  D3 runs over all windows and counts transitions")
    print("\n  (D3's fixture is deliberately sparse -- a 'NOT VIABLE' verdict there is"
          "\n   the expected self-check outcome, not a failure.)")
    return ok1 and ok2 and ok2b and ok3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/kaggle/input/kushflq")
    ap.add_argument("--emb-file", default="poi_hyperbolic_embs.npy")
    ap.add_argument("--meta-file", default="poi_metadata_NYC.csv")
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        sys.exit(0 if _self_check() else 1)

    embs, paths, df = _load(a.data_dir, a.emb_file, a.meta_file)
    print(f"loaded embeddings {embs.shape} and {len(paths)} category paths "
          f"from {a.data_dir}")
    d1_radial_hierarchy(embs, paths, a.curvature)
    d2_consensus_semantics(embs, paths, a.curvature)
    if df is not None:
        d3_group_feasibility(df)
    else:
        print("\nD3 skipped: no train/val/test_NYC.csv found in --data-dir")


if __name__ == "__main__":
    main()
