"""
Multi-signal user-user affinity for meaningful group construction, validated against real
co-presence.

The problem it solves
---------------------
One-hour co-location is the obvious group source and it is too thin: on FSQ-NYC it yields
3,750 groups of which 80% are pairs, only 250 reach size 5, and *none* reach size 20. It also
yields just 21 group->group transitions, so it cannot supply next-POI targets.

The fix is to stop using the scarce co-presence signal as the group source and use it as the
**label** for an affinity function built from abundant signals. Measured over all 575,128 user
pairs, with every signal recomputed so it cannot contain the co-presence events that produced
the labels (encounters forced >= `far_gap_min` apart):

    taste (category L2 cosine)          AUC 0.6477     <- best single signal
    rhythm (hour-of-week cosine)        AUC 0.6459
    territory (locality cosine)         AUC 0.6218
    same POI same day, >=3h apart       AUC 0.5982
    same POI, never same day (idf)      AUC 0.5982
    ----------------------------------------------
    z-sum of the four                   AUC 0.7373

    top 0.5% of pairs -> 2,876 pairs / 763 users, 16.7% precision vs real co-presence (23.3x lift)
    top 0.1% of pairs ->   576 pairs / 326 users, 41.7% precision                      (58.0x lift)

A methodological warning, because it cost a wrong conclusion once already: scoring `same POI
same day` WITHOUT the gap constraint gives AUC 0.98, which looks like a triumph and is pure
tautology -- two users co-present within an hour are necessarily at the same POI on the same
day, so the feature contains the label. Always keep `far_gap_min` > the co-presence window.

Why cliques
-----------
A clique in the thresholded affinity graph is a group in which *every pair* is high-affinity,
not merely connected to a hub. That is what makes the group a plausible joint-decision unit.
At the top 1% threshold FSQ-NYC yields 3,124 triangles, 1,704 5-cliques, and a maximum clique
of 18 -- so KCGRS's group-size 5-10 protocol becomes reachable, which one-hour co-location
never was.

Groups are grown rather than enumerated (`find_cliques` explodes past the 2% threshold: 166k
maximal cliques at 5%). `grow_group` starts from a seed and repeatedly adds the candidate with
the highest minimum affinity to every current member, which yields a clique of exactly size k
in O(k * n) per group.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

SEP = ">"


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------

def _onehot_cosine(user_ids, keys, ui, n):
    """L2-normalised histogram cosine over an arbitrary categorical key."""
    uniq = sorted(set(keys))
    ki = {k: i for i, k in enumerate(uniq)}
    M = np.zeros((n, len(uniq)))
    for u, k in zip(user_ids, keys):
        M[ui[u], ki[k]] += 1.0
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    # NumPy's SIMD matmul raises spurious FP flags here; M is verified finite with unit rows.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = M @ M.T
    assert np.isfinite(S).all()
    return S


def far_covisit(df, ui, n, far_gap_min=180, max_venue_users=60):
    """Same POI, same day, but >= far_gap_min apart -- i.e. explicitly NOT co-present.

    Weighted by venue rarity: turning up at a quiet venue on the same day as someone else is
    weak evidence of a shared routine; doing it at Times Square is none.
    """
    rar = df.groupby("poi_idx")["user_id"].nunique().to_dict()
    A = np.zeros((n, n))
    for (poi, day), g in df.groupby(["poi_idx", "day"], sort=False):
        if rar[poi] > max_venue_users:
            continue
        rows = sorted(zip(g["ts"].to_numpy(), g["user_id"].to_numpy()))
        if len(rows) < 2:
            continue
        w = 1.0 / np.log(1.0 + rar[poi])
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                if rows[a][1] == rows[b][1] or rows[b][0] - rows[a][0] < far_gap_min:
                    continue
                A[ui[rows[a][1]], ui[rows[b][1]]] += w
    return A + A.T


def build_affinity(df, cat_of, loc_of, users, cat_level=2, far_gap_min=180,
                   weights=None, verbose=True):
    """Combine taste / rhythm / territory / far-co-visit into one z-scored affinity matrix.

    `df` must be TRAIN-split rows only and carry ts / day / hour columns. Returns
    (affinity [n,n], components dict).
    """
    from build_groups import cat_prefix       # local import to avoid a circular top-level one

    ui = {u: i for i, u in enumerate(users)}
    n = len(users)
    uid = df["user_id"].to_numpy()

    cat2 = [cat_prefix(cat_of, p, cat_level) for p in df["poi_idx"]]
    locs = [str(loc_of.get(int(p), "Unknown")) for p in df["poi_idx"]]
    how = (df["utc_time"].dt.dayofweek * 24 + df["utc_time"].dt.hour).to_numpy()

    comp = {
        "taste": _onehot_cosine(uid, cat2, ui, n),
        "rhythm": _onehot_cosine(uid, how, ui, n),
        "territory": _onehot_cosine(uid, locs, ui, n),
        "far_covisit": far_covisit(df, ui, n, far_gap_min),
    }
    weights = weights or {k: 1.0 for k in comp}

    iu = np.triu_indices(n, 1)

    def z(M):
        v = np.nan_to_num(M[iu].astype(float), nan=0.0)
        return (v - v.mean()) / (v.std() + 1e-12)

    flat = sum(weights.get(k, 1.0) * z(M) for k, M in comp.items())
    A = np.zeros((n, n))
    A[iu] = flat
    A = A + A.T
    np.fill_diagonal(A, -np.inf)
    if verbose:
        print(f"affinity built from {', '.join(comp)}  "
              f"(far_gap_min={far_gap_min}, cat_level={cat_level})")
    return A, comp


# --------------------------------------------------------------------------
# validation against real co-presence
# --------------------------------------------------------------------------

def copresence_labels(groups, users):
    """Boolean [n,n]: did this pair ever share a real co-presence group?"""
    ui = {u: i for i, u in enumerate(users)}
    n = len(users)
    Y = np.zeros((n, n), dtype=bool)
    if groups is None or groups.empty:
        return Y
    for r in groups.itertuples(index=False):
        m = [ui[x] for x in r.members if x in ui]
        for a in range(len(m)):
            for b in range(a + 1, len(m)):
                Y[m[a], m[b]] = Y[m[b], m[a]] = True
    return Y


def auc_vs_copresence(score, Y):
    """Rank-based AUC of `score` at separating co-present pairs from the rest."""
    n = Y.shape[0]
    iu = np.triu_indices(n, 1)
    y = Y[iu]
    if y.sum() == 0 or y.all():
        return float("nan")
    s = np.nan_to_num(score[iu].astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def validate(A, comp, Y, percentiles=(99.9, 99.5, 99.0, 98.0), verbose=True):
    """Report per-signal AUC, combined AUC, and precision/lift at each threshold."""
    n = Y.shape[0]
    iu = np.triu_indices(n, 1)
    y = Y[iu]
    out = {"n_pairs": int(len(y)), "n_copresent": int(y.sum()),
           "base_rate": float(y.mean()), "auc": {}, "thresholds": {}}
    if verbose:
        print(f"\n  affinity validation against {int(y.sum()):,} real co-present pairs "
              f"of {len(y):,} ({y.mean():.3%} base rate)")
        print(f"    {'signal':<34} {'AUC':>7}")
    for k, M in sorted(comp.items(), key=lambda kv: -auc_vs_copresence(kv[1], Y)):
        out["auc"][k] = auc_vs_copresence(M, Y)
        if verbose:
            print(f"    {k:<34} {out['auc'][k]:>7.4f}")
    out["auc"]["COMBINED"] = auc_vs_copresence(A, Y)
    if verbose:
        print(f"    {'COMBINED':<34} {out['auc']['COMBINED']:>7.4f}"
              f"   <- the number to report")

    flat = A[iu]
    for p in percentiles:
        thr = float(np.percentile(flat, p))
        sel = flat >= thr
        deg = np.zeros(n)
        np.add.at(deg, iu[0][sel], 1)
        np.add.at(deg, iu[1][sel], 1)
        prec = float(y[sel].mean()) if sel.any() else 0.0
        out["thresholds"][p] = dict(
            threshold=thr, n_pairs=int(sel.sum()), n_users=int((deg > 0).sum()),
            precision=prec, lift=float(prec / y.mean()) if y.mean() else float("nan"))
        if verbose:
            r = out["thresholds"][p]
            print(f"    top {100-p:>4.1f}%  pairs={r['n_pairs']:>7,}  users={r['n_users']:>5}  "
                  f"precision={r['precision']:>6.2%}  lift={r['lift']:>5.1f}x")
    if out["auc"]["COMBINED"] < 0.60 and verbose:
        print("    WARNING: combined AUC < 0.60 -- the affinity barely beats chance at "
              "recovering real companions; established-regime groups are weakly justified")
    return out


# --------------------------------------------------------------------------
# group formation
# --------------------------------------------------------------------------

def threshold_graph(A, percentile):
    """Adjacency (bool) keeping the top `100-percentile`% of pairs, plus the cut value."""
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    thr = float(np.percentile(A[iu], percentile))
    G = A >= thr
    np.fill_diagonal(G, False)
    return G, thr


def grow_group(seed, k, A, G, rng, max_candidates=64):
    """Grow a clique of exactly size k from `seed`, or return None.

    At each step the candidate maximising the *minimum* affinity to every current member wins,
    so the result is a clique in G rather than a star around the seed. Candidates are restricted
    to the seed's neighbourhood, which is what keeps this O(k * deg) rather than O(k * n).
    """
    members = [seed]
    nbrs = np.flatnonzero(G[seed])
    if len(nbrs) < k - 1:
        return None
    rng.shuffle(nbrs)
    nbrs = nbrs[:max_candidates]
    while len(members) < k:
        best, best_score = None, -np.inf
        for c in nbrs:
            if c in members or not G[c, members].all():
                continue
            score = float(A[c, members].min())
            if score > best_score:
                best, best_score = int(c), score
        if best is None:
            return None
        members.append(best)
    return members


def sample_affinity_groups(A, percentile, sizes, n_groups, seed=42, verbose=True):
    """Sample up to `n_groups` cliques with sizes drawn from `sizes`."""
    rng = np.random.default_rng(seed)
    G, thr = threshold_graph(A, percentile)
    deg = G.sum(1)
    eligible = np.flatnonzero(deg >= max(sizes) - 1)
    if len(eligible) == 0:
        eligible = np.flatnonzero(deg >= min(sizes) - 1)
    out, attempts = [], 0
    while len(out) < n_groups and attempts < n_groups * 20:
        attempts += 1
        if len(eligible) == 0:
            break
        s = int(rng.choice(eligible))
        k = int(rng.choice(sizes))
        g = grow_group(s, k, A, G, rng)
        if g is not None:
            out.append(sorted(g))
    if verbose:
        print(f"  affinity groups: {len(out):,} cliques (top {100-percentile:.1f}% threshold "
              f"={thr:.3f}, {int((deg>0).sum())} users eligible, {attempts:,} attempts)")
    return out, thr


def clique_neighbourhood(A, percentile):
    """For each user, the set of users they are clique-eligible with (thresholded neighbours).

    Used by build_groups' `established` regime to pick companions for an anchor: the anchor's
    companions must be mutually high-affinity, not just high-affinity to the anchor.
    """
    G, thr = threshold_graph(A, percentile)
    return {i: set(np.flatnonzero(G[i]).tolist()) for i in range(A.shape[0])}, G, thr
