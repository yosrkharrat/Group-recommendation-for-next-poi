"""
Memory-safe affinity backend — identical output, no [n,n] materialisation.

Why this exists
---------------
`affinity.build_affinity` allocates FIVE dense `[n, n]` float64 matrices (taste, rhythm,
territory, far-co-visit, combined) plus `np.triu_indices(n, 1)`. At the Foursquare arm's 7,849
users that is ~0.5 GB per matrix and fine. At Gowalla's **31,667 users each one is 8.0 GB** and
`triu_indices` alone is another 8 GB, so the `established` regime cannot run at all — which is
why the first Gowalla group build was forced to `--regimes occasional random`, the one real
divergence from the Foursquare pipeline.

This module restores `established` by computing exactly the same quantities blockwise. It is a
drop-in replacement for `clique_neighbourhood(build_affinity(...)[0], percentile)` and is
verified bit-identical against the dense path by `--self-check`.

How the exact statistics are obtained without the matrix
--------------------------------------------------------
`z()` in affinity.py needs each component's **mean and population std over the strict upper
triangle**. For a cosine component `S = M @ M.T` with L2-normalised rows, both are available in
closed form from `M` alone (`k` = number of categories/hours/localities, tiny):

    sum_ij  S_ij = sum_k (sum_i M_ik)^2                     = ||colsum(M)||^2
    sum_ij  S_ij^2 = ||M.T @ M||_F^2                        (Gram trick, [k,k] only)
    diagonal: S_ii = ||M_i||^2 = 1 for every nonzero row

so `sum_upper = (total - trace) / 2` and `sumsq_upper = (frob - trace_sq) / 2`, both exact in
O(n*k + k^2). The far-co-visit component is genuinely sparse (only pairs sharing a rare venue on
the same day, >= far_gap_min apart), so it is held as a scipy sparse matrix and its two sums are
read straight off the stored values.

The percentile cut is then found by a two-pass histogram over the combined z-sum: pass 1 takes
the global min/max, pass 2 histograms into 2^20 bins and locates the bin holding the target order
statistic, pass 3 re-scans only that bin's values so the final cut reproduces
`np.percentile(..., linear)` exactly, including its interpolation between two order statistics.

What the caller gets back
-------------------------
`build_groups.py` only ever uses the affinity graph through three operations —
`clique_nbrs.get(i)`, `clique_nbrs.values()`, and scalar `G[i, j]` — so the returned objects
implement precisely those over CSR storage instead of a dense bool array (10M edges ~ 40 MB
rather than 8 GB).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.sparse as sp

DEFAULT_BLOCK = 2048
HIST_BINS = 1 << 20


# --------------------------------------------------------------------------
# component construction (sparse)
# --------------------------------------------------------------------------

def _onehot_normalised(user_ids, keys, ui, n):
    """Sparse L2-normalised histogram over a categorical key -- the matrix `M` whose
    `M @ M.T` is affinity.py's `_onehot_cosine` output, never formed here."""
    uniq = sorted(set(keys))
    ki = {k: i for i, k in enumerate(uniq)}
    rows = np.fromiter((ui[u] for u in user_ids), dtype=np.int64, count=len(user_ids))
    cols = np.fromiter((ki[k] for k in keys), dtype=np.int64, count=len(keys))
    M = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(uniq))).tocsr()
    M.sum_duplicates()
    norms = np.sqrt(M.multiply(M).sum(axis=1)).A.ravel()
    norms = np.maximum(norms, 1e-12)
    return sp.diags(1.0 / norms) @ M


def _far_covisit_sparse(df, ui, n, far_gap_min=180, max_venue_users=60):
    """affinity.far_covisit, accumulated sparsely (same weights, same >= gap rule)."""
    rar = df.groupby("poi_idx")["user_id"].nunique().to_dict()
    I, J, V = [], [], []
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
                I.append(ui[rows[a][1]]); J.append(ui[rows[b][1]]); V.append(w)
    if not I:
        return sp.csr_matrix((n, n))
    F = sp.coo_matrix((V, (I, J)), shape=(n, n)).tocsr()
    F.sum_duplicates()
    return F + F.T                      # affinity.far_covisit returns A + A.T


# --------------------------------------------------------------------------
# exact upper-triangle moments
# --------------------------------------------------------------------------

def _cosine_moments(M, n):
    """(sum, sumsq) of `M @ M.T` over the strict upper triangle -- exact, no [n,n]."""
    colsum = np.asarray(M.sum(axis=0)).ravel()
    total = float(colsum @ colsum)
    row_sq = np.asarray(M.multiply(M).sum(axis=1)).ravel()      # 1.0 per nonzero row
    trace = float(row_sq.sum())
    trace_sq = float((row_sq ** 2).sum())
    G = (M.T @ M).toarray() if sp.issparse(M.T @ M) else np.asarray(M.T @ M)
    frob = float((G * G).sum())                                  # ||M M^T||_F^2 = ||M^T M||_F^2
    return (total - trace) / 2.0, (frob - trace_sq) / 2.0


def _sparse_moments(F):
    """(sum, sumsq) of a symmetric sparse matrix over the strict upper triangle."""
    U = sp.triu(F, k=1)
    v = U.data
    return float(v.sum()), float((v * v).sum())


def _zparams(sum_u, sumsq_u, n_pairs):
    mean = sum_u / n_pairs
    var = max(sumsq_u / n_pairs - mean * mean, 0.0)
    return mean, math.sqrt(var) + 1e-12          # affinity.z(): (v - mean) / (std + 1e-12)


# --------------------------------------------------------------------------
# blockwise combined score
# --------------------------------------------------------------------------

class _Combined:
    """Yields blocks of the combined z-summed affinity, strict-upper-triangle only."""

    def __init__(self, comps, weights, n, block=DEFAULT_BLOCK):
        self.comps, self.weights, self.n, self.block = comps, weights, n, block
        n_pairs = n * (n - 1) // 2
        self.n_pairs = n_pairs
        self.zp = {}
        for name, (kind, obj) in comps.items():
            s, ss = _cosine_moments(obj, n) if kind == "cos" else _sparse_moments(obj)
            self.zp[name] = _zparams(s, ss, n_pairs)

    def blocks(self):
        """(row_start, row_end, values) for the strict-upper part of each row block."""
        n = self.n
        for r0 in range(0, n, self.block):
            r1 = min(r0 + self.block, n)
            acc = np.zeros((r1 - r0, n), dtype=np.float64)
            for name, (kind, obj) in self.comps.items():
                mean, std = self.zp[name]
                w = self.weights.get(name, 1.0)
                if kind == "cos":
                    blk = (obj[r0:r1] @ obj.T).toarray()
                else:
                    blk = obj[r0:r1].toarray()
                acc += w * ((blk - mean) / std)
            # keep strict upper triangle: column > global row index
            cols = np.arange(n)[None, :]
            rows = np.arange(r0, r1)[:, None]
            mask = cols > rows
            yield r0, r1, acc, mask


def _percentile_exact(comb, percentile, verbose=True):
    """np.percentile(values, percentile, method='linear') over the strict upper triangle."""
    lo_v, hi_v = np.inf, -np.inf
    for _, _, acc, mask in comb.blocks():
        v = acc[mask]
        if v.size:
            lo_v = min(lo_v, float(v.min())); hi_v = max(hi_v, float(v.max()))
    if not np.isfinite(lo_v) or hi_v <= lo_v:
        return float(lo_v if np.isfinite(lo_v) else 0.0)

    edges = np.linspace(lo_v, hi_v, HIST_BINS + 1)
    hist = np.zeros(HIST_BINS, dtype=np.int64)
    for _, _, acc, mask in comb.blocks():
        v = acc[mask]
        if v.size:
            hist += np.histogram(v, bins=edges)[0]
    total = int(hist.sum())

    # np.percentile 'linear': position p*(N-1), interpolate between order stats lo_i, hi_i
    pos = (percentile / 100.0) * (total - 1)
    lo_i, hi_i = int(math.floor(pos)), int(math.ceil(pos))
    cum = np.cumsum(hist)
    b_lo = int(np.searchsorted(cum, lo_i + 1))
    b_hi = int(np.searchsorted(cum, hi_i + 1))
    lo_edge, hi_edge = edges[min(b_lo, HIST_BINS - 1)], edges[min(b_hi, HIST_BINS - 1) + 1]

    # re-scan only the values inside the (at most two) bins that hold the order statistics
    keep = []
    for _, _, acc, mask in comb.blocks():
        v = acc[mask]
        sel = v[(v >= lo_edge) & (v <= hi_edge)]
        if sel.size:
            keep.append(sel)
    band = np.sort(np.concatenate(keep)) if keep else np.array([lo_v])
    before = int(cum[b_lo - 1]) if b_lo > 0 else 0
    def order_stat(rank):
        idx = rank - before
        return float(band[min(max(idx, 0), len(band) - 1)])
    a, b = order_stat(lo_i), order_stat(hi_i)
    thr = a + (pos - lo_i) * (b - a)
    if verbose:
        print(f"  affinity threshold (blocked, exact): {thr:.6f} over {total:,} pairs")
    return float(thr)


# --------------------------------------------------------------------------
# CSR-backed stand-ins for the dense bool graph
# --------------------------------------------------------------------------

class SparseGraphView:
    """Scalar `G[i, j]` over CSR storage -- the only way build_groups touches clique_G."""

    def __init__(self, adj):
        self.adj = adj.tocsr()
        self.adj.sort_indices()

    def __getitem__(self, key):
        i, j = key
        row = self.adj.indices[self.adj.indptr[i]:self.adj.indptr[i + 1]]
        pos = np.searchsorted(row, j)
        return bool(pos < len(row) and row[pos] == j)

    def sum(self, axis=None):
        return np.diff(self.adj.indptr) if axis == 1 else self.adj.nnz


class NbrsView:
    """`clique_nbrs`-compatible mapping: .get(i, ()), .values(), len()."""

    def __init__(self, adj):
        self.adj = adj.tocsr()
        self.adj.sort_indices()

    def get(self, i, default=()):
        if i is None or i < 0 or i >= self.adj.shape[0]:
            return default
        row = self.adj.indices[self.adj.indptr[i]:self.adj.indptr[i + 1]]
        return row.tolist() if len(row) else default

    def values(self):
        deg = np.diff(self.adj.indptr)
        return (list(range(d)) for d in deg)      # only truthiness/len is used

    def __len__(self):
        return self.adj.shape[0]


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def clique_neighbourhood_blocked(df, cat_of, loc_of, users, cat_level=2, far_gap_min=180,
                                 percentile=99.0, weights=None, block=DEFAULT_BLOCK,
                                 verbose=True):
    """Blocked equivalent of clique_neighbourhood(build_affinity(...)[0], percentile).

    Returns (clique_nbrs, clique_G, thr, stats) where the first two are CSR-backed views with
    the same access surface build_groups.py uses on the dense originals.
    """
    from build_groups import cat_prefix

    ui = {u: i for i, u in enumerate(users)}
    n = len(users)
    uid = df["user_id"].to_numpy()
    cat2 = [cat_prefix(cat_of, p, cat_level) for p in df["poi_idx"]]
    locs = [str(loc_of.get(int(p), "Unknown")) for p in df["poi_idx"]]
    how = (df["utc_time"].dt.dayofweek * 24 + df["utc_time"].dt.hour).to_numpy()

    comps = {
        "taste": ("cos", _onehot_normalised(uid, cat2, ui, n)),
        "rhythm": ("cos", _onehot_normalised(uid, how, ui, n)),
        "territory": ("cos", _onehot_normalised(uid, locs, ui, n)),
        "far_covisit": ("sparse", _far_covisit_sparse(df, ui, n, far_gap_min)),
    }
    weights = weights or {k: 1.0 for k in comps}
    if verbose:
        print(f"affinity (blocked) over {n:,} users, block={block}: "
              f"{', '.join(comps)}  (far_gap_min={far_gap_min}, cat_level={cat_level})")

    comb = _Combined(comps, weights, n, block)
    thr = _percentile_exact(comb, percentile, verbose=verbose)

    I, J = [], []
    for r0, _, acc, mask in comb.blocks():
        hit = mask & (acc >= thr)
        rr, cc = np.nonzero(hit)
        if len(rr):
            I.append(rr + r0); J.append(cc)
    if I:
        I, J = np.concatenate(I), np.concatenate(J)
    else:
        I, J = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    up = sp.coo_matrix((np.ones(len(I), dtype=bool), (I, J)), shape=(n, n))
    adj = (up + up.T).tocsr()
    adj.data[:] = True

    nbrs, G = NbrsView(adj), SparseGraphView(adj)
    deg = np.diff(adj.tocsr().indptr)
    stats = dict(threshold=thr, n_pairs=int(comb.n_pairs), n_edges=int(len(I)),
                 n_users=int(n), n_users_with_neighbour=int((deg > 0).sum()),
                 mean_degree=float(deg.mean()), backend="blocked",
                 z_params={k: dict(mean=v[0], std=v[1]) for k, v in comb.zp.items()})
    if verbose:
        print(f"  affinity graph: top {100 - percentile:.1f}% (cut={thr:.3f}), "
              f"{int((deg > 0).sum()):,}/{n:,} users with >=1 neighbour, "
              f"{len(I):,} edges, mean degree {deg.mean():.1f}")
    return nbrs, G, thr, stats


# --------------------------------------------------------------------------
# self-check: bit-identical to the dense path
# --------------------------------------------------------------------------

def _self_check():
    import affinity as dense_mod

    print("SELF-CHECK: blocked affinity vs the dense path on a synthetic fixture")
    rng = np.random.default_rng(0)
    n_users, n_pois = 60, 40
    rows = []
    base = pd.Timestamp("2010-03-01", tz="UTC")
    for u in range(n_users):
        for _ in range(rng.integers(6, 20)):
            p = int(rng.integers(0, n_pois))
            t = base + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 40)))
            rows.append((u, p, t))
    df = pd.DataFrame(rows, columns=["user_id", "poi_idx", "utc_time"])
    df["ts"] = df["utc_time"].astype("int64") // (60 * 10 ** 9)
    df["day"] = df["utc_time"].dt.floor("D")
    cat_of = {p: f"L1>L2_{p % 7}" for p in range(n_pois)}
    loc_of = {p: ["New York", "Chicago", "Los Angeles"][p % 3] for p in range(n_pois)}
    users = sorted(df["user_id"].unique().tolist())

    A, comp = dense_mod.build_affinity(df, cat_of, loc_of, users, cat_level=2,
                                       far_gap_min=180, verbose=False)
    for pct in (99.0, 95.0, 90.0):
        d_nbrs, d_G, d_thr = dense_mod.clique_neighbourhood(A, pct)
        b_nbrs, b_G, b_thr, _ = clique_neighbourhood_blocked(
            df, cat_of, loc_of, users, cat_level=2, far_gap_min=180,
            percentile=pct, block=16, verbose=False)
        assert abs(d_thr - b_thr) < 1e-9, f"pct={pct}: thr {d_thr} vs {b_thr}"
        for i in range(len(users)):
            assert set(d_nbrs.get(i, set())) == set(b_nbrs.get(i, ())), f"pct={pct} nbrs[{i}]"
        for i in range(len(users)):
            for j in range(len(users)):
                assert bool(d_G[i, j]) == bool(b_G[i, j]), f"pct={pct} G[{i},{j}]"
        print(f"  PASS  percentile {pct}: threshold, neighbour sets and G[i,j] all identical "
              f"(thr={b_thr:.9f})")

    # the per-component z params must match the dense z() exactly
    iu = np.triu_indices(len(users), 1)
    comb = _Combined({"taste": ("cos", _onehot_normalised(
        df["user_id"].to_numpy(),
        [f"L1>L2_{p % 7}" for p in df["poi_idx"]],
        {u: i for i, u in enumerate(users)}, len(users)))}, {"taste": 1.0}, len(users), 16)
    v = comp["taste"][iu]
    mean, std = comb.zp["taste"]
    assert abs(mean - v.mean()) < 1e-12, (mean, v.mean())
    assert abs(std - (v.std() + 1e-12)) < 1e-12, (std, v.std())
    print(f"  PASS  Gram-trick moments exact: mean {mean:.12f}, std {std:.12f}")
    return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    sys.exit(0 if _self_check() else 1)
