"""
Hyperbolic group aggregation and scoring for group next-POI recommendation.

This module is the geometric core of the group extension: it turns a set of
member embeddings living on the Poincare ball (the RotH output space,
`poi_hyperbolic_embs.npy`, c = 1.0, d = 64) into a single *group* embedding,
and scores POIs against it by geodesic distance.

Why the Poincare ball rather than a Euclidean average (KCGRS Eq. 9):

  * The weighted *gyromidpoint* (equivalently, the Einstein midpoint in the
    Klein model) is the natural hyperbolic analogue of a weighted mean. It is
    closed-form, differentiable, and permutation invariant.
  * In a hierarchy-bearing ball, the radius ||z|| encodes specificity: leaves
    sit near the boundary, roots near the origin. The midpoint of two members
    in *different* subtrees is pulled toward the origin, i.e. toward the
    common ancestor. Consensus-by-generalisation therefore falls out of the
    geometry instead of being imposed by a loss.
  * Setting c -> 0 recovers the Euclidean weighted mean exactly, so KCGRS's
    aggregation is available as a one-knob ablation on the same code path.

All functions take/return points on the ball D^d_c = {x : c||x||^2 < 1}.
Batched: a leading batch dimension is allowed everywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Points are clamped this far inside the boundary; matches the stage-2
# hyperbolic-ops module's (1 - 1e-5)/sqrt(c) convention.
BALL_EPS = 1e-5
MIN_NORM = 1e-15


# --------------------------------------------------------------------------
# Ball primitives
# --------------------------------------------------------------------------

def project_to_ball(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Clamp x strictly inside the ball of curvature -c."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    max_norm = (1.0 - BALL_EPS) / (c ** 0.5)
    return torch.where(norm > max_norm, x / norm * max_norm, x)


def geodesic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Poincare distance d_c(x, y) = (2/sqrt(c)) artanh(sqrt(c) ||(-x) (+)_c y||)."""
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c)
    norm = diff.norm(dim=-1).clamp(MIN_NORM, (1.0 - BALL_EPS) / sqrt_c)
    return 2.0 / sqrt_c * torch.atanh(sqrt_c * norm)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Mobius addition on the ball."""
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return project_to_ball(num / den.clamp_min(MIN_NORM), c)


def radius(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Hyperbolic distance from the origin: the 'depth' / specificity of a point."""
    sqrt_c = c ** 0.5
    n = x.norm(dim=-1).clamp(0.0, (1.0 - BALL_EPS) / sqrt_c)
    return 2.0 / sqrt_c * torch.atanh(sqrt_c * n)


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Tangent space at the origin -> ball."""
    sqrt_c = c ** 0.5
    n = v.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    return project_to_ball(torch.tanh(sqrt_c * n) * v / (sqrt_c * n), c)


def logmap0(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Ball -> tangent space at the origin. Inverse of `expmap0`.

    This is the map already used in the LLaDA injection path (`logmap0` in the
    Stage 6b notebooks); it is radial, so the hierarchy survives as vector norm.
    """
    sqrt_c = c ** 0.5
    n = x.norm(dim=-1, keepdim=True).clamp(MIN_NORM, (1.0 - BALL_EPS) / sqrt_c)
    return torch.atanh(sqrt_c * n) * x / (sqrt_c * n)


# --------------------------------------------------------------------------
# Klein <-> Poincare, and the weighted gyromidpoint
# --------------------------------------------------------------------------

def poincare_to_klein(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    x2 = (x * x).sum(-1, keepdim=True)
    return 2.0 * x / (1.0 + c * x2)


def klein_to_poincare(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    x2 = (x * x).sum(-1, keepdim=True)
    denom = 1.0 + torch.sqrt((1.0 - c * x2).clamp_min(MIN_NORM))
    return project_to_ball(x / denom, c)


def gyromidpoint(
    x: torch.Tensor,
    weights: torch.Tensor | None = None,
    c: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted hyperbolic mean (Einstein midpoint computed in the Klein model).

    Args:
        x:       (..., k, d) points on the Poincare ball.
        weights: (..., k) non-negative weights. Uniform if None.
        c:       curvature magnitude. c -> 0 gives the Euclidean weighted mean.
        mask:    (..., k) boolean; False entries are excluded (padded members).

    Returns:
        (..., d) point on the ball.

    Properties (verified in `_self_test`):
        * permutation invariant;
        * idempotent: all members equal -> that same point;
        * for c -> 0 it converges to sum_i w_i x_i / sum_i w_i;
        * the midpoint of points in different directions has *smaller* radius
          than its members -- the geometric statement of "consensus is more
          general than any individual preference".
    """
    if weights is None:
        weights = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    if mask is not None:
        weights = weights * mask.to(weights.dtype)

    if c <= 0.0:  # exact Euclidean limit, used as the KCGRS ablation
        w = weights.unsqueeze(-1)
        return (w * x).sum(-2) / w.sum(-2).clamp_min(MIN_NORM)

    x = project_to_ball(x, c)
    xk = poincare_to_klein(x, c)                                   # (..., k, d)
    xk2 = (xk * xk).sum(-1)                                        # (..., k)
    gamma = 1.0 / torch.sqrt((1.0 - c * xk2).clamp_min(MIN_NORM))  # Lorentz factors
    wg = (weights * gamma).unsqueeze(-1)                           # (..., k, 1)
    mk = (wg * xk).sum(-2) / wg.sum(-2).clamp_min(MIN_NORM)        # (..., d)
    return klein_to_poincare(mk, c)


def frechet_mean(
    x: torch.Tensor,
    weights: torch.Tensor | None = None,
    c: float = 1.0,
    iters: int = 25,
    step: float = 0.5,
) -> torch.Tensor:
    """Karcher/Frechet mean by damped Riemannian gradient descent, initialised
    at the gyromidpoint.

    Only used to verify that the closed-form midpoint is a good surrogate --
    not for training (it is iterative, and backprop through it is expensive).
    Measured on random 5-member groups in d = 64: the gyromidpoint lands within
    ~5% of the group's own spread of this mean, while being closed-form.

    `step` must be < 1: an undamped unit step overshoots and the objective
    creeps back up after ~20 iterations.
    """
    if weights is None:
        weights = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    w = (weights / weights.sum(-1, keepdim=True).clamp_min(MIN_NORM)).unsqueeze(-1)
    m = gyromidpoint(x, weights, c)
    for _ in range(iters):
        # tangent-space average at m, then move along the geodesic
        u = logmap_x(m.unsqueeze(-2), x, c)          # (..., k, d)
        m = expmap_x(m, step * (w * u).sum(-2), c)
    return m


def logmap_x(base: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """log map at an arbitrary base point."""
    sqrt_c = c ** 0.5
    sub = mobius_add(-base, x, c)
    sub_norm = sub.norm(dim=-1, keepdim=True).clamp(MIN_NORM, (1.0 - BALL_EPS) / sqrt_c)
    lam = 2.0 / (1.0 - c * (base * base).sum(-1, keepdim=True)).clamp_min(MIN_NORM)
    return 2.0 / (sqrt_c * lam) * torch.atanh(sqrt_c * sub_norm) * sub / sub_norm


def expmap_x(base: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """exp map at an arbitrary base point."""
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    lam = 2.0 / (1.0 - c * (base * base).sum(-1, keepdim=True)).clamp_min(MIN_NORM)
    second = torch.tanh(sqrt_c * lam * v_norm / 2.0) * v / (sqrt_c * v_norm)
    return mobius_add(base, second, c)


# --------------------------------------------------------------------------
# Group heterogeneity
# --------------------------------------------------------------------------

def group_heterogeneity(
    x: torch.Tensor,
    weights: torch.Tensor | None = None,
    c: float = 1.0,
    mask: torch.Tensor | None = None,
) -> dict:
    """Geometric descriptors of how divided a group is.

    Returns a dict of (...) shaped tensors:
        dispersion   mean weighted geodesic distance from members to the consensus
                     (the Frechet variance's square root; 0 iff all members agree)
        spread       mean pairwise geodesic distance between members
        depth_drop   mean member radius - consensus radius. Positive means the
                     consensus is *more generic* than the members; this is the
                     quantity that should grow with disagreement, and it is the
                     mechanistic version of KCGRS's descriptive H_gr.
        consensus_r  radius of the consensus point (absolute specificity)
    """
    if weights is None:
        weights = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    if mask is not None:
        weights = weights * mask.to(weights.dtype)
    w = weights / weights.sum(-1, keepdim=True).clamp_min(MIN_NORM)

    g = gyromidpoint(x, weights, c, mask=mask)
    d_to_g = geodesic_distance(x, g.unsqueeze(-2), c)              # (..., k)
    pair = geodesic_distance(x.unsqueeze(-2), x.unsqueeze(-3), c)  # (..., k, k)

    k = x.shape[-2]
    if k > 1:
        offdiag = pair.sum((-1, -2)) / (k * (k - 1))
    else:
        offdiag = torch.zeros_like(d_to_g[..., 0])

    r_members = radius(x, c)
    return dict(
        dispersion=(w * d_to_g).sum(-1),
        spread=offdiag,
        depth_drop=(w * r_members).sum(-1) - radius(g, c),
        consensus_r=radius(g, c),
    )


# --------------------------------------------------------------------------
# Trainable pieces
# --------------------------------------------------------------------------

class GeometricAttention(nn.Module):
    """Learns each member's influence on the group consensus.

    Unlike KCGRS's free per-group scalar weights (which cannot generalise to an
    unseen group), the weights here are a *function* of features that exist for
    any group: the member's own embedding, its geodesic distance to a
    provisional consensus, and scalar side-information (activity count, recency).
    One refinement round is usually enough; `rounds=2` lets an outlier be
    down-weighted after the first consensus estimate.
    """

    def __init__(self, dim: int, n_side: int = 2, hidden: int = 64,
                 c: float = 1.0, rounds: int = 2, temperature: float = 1.0):
        super().__init__()
        self.c = c
        self.rounds = rounds
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + n_side, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, side: torch.Tensor | None = None,
                mask: torch.Tensor | None = None):
        """x: (B, k, d) on the ball. side: (B, k, n_side). mask: (B, k) bool."""
        B, k, _ = x.shape
        if side is None:
            side = x.new_zeros(B, k, 0)
        w = x.new_ones(B, k)
        if mask is not None:
            w = w * mask.to(w.dtype)

        g = gyromidpoint(x, w, self.c, mask=mask)
        for _ in range(self.rounds):
            d = geodesic_distance(x, g.unsqueeze(1), self.c).unsqueeze(-1)  # (B,k,1)
            logits = self.net(torch.cat([logmap0(x, self.c), d, side], -1)).squeeze(-1)
            if mask is not None:
                logits = logits.masked_fill(~mask, float("-inf"))
            w = torch.softmax(logits / self.temperature, dim=-1)
            g = gyromidpoint(x, w, self.c, mask=mask)
        return g, w


class HyperbolicScorer(nn.Module):
    """Scores POIs by *negative geodesic distance* on the ball.

    Motivation: the current LLaDA scorer's `tied` path is
    `MLP(h) . logmap0(z_p)`, a Euclidean inner product against a randomly
    projected tangent vector. An inner product cannot express a geodesic
    distance, so the hierarchy only ever reaches the decision function as
    feature correlation. This head maps the hidden state into the 64-d tangent
    space, exponentiates onto the ball, and ranks by distance -- making the
    geometry part of the decision surface rather than only of the input.

    Add its output as a third term alongside `poi_head` and the tied path; its
    ablation is a direct test of the paper's central geometric claim.
    """

    def __init__(self, hidden_size: int, poi_ball: torch.Tensor, c: float = 1.0,
                 bottleneck: int = 256, learn_curvature: bool = True):
        super().__init__()
        d = poi_ball.shape[-1]
        self.register_buffer("poi_ball", project_to_ball(poi_ball, c))  # (N_POI, d) frozen
        self.to_tangent = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d),
        )
        self.log_c = nn.Parameter(torch.tensor(float(c)).log(), requires_grad=learn_curvature)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(poi_ball.shape[0]))

    @property
    def c(self) -> torch.Tensor:
        return self.log_c.exp()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, hidden) -> (B, N_POI) logits."""
        c = float(self.c.detach())
        q = expmap0(self.to_tangent(h), c)                       # (B, d) on the ball
        d = geodesic_distance(q.unsqueeze(1), self.poi_ball.unsqueeze(0), c)
        return -self.scale * d + self.bias


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def least_misery_loss(member_logits: torch.Tensor, group_logits: torch.Tensor,
                      mask: torch.Tensor | None = None, tau: float = 1.0):
    """Penalise recommending a POI that the least-satisfied member dislikes.

    member_logits: (B, k, N_POI) each member's own scores (frozen individual model)
    group_logits:  (B, N_POI)    the group model's scores
    Uses a soft-min over members of their normalised score for the group's
    argmax POI, so the gradient reaches the aggregation weights.
    """
    top = group_logits.argmax(-1)                                     # (B,)
    p_member = torch.log_softmax(member_logits, dim=-1)               # (B,k,N)
    s = p_member.gather(-1, top[:, None, None].expand(-1, member_logits.shape[1], -1)).squeeze(-1)
    if mask is not None:
        s = s.masked_fill(~mask, float("inf"))
    soft_min = -tau * torch.logsumexp(-s / tau, dim=-1)                # (B,)
    return -soft_min.mean()


def frechet_variance_loss(x: torch.Tensor, g: torch.Tensor,
                          weights: torch.Tensor, c: float = 1.0):
    """Keeps the learned consensus geometrically faithful: its minimiser over g
    *is* the weighted Frechet mean, so this regulariser aligns whatever the
    attention learns with the hyperbolic barycentre."""
    d = geodesic_distance(x, g.unsqueeze(-2), c)
    w = weights / weights.sum(-1, keepdim=True).clamp_min(MIN_NORM)
    return (w * d.pow(2)).sum(-1).mean()


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------

def random_ball_points(*shape, d: int = 64, c: float = 1.0, r_max: float = 0.9,
                       dtype=torch.float32) -> torch.Tensor:
    """Uniform directions with radii spread over [0, r_max] / sqrt(c).

    Note for test authors: `randn(d) * s` is *not* a usable ball sample, because
    its norm concentrates at s*sqrt(d) -- in d = 64 almost every draw lands
    outside the ball and gets clamped onto the boundary, where float32 artanh
    saturates. Sample the radius explicitly instead.
    """
    v = torch.randn(*shape, d, dtype=dtype)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    r = torch.rand(*shape, 1, dtype=dtype) * (r_max / (c ** 0.5))
    return v * r


def _self_test():
    torch.manual_seed(0)
    d, c = 64, 1.0
    ok = lambda name, cond: print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    x = random_ball_points(8, 5, d=d, c=c)
    w = torch.rand(8, 5)

    # 1. permutation invariance
    perm = torch.randperm(5)
    a = gyromidpoint(x, w, c)
    b = gyromidpoint(x[:, perm], w[:, perm], c)
    ok("gyromidpoint is permutation invariant", torch.allclose(a, b, atol=1e-6))

    # 2. idempotence
    same = x[:, :1].expand(-1, 5, -1).contiguous()
    ok("gyromidpoint of identical members returns the member",
       torch.allclose(gyromidpoint(same, w, c), x[:, 0], atol=1e-6))

    # 3. Euclidean limit
    tiny = 1e-6
    xs = random_ball_points(8, 5, d=d, c=1.0, r_max=0.01)
    euc = (w.unsqueeze(-1) * xs).sum(-2) / w.sum(-1, keepdim=True)
    ok("c -> 0 recovers the Euclidean weighted mean",
       torch.allclose(gyromidpoint(xs, w, tiny), euc, atol=1e-4))

    # 4. the consensus-generalisation property: two leaves in different
    #    directions -> midpoint strictly closer to the origin than either
    u = torch.zeros(1, d); u[0, 0] = 0.90
    v = torch.zeros(1, d); v[0, 1] = 0.90
    m = gyromidpoint(torch.stack([u, v], 1), c=c)
    ok("consensus of divergent members is more generic (smaller radius)",
       bool((radius(m, c) < radius(u, c)).all() and (radius(m, c) < radius(v, c)).all()))

    # 5. ... but the consensus of *aligned* members stays specific
    v2 = torch.zeros(1, d); v2[0, 0] = 0.85
    m2 = gyromidpoint(torch.stack([u, v2], 1), c=c)
    ok("consensus of aligned members stays specific",
       bool((radius(m2, c) > radius(m, c)).all()))

    # 6. heterogeneity descriptors move in the right direction
    het = group_heterogeneity(torch.stack([u, v], 1), c=c)
    hom = group_heterogeneity(torch.stack([u, v2], 1), c=c)
    ok("depth_drop is larger for the heterogeneous group",
       bool((het["depth_drop"] > hom["depth_drop"]).all()))
    ok("spread is larger for the heterogeneous group",
       bool((het["spread"] > hom["spread"]).all()))

    # 7. gyromidpoint is close to the true Frechet mean, measured against the
    #    group's own spread (radius is the wrong normaliser -- it collapses to
    #    ~0 for exactly the heterogeneous groups we care about)
    xd, wd = x.double(), w.double()
    fm = frechet_mean(xd, wd, c)
    gm = gyromidpoint(xd, wd, c)
    spread = group_heterogeneity(xd, wd, c)["spread"].clamp_min(1e-6)
    rel = (geodesic_distance(fm, gm, c) / spread).mean()
    ok(f"gyromidpoint approximates the Frechet mean "
       f"(gap = {rel:.1%} of group spread)", bool(rel < 0.10))
    # and it is a descent direction: the iterative mean cannot be much worse
    fvar = lambda m: (wd / wd.sum(-1, keepdim=True)
                      * geodesic_distance(xd, m.unsqueeze(-2), c).pow(2)).sum(-1).mean()
    ok("Frechet iteration decreases the Frechet variance from the gyromidpoint",
       bool(fvar(fm) <= fvar(gm)))

    # 8. exp/log maps are mutually inverse, at the origin and at a base point
    ok("expmap0 and logmap0 are mutually inverse",
       torch.allclose(expmap0(logmap0(x, c), c), x, atol=1e-5))
    base = random_ball_points(4, d=d, c=c, r_max=0.5)
    y = random_ball_points(4, d=d, c=c, r_max=0.5)
    ok("expmap_x and logmap_x are mutually inverse",
       torch.allclose(expmap_x(base, logmap_x(base, y, c), c), y, atol=1e-5))

    # 8b. near-boundary precision. RotH embeddings may sit close to ||z|| = 1,
    #     where float32 artanh saturates; the aggregation must be done in
    #     float64 in that regime. This test documents the threshold.
    xb = random_ball_points(64, 4, d=d, c=c, r_max=0.999)
    g32 = gyromidpoint(xb, c=c)
    g64 = gyromidpoint(xb.double(), c=c).float()
    gap32 = (g32 - g64).abs().max().item()
    ok(f"float32 gyromidpoint stays within 1e-3 of float64 at r<=0.999 "
       f"(max gap {gap32:.2e})", gap32 < 1e-3)

    # 9. modules run and shapes line up
    att = GeometricAttention(dim=d, n_side=2, c=c)
    g, weights = att(x, side=torch.rand(8, 5, 2), mask=torch.ones(8, 5, dtype=torch.bool))
    ok("GeometricAttention returns (B,d) consensus and (B,k) simplex weights",
       g.shape == (8, d) and torch.allclose(weights.sum(-1), torch.ones(8), atol=1e-5))

    poi = project_to_ball(torch.randn(300, d) * 0.2, c)
    sc = HyperbolicScorer(hidden_size=2048, poi_ball=poi, c=c)
    ok("HyperbolicScorer emits (B, N_POI) logits", sc(torch.randn(4, 2048)).shape == (4, 300))

    lm = least_misery_loss(torch.randn(4, 3, 300), torch.randn(4, 300),
                           mask=torch.ones(4, 3, dtype=torch.bool))
    ok("least_misery_loss is a finite scalar", lm.dim() == 0 and torch.isfinite(lm))

    fv = frechet_variance_loss(x, gyromidpoint(x, w, c), w, c)
    ok("frechet_variance_loss is a finite scalar", fv.dim() == 0 and torch.isfinite(fv))


if __name__ == "__main__":
    print("hyperbolic_group self-tests (c = 1.0, d = 64):")
    _self_test()
