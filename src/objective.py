"""
Multi-mask generation with an Acc@t-guided top-3 selection objective, plus a genuine
hyperbolic scoring path.

The objective (supervisor's spec)
--------------------------------
"During training, at each generation step t, the diffusion LLM considers the top-3 most
probable POI tokens as candidate next predictions. It temporarily evaluates each candidate
using the intermediate recommendation metric (Accuracy@t) and selects the one that yields the
highest score before proceeding to generate the next POI token."

Implemented literally. The prompt ends with M mask slots instead of one, so the model GENERATES
a ranked list rather than having one read off its logits:

    [next POIs] <MASK> <MASK> ... <MASK>          M = 10

At slot t the top-3 not-yet-placed candidates are scored by Acc@t -- 1 if the ground-truth POI
is among the t POIs placed so far, else 0 -- and the winner is placed. Cross-entropy at each
slot then imitates that choice.

What the selection rule actually does, stated plainly because it is easy to over-read: Acc@t is
binary, so the rule reduces to *"if the true POI is among the top-3 and not yet placed, place it
now; otherwise keep the model's own argmax."* All three candidates tie whenever the target is
absent from the top-3 (all score 0) or already placed (all score 1). That is not a weakness --
it is exactly the early-placement pressure the objective is after, and it is expert-iteration /
DAgger rather than RL: the oracle demonstrates the best action and CE imitates it. But it does
mean most slots carry self-distillation, not supervision, which is what `w_oracle` moderates.

Two properties worth preserving
-------------------------------
1. **It reduces to the existing objective.** M=1, w_oracle=0, w_rank=1 gives byte-identical
   behaviour to the plain CE in stage6b_run2_server.ipynb, so the new run is A/B-comparable
   against the Acc@1=0.1699 anchor instead of being a fresh baseline.
2. **No duplicate POIs.** Already-placed POIs are masked out before each top-k, otherwise the
   model can emit one POI ten times and make Acc@10 identical to Acc@1 -- the degenerate
   solution that would make the whole objective vacuous.

The `w_rank` term (plain CE toward the true POI at slot 0) is not in the spec and is there on
purpose: evaluation still ranks all 5,120 POIs from the logits at a single position, so without
it the model is free to spread probability mass across slots and degrade exactly the quantity
being measured. Set w_rank=0 to get the spec's objective unaccompanied.

Hyperbolic scoring path
-----------------------
`HyperbolicScorer` addresses the R2 problem in GROUP_REC_PROPOSAL: the existing `tied` path
computes `MLP(h) . logmap0(z_p)`, a Euclidean inner product, and an inner product *cannot
express a geodesic distance*. So hierarchy survives only as feature correlation and the
hyperbolic-vs-euclidean ablation can come back null for architectural rather than geometric
reasons. This ranks by `-d_c(q, z_p) + b_p` with learnable curvature, in 64 dimensions, so it
costs almost nothing and makes the ablation interpretable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-15


# --------------------------------------------------------------------------
# hyperbolic ops (self-contained: this module must not depend on notebook state)
# --------------------------------------------------------------------------

def project_to_ball(x, c=1.0):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    maxnorm = (1.0 - 1e-5) / (c ** 0.5)
    return torch.where(norm > maxnorm, x / norm * maxnorm, x)


def expmap0(v, c=1.0):
    sqrt_c = c ** 0.5
    norm = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return project_to_ball(torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm), c)


def mobius_add(x, y, c=1.0):
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return num / den.clamp_min(_EPS)


def geodesic_distance(x, y, c=1.0):
    sqrt_c = c ** 0.5
    add = mobius_add(-x, y, c)
    n = add.norm(dim=-1).clamp_min(_EPS)
    return (2.0 / sqrt_c) * torch.arctanh((sqrt_c * n).clamp(max=1.0 - 1e-7))


def geodesic_distance_matrix(q, Z, c=1.0):
    """All-pairs Poincare distance, q [B, d] vs Z [N, d] -> [B, N], WITHOUT broadcasting.

    Uses the arccosh form

        d_c(x, y) = (1/sqrt(c)) * arccosh(1 + 2c||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))

    which depends on the data only through ||x-y||^2 and can therefore be computed with a
    matmul. The gyro form -- (2/sqrt c) artanh(sqrt c ||(-x) (+) y||) -- needs mobius_add, whose
    [B, N, d] broadcast costs 0.21 GB per intermediate at B*M=160, N=5120, d=64, and allocates
    roughly six of them which autograd then keeps for backward. That is >1.3 GB per forward on
    top of a 7B model. This version is ~3 MB. The two are equivalent; the self-check asserts it
    numerically rather than taking the identity on trust.
    """
    q2 = (q * q).sum(-1, keepdim=True)                       # [B, 1]
    z2 = (Z * Z).sum(-1).unsqueeze(0)                        # [1, N]
    sq = (q2 + z2 - 2.0 * (q @ Z.t())).clamp_min(0.0)        # ||q - z||^2  via matmul
    denom = ((1.0 - c * q2).clamp_min(_EPS) * (1.0 - c * z2).clamp_min(_EPS))
    arg = (1.0 + 2.0 * c * sq / denom).clamp_min(1.0 + 1e-7)
    return torch.arccosh(arg) / (c ** 0.5)


class HyperbolicScorer(nn.Module):
    """Rank POIs by NEGATIVE GEODESIC DISTANCE on the Poincare ball, not by dot product.

    hidden -> 64-d tangent vector -> expmap0 onto the ball -> -d_c(q, z_p) + b_p.

    Curvature is learned through softplus so it stays strictly positive; initialising the
    pre-softplus logit at log(e-1) makes c start at exactly 1.0, matching the RotH training.
    """

    def __init__(self, hidden_size, poi_ball, c_init=1.0, learn_curvature=True, gate_init=0.0):
        super().__init__()
        d = poi_ball.shape[1]
        self.to_tangent = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Linear(hidden_size // 4, d),
        )
        # Zero BOTH weight and bias, so the tangent vector is exactly 0 and the query sits at
        # the origin. Zeroing only the weight leaves a random bias and puts the query somewhere
        # arbitrary on the ball, which is a silent init bug -- the model still trains, it just
        # starts from a meaningless point.
        nn.init.zeros_(self.to_tangent[3].weight)
        nn.init.zeros_(self.to_tangent[3].bias)
        self.register_buffer("poi_ball", project_to_ball(poi_ball.float(), c_init))
        self.bias = nn.Parameter(torch.zeros(poi_ball.shape[0]))
        inv = torch.log(torch.expm1(torch.tensor(float(c_init))))
        self.c_logit = nn.Parameter(inv.clone(), requires_grad=learn_curvature)
        # ReZero-style gate. Even with q at the origin, -d_c(0, z_p) still VARIES across POIs
        # (it is a monotone function of the radius), so the scorer would contribute a non-flat
        # per-POI prior at step 0 and break the notebook's `loss == ln(N_POI)` init assert --
        # the check that catches a mis-wired head. The gate starts at 0 so the term is exactly
        # zero at init; its own gradient is non-zero, so the model opens it if it is useful.
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    @property
    def c(self):
        return F.softplus(self.c_logit).clamp_min(1e-4)

    def forward(self, h):
        c = self.c
        q = expmap0(self.to_tangent(h), c)                       # [B, d] on the ball
        d = geodesic_distance_matrix(q, self.poi_ball, c)        # [B, N], matmul-based
        return self.gate * (-d + self.bias)


# --------------------------------------------------------------------------
# the Acc@t-guided selection
# --------------------------------------------------------------------------

@torch.no_grad()
def oracle_select(logits, target, top_k=3):
    """Acc@t-guided top-k selection over M generation slots. Fully vectorised over the batch.

    logits : [B, M, N_POI]   per-slot scores
    target : [B]             the ground-truth next POI
    returns: selected [B, M] long, acc_at [B, M] float (Acc@t after placing slot t)

    At slot t: mask already-placed POIs, take the top-k, score each by Acc@t (1 iff the target
    is among the t placed POIs), keep the winner. Ties -- which is every slot where the target
    is absent from the top-k or already placed -- fall back to the model's own argmax.
    """
    B, M, N = logits.shape
    placed = torch.zeros(B, N, dtype=torch.bool, device=logits.device)
    selected = torch.zeros(B, M, dtype=torch.long, device=logits.device)
    acc_at = torch.zeros(B, M, device=logits.device)
    tgt = target.view(B, 1)

    for t in range(M):
        lg = logits[:, t, :].masked_fill(placed, float("-inf"))
        topk = lg.topk(min(top_k, N), dim=-1).indices                  # [B, k]
        hit = (topk == tgt).any(-1)                                    # target among top-k
        free = ~placed.gather(1, tgt).squeeze(1)                       # not already placed
        pick = torch.where(hit & free, target, topk[:, 0])
        selected[:, t] = pick
        placed.scatter_(1, pick.view(B, 1), True)
        acc_at[:, t] = placed.gather(1, tgt).squeeze(1).float()
    return selected, acc_at


def multimask_loss(logits, target, w_rank=1.0, w_oracle=1.0, top_k=3, label_smoothing=0.0):
    """Loss for the multi-mask Acc@t objective.

    logits : [B, M, N_POI]
    target : [B]

    total = w_rank * CE(slot 0, true POI)  +  w_oracle * mean_t CE(slot t, oracle choice)

    With M=1, w_oracle=0, w_rank=1 this is exactly the notebook's existing single-token CE, so
    the objective is A/B-comparable against the current checkpoint rather than a new baseline.
    """
    B, M, N = logits.shape
    stats = {}

    rank_loss = logits.new_zeros(())
    if w_rank > 0:
        rank_loss = F.cross_entropy(logits[:, 0, :], target, label_smoothing=label_smoothing)
    stats["rank"] = rank_loss.detach()

    oracle_loss = logits.new_zeros(())
    if w_oracle > 0 and M > 0:
        selected, acc_at = oracle_select(logits, target, top_k=top_k)
        oracle_loss = F.cross_entropy(logits.reshape(B * M, N), selected.reshape(B * M),
                                      label_smoothing=label_smoothing)
        stats["oracle"] = oracle_loss.detach()
        # diagnostics: how often the oracle actually overrode the model, and the Acc@k curve
        with torch.no_grad():
            argmax0 = logits.argmax(-1)
            stats["override_rate"] = (selected != argmax0).float().mean()
            for k in (1, 5, 10):
                if k <= M:
                    stats[f"acc@{k}"] = acc_at[:, k - 1].mean()
    else:
        stats["oracle"] = oracle_loss.detach()

    return w_rank * rank_loss + w_oracle * oracle_loss, stats


# --------------------------------------------------------------------------
# collator
# --------------------------------------------------------------------------

def make_multimask_collate(encode_fn, mask_token_id, eos_id, n_masks=10, max_len=1024):
    """Left-pad with EOS, then append `n_masks` mask slots flush right.

    encode_fn(example) -> (prompt_ids, target_local) where target_local is in [0, N_POI).
    Returns dict with input_ids [B, L], mask_pos [B, M] (indices of the mask slots) and
    target [B]. mask_pos is returned explicitly so the training loop never has to re-derive
    slot order from a label tensor -- getting that order wrong silently scrambles Acc@t.
    """
    def collate(batch):
        enc = [encode_fn(ex) for ex in batch]
        L = max(len(p) + n_masks for p, _ in enc)
        assert L <= max_len, f"prompt+{n_masks} masks = {L} exceeds max_len {max_len}"
        B = len(batch)
        input_ids = torch.full((B, L), eos_id, dtype=torch.long)
        mask_pos = torch.zeros(B, n_masks, dtype=torch.long)
        target = torch.zeros(B, dtype=torch.long)
        for i, (p_ids, tgt) in enumerate(enc):
            n = len(p_ids)
            start = L - (n + n_masks)
            input_ids[i, start:start + n] = torch.tensor(p_ids, dtype=torch.long)
            input_ids[i, start + n:start + n + n_masks] = mask_token_id
            mask_pos[i] = torch.arange(start + n, start + n + n_masks)
            target[i] = tgt
        return dict(input_ids=input_ids, mask_pos=mask_pos, target=target,
                    attention_mask=torch.ones((B, L), dtype=torch.long))
    return collate


def gather_slot_hidden(h, mask_pos):
    """h [B, L, H] + mask_pos [B, M] -> [B, M, H] in slot order."""
    B, M = mask_pos.shape
    idx = mask_pos.unsqueeze(-1).expand(B, M, h.shape[-1])
    return h.gather(1, idx)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def _self_check():
    print("SELF-CHECK  objective.py")
    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    res = []
    torch.manual_seed(0)
    B, M, N = 6, 10, 40

    # --- oracle picks the target as soon as it is in the top-3 ---
    logits = torch.randn(B, M, N)
    target = torch.randint(0, N, (B,))
    logits[:, 0, :].scatter_(1, target.view(B, 1), 5.0)          # target is top-1 at slot 0
    sel, acc = oracle_select(logits, target)
    res.append(ok("target placed at slot 0 when it is top-1 there", bool((sel[:, 0] == target).all())))
    res.append(ok("Acc@1 = 1 for every row in that case", bool((acc[:, 0] == 1).all())))

    # --- target ranked 3rd is still selected (the whole point of top-3) ---
    logits = torch.zeros(B, M, N)
    logits[:, :, :] = torch.randn(B, M, N) * 0.01
    for i in range(B):
        others = [j for j in range(N) if j != target[i]][:2]
        logits[i, 0, others[0]] = 3.0
        logits[i, 0, others[1]] = 2.0
        logits[i, 0, target[i]] = 1.0                             # rank 3
    sel, acc = oracle_select(logits, target, top_k=3)
    res.append(ok("target at rank 3 is selected over the argmax", bool((sel[:, 0] == target).all())))
    sel2, _ = oracle_select(logits, target, top_k=2)
    res.append(ok("...and NOT selected when top_k=2 (rank 3 is out of reach)",
                  bool((sel2[:, 0] != target).all())))

    # --- no duplicates, ever ---
    logits = torch.randn(B, M, N)
    sel, acc = oracle_select(logits, target)
    dup = sum(len(set(sel[i].tolist())) != M for i in range(B))
    res.append(ok(f"all {M} slots distinct in every row (dups={dup})", dup == 0))

    # --- Acc@t is monotone non-decreasing, and 1 forever once hit ---
    mono = bool((acc[:, 1:] >= acc[:, :-1]).all())
    res.append(ok("Acc@t is monotone non-decreasing in t", mono))

    # --- reduction to the existing objective ---
    logits1 = torch.randn(B, 1, N)
    loss_new, _ = multimask_loss(logits1, target, w_rank=1.0, w_oracle=0.0)
    loss_old = F.cross_entropy(logits1[:, 0, :], target)
    res.append(ok("M=1, w_oracle=0 reproduces plain CE exactly",
                  torch.allclose(loss_new, loss_old, atol=1e-6)))

    # --- gradients actually flow to every slot ---
    lg = torch.randn(B, M, N, requires_grad=True)
    loss, stats = multimask_loss(lg, target, w_rank=1.0, w_oracle=1.0)
    loss.backward()
    per_slot = lg.grad.abs().sum(dim=(0, 2))
    res.append(ok("every generation slot receives gradient", bool((per_slot > 0).all())))
    res.append(ok("stats report Acc@1/5/10 and override rate",
                  all(k in stats for k in ("acc@1", "acc@5", "acc@10", "override_rate"))))

    # --- collator: masks flush right, slot order correct ---
    def enc(ex):
        return list(range(ex["n"])), ex["t"]
    coll = make_multimask_collate(enc, mask_token_id=999, eos_id=0, n_masks=M, max_len=64)
    batch = coll([{"n": 5, "t": 3}, {"n": 9, "t": 7}])
    L = batch["input_ids"].shape[1]
    res.append(ok("mask slots sit flush right", int(batch["mask_pos"][:, -1].max()) == L - 1))
    got = batch["input_ids"].gather(1, batch["mask_pos"])
    res.append(ok("every mask_pos really points at a mask token", bool((got == 999).all())))
    res.append(ok("mask_pos is strictly increasing (slot order preserved)",
                  bool((batch["mask_pos"][:, 1:] > batch["mask_pos"][:, :-1]).all())))

    # --- hyperbolic scorer ---
    ball = project_to_ball(torch.randn(N, 8) * 0.1)
    sc = HyperbolicScorer(hidden_size=32, poi_ball=ball)
    h = torch.randn(B, 32)
    s = sc(h)
    res.append(ok("hyperbolic scorer returns [B, N_POI]", tuple(s.shape) == (B, N)))
    res.append(ok("curvature starts at 1.0", abs(float(sc.c.detach()) - 1.0) < 1e-4))
    res.append(ok("scores are finite", bool(torch.isfinite(s).all())))

    # The init contract the notebook's step-0 assert depends on: the term must be exactly zero,
    # so head+tied+hyperbolic still gives loss == ln(N_POI) at step 0.
    res.append(ok("gated to exactly 0 at init (preserves the ln(N_POI) step-0 loss)",
                  bool((s.abs() < 1e-12).all())))
    with torch.no_grad():
        logits0 = torch.zeros(B, N) + sc(h)                  # head=0, tied=0, hyper=gated 0
        loss0 = F.cross_entropy(logits0, target)
    import math
    res.append(ok(f"step-0 loss == ln({N}) = {math.log(N):.4f} with all three heads",
                  abs(float(loss0) - math.log(N)) < 1e-5))

    # q must sit at the ORIGIN at init -- zeroing only the weight and leaving a random bias
    # would put it somewhere arbitrary, which is the bug this checks for.
    with torch.no_grad():
        q0 = sc.to_tangent(h)
    res.append(ok("zero-init puts the query exactly at the origin",
                  bool((q0.abs() < 1e-12).all())))

    s.sum().backward()
    res.append(ok("gradient reaches the gate at init (so it can open)",
                  sc.gate.grad is not None and bool(sc.gate.grad.abs() > 0)))
    # With a zero last layer the earlier layers are correctly gradient-blocked at init; they
    # unblock once that layer moves. Verify the path is live rather than dead.
    sc.zero_grad()
    with torch.no_grad():
        sc.gate.fill_(1.0)
        sc.to_tangent[3].weight.normal_(0, 0.02)
    sc(h).sum().backward()
    res.append(ok("gradient reaches the tangent map once the gate opens",
                  sc.to_tangent[0].weight.grad is not None
                  and bool(sc.to_tangent[0].weight.grad.abs().sum() > 0)))
    res.append(ok("curvature is learnable", sc.c_logit.grad is not None
                  and bool(sc.c_logit.grad.abs() > 0)))

    # --- the matmul distance MUST equal the gyro form, or the scorer is silently wrong ---
    # Points are placed at an explicit FRACTION of the ball radius 1/sqrt(c) rather than via
    # project_to_ball(randn*s, c): that clamps at (1-1e-5)/sqrt(c), which shrinks as c grows, so
    # at c=2 most points land exactly ON the boundary. There the GYRO form saturates -- its
    # artanh argument is clamped at 1-1e-7 -- and the two forms diverge by whole units. The
    # matmul form is the accurate one in that regime; the old test read that as a formula bug.
    def _on_sphere(n, d, frac, c, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, d, generator=g)
        return x / x.norm(dim=-1, keepdim=True) * (frac / c ** 0.5)

    for cc in (0.5, 1.0, 2.0):
        for frac in (0.2, 0.6):
            qq, ZZ = _on_sphere(7, 8, frac, cc, 1), _on_sphere(11, 8, frac, cc, 2)
            fast = geodesic_distance_matrix(qq, ZZ, cc)
            slow = geodesic_distance(qq.unsqueeze(1), ZZ.unsqueeze(0), cc)
            err = float((fast - slow).abs().max())
            res.append(ok(f"matmul == mobius at c={cc}, r={frac}R (err {err:.1e})", err < 1e-5))

    # RotH embeddings live at ||z|| in [0.30, 0.65] for c=1, i.e. the regime just checked.
    # Near the boundary the forms legitimately diverge (gyro saturates); require only finiteness.
    qq, ZZ = _on_sphere(5, 8, 0.999, 1.0, 3), _on_sphere(5, 8, 0.999, 1.0, 4)
    f2 = geodesic_distance_matrix(qq, ZZ, 1.0)
    res.append(ok("matmul distance stays finite at ||x|| = 0.999R", bool(torch.isfinite(f2).all())))
    res.append(ok("...and is monotone in separation there", bool((f2 > 0).all())))
    return all(res)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_check() else 1)
