"""
RotH (rotation-based hyperbolic KG embedding) on the merged POI + group graph, WITH a depth
regulariser.

Why the regulariser exists
--------------------------
The previously shipped `poi_hyperbolic_embs.npy` has essentially no radial hierarchy. Diagnostic
D1 over all 5,120 POIs:

    Spearman(taxonomy depth, hyperbolic radius) = +0.0189      VERDICT: ABSENT
    depth 1 mean radius 1.2557    depth 3 mean radius 1.2577
    depth 2 mean radius 1.2529    depth 5 mean radius 1.2732      (all norms in [0.465, 0.638])

Every POI sits in the same thin shell, so "general categories near the origin, specific ones near
the boundary" -- the entire premise of using hyperbolic space, and the substrate any
consensus-by-generalisation argument needs -- is simply not there. That is not a bug in RotH: RotH
optimises link prediction, and nothing in a link-prediction loss asks the radial coordinate to
encode depth. It is a hoped-for by-product, and here it did not happen.

The fix is one extra term over the graph's parent->child edges:

    L_depth = mean over (parent, child) of  relu( margin - (r(child) - r(parent)) )
    with r(z) = 2 * artanh(||z||)         -- the same quantity D1 reports

i.e. a child must sit at least `margin` further from the origin than its parent, or it is
penalised. Cheap, differentiable, and it targets exactly the coordinate the link-prediction loss
ignores. Everything else follows the recipe that produced the project's best RotH run
(HAS_CATEGORY MRR 0.120 -> 0.191): d=64, learnable per-relation curvature, NSSA self-adversarial
loss, type-restricted negatives for the hierarchical relations, inverse-frequency relation
weighting, cosine LR.

Type-restricted negatives are not optional. The Stage-2 audit found that under uniform sampling
only 7.0-7.4% of negatives for the hierarchical relations were even the correct node type, so the
model was rewarded for separating types rather than for learning hierarchy.

Usage
-----
    python train_roth.py --kg-dir ./data/kg --out-dir ./data/kg --epochs 150
    python train_roth.py --self-check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-15
MAX_NORM = 1.0 - 1e-5


# --------------------------------------------------------------------------
# Poincare-ball ops (mirrors stage2-hyperbolic-ops)
# --------------------------------------------------------------------------

def project(x, c):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    maxn = MAX_NORM / c.clamp_min(EPS).sqrt()
    return torch.where(norm > maxn, x / norm * maxn, x)


def expmap0(v, c):
    sqrt_c = c.clamp_min(EPS).sqrt()
    norm = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return project(torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm), c)


def mobius_add(x, y, c):
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return project(num / den.clamp_min(EPS), c)


def geodesic_dist(x, y, c):
    sqrt_c = c.clamp_min(EPS).sqrt()
    diff = mobius_add(-x, y, c).norm(dim=-1).clamp_min(EPS)
    return 2.0 / sqrt_c.squeeze(-1) * torch.atanh((sqrt_c.squeeze(-1) * diff).clamp(max=MAX_NORM))


def givens_rotation(theta, x):
    """Block-diagonal 2x2 rotations -- the 'Rot' in RotH, and an isometry of the ball."""
    d = x.shape[-1]
    x = x.view(*x.shape[:-1], d // 2, 2)
    cos, sin = torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)
    x0, x1 = x[..., 0:1], x[..., 1:2]
    out = torch.cat([cos * x0 - sin * x1, sin * x0 + cos * x1], dim=-1)
    return out.view(*out.shape[:-2], d)


def hyp_radius(z, c=1.0):
    """Hyperbolic distance from the origin -- the quantity D1 reports as 'radius'."""
    n = z.norm(dim=-1).clamp(max=MAX_NORM)
    return 2.0 * torch.atanh(n * math.sqrt(c))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class RotH(nn.Module):
    def __init__(self, n_ent, n_rel, dim=64, init_scale=1e-3):
        super().__init__()
        assert dim % 2 == 0, "RotH needs an even dim for its 2x2 rotation blocks"
        self.dim = dim
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)            # translation, in tangent space
        self.theta = nn.Embedding(n_rel, dim // 2)     # rotation angles
        self.bh = nn.Embedding(n_ent, 1)
        self.bt = nn.Embedding(n_ent, 1)
        # learnable per-relation curvature via softplus, so c stays strictly positive
        self.c_logit = nn.Parameter(torch.zeros(n_rel))
        nn.init.normal_(self.ent.weight, std=init_scale)
        nn.init.normal_(self.rel.weight, std=init_scale)
        nn.init.uniform_(self.theta.weight, -0.1, 0.1)
        nn.init.zeros_(self.bh.weight)
        nn.init.zeros_(self.bt.weight)

    def curvature(self, r_id):
        return nn.functional.softplus(self.c_logit[r_id]).unsqueeze(-1) + 1e-4

    def score(self, h, r, t):
        """h, t: [B] or [B, N] entity ids; r: [B] relation ids. Returns matching-shape scores."""
        if t.dim() > h.dim():
            h = h.unsqueeze(-1).expand_as(t)
            r_exp = r.unsqueeze(-1).expand_as(t)
        else:
            r_exp = r
        c = self.curvature(r_exp)
        x_h = expmap0(self.ent(h), c)
        x_t = expmap0(self.ent(t), c)
        trans = expmap0(self.rel(r_exp), c)
        q = givens_rotation(self.theta(r_exp), mobius_add(x_h, trans, c))
        d = geodesic_dist(q, x_t, c)
        return -(d ** 2) + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)

    def ball_points(self, c=1.0):
        """All entity embeddings as Poincare-ball points at a common reference curvature.

        The saved artifact must be at one curvature, not each relation's own -- the downstream
        LLaDA pipeline applies logmap0 with c=CURVATURE_C=1.0.
        """
        cc = torch.full((1,), float(c))
        return expmap0(self.ent.weight, cc)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def build_type_pools(ent_types, id_of_type):
    """entity ids grouped by node type, for type-restricted negative sampling."""
    pools = {}
    for t, ids in id_of_type.items():
        pools[t] = torch.tensor(sorted(ids), dtype=torch.long)
    return pools


def sample_negatives(t_ids, r_ids, n_neg, n_ent, tail_pool_of_rel, generator):
    """Corrupt tails. Hierarchical relations draw from the true tail's TYPE pool only."""
    B = len(t_ids)
    neg = torch.randint(0, n_ent, (B, n_neg), generator=generator)
    for r, pool in tail_pool_of_rel.items():
        m = (r_ids == r)
        if not m.any() or len(pool) == 0:
            continue
        k = int(m.sum())
        pick = torch.randint(0, len(pool), (k, n_neg), generator=generator)
        neg[m] = pool[pick]
    return neg


def nssa_loss(pos, neg, gamma, alpha, weights=None):
    """Self-adversarial negative sampling loss (Sun et al., RotatE)."""
    pos_term = -nn.functional.logsigmoid(gamma + pos)
    with torch.no_grad():
        w = torch.softmax(alpha * neg, dim=-1)
    neg_term = -(w * nn.functional.logsigmoid(-(gamma + neg))).sum(-1)
    per = pos_term + neg_term
    if weights is not None:
        per = per * weights
    return per.mean()


def depth_loss(model, hier, margin, c=1.0, rel_groups=None, root_pull=0.0):
    """relu(margin - (r_child - r_parent)) over parent->child pairs. THE fix for D1.

    Averaged PER RELATION, not per pair. A flat mean is dominated by whichever hierarchy relation
    happens to be largest: HAS_CATEGORY contributes 5,120 pairs and SUBCATEGORY_OF only 417, so a
    flat mean gives the taxonomy spine 2.9% of the gradient. Measured consequence of that -- the
    cross-type ordering came out right (REGION 0.31 < LOCALITY 0.59 < CATEGORY 0.66 < POI 0.76)
    while the category chain itself stayed unordered (depth1 0.708 > depth2 0.617), and D1 is
    precisely a statement about that chain.

    `root_pull` adds a weak penalty on every entity's radius. The relu is one-sided, so on its own
    it can be satisfied by inflating all radii together; pulling everything toward the origin makes
    the margin the binding constraint and is how Poincare embeddings get their hierarchy for free.
    """
    if len(hier) == 0:
        return torch.zeros((), dtype=torch.float32)
    z = model.ball_points(c)
    r_par = hyp_radius(z[hier[:, 0]], c)
    r_chi = hyp_radius(z[hier[:, 1]], c)
    viol = torch.relu(margin - (r_chi - r_par))

    if rel_groups:
        terms = [viol[m].mean() for m in rel_groups if m.any()]
        loss = torch.stack(terms).mean() if terms else viol.mean()
    else:
        loss = viol.mean()

    if root_pull > 0:
        loss = loss + root_pull * hyp_radius(z, c).pow(2).mean()
    return loss


@torch.no_grad()
def link_prediction(model, triples, n_ent, filt, per_relation=None, max_eval=3000, seed=0,
                    device=None):
    """Filtered MRR / Hits@1 by corrupting the tail against all entities."""
    device = device or torch.device("cpu")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(triples), generator=g)[:max_eval]
    sub = triples[idx]
    all_ent = torch.arange(n_ent, device=device)
    ranks, by_rel = [], {}
    for h, r, t in sub.tolist():
        s = model.score(torch.tensor([h], device=device), torch.tensor([r], device=device),
                        all_ent.unsqueeze(0))[0]
        known = filt.get((h, r), ())
        if known:
            k = torch.tensor([x for x in known if x != t], dtype=torch.long, device=device)
            if len(k):
                s[k] = -1e9
        rank = int((s > s[t]).sum().item()) + 1
        ranks.append(rank)
        by_rel.setdefault(r, []).append(rank)
    ranks = np.array(ranks, dtype=float)
    out = dict(mrr=float((1.0 / ranks).mean()), hits1=float((ranks <= 1).mean()),
               hits10=float((ranks <= 10).mean()), n=len(ranks))
    if per_relation is not None:
        out["per_relation"] = {
            per_relation[r]: dict(mrr=float(np.mean(1.0 / np.array(v))),
                                  hits1=float(np.mean(np.array(v) <= 1)), n=len(v))
            for r, v in sorted(by_rel.items())}
    return out


def d1_radial_hierarchy(z_poi, depths):
    """Spearman(taxonomy depth, hyperbolic radius) -- the diagnostic this whole file targets."""
    r = hyp_radius(torch.as_tensor(z_poi), 1.0).numpy()
    d = np.asarray(depths, dtype=float)
    ok = np.isfinite(r) & np.isfinite(d)
    r, d = r[ok], d[ok]

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        rk = np.empty(len(a), dtype=float)
        rk[order] = np.arange(len(a), dtype=float)
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, rk)
        return (sums / cnt)[inv]

    ra, rb = rank(d), rank(r)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    rho = float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))
    by_depth = {int(k): dict(n=int((d == k).sum()), mean_radius=float(r[d == k].mean()))
                for k in sorted(set(d.tolist()))}
    verdict = ("STRONG" if rho > 0.30 else "WEAK" if rho > 0.10 else "ABSENT")
    return dict(spearman=rho, verdict=verdict, by_depth=by_depth,
                norm_min=float(np.linalg.norm(z_poi, axis=1).min()),
                norm_max=float(np.linalg.norm(z_poi, axis=1).max()))


def train(triples, hier, n_ent, n_rel, rel_names, ent_type_ids, a, poi_rows=None, depths=None):
    device = torch.device(getattr(a, "device", None) or "cpu")
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)      # kept on CPU: randperm/randint stay CPU-side,
                                                      # only the small per-batch slices move to device

    # held-out triples for link prediction
    perm = torch.randperm(len(triples), generator=gen)
    n_val = max(1, int(len(triples) * a.val_frac))
    val_t, train_t = triples[perm[:n_val]], triples[perm[n_val:]]

    filt = {}
    for h, r, t in triples.tolist():
        filt.setdefault((h, r), []).append(t)

    # type-restricted negatives for the hierarchical relations
    name_of = {v: k for k, v in rel_names.items()}
    tail_types = {}
    for h, r, t in train_t.tolist():
        tail_types.setdefault(r, set()).add(ent_type_ids[t])
    # Type-restricted negatives. `hierarchy` (the original setting) leaves the flat relations
    # sampling uniformly over all entities -- but FOLLOWED_BY and IS_NEAR_TO are POI->POI and
    # only 54% of the graph is POI, so 46% of their negatives are wrong by node type alone and
    # teach nothing. Measured on the trained model: a positive beats a uniform negative 100.0%
    # of the time and a HARD negative only 43.2%, and with n_neg=25 the expected number of
    # actually-competitive negatives per batch item is 0.22 -- i.e. ~78% of steps carry no
    # learning signal at all, which is why the loss plateaus by epoch 20 at Hits@1 = 0.056.
    pools = {}
    if a.typed_negatives != "none":
        for r, types in tail_types.items():
            if a.typed_negatives == "all" or name_of[r] in a.hierarchy_relations:
                ids = [i for i, ty in enumerate(ent_type_ids) if ty in types]
                pools[r] = torch.tensor(ids, dtype=torch.long)
    if pools:
        sizes = {name_of[r]: len(p) for r, p in sorted(pools.items())}
        print(f"type-restricted negatives ({a.typed_negatives}): " +
              "  ".join(f"{k}<-{v:,}" for k, v in sizes.items()))
    else:
        print("negatives: uniform over all entities")

    # inverse-frequency relation weights (IS_NEAR_TO is 36% of edges on its own)
    cnt = torch.bincount(train_t[:, 1], minlength=n_rel).float().clamp_min(1)
    rel_w = (cnt.sum() / cnt)
    rel_w = (rel_w / rel_w.mean()).clamp(max=a.max_rel_weight)

    model = RotH(n_ent, n_rel, a.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.02)

    # boolean mask per hierarchy relation, so the depth loss weights each type equally
    rel_groups = []
    hier = hier.to(device)
    if hier.numel() and hier.shape[1] == 3:
        for r in sorted(set(hier[:, 2].tolist())):
            rel_groups.append(hier[:, 2] == r)
        print("depth-loss hierarchy relations: " + "  ".join(
            f"{name_of[r]}={int((hier[:,2]==r).sum()):,}" for r in sorted(set(hier[:, 2].tolist()))))

    n_batches = math.ceil(len(train_t) / a.batch_size)
    print(f"device={device}  entities={n_ent:,} relations={n_rel} dim={a.dim}  "
          f"train triples={len(train_t):,} val={len(val_t):,}  hierarchy pairs={len(hier):,}")
    print(f"epochs={a.epochs} batch={a.batch_size} negatives={a.n_neg} "
          f"depth_weight={a.depth_weight} margin={a.depth_margin}\n")

    history = []
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = torch.randperm(len(train_t), generator=gen)
        tot = tot_kge = tot_depth = 0.0
        for b in range(n_batches):
            bi = order[b * a.batch_size:(b + 1) * a.batch_size]
            h_cpu, r_cpu, t_cpu = train_t[bi, 0], train_t[bi, 1], train_t[bi, 2]
            neg_cpu = sample_negatives(t_cpu, r_cpu, a.n_neg, n_ent, pools, gen)
            w_r = rel_w[r_cpu].to(device)
            h, r, t, neg = (x.to(device) for x in (h_cpu, r_cpu, t_cpu, neg_cpu))

            pos_s = model.score(h, r, t)
            neg_s = model.score(h, r, neg)
            l_kge = nssa_loss(pos_s, neg_s, a.gamma, a.alpha, w_r)
            l_depth = depth_loss(model, hier, a.depth_margin,
                                 rel_groups=rel_groups, root_pull=a.root_pull)
            loss = l_kge + a.depth_weight * l_depth

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); tot_kge += l_kge.item(); tot_depth += float(l_depth.detach())
        sched.step()

        if epoch % a.log_every == 0 or epoch == a.epochs or epoch == 1:
            msg = (f"epoch {epoch:>4}/{a.epochs}  loss={tot/n_batches:.4f} "
                   f"kge={tot_kge/n_batches:.4f} depth={tot_depth/n_batches:.4f} "
                   f"lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
            if depths is not None and poi_rows is not None:
                model.eval()
                with torch.no_grad():
                    z = model.ball_points(1.0).cpu().numpy()[poi_rows]
                d1 = d1_radial_hierarchy(z, depths)
                msg += f"  D1_rho={d1['spearman']:+.4f} ({d1['verdict']})"
                history.append(dict(epoch=epoch, loss=tot / n_batches,
                                    depth=tot_depth / n_batches, d1=d1["spearman"]))
            print(msg)

    model.eval()
    lp = link_prediction(model, val_t, n_ent, filt, per_relation=name_of,
                         max_eval=a.max_eval, seed=a.seed, device=device)
    return model, lp, history


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(a):
    triples = torch.load(os.path.join(a.kg_dir, "kg_triples.pt"))
    hier = torch.load(os.path.join(a.kg_dir, "kg_hierarchy.pt"))
    with open(os.path.join(a.kg_dir, "kg_entities.json")) as f:
        ents = json.load(f)
    with open(os.path.join(a.kg_dir, "kg_relations.json")) as f:
        rels = json.load(f)
    with open(os.path.join(a.kg_dir, "kg_poi_rows.json")) as f:
        poi_rows_map = {int(k): int(v) for k, v in json.load(f).items()}

    n_ent = len(ents)
    rel_names = rels["relation_to_id"]
    a.hierarchy_relations = rels.get("hierarchy_relations", [])
    ent_type_ids = [None] * n_ent
    for name, rec in ents.items():
        ent_type_ids[rec["id"]] = rec["type"]

    # POI rows in poi_idx order + their taxonomy depths, for D1
    meta = None
    depths = poi_rows = None
    mp = os.path.join(a.data_dir, f"poi_metadata_{a.dataset}.csv")
    if os.path.exists(mp):
        import pandas as pd
        meta = pd.read_csv(mp)
        col = next((c for c in ("category_path", "category") if c in meta.columns), None)
        poi_rows = np.array([poi_rows_map[int(p)] for p in meta["poi_idx"]], dtype=int)
        depths = np.array([len([x for x in str(s).split(">") if x.strip()])
                           for s in meta[col].fillna("")], dtype=float)

    model, lp, history = train(triples, hier, n_ent, len(rel_names), rel_names,
                               ent_type_ids, a, poi_rows, depths)

    print(f"\nlink prediction (filtered, n={lp['n']}): "
          f"MRR={lp['mrr']:.4f}  Hits@1={lp['hits1']:.4f}  Hits@10={lp['hits10']:.4f}")
    print(f"  {'relation':<20} {'MRR':>7} {'Hits@1':>7} {'n':>6}")
    for r, v in sorted(lp["per_relation"].items(), key=lambda kv: -kv[1]["mrr"]):
        print(f"  {r:<20} {v['mrr']:>7.4f} {v['hits1']:>7.4f} {v['n']:>6}")

    os.makedirs(a.out_dir, exist_ok=True)
    with torch.no_grad():
        z_all = model.ball_points(a.curvature).cpu().numpy().astype(np.float32)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "dim": a.dim,
                "curvature_per_relation": torch.nn.functional.softplus(
                    model.c_logit.detach()).tolist()},
               os.path.join(a.out_dir, "roth_best.pt"))
    np.save(os.path.join(a.out_dir, "entity_hyperbolic_embs.npy"), z_all)

    result = dict(link_prediction=lp, history=history, config=vars(a))
    if poi_rows is not None:
        z_poi = z_all[poi_rows]
        np.save(os.path.join(a.out_dir, f"poi_hyperbolic_embs_{a.dataset}.npy"), z_poi)
        d1 = d1_radial_hierarchy(z_poi, depths)
        result["d1"] = d1
        print(f"\nD1  Spearman(taxonomy depth, hyperbolic radius) = {d1['spearman']:+.4f}   "
              f"VERDICT: {d1['verdict']}   (was +0.0189 ABSENT)")
        print(f"  norms in [{d1['norm_min']:.4f}, {d1['norm_max']:.4f}]")
        for d, v in d1["by_depth"].items():
            print(f"    depth {d}: n={v['n']:>5}  mean radius = {v['mean_radius']:.4f}")
        print(f"\n  saved poi_hyperbolic_embs_{a.dataset}.npy {z_poi.shape} "
              f"(rows in poi_idx order -- drop-in for EMB_FILE)")

    with open(os.path.join(a.out_dir, "roth_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved roth_best.pt, entity_hyperbolic_embs.npy, roth_results.json -> {a.out_dir}")


def _self_check():
    """Synthetic 3-level tree. The depth regulariser must recover a radial ordering that plain
    link-prediction training does not."""
    print("SELF-CHECK: does the depth regulariser create a radial hierarchy?")
    torch.manual_seed(0)
    # tree: 1 root -> 4 mid -> 16 leaf, plus flat sibling edges to give the KGE something else
    ents, depth = {}, {}
    ents["root"] = 0; depth[0] = 1
    tri, hier = [], []
    nid = 1
    for m in range(4):
        ents[f"m{m}"] = nid; depth[nid] = 2
        tri.append((nid, 0, 0)); hier.append((0, nid)); mid = nid; nid += 1
        for l in range(4):
            ents[f"l{m}_{l}"] = nid; depth[nid] = 3
            tri.append((nid, 0, mid)); hier.append((mid, nid)); nid += 1
    for i in range(1, nid):                          # flat relation
        tri.append((i, 1, 1 + (i % (nid - 1))))
    triples = torch.tensor([(h, r, t) for h, r, t in tri], dtype=torch.long)
    hier_t = torch.tensor([(p_, c_, 0) for p_, c_ in hier], dtype=torch.long)
    n_ent = nid
    ent_type_ids = ["NODE"] * n_ent
    depths = np.array([depth[i] for i in range(n_ent)], dtype=float)

    class A: pass
    res = {}
    for name, w in (("WITHOUT depth reg", 0.0), ("WITH depth reg", 5.0)):
        a = A()
        a.dim, a.epochs, a.batch_size, a.n_neg = 16, 120, 32, 8
        a.lr, a.gamma, a.alpha, a.seed = 5e-2, 3.0, 1.0, 0
        a.depth_weight, a.depth_margin, a.root_pull = w, 0.3, 0.01
        a.val_frac, a.max_eval, a.log_every = 0.1, 100, 10 ** 9
        a.typed_negatives = 'hierarchy'
        a.max_rel_weight, a.hierarchy_relations = 5.0, ["PARENT_OF"]
        model, lp, _ = train(triples, hier_t, n_ent, 2, {"PARENT_OF": 0, "FLAT": 1},
                             ent_type_ids, a, np.arange(n_ent), depths)
        with torch.no_grad():
            z = model.ball_points(1.0).numpy()
        d1 = d1_radial_hierarchy(z, depths)
        res[name] = d1
        print(f"  {name:<20} D1 rho={d1['spearman']:+.4f} ({d1['verdict']})  "
              f"LP MRR={lp['mrr']:.3f}  radii by depth: " +
              " ".join(f"d{k}={v['mean_radius']:.3f}" for k, v in d1["by_depth"].items()))

    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    print()
    wi, wo = res["WITH depth reg"], res["WITHOUT depth reg"]
    rad = [v["mean_radius"] for v in wi["by_depth"].values()]
    return all([
        ok("depth regulariser raises D1 rho", wi["spearman"] > wo["spearman"]),
        ok(f"D1 rho reaches STRONG with the regulariser ({wi['spearman']:+.3f})",
           wi["spearman"] > 0.30),
        ok("mean radius increases monotonically with depth",
           all(rad[i] < rad[i + 1] for i in range(len(rad) - 1))),
        ok("embeddings stay strictly inside the ball", wi["norm_max"] < 1.0),
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kg-dir", default="./data/kg")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./data/kg")
    p.add_argument("--dataset", default="NYC")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-neg", type=int, default=128,
                   help="negatives per positive. 25 leaves ~78%% of steps with no competitive "
                        "negative at all (measured); 128 raises the expected count to ~1.1")
    p.add_argument("--typed-negatives", choices=["none", "hierarchy", "all"], default="all",
                   help="'hierarchy' was the original setting and lets POI->POI relations draw "
                        "46%% of their negatives from the wrong node type")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=6.0, help="NSSA margin")
    p.add_argument("--alpha", type=float, default=1.0, help="NSSA adversarial temperature")
    p.add_argument("--depth-weight", type=float, default=1.0,
                   help="weight on the hierarchy depth regulariser (0 reproduces the old run)")
    p.add_argument("--depth-margin", type=float, default=0.1,
                   help="minimum radius gap a child must keep from its parent")
    p.add_argument("--root-pull", type=float, default=0.01,
                   help="weak penalty on every entity radius; makes the depth margin binding "
                        "instead of satisfiable by inflating all radii together")
    p.add_argument("--max-rel-weight", type=float, default=5.0)
    p.add_argument("--curvature", type=float, default=1.0,
                   help="reference curvature for the saved ball points (must match CURVATURE_C)")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-eval", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="'cuda' or 'cpu'; default auto-picks cuda if torch.cuda.is_available()")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        sys.exit(0 if _self_check() else 1)
    a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run(a)


if __name__ == "__main__":
    main()
