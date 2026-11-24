"""One harness for the POI-KG model comparison (RotH / RefH / AttH / MuRP), on one split.

Why this exists
---------------
Comparing the four hyperbolic models fairly needs one fixed protocol. `train_roth.py` can't
provide it on its own: it scores one random 5% split once, at the last epoch, with symmetric pairs
split across train/val and the depth regulariser fed the held-out hierarchy edges. This harness
fixes the protocol and keeps the models:

* split: ~90/5/5, stratified by relation, a triple and its reverse always land in the same part
  (IS_NEAR_TO is 100% symmetric, FOLLOWED_BY 28%), so nothing is answered by its own reverse;
* the depth regulariser only sees hierarchy pairs whose triple is in TRAIN;
* every epoch: validation MRR on a fixed sample + D1, so the real peak and D1's trajectory show;
* checkpoint = best validation MRR; test = every test triple, filtered, ties at their mean rank;
* per-relation test MRR, wall time per epoch, peak accelerator memory, parameter count.

Models (all share the entity table, NSSA loss, typed negatives, inverse-frequency relation
weights, cosine LR and the depth regulariser, so the only thing that varies is the score):

    hyperbolic  roth, refh, atth   (train_roth.py, learnable per-relation curvature)
                murp               (Balazevic et al. 2019: diagonal head scaling, tail translation)
                roth_c<value>      (RotH at a FIXED curvature: the c -> 0 ablation of the same
                                    code path, e.g. roth_c1, roth_c0.1, roth_c0.0001)
    euclidean   transe, rotate, complex, distmult   (available as flat references; NOT part of
                                                     the 2026-11-19 comparison, which is the four
                                                     hyperbolic models)

D1 = Spearman(taxonomy depth of the POI's category, radius of the POI). For a Euclidean model the
radius is the entity vector's norm; for the hyperbolic ones it is the radius at c = 1, which is a
monotone function of the same norm, so the rank correlation is directly comparable.

Usage
-----
    python src/kge_bench.py --kg-dir <dir> --data-dir <dir> --dataset LLMGPR \
        --model roth --epochs 30 --depth-weight 1.0 --depth-margin 0.1 --out runs/x.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_roth import (AttH, RefH, RotH, d1_radial_hierarchy, expmap0, geodesic_dist,  # noqa: E402
                        hyp_radius, mobius_add, nssa_loss, sample_negatives)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

class FixedCurvature:
    """Mixin: replace the learnable per-relation curvature with one constant c."""
    fixed_c = 1.0

    def curvature(self, r_id):
        return torch.full((*r_id.shape, 1), self.fixed_c, device=r_id.device)

    def radius(self):
        return hyp_radius(self.ball_points(1.0), 1.0)


def _hyperbolic_radius(self):
    return hyp_radius(self.ball_points(1.0), 1.0)


for _cls in (RotH, RefH, AttH):
    _cls.radius = _hyperbolic_radius
    _cls.family = "hyperbolic"
    _cls.gamma_override = None


class MuRP(RotH):
    """MuRP: d_c( exp0(R_r * log0(h)), t (+)_c r )^2 with a diagonal R_r.

    Entities live in the tangent space at the origin (like the other three), so log0(exp0(e_h))
    is e_h itself and the head transform is exp0(R_r * e_h). Biases and the per-relation
    curvature are inherited, which keeps it on the same footing as RotH/RefH/AttH here.
    """

    def _init_relation_params(self, n_rel, dim):
        self.R = nn.Embedding(n_rel, dim)

    def _reset_relation_params(self):
        nn.init.ones_(self.R.weight)

    def score(self, h, r, t):
        if t.dim() > h.dim():
            h = h.unsqueeze(-1).expand_as(t)
            r_exp = r.unsqueeze(-1).expand_as(t)
        else:
            r_exp = r
        c = self.curvature(r_exp)
        q = expmap0(self.R(r_exp) * self.ent(h), c)
        x_t = mobius_add(expmap0(self.ent(t), c), expmap0(self.rel(r_exp), c), c)
        d = geodesic_dist(q, x_t, c)
        return -(d ** 2) + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)


class EuclideanKGE(nn.Module):
    family = "euclidean"
    gamma_override = None          # None = use --gamma (distance models)

    def __init__(self, n_ent, n_rel, dim=64, init_scale=1e-3):
        super().__init__()
        self.dim = dim
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.normal_(self.ent.weight, std=init_scale)
        nn.init.normal_(self.rel.weight, std=init_scale)

    def radius(self):
        return self.ent.weight.norm(dim=-1)

    @staticmethod
    def _expand(h, r, t):
        if t.dim() > h.dim():
            return h.unsqueeze(-1).expand_as(t), r.unsqueeze(-1).expand_as(t)
        return h, r


class TransE(EuclideanKGE):
    """-||h + r - t||^2 + b_h + b_t (squared L2, the same form as the hyperbolic scores)."""

    def __init__(self, n_ent, n_rel, dim=64, init_scale=1e-3):
        super().__init__(n_ent, n_rel, dim, init_scale)
        self.bh = nn.Embedding(n_ent, 1)
        self.bt = nn.Embedding(n_ent, 1)
        nn.init.zeros_(self.bh.weight)
        nn.init.zeros_(self.bt.weight)

    def score(self, h, r, t):
        h, r = self._expand(h, r, t)
        d = self.ent(h) + self.rel(r) - self.ent(t)
        return -(d * d).sum(-1) + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)


class RotatE(TransE):
    """Complex rotation, then squared distance: -||h o e^{i theta_r} - t||^2 + biases.
    (The dim is split into dim/2 complex coordinates, so parameter count matches the others.)"""

    def __init__(self, n_ent, n_rel, dim=64, init_scale=1e-3):
        super().__init__(n_ent, n_rel, dim, init_scale)
        nn.init.uniform_(self.rel.weight, -math.pi, math.pi)   # only the first dim/2 are used

    def score(self, h, r, t):
        h, r = self._expand(h, r, t)
        k = self.dim // 2
        eh, et = self.ent(h), self.ent(t)
        th = self.rel(r)[..., :k]
        hr, hi, tr, ti = eh[..., :k], eh[..., k:], et[..., :k], et[..., k:]
        cr, si = torch.cos(th), torch.sin(th)
        dr, di = hr * cr - hi * si - tr, hr * si + hi * cr - ti
        return -(dr * dr + di * di).sum(-1) + self.bh(h).squeeze(-1) + self.bt(t).squeeze(-1)


class DistMult(EuclideanKGE):
    """<h, r, t>. Semantic matching, so NSSA runs at gamma 0 (plain logistic loss): with
    gamma 6 the positive term is satisfied at initialisation and only negatives would train."""
    gamma_override = 0.0

    def __init__(self, n_ent, n_rel, dim=64, init_scale=1e-1):
        super().__init__(n_ent, n_rel, dim, init_scale)

    def score(self, h, r, t):
        h, r = self._expand(h, r, t)
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)


class ComplEx(DistMult):
    """Re(<h, r, conj(t)>) over dim/2 complex coordinates."""

    def score(self, h, r, t):
        h, r = self._expand(h, r, t)
        k = self.dim // 2
        eh, er, et = self.ent(h), self.rel(r), self.ent(t)
        hr, hi, rr, ri, tr, ti = (eh[..., :k], eh[..., k:], er[..., :k], er[..., k:],
                                  et[..., :k], et[..., k:])
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)


BASE_MODELS = {"roth": RotH, "refh": RefH, "atth": AttH, "murp": MuRP,
               "transe": TransE, "rotate": RotatE, "complex": ComplEx, "distmult": DistMult}


def build_model(name, n_ent, n_rel, dim):
    if name.startswith("roth_c"):
        c = float(name[len("roth_c"):])
        cls = type(f"RotH_c{c:g}", (FixedCurvature, RotH), {"fixed_c": c, "family": "hyperbolic"})
        return cls(n_ent, n_rel, dim)
    if name not in BASE_MODELS:
        raise ValueError(f"unknown model {name!r}: {sorted(BASE_MODELS)} or roth_c<value>")
    return BASE_MODELS[name](n_ent, n_rel, dim)


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------

def shared_split(triples, seed, val_frac, test_frac):
    """Stratified by relation; a triple and its reverse (same relation) share a part."""
    g = np.random.default_rng(seed)
    tl = triples.tolist()
    key_of = {}
    for i, (h, r, t) in enumerate(tl):
        key_of.setdefault((min(h, t), r, max(h, t)), []).append(i)
    by_rel = {}
    for k, idx in key_of.items():
        by_rel.setdefault(k[1], []).append(idx)
    part = np.zeros(len(tl), dtype=np.int8)          # 0 train, 1 val, 2 test
    for r, groups in sorted(by_rel.items()):
        order = g.permutation(len(groups))
        n_val = int(round(len(groups) * val_frac))
        n_test = int(round(len(groups) * test_frac))
        for j, gi in enumerate(order):
            p = 1 if j < n_val else 2 if j < n_val + n_test else 0
            part[groups[gi]] = p
    part = torch.from_numpy(part)
    return triples[part == 0], triples[part == 1], triples[part == 2]


def depth_loss(model, hier, margin, rel_groups, root_pull):
    """train_roth.depth_loss, but on model.radius() so the Euclidean models get the same term."""
    if len(hier) == 0:
        return torch.zeros((), device=model.ent.weight.device)
    rad = model.radius()
    viol = torch.relu(margin - (rad[hier[:, 1]] - rad[hier[:, 0]]))
    loss = torch.stack([viol[m].mean() for m in rel_groups if m.any()]).mean()
    if root_pull > 0:
        loss = loss + root_pull * rad.pow(2).mean()
    return loss


@torch.no_grad()
def rank_queries(model, q, n_ent, filt, device, batch=16):
    """Filtered tail ranks for q [N,3]; ties take the mean rank."""
    all_ent = torch.arange(n_ent, device=device)
    ranks = np.empty(len(q))
    for s in range(0, len(q), batch):
        sub = q[s:s + batch]
        h, r, t = (sub[:, i].to(device) for i in range(3))
        sc = model.score(h, r, all_ent.unsqueeze(0).expand(len(sub), -1))
        for j, (hh, rr, tt) in enumerate(sub.tolist()):
            known = [x for x in filt.get((hh, rr), ()) if x != tt]
            if known:
                sc[j, known] = -float("inf")
        true = sc.gather(1, t.unsqueeze(1))
        gt = (sc > true).sum(1).float()
        eq = (sc == true).sum(1).float() - 1
        ranks[s:s + len(sub)] = (gt + eq / 2 + 1).cpu().numpy()
    return ranks


def summarise(ranks, rel_ids=None, name_of=None):
    out = dict(mrr=float(np.mean(1 / ranks)), hits1=float(np.mean(ranks <= 1)),
               hits3=float(np.mean(ranks <= 3)), hits10=float(np.mean(ranks <= 10)),
               mr=float(np.mean(ranks)), n=int(len(ranks)))
    if rel_ids is not None:
        out["per_relation"] = {
            name_of[r]: dict(mrr=float(np.mean(1 / ranks[rel_ids == r])),
                             hits1=float(np.mean(ranks[rel_ids == r] <= 1)),
                             hits10=float(np.mean(ranks[rel_ids == r] <= 10)),
                             n=int((rel_ids == r).sum()))
            for r in sorted(set(rel_ids.tolist()))}
    return out


def d1_of(model, poi_rows, depths):
    rad = model.radius().detach().cpu().numpy()[poi_rows]
    # d1_radial_hierarchy maps its input through hyp_radius(.,1); feed it points whose norm is a
    # monotone function of the model's radius so the Spearman is the model's own.
    z = np.tanh(rad / (1 + rad.max()))[:, None]
    return d1_radial_hierarchy(z, depths)["spearman"]


@torch.no_grad()
def poi_points(model, poi_rows):
    """POI embeddings as the downstream pipeline would consume them: ball points at c = 1 for
    the hyperbolic models (what EMB_FILE has always held), raw vectors for the Euclidean ones."""
    if model.family == "hyperbolic":
        return model.ball_points(1.0)[poi_rows]
    return model.ent.weight[poi_rows]


@torch.no_grad()
def knn_purity(model, poi_rows, labels, k=10, chunk=512):
    """The model-selection notebook's `purity_hyp@10` on the source space: the share of each POI's k nearest POIs that
    carry its category. Distances are the model's own geometry: Poincare distance at c = 1 for
    the hyperbolic models (ranked via ||x-y||^2 / (1-||y||^2), monotone in arcosh for a fixed
    query), Euclidean for the rest."""
    z = poi_points(model, poi_rows)
    n = len(z)
    if model.family == "hyperbolic":
        inv = 1.0 / (1 - z.pow(2).sum(-1)).clamp_min(1e-9)
    else:
        inv = torch.ones(n, device=z.device)
    lab = torch.as_tensor(labels, device=z.device)
    hits = 0.0
    for s in range(0, n, chunk):
        d = torch.cdist(z[s:s + chunk], z).pow(2) * inv[None, :]
        rows = torch.arange(len(d), device=z.device)
        d[rows, s + rows] = float("inf")
        nn_idx = d.topk(k, largest=False).indices
        hits += float((lab[nn_idx] == lab[s:s + chunk, None]).float().mean(1).sum())
    return hits / n


def purity_floor(labels):
    p = np.bincount(labels) / len(labels)
    return float((p * p).sum())


def peak_memory(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2 ** 20
    return float("nan")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run(a):
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else "cpu"))
    triples = torch.load(os.path.join(a.kg_dir, "kg_triples.pt"))
    hier_all = torch.load(os.path.join(a.kg_dir, "kg_hierarchy.pt"))
    ents = json.load(open(os.path.join(a.kg_dir, "kg_entities.json")))
    rels = json.load(open(os.path.join(a.kg_dir, "kg_relations.json")))
    rows_map = {int(k): int(v) for k, v in
                json.load(open(os.path.join(a.kg_dir, "kg_poi_rows.json"))).items()}
    rel_names = rels["relation_to_id"]
    name_of = {v: k for k, v in rel_names.items()}
    n_ent, n_rel = len(ents), len(rel_names)
    ent_type = [None] * n_ent
    for v in ents.values():
        ent_type[v["id"]] = v["type"]

    meta = pd.read_csv(os.path.join(a.data_dir, f"poi_metadata_{a.dataset}.csv"))
    col = next(c for c in ("category_path", "category") if c in meta.columns)
    poi_rows = np.array([rows_map[int(x)] for x in meta["poi_idx"]], dtype=int)
    depths = np.array([len([x for x in str(s).split(">") if x.strip()])
                       for s in meta[col].fillna("")], dtype=float)
    leaf = pd.factorize(meta[col].fillna("?").astype(str))[0]
    top = pd.factorize(meta[col].fillna("?").astype(str).str.split(">").str[0].str.strip())[0]

    train_t, val_t, test_t = shared_split(triples, a.split_seed, a.val_frac, a.test_frac)
    train_set = set(map(tuple, train_t.tolist()))
    hier = hier_all[torch.tensor([(c, r, p) in train_set for p, c, r in hier_all.tolist()])]
    filt = {}
    for h, r, t in triples.tolist():
        filt.setdefault((h, r), []).append(t)
    gv = torch.Generator().manual_seed(a.split_seed)
    val_q = val_t[torch.randperm(len(val_t), generator=gv)[:a.max_val]]
    test_q = test_t if a.max_test <= 0 else test_t[:a.max_test]

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)  # seeds CPU and CUDA generators
    gen = torch.Generator().manual_seed(a.seed)
    tail_types = {}
    for h, r, t in train_t.tolist():
        tail_types.setdefault(r, set()).add(ent_type[t])
    pools = {r: torch.tensor([i for i, ty in enumerate(ent_type) if ty in tys])
             for r, tys in tail_types.items()}
    cnt = torch.bincount(train_t[:, 1], minlength=n_rel).float().clamp_min(1)
    rel_w = (cnt.sum() / cnt)
    rel_w = (rel_w / rel_w.mean()).clamp(max=a.max_rel_weight)

    model = build_model(a.model, n_ent, n_rel, a.dim).to(device)
    gamma = a.gamma if model.gamma_override is None else model.gamma_override
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.02)
    hier = hier.to(device)
    rel_groups = [hier[:, 2] == r for r in sorted(set(hier[:, 2].tolist()))]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.model} [{model.family}] on {device}: {n_ent:,} entities, {n_rel} relations, "
          f"train/val/test {len(train_t):,}/{len(val_t):,}/{len(test_t):,}, "
          f"train-only hierarchy pairs {len(hier):,}, params {n_params:,}, gamma {gamma}, "
          f"depth_weight {a.depth_weight} margin {a.depth_margin}", flush=True)

    history, best, best_state, since_best = [], -1.0, None, 0
    n_batches = math.ceil(len(train_t) / a.batch_size)
    t_start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(train_t), generator=gen)
        tot_kge = tot_depth = 0.0
        for b in range(n_batches):
            bi = order[b * a.batch_size:(b + 1) * a.batch_size]
            h, r, t = train_t[bi, 0], train_t[bi, 1], train_t[bi, 2]
            neg = sample_negatives(t, r, a.n_neg, n_ent, pools, gen)
            w = rel_w[r].to(device)
            h, r, t, neg = (x.to(device) for x in (h, r, t, neg))
            l_kge = nssa_loss(model.score(h, r, t), model.score(h, r, neg), gamma, a.alpha, w)
            loss = l_kge
            if a.depth_weight > 0:
                l_depth = depth_loss(model, hier, a.depth_margin, rel_groups, a.root_pull)
                loss = loss + a.depth_weight * l_depth
                tot_depth += float(l_depth.detach())
            if a.l2 > 0:
                loss = loss + a.l2 * model.ent.weight[torch.cat([h, t])].pow(2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_kge += float(l_kge.detach())
        sched.step()
        train_s = time.time() - t0
        model.eval()
        row = dict(epoch=epoch, kge=tot_kge / n_batches, depth=tot_depth / n_batches,
                   train_s=round(train_s, 1))
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            vm = summarise(rank_queries(model, val_q, n_ent, filt, device))
            row.update(val_mrr=vm["mrr"], val_hits10=vm["hits10"], d1=d1_of(model, poi_rows, depths))
            if vm["mrr"] > best:
                best, since_best = vm["mrr"], 0
                best_state = copy.deepcopy(model.state_dict())
                row["best"] = True
            else:
                since_best += a.eval_every
        history.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in row.items()), flush=True)
        if a.patience and since_best >= a.patience:
            print(f"early stop: no validation gain for {since_best} epochs", flush=True)
            break

    final_d1 = d1_of(model, poi_rows, depths)
    model.load_state_dict(best_state)
    best_epoch = max(h["epoch"] for h in history if h.get("best"))
    test_ranks = rank_queries(model, test_q, n_ent, filt, device)
    res = dict(model=a.model, family=model.family, config=vars(a), n_params=n_params,
               best_epoch=best_epoch, best_val_mrr=best,
               test=summarise(test_ranks, test_q[:, 1].numpy(), name_of),
               d1_best=d1_of(model, poi_rows, depths), d1_final=final_d1,
               purity10_leaf=knn_purity(model, poi_rows, leaf),
               purity10_top=knn_purity(model, poi_rows, top),
               purity_floor_leaf=purity_floor(leaf), purity_floor_top=purity_floor(top),
               train_seconds=round(time.time() - t_start, 1),
               sec_per_epoch=round(float(np.mean([h["train_s"] for h in history])), 1),
               peak_mem_mb=round(peak_memory(device), 1), history=history,
               split_sizes=dict(train=len(train_t), val=len(val_t), test=len(test_t),
                                hierarchy_train=len(hier)))
    print(f"BEST epoch {best_epoch}: val MRR {best:.4f}  test MRR {res['test']['mrr']:.4f}  "
          f"D1 best {res['d1_best']:+.4f} final {final_d1:+.4f}  purity@10 leaf "
          f"{res['purity10_leaf']:.4f} (floor {res['purity_floor_leaf']:.4f}) top "
          f"{res['purity10_top']:.4f} (floor {res['purity_floor_top']:.4f})", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        if a.save_embs:
            # rows in poi_idx order (meta order), best checkpoint -- for Stage 2 alignment later
            np.save(a.out[:-5] + "_poi_embs.npy",
                    poi_points(model, poi_rows).detach().cpu().numpy().astype(np.float32))
            # Travel with the selected embedding when it is copied/renamed for stage 5.
            # Prevent a rebuilt KG being paired with embeddings trained on an older guard.
            from group_outing_guard import file_sha256
            emb_path = a.out[:-5] + "_poi_embs.npy"
            kg_manifest_path = os.path.join(a.kg_dir, "kg_manifest.json")
            provenance = dict(version="kge-embedding-v1", model=a.model, seed=a.seed,
                              embedding_sha256=file_sha256(emb_path),
                              kg_triples_sha256=file_sha256(os.path.join(a.kg_dir, "kg_triples.pt")),
                              kg_manifest_sha256=file_sha256(kg_manifest_path),
                              metadata_sha256=file_sha256(os.path.join(a.data_dir, f"poi_metadata_{a.dataset}.csv")),
                              outing_guard_policy=json.load(open(kg_manifest_path)).get("outing_guard_policy"))
            with open(emb_path + ".manifest.json", "w") as f:
                json.dump(provenance, f, indent=2)
    return res


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--kg-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dataset", default="LLMGPR")
    p.add_argument("--model", default="roth")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-neg", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=6.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--depth-weight", type=float, default=1.0)
    p.add_argument("--depth-margin", type=float, default=0.1)
    p.add_argument("--root-pull", type=float, default=0.01)
    p.add_argument("--max-rel-weight", type=float, default=5.0)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--max-val", type=int, default=2000)
    p.add_argument("--max-test", type=int, default=0, help="0 = every test triple")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--patience", type=int, default=0, help="0 = never stop early")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--save-embs", action="store_true",
                   help="also save the best checkpoint's POI embeddings next to --out")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())
