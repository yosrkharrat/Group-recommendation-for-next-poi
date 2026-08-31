"""
GBSR (Graph Bottlenecked Social Recommendation, KDD'24) -- denoise the real friendship graph
before it feeds `build_groups.py --group-source social` and `build_kg_lbsn.py`'s FRIEND_OF edges.

Why this exists (read LLMGPR_TRACK.md section 2 first)
--------------------------------------------------------
That doc settles three things about the upstream repo (`yimutianyang/KDD24-GBSR`):

1. GBSR has nothing to do with groups -- it is a social-graph denoiser. "Run GBSR on the social
   graph *first*, then induce groups from the cleaned graph. Group quality inherits the
   denoising -- a cleaner story than bolting a denoiser onto groups it was never built for."
2. Its own `torch_version/GBSR.py` never trains: `weights = weights.detach()` right before the
   masked graph is built severs every gradient path back to `linear_1`/`linear_2`, so the "mask
   learner" is a frozen random MLP plus Gumbel noise for the whole run. Confirmed by diffing it
   against `models/GBSR.py` (the TensorFlow original), which has no such detach -- gradients
   flow from the BPR + HSIC loss straight into the mask MLP there. **Fixed here**: no detach.
3. Social coverage on this project's friendship_old graph is thin (2,196/7,849 users, mean
   degree 3.2) versus GBSR's own yelp benchmark (18,862/19,539, mean degree 38.6). Pruning
   redundant edges is GBSR's whole premise; on a graph this sparse there may be little
   redundancy to prune. Run this script, look at the mask-weight separation it reports, and
   decide from that measurement rather than assuming denoising helps.

What this script does, concretely
----------------------------------
Ports the architecture from `models/GBSR.py` / `torch_version/GBSR.py` (LightGCN-S: one shared
graph over users+items+social edges, propagated with mean-pooled GCN layers) with two changes
from the upstream code:

  * the detach bug above is removed, so the mask MLP actually trains;
  * the mask is symmetric per undirected friendship edge. Upstream stores the social graph as
    two directed entries (u->v, v->u) and lets the graph_learner sample each independently,
    which is faithful to the paper but produces two possibly-different weights for one
    friendship. Every downstream consumer here (`build_groups.py`'s `nx.Graph`, `build_kg_lbsn`'s
    FRIEND_OF) treats friendship as undirected, so this script exports one weight per pair --
    the mean of the model's two directional (deterministic) mask values.

The "recommendation" signal GBSR needs (BPR over a user-item graph, used only to shape which
social edges survive) is this project's own train-split check-ins: (user_id, poi_idx). The
social graph to denoise is `friendship_old_<DS>.csv` -- the BEFORE-period snapshot that
`build_kg_lbsn.py` already treats as safe to use (see its own leakage guard). `friendship_new`
is never read here, on the same causality grounds.

Model selection uses held-out VAL check-ins (never test) -- upstream's own `run_GBSR.py` selects
on `testdata` directly (LLMGPR_TRACK.md trap #2); this script does not repeat that mistake.

Outputs (--out-dir)
--------------------
    social_edge_weights_<DS>.csv     every input friendship_old pair + its learned mask weight
    friendship_old_denoised_<DS>.csv same schema as friendship_old_<DS>.csv, edges pruned below
                                     --keep-threshold (default: this run's own median weight)
    gbsr_denoise_manifest.json       hyperparams, edge counts, mask-weight summary, val NDCG@k

Usage
-----
    python denoise_social_gbsr.py --data-dir ./data/llmgpr --csv-dir ./data/llmgpr \\
        --dataset LLMGPR --out-dir ./data/llmgpr
    python denoise_social_gbsr.py --self-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import load_checkins  # noqa: E402  (train/val/test loader, shared with groups)

EPS = 1e-12
GUMBEL_TEMP = 0.2       # matches upstream's hardcoded Concrete-distribution temperature


# --------------------------------------------------------------------------
# graph construction: one LightGCN-S adjacency over users + items + social edges
# --------------------------------------------------------------------------

def build_graph(n_users, n_items, train_pairs, social_edges, device):
    """R block = user-item (train interactions), S block = user-user (friendship_old).
    Returns a symmetric D^-1/2 A D^-1/2 sparse tensor plus the positions of each social edge's
    two directed entries (u,v) and (v,u) inside it, aligned with `social_edges`'s own order.

    `coalesce()` sorts the sparse tensor's indices (row-major) and can reorder or merge entries,
    so a position computed from *pre-coalesce* insertion order does not generally point at the
    right value afterwards -- entries must be located in the *post-coalesce* index array by an
    explicit (row, col) lookup instead.
    """
    n = n_users + n_items
    rows, cols = [], []

    for u, i in train_pairs:
        v = n_users + i
        rows += [u, v]
        cols += [v, u]

    for u, v in social_edges:
        rows += [u, v]
        cols += [v, u]

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    deg = np.zeros(n, dtype=np.float64)
    np.add.at(deg, rows, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, EPS))
    values = d_inv_sqrt[rows] * d_inv_sqrt[cols]

    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    val = torch.tensor(values, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

    coo = adj.indices().cpu().numpy()
    pos_of = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(coo[0], coo[1]))}
    fwd = np.array([pos_of[(u, v)] for u, v in social_edges], dtype=np.int64)
    bwd = np.array([pos_of[(v, u)] for u, v in social_edges], dtype=np.int64)
    social_index = np.concatenate([fwd, bwd])          # [:len(social_edges)] = fwd, rest = bwd
    return adj, torch.tensor(social_index, dtype=torch.long, device=device)


# --------------------------------------------------------------------------
# model (ported from models/GBSR.py + torch_version/GBSR.py, detach bug removed)
# --------------------------------------------------------------------------

def kernel_matrix(x, sigma):
    """exp((cosine_sim - 1) / sigma) on L2-normalised rows -- verbatim from the paper's code."""
    return torch.exp((x @ x.t() - 1.0) / sigma)


def hsic(kx, ky, m):
    kxy = kx @ ky
    h = torch.trace(kxy) / m ** 2 + kx.mean() * ky.mean() - 2 * kxy.mean() / m
    return h * (m / (m - 1)) ** 2


class GBSR(nn.Module):
    def __init__(self, n_users, n_items, adj, social_index, dim=64, gcn_layer=3, edge_bias=0.5,
                init_scale=0.01):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.gcn_layer = gcn_layer
        self.edge_bias = edge_bias
        self.adj = adj                                  # unmasked graph, fixed
        self.social_index = social_index
        self.social_u = adj.indices()[0][social_index]
        self.social_v = adj.indices()[1][social_index]
        self.social_weight = adj.values()[social_index]

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=init_scale)
        nn.init.normal_(self.item_emb.weight, std=init_scale)

        self.mask_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def graph_learner(self):
        """Gumbel-sigmoid mask over the social edges. NOT detached -- this is the upstream fix:
        torch_version/GBSR.py's `weights.detach()` here would sever every gradient into
        mask_mlp, leaving it at initialisation for the whole run (LLMGPR_TRACK.md trap #1)."""
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        cat = torch.cat([ego[self.social_u], ego[self.social_v]], dim=1)
        logit = self.mask_mlp(cat).view(-1)
        eps = torch.rand_like(logit).clamp(EPS, 1 - EPS)
        gumbel = torch.log(eps) - torch.log(1 - eps)
        gate = torch.sigmoid((logit + gumbel) / GUMBEL_TEMP) + self.edge_bias

        weights = torch.ones(self.adj.values().shape[0], device=logit.device)
        weights = weights.index_copy(0, self.social_index, gate)
        # returns the masked edge VALUES, not a sparse tensor: autograd through
        # torch.sparse.mm w.r.t. the values materialises a dense [n,n] intermediate
        # (SparseAddmmBackward0 -> full at::mm; ~25 GB at n=79,450, measured as THE
        # bottleneck by stack sampling). propagate() consumes values via gather/index_add,
        # which is the same D^-1/2 A D^-1/2 product with O(nnz*d) forward AND backward.
        return self.adj.values() * weights

    def deterministic_gate(self):
        """Expectation-trick mask for export: drop the Gumbel noise term (eps=0.5 -> logit
        contributes alone), so the reported weight is reproducible instead of one stochastic
        sample. Standard practice for reading out a Concrete/Gumbel-sigmoid relaxation."""
        with torch.no_grad():
            ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
            cat = torch.cat([ego[self.social_u], ego[self.social_v]], dim=1)
            logit = self.mask_mlp(cat).view(-1)
            return (torch.sigmoid(logit / GUMBEL_TEMP) + self.edge_bias).cpu().numpy()

    def propagate(self, values=None):
        """Mean-pooled LightGCN layers over the shared graph, from an edge-value vector
        (None = the unmasked graph). y = A_norm @ x is computed as a gather + index_add over
        the coalesced edge list -- identical arithmetic to torch.sparse.mm, but its backward
        w.r.t. `values` stays O(nnz*d) instead of materialising a dense [n,n] product."""
        v = self.adj.values() if values is None else values
        row, col = self.adj.indices()[0], self.adj.indices()[1]
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layers = [ego]
        for _ in range(self.gcn_layer):
            x = layers[-1]
            layers.append(torch.zeros_like(x).index_add_(0, row, x[col] * v.unsqueeze(1)))
        mean = torch.stack(layers, dim=1).mean(dim=1)
        return mean[:self.n_users], mean[self.n_users:]

    def forward(self, users, pos, neg, beta, sigma, l2_reg, hsic_cap=None):
        masked_values = self.graph_learner()
        u_old, i_old = self.propagate()
        u_new, i_new = self.propagate(masked_values)

        ue, pe, ne = u_new[users], i_new[pos], i_new[neg]
        pos_s = (ue * pe).sum(-1)
        neg_s = (ue * ne).sum(-1)
        bpr = -F.logsigmoid(pos_s - neg_s).mean()
        auc = (pos_s > neg_s).float().mean()

        ue0 = self.user_emb(users)
        pe0, ne0 = self.item_emb(pos), self.item_emb(neg)
        reg = 0.5 * (ue0.norm(2).pow(2) + pe0.norm(2).pow(2) + ne0.norm(2).pow(2)) / len(users)
        reg = reg * l2_reg

        uu = torch.unique(users)
        ii = torch.unique(pos)
        # HSIC's kernel matrices are [b,b]; at large batches they dominate the step (measured
        # 4.6s vs 1.3s per forward at batch 4096 vs 1024 on this graph). `hsic_cap` estimates
        # the same HSIC on a uniform subsample of the batch's unique users/items -- the
        # regulariser's target is unchanged, only the estimator's variance grows.
        if hsic_cap is not None:
            if len(uu) > hsic_cap:
                uu = uu[torch.randperm(len(uu), device=uu.device)[:hsic_cap]]
            if len(ii) > hsic_cap:
                ii = ii[torch.randperm(len(ii), device=ii.device)[:hsic_cap]]
        hx = F.normalize(u_old[uu], dim=1)
        hy = F.normalize(u_new[uu], dim=1)
        ib_u = hsic(kernel_matrix(hx, sigma), kernel_matrix(hy, sigma), len(uu))
        hx = F.normalize(i_old[ii], dim=1)
        hy = F.normalize(i_new[ii], dim=1)
        ib_i = hsic(kernel_matrix(hx, sigma), kernel_matrix(hy, sigma), len(ii))
        ib = (ib_u + ib_i) * beta

        return dict(loss=bpr + reg + ib, bpr=bpr, reg=reg, ib=ib, auc=auc)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def sample_negatives(users, pos_keys, n_items, gen):
    """Reject-sample until the negative is genuinely not a positive -- matches upstream
    rec_dataset.py's `negative_sampling` (unbounded retry), rather than silently accepting an
    occasional false negative after a fixed number of tries.

    Vectorized (the original per-element Python loop cost ~1 min/epoch at 469k pairs): all
    collisions are redrawn together each round, which is the same rejection sampler -- uniform
    over each user's non-positives -- consuming the generator in a different order.
    `pos_keys` is the SORTED int64 array of (user * n_items + item) positives."""
    u64 = users.to(torch.int64) * n_items
    neg = torch.randint(0, n_items, (len(users),), generator=gen)
    if len(pos_keys) == 0:
        return neg
    while True:
        loc = np.searchsorted(pos_keys, (u64 + neg).numpy())
        loc[loc >= len(pos_keys)] = len(pos_keys) - 1
        bad = torch.from_numpy(pos_keys[loc] == (u64 + neg).numpy())
        if not bad.any():
            return neg
        neg[bad] = torch.randint(0, n_items, (int(bad.sum()),), generator=gen)


@torch.no_grad()
def ndcg_at_k(user_emb, item_emb, train_pos, val_pos, k=20, max_users=2000, seed=0):
    """Brute-force filtered NDCG@k -- small enough graphs here that faiss buys nothing."""
    rng = np.random.RandomState(seed)
    users = list(val_pos.keys())
    if len(users) > max_users:
        users = list(rng.choice(users, max_users, replace=False))
    # float64 + finite floor: fp32 overflow here used to yield non-finite scores that silently
    # corrupted the val NDCG used for MODEL SELECTION (LLMGPR_FINETUNE_HANDOFF.md, known issue b).
    # Non-finite -> a floor score, so a diverged epoch ranks near 0 and is never chosen as best.
    scores_all = item_emb.astype(np.float64) @ user_emb.astype(np.float64).T   # [n_items, n_users_scored]
    bad = ~np.isfinite(scores_all)
    if bad.any():
        print(f"  WARNING: {int(bad.sum()):,} non-finite scores in ndcg_at_k -- flooring them")
        scores_all[bad] = -1e18
    ndcgs = []
    for j, u in enumerate(users):
        s = scores_all[:, j].copy()
        s[list(train_pos.get(u, ()))] = -1e9
        order = np.argsort(-s)[:k]
        targets = val_pos[u]
        gains = np.array([1.0 / np.log2(r + 2) for r in range(k)])
        dcg = sum(gains[r] for r, it in enumerate(order) if it in targets)
        idcg = gains[:min(len(targets), k)].sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def train(a, n_users, n_items, train_pairs, social_edges, val_pos, device):
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    pos_by_user = {}
    for u, i in train_pairs:
        pos_by_user.setdefault(u, set()).add(i)
    train_pos = {u: tuple(v) for u, v in pos_by_user.items()}
    pos_keys = np.sort(np.array([u * n_items + i for u, i in train_pairs], dtype=np.int64))
    users_t = torch.tensor([u for u, _ in train_pairs], dtype=torch.long)
    pos_t = torch.tensor([i for _, i in train_pairs], dtype=torch.long)

    adj, social_index = build_graph(n_users, n_items, train_pairs, social_edges, device)
    model = GBSR(n_users, n_items, adj, social_index, dim=a.dim, gcn_layer=a.gcn_layer,
                edge_bias=a.edge_bias).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    n = len(train_pairs)
    n_batches = max(1, n // a.batch_size)
    best_ndcg, best_state, since_improve = -1.0, None, 0
    print(f"users={n_users:,} items={n_items:,} train_interactions={n:,} "
          f"social_edges={len(social_edges):,}  device={device}")

    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot = dict(loss=0.0, bpr=0.0, reg=0.0, ib=0.0, auc=0.0)
        for b in range(n_batches):
            bi = perm[b * a.batch_size:(b + 1) * a.batch_size]
            if len(bi) == 0:
                continue
            u, p = users_t[bi], pos_t[bi]
            neg = sample_negatives(u, pos_keys, n_items, gen)
            u, p, neg = u.to(device), p.to(device), neg.to(device)
            out = model(u, p, neg, a.beta, a.sigma, a.l2_reg,
                        hsic_cap=getattr(a, "hsic_sample", None))
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            for k in tot:
                tot[k] += float(out[k].detach())

        if epoch % a.eval_every == 0 or epoch == a.epochs:
            model.eval()
            with torch.no_grad():
                u_emb, i_emb = model.propagate(model.graph_learner())
            val_ndcg = ndcg_at_k(u_emb.cpu().numpy(), i_emb.cpu().numpy(), train_pos, val_pos,
                                a.topk, seed=a.seed)
            # mask stats every eval, not just at export: a mask that is already saturated at
            # the best-val epoch cannot be discriminating edges, and without the trajectory a
            # flat FINAL mask is indistinguishable from one that never had variance at all.
            gs = model.deterministic_gate()
            print(f"epoch {epoch:>4}/{a.epochs}  loss={tot['loss']/n_batches:.4f} "
                  f"bpr={tot['bpr']/n_batches:.4f} ib={tot['ib']/n_batches:.4f} "
                  f"auc={tot['auc']/n_batches:.4f}  val_NDCG@{a.topk}={val_ndcg:.4f}  "
                  f"mask std={gs.std():.2e} range={gs.max()-gs.min():.2e} "
                  f"uniq={len(np.unique(gs)):,}")
            if val_ndcg >= best_ndcg:
                best_ndcg, since_improve = val_ndcg, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since_improve += 1
            if epoch > a.min_epochs and since_improve >= a.early_stop:
                print(f"early stop at epoch {epoch} (no val improvement for {a.early_stop} evals)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ndcg


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(a):
    df = load_checkins(a.data_dir, a.dataset)
    users = sorted(df["user_id"].unique().tolist())
    pois = sorted(df["poi_idx"].unique().tolist())
    n_users = max(users) + 1
    n_items = max(pois) + 1

    train_df = df[df["orig_split"] == "train"]
    val_df = df[df["orig_split"] == "val"]
    train_pairs = list({(int(u), int(i)) for u, i in zip(train_df.user_id, train_df.poi_idx)})
    val_pos = {}
    for u, i in zip(val_df.user_id, val_df.poi_idx):
        val_pos.setdefault(int(u), set()).add(int(i))

    fpath = os.path.join(a.csv_dir, f"friendship_old_{a.dataset}.csv")
    friends = pd.read_csv(fpath)
    social_edges = [(int(r.u1), int(r.u2)) for r in friends.itertuples(index=False)]
    n_users = max(n_users, max((max(u, v) for u, v in social_edges), default=0) + 1)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, best_ndcg = train(a, n_users, n_items, train_pairs, social_edges, val_pos, device)

    model.eval()
    gate = model.deterministic_gate()          # one value per directed (u,v)/(v,u) entry
    half = len(gate) // 2
    sym = (gate[:half] + gate[half:]) / 2.0     # (u,v) and (v,u) were appended in that order
    weights = pd.DataFrame(social_edges, columns=["u1", "u2"])
    weights["mask_weight"] = sym

    threshold = a.keep_threshold if a.keep_threshold is not None else float(np.median(sym))
    kept = weights[weights["mask_weight"] >= threshold]
    pruned = weights[weights["mask_weight"] < threshold]

    print(f"\nmask weight: mean={sym.mean():.4f} std={sym.std():.4f} "
          f"min={sym.min():.4f} max={sym.max():.4f}  threshold={threshold:.4f}")
    print(f"kept {len(kept):,}/{len(weights):,} edges "
          f"({100*len(kept)/max(len(weights),1):.1f}%), pruned {len(pruned):,}")

    os.makedirs(a.out_dir, exist_ok=True)
    weights.to_csv(os.path.join(a.out_dir, f"social_edge_weights_{a.dataset}.csv"), index=False)
    kept[["u1", "u2"]].to_csv(
        os.path.join(a.out_dir, f"friendship_old_denoised_{a.dataset}.csv"), index=False)

    manifest = dict(config=vars(a), n_users=n_users, n_items=n_items,
                    n_train_interactions=len(train_pairs), n_social_edges=len(social_edges),
                    best_val_ndcg=best_ndcg, threshold=threshold,
                    n_kept=int(len(kept)), n_pruned=int(len(pruned)),
                    mask_weight=dict(mean=float(sym.mean()), std=float(sym.std()),
                                     min=float(sym.min()), max=float(sym.max())))
    with open(os.path.join(a.out_dir, "gbsr_denoise_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote social_edge_weights_{a.dataset}.csv, "
          f"friendship_old_denoised_{a.dataset}.csv, gbsr_denoise_manifest.json -> {a.out_dir}")
    print("\nfeed the denoised file to:\n"
          f"  build_groups.py --group-source social --friend-old "
          f"{a.out_dir}/friendship_old_denoised_{a.dataset}.csv\n"
          "  (build_kg_lbsn.py's FRIEND_OF: swap its friendship_old_<DS>.csv for this file, or "
          "keep both and compare -- see LLMGPR_TRACK.md trap #3 before trusting either alone)")


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def _self_check():
    """Four taste groups (5 users each), each with its own 5-item pool; every user draws from
    their group's pool with an individually random (Dirichlet) weighting, so users within a
    group still have distinct embeddings rather than collapsing to one group prototype -- an
    earlier, simpler version of this fixture (two clusters, identical sampling for every member)
    let every same-group user become embedding-identical, which made the mask degenerate into a
    coarse "which group is on which side of the edge" rule instead of judging the tie itself.
    'signal' friendships connect two members of the SAME group (a real shared-taste tie);
    'noise' friendships connect members of DIFFERENT groups (spurious -- no shared preference).
    GBSR should learn higher mask weight for signal edges: propagating through a cross-group
    edge pulls a user's embedding toward a group whose items they never interact with, which
    hurts the BPR objective the mask is trained against."""
    print("SELF-CHECK: does GBSR separate shared-taste friendships from unrelated ones?")
    rng = np.random.RandomState(0)
    n_groups, group_size, items_per_group = 4, 10, 5
    n_users, n_items = n_groups * group_size, n_groups * items_per_group

    all_ids = list(range(n_users))
    rng.shuffle(all_ids)
    groups = {g: sorted(all_ids[g * group_size:(g + 1) * group_size]) for g in range(n_groups)}
    items_of = {g: list(range(g * items_per_group, (g + 1) * items_per_group))
               for g in range(n_groups)}

    train_pairs = []
    for g, us in groups.items():
        pool = items_of[g]
        for u in us:
            w = rng.dirichlet(np.ones(len(pool)) * 0.5)         # per-user idiosyncratic taste
            for i in rng.choice(pool, size=6, replace=True, p=w):
                train_pairs.append((u, int(i)))
    train_pairs = list(set(train_pairs))

    val_pos = {}
    for g, us in groups.items():
        for u in us:
            val_pos[u] = set(rng.choice(items_of[g], size=2, replace=False).tolist())

    signal_edges = []
    for us in groups.values():
        for _ in range(8):
            signal_edges.append(tuple(sorted(rng.choice(us, 2, replace=False))))
    signal_edges = list(set(signal_edges))
    noise_edges = []
    while len(noise_edges) < 30:
        g1, g2 = rng.choice(n_groups, 2, replace=False)
        pair = tuple(sorted((int(rng.choice(groups[g1])), int(rng.choice(groups[g2])))))
        if pair not in noise_edges:
            noise_edges.append(pair)
    social_edges = signal_edges + noise_edges

    class A:
        pass

    a = A()
    a.dim, a.gcn_layer, a.edge_bias = 16, 3, 0.5
    a.lr, a.beta, a.sigma, a.l2_reg = 5e-3, 0.5, 0.25, 1e-4
    a.batch_size, a.epochs, a.min_epochs, a.early_stop = 128, 500, 500, 8
    a.eval_every, a.topk, a.seed = 500, 5, 0

    device = torch.device("cpu")
    model, best_ndcg = train(a, n_users, n_items, train_pairs, social_edges, val_pos, device)

    gate = model.deterministic_gate()
    half = len(gate) // 2
    sym = (gate[:half] + gate[half:]) / 2.0
    edge_of = {e: w for e, w in zip(social_edges, sym)}
    signal_w = np.array([edge_of[e] for e in signal_edges])
    noise_w = np.array([edge_of[e] for e in noise_edges])

    threshold = float(np.median(sym))
    signal_kept = (signal_w >= threshold).mean()
    noise_kept = (noise_w >= threshold).mean()

    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    print(f"\n  mean mask weight: signal={signal_w.mean():.4f}  noise={noise_w.mean():.4f}")
    print(f"  fraction kept @ median threshold: signal={signal_kept:.2f}  noise={noise_kept:.2f}")
    return all([
        ok("val NDCG improved above 0 (BPR signal is learnable at all)", best_ndcg > 0.0),
        ok("signal edges get a higher mean mask weight than noise edges",
           signal_w.mean() > noise_w.mean()),
        ok("median threshold keeps signal edges more often than noise edges",
           signal_kept > noise_kept),
        _check_detach_bug_is_fixed(),
    ])


def _check_detach_bug_is_fixed():
    """Direct, deterministic check of LLMGPR_TRACK.md trap #1: torch_version/GBSR.py calls
    `weights.detach()` right before building the masked graph, which severs every gradient path
    back into the mask MLP -- so it never trains and stays at initialisation for the whole run.
    Verifies that mask_mlp.parameters() receive a nonzero gradient here (the fix), and that
    reproducing the upstream line makes it exactly zero (the bug), on the same tiny graph."""
    torch.manual_seed(0)
    n_users, n_items = 6, 4
    train_pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (5, 1)]
    social_edges = [(0, 1), (2, 3), (4, 5)]
    adj, social_index = build_graph(n_users, n_items, train_pairs, social_edges,
                                     torch.device("cpu"))
    model = GBSR(n_users, n_items, adj, social_index, dim=8, gcn_layer=2)
    users = torch.tensor([0, 1, 2])
    pos = torch.tensor([0, 1, 2])
    neg = torch.tensor([1, 2, 3])

    out = model(users, pos, neg, beta=1.0, sigma=0.25, l2_reg=1e-4)
    out["loss"].backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.mask_mlp.parameters())

    model.zero_grad()
    ego = torch.cat([model.user_emb.weight, model.item_emb.weight], dim=0)
    cat = torch.cat([ego[model.social_u], ego[model.social_v]], dim=1)
    logit = model.mask_mlp(cat).view(-1)
    gate = torch.sigmoid(logit / GUMBEL_TEMP) + model.edge_bias
    weights = torch.ones(model.adj.values().shape[0])
    weights = weights.index_copy(0, model.social_index, gate)
    weights = weights.detach()                         # upstream's exact bug, reproduced
    masked_values = model.adj.values() * weights
    u_new, i_new = model.propagate(masked_values)
    bpr = -F.logsigmoid((u_new[users] * i_new[pos]).sum(-1) -
                        (u_new[users] * i_new[neg]).sum(-1)).mean()
    bpr.backward()
    grad_bug = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.mask_mlp.parameters())

    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    return all([
        ok("no detach(): mask MLP receives a nonzero gradient from the loss", grad_ok),
        ok("with upstream's detach() reproduced: mask MLP gradient is exactly zero", not grad_bug),
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data", help="has train/val/test_<DS>.csv")
    p.add_argument("--csv-dir", default=None,
                   help="has friendship_old_<DS>.csv (default: same as --data-dir)")
    p.add_argument("--out-dir", default="./data")
    p.add_argument("--dataset", default="NYC")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--gcn-layer", type=int, default=3)
    p.add_argument("--edge-bias", type=float, default=0.5,
                   help="prior mask offset -- upstream default, keeps edges alive at init")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=5.0, help="HSIC (info-bottleneck) weight")
    p.add_argument("--sigma", type=float, default=0.25, help="HSIC kernel bandwidth")
    p.add_argument("--l2-reg", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hsic-sample", type=int, default=None,
                   help="cap the HSIC estimator to this many unique users/items per batch "
                        "(the [b,b] kernels dominate large-batch steps; subsampling keeps the "
                        "estimator's target and adds variance)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--min-epochs", type=int, default=30,
                   help="no early stop before this many epochs, mirrors upstream's epoch>50 gate")
    p.add_argument("--early-stop", type=int, default=10,
                   help="stop after this many evals with no val NDCG improvement")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--keep-threshold", type=float, default=None,
                   help="mask weight cutoff for the denoised export; default = this run's own "
                        "median weight (self-calibrating, no magic constant)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    a.csv_dir = a.csv_dir or a.data_dir

    if a.self_check:
        sys.exit(0 if _self_check() else 1)
    run(a)


if __name__ == "__main__":
    main()
