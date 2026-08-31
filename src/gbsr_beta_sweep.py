"""
Is GBSR *capable* of pruning this social graph, at any bottleneck strength?

Context. On Foursquare (`LLMGPR_TRACK.md` §2) GBSR's mask collapsed to a constant and the
proposed explanation was graph sparsity: mean degree 3.2 against GBSR's own yelp at 38.6, so
"on a graph this sparse, there may be little redundancy to prune". Gowalla's friendship graph
is 2.8x denser (mean degree 8.8, 117,949 edges over 26,779 users) and the mask collapsed
identically -- mean 1.5000, std 2.2e-05, 15 distinct values over 117,949 edges, 99% of them at
exactly the sigmoid ceiling. **So the sparsity explanation is falsified and the cause must be
sought elsewhere.**

The mechanism this script tests. The gate is
    gate = sigmoid((logit + gumbel) / GUMBEL_TEMP) + edge_bias,  GUMBEL_TEMP = 0.2
so the MLP only has to push logits past ~3 for sigmoid to saturate at 1.0 and every edge to
receive the identical weight 1 + edge_bias = 1.5. BPR always prefers keeping edges (an edge is
information for the recommender), so saturation is the path of least resistance; the only force
opposing it is the HSIC information-bottleneck term, weighted by `beta` (default 5.0).

Hypothesis: the default beta is far too weak to oppose BPR, and the collapse is a
hyperparameter regime rather than a property of the data. Prediction if TRUE: raising beta
should produce a mask with real variance (and cost val NDCG). Prediction if FALSE: the mask
saturates at every beta, and GBSR is inapplicable to this graph -- which is the stronger form
of the no-op finding, since it then holds independent of tuning.

Reports, per beta, per epoch: mask std / range / distinct values / fraction at the ceiling,
plus val NDCG@20, so the mask trajectory and the accuracy cost are visible together.

Usage
-----
    python src/gbsr_beta_sweep.py --betas 5 100 2000 --epochs 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import denoise_social_gbsr as G


def load(data_dir, csv_dir, dataset):
    df = G.load_checkins(data_dir, dataset)
    train_df, val_df = df[df["orig_split"] == "train"], df[df["orig_split"] == "val"]
    train_pairs = list({(int(u), int(i)) for u, i in zip(train_df.user_id, train_df.poi_idx)})
    val_pos = {}
    for u, i in zip(val_df.user_id, val_df.poi_idx):
        val_pos.setdefault(int(u), set()).add(int(i))
    friends = pd.read_csv(os.path.join(csv_dir, f"friendship_old_{dataset}.csv"))
    social_edges = [(int(r.u1), int(r.u2)) for r in friends.itertuples(index=False)]
    n_users = max(int(df.user_id.max()) + 1,
                  max((max(u, v) for u, v in social_edges), default=0) + 1)
    n_items = int(df.poi_idx.max()) + 1
    return n_users, n_items, train_pairs, social_edges, val_pos


def run_one(beta, n_users, n_items, train_pairs, social_edges, val_pos, a, device):
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    pos_by_user = {}
    for u, i in train_pairs:
        pos_by_user.setdefault(u, set()).add(i)
    train_pos = {u: tuple(v) for u, v in pos_by_user.items()}
    pos_keys = np.sort(np.array([u * n_items + i for u, i in train_pairs], dtype=np.int64))
    users_t = torch.tensor([u for u, _ in train_pairs], dtype=torch.long)
    pos_t = torch.tensor([i for _, i in train_pairs], dtype=torch.long)

    adj, social_index = G.build_graph(n_users, n_items, train_pairs, social_edges, device)
    model = G.GBSR(n_users, n_items, adj, social_index, dim=a.dim, gcn_layer=a.gcn_layer,
                   edge_bias=a.edge_bias).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n = len(train_pairs)
    n_batches = max(1, n // a.batch_size)

    g0 = model.deterministic_gate()
    rows = [dict(beta=beta, epoch=0, std=float(g0.std()), rng=float(g0.max() - g0.min()),
                 uniq=int(len(np.unique(g0))),
                 frac_ceiling=float((g0 >= g0.max() - 1e-9).mean()), val_ndcg=None, auc=None)]
    print(f"\n=== beta={beta}")
    print(f"  epoch 0 (init)   mask std={g0.std():.3e} range={g0.max()-g0.min():.3e} "
          f"uniq={len(np.unique(g0)):,} at-ceiling={100*(g0 >= g0.max()-1e-9).mean():.1f}%")

    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot_auc = 0.0
        for b in range(n_batches):
            bi = perm[b * a.batch_size:(b + 1) * a.batch_size]
            if len(bi) == 0:
                continue
            u, p = users_t[bi], pos_t[bi]
            neg = G.sample_negatives(u, pos_keys, n_items, gen)
            u, p, neg = u.to(device), p.to(device), neg.to(device)
            out = model(u, p, neg, beta, a.sigma, a.l2_reg, hsic_cap=a.hsic_sample)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            tot_auc += float(out["auc"].detach())

        model.eval()
        with torch.no_grad():
            u_emb, i_emb = model.propagate(model.graph_learner())
        nd = G.ndcg_at_k(u_emb.cpu().numpy(), i_emb.cpu().numpy(), train_pos, val_pos,
                         a.topk, max_users=a.max_eval_users, seed=a.seed)
        gs = model.deterministic_gate()
        ceil = float((gs >= gs.max() - 1e-9).mean())
        rows.append(dict(beta=beta, epoch=epoch, std=float(gs.std()),
                         rng=float(gs.max() - gs.min()), uniq=int(len(np.unique(gs))),
                         frac_ceiling=ceil, val_ndcg=nd, auc=tot_auc / n_batches))
        print(f"  epoch {epoch:<3}         mask std={gs.std():.3e} range={gs.max()-gs.min():.3e} "
              f"uniq={len(np.unique(gs)):,} at-ceiling={100*ceil:.1f}%   "
              f"auc={tot_auc/n_batches:.4f} val_NDCG@{a.topk}={nd:.4f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data/gowalla")
    p.add_argument("--csv-dir", default="./data/gowalla")
    p.add_argument("--dataset", default="GOWALLA")
    p.add_argument("--out", default="./data/gowalla/gbsr_beta_sweep.json")
    p.add_argument("--betas", type=float, nargs="+", default=[5.0, 100.0, 2000.0])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--gcn-layer", type=int, default=3)
    p.add_argument("--edge-bias", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--l2-reg", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--hsic-sample", type=int, default=1024)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--max-eval-users", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = torch.device("cpu")   # sparse/scatter path; MPS has no sparse support
    n_users, n_items, train_pairs, social_edges, val_pos = load(a.data_dir, a.csv_dir, a.dataset)
    print(f"users={n_users:,} items={n_items:,} train={len(train_pairs):,} "
          f"social_edges={len(social_edges):,}  betas={a.betas}  epochs={a.epochs}")
    print(f"gate = sigmoid((logit+gumbel)/{G.GUMBEL_TEMP}) + {a.edge_bias}  "
          f"-> ceiling {1.0 + a.edge_bias}")

    all_rows = []
    for beta in a.betas:
        all_rows += run_one(beta, n_users, n_items, train_pairs, social_edges, val_pos, a, device)

    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 78)
    print("VERDICT -- does any bottleneck strength give the mask real variance?")
    print("=" * 78)
    print(f"{'beta':>8}{'final std':>12}{'final range':>13}{'uniq':>7}{'at-ceiling':>12}"
          f"{'val NDCG':>10}")
    for beta in a.betas:
        last = df[(df.beta == beta) & (df.epoch == df[df.beta == beta].epoch.max())].iloc[0]
        print(f"{beta:>8.0f}{last['std']:>12.2e}{last['rng']:>13.2e}{last['uniq']:>7,}"
              f"{100*last['frac_ceiling']:>11.1f}%{last['val_ndcg']:>10.4f}")
    # The verdict must be about the CONVERGED mask, which is what gets exported and what the
    # denoised arm would be built from. An earlier version of this check asked "any epoch with
    # std > 1e-3" and reported YES for every beta -- but that counts epochs 1-2, where the mask
    # still carries its initialisation noise (uniq ~113k at epoch 0, before any training). The
    # mask losing that variance IS the collapse, so early-epoch variance is the opposite of
    # evidence for discrimination.
    THR = 1e-3
    finals = {}
    for beta in a.betas:
        sub = df[df.beta == beta]
        finals[beta] = sub[sub.epoch == sub.epoch.max()].iloc[0]
    disc = [b for b, r in finals.items() if r["std"] > THR]
    print(f"\nconverged mask std > {THR:g} at any beta: {'YES ' + str(disc) if disc else 'NO'}")
    if not disc:
        print("  -> the mask collapses to a CONSTANT at every bottleneck strength tested;")
        print("     GBSR cannot prune this graph, and the collapse is not a tuning miss.")
        worst = min(finals.items(), key=lambda kv: kv[1]["val_ndcg"])
        print(f"     raising beta also destroys the recommender: "
              f"val NDCG {finals[a.betas[0]]['val_ndcg']:.4f} (beta={a.betas[0]:g}) "
              f"-> {worst[1]['val_ndcg']:.4f} (beta={worst[0]:g})")
    early = df[(df.epoch <= 2) & (df["std"] > THR)]
    print(f"  (mask std > {THR:g} in epochs <=2 for betas {sorted(set(early.beta))} -- "
          f"initialisation noise, washed out by training, NOT discrimination)")
    with open(a.out, "w") as f:
        json.dump(dict(config=vars(a), gumbel_temp=G.GUMBEL_TEMP, rows=all_rows), f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
