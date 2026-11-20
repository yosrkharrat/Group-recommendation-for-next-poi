"""
Stage 2 of the supervisor's KGE model selection: does each model's structure survive the alignment
into the LLM's embedding space?

Why
---
`model_selection_hyperbolic_kge.ipynb` (the supervisor's) chooses the hyperbolic KGE in two stages:

    Stage 1  intrinsic link prediction (MRR, Hits@k) on held-out triples;
    Stage 2  structure preservation AFTER the alignment adapter, i.e. on the vectors the LLM
             actually receives: overlap@10 of hyperbolic vs aligned neighbours, and category
             purity@10 of the aligned neighbours.

Her rule: *"structure preservation on the post-alignment artifact is authoritative over intrinsic
MRR"* -> carry forward the max `purity_aligned@10`. The September selection
(`compare_kge_models.py`) could only run Stage 1, on the belief that Stage 2 needs the LLM. It
does not. Stage 5's alignment (section 6b, `run_geometry_alignment`) never runs the LLM. It needs:

- the hyperbolic POI embeddings;
- the POI-POI triples;
- the LLM's hidden size (2048);
- the per-dimension mean / std of the LLM's ORIGINAL input-embedding rows, which
  `match_llm_embedding_distribution` matches to.

Those two vectors are shipped as an .npz (`--llm-stats`), read once from the LLaDA-MoE safetensors
(`model.embed_tokens.weight`, rows [:157153] = the vocabulary before the POI tokens; bf16 -> fp32
exact; std with ddof=1 like torch.std). So Stage 2 runs on any GPU, or slowly on CPU.

Checked against stage 5's own log (Kaggle smoke run, old RefH embeddings, 20 alignment epochs):
D1 +0.3203 and purity_hyp@10 0.3495 reproduced exactly; overlap@10 0.7944 vs 0.7955 and
purity_aligned@10 0.2968 vs 0.2970 (the adapter's random init differs, nothing else).

This script reproduces stage 5 cells 6b exactly:
- the same ManifoldAwareAdapter / RadiusPredictor / RelationScorer / GeometryPreservingLoss
  (src/alignment.py);
- the same ALIGN_* values (notebook cell 9);
- the same notebook log map;
- the same verification (`_geo_topk` on the ball, `_euc_topk` on the aligned rows, overlap@10,
  purity@10 with the full category path as the label, an empirical random floor).

Each variant is aligned from the same seed, so the variants differ only by their embeddings.

Decision rule written to the output (supervisor's rule + stage 5's hard gate)
-----------------------------------------------------------------------------
1. D1 gate, as in stage 5 (it aborts otherwise): Spearman(depth, radius) > 0.30 and mean radius
   non-decreasing with depth (adjacent depths within 1% count as tied).
2. Among the variants that pass: max purity_aligned@10 (Stage 2, authoritative).
3. Stage 1 (max test MRR, from kge_bench.py's JSON) is reported next to it, with a flag when the
   two stages disagree.

With several seeds per model, metrics are averaged over seeds before the decision.

Usage
-----
    python src/kge_alignment_eval.py --runs out/*_s4?.json --kg-dir kg_poi \
        --meta data/poi_metadata_LLMGPR.csv --llm-stats llada_moe_embed_stats.npz --out out/stage2
`--runs` are kge_bench.py outputs run with --save-embs (each has <run>_poi_embs.npy beside it).
Models split across machines: run Stage 2 on each, then
    python src/kge_alignment_eval.py --merge a/stage2_selection.csv b/stage2_selection.csv --out merged
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment import (GeometryPreservingLoss, ManifoldAwareAdapter, RadiusPredictor,  # noqa: E402
                       RelationScorer)

# stage 5, notebook cell 9 (do not change one without the other)
ALIGN = dict(HIDDEN_DIM=1024, NUM_LAYERS=4, DROPOUT=0.1, EPOCHS=500, BATCH_SIZE=512, TRIPLE_SAMPLE=256,
             LR=1e-3, WEIGHT_DECAY=1e-4, W_RANKING=1.0, W_RADIUS=0.2, W_TRIPLE=1.0, W_SPECTRAL=0.01,
             SPECTRAL_TOPM=32, SPECTRAL_START_EPOCH=100)
CURVATURE_C = 1.0
D_MODEL = 2048          # LLaDA-MoE-7B-A1B hidden size
D1_GATE, D1_TIE_TOL = 0.30, 0.01


def logmap0(x: torch.Tensor, c: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
    """Stage 5's own log map (notebook cell 17), byte-for-byte, so W_POI's tangent vectors match."""
    sqrt_c = c ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1.0 - 1e-5) / sqrt_c
    x = torch.where(norm > max_norm, x / norm * max_norm, x)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.atanh((sqrt_c * norm).clamp(max=1 - 1e-7)) * x / (sqrt_c * norm)


def run_geometry_alignment(hyp_embs_np, triples, num_relations, d_model, device, epochs, log=print):
    """Notebook cell 24, with the ALIGN_* globals passed in from ALIGN."""
    A = ALIGN
    hyp_dim, n_pois = hyp_embs_np.shape[1], hyp_embs_np.shape[0]
    hyp_raw = torch.tensor(hyp_embs_np, dtype=torch.float32, device=device)
    tangent = logmap0(hyp_raw, c=CURVATURE_C)
    triples = triples.to(device)
    adapter = ManifoldAwareAdapter(hyp_dim, d_model, hidden_dim=A["HIDDEN_DIM"],
                                   num_layers=A["NUM_LAYERS"], dropout=A["DROPOUT"]).to(device)
    radius_predictor = RadiusPredictor(d_model).to(device)
    relation_scorer = RelationScorer(num_relations, d_model).to(device)
    geom_loss_fn = GeometryPreservingLoss(
        radius_predictor=radius_predictor, relation_scorer=relation_scorer,
        weight_ranking=A["W_RANKING"], weight_radius=A["W_RADIUS"], weight_triple=A["W_TRIPLE"],
        weight_spectral=A["W_SPECTRAL"], top_m_spectral=A["SPECTRAL_TOPM"]).to(device)
    trainable = (list(adapter.parameters()) + list(radius_predictor.parameters())
                 + list(relation_scorer.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=A["LR"], weight_decay=A["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loader = DataLoader(TensorDataset(hyp_raw, tangent, torch.arange(n_pois, device=device)),
                        batch_size=A["BATCH_SIZE"], shuffle=True)
    adapter.train(); radius_predictor.train(); relation_scorer.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        use_spectral = epoch > A["SPECTRAL_START_EPOCH"]
        for hyp_b, tan_b, _idx in loader:
            aligned_b = adapter(tan_b)
            _, loss_dict = geom_loss_fn(hyp_emb=hyp_b, aligned_emb=aligned_b,
                                        curvature=CURVATURE_C, use_spectral=use_spectral)
            triple_idx = torch.randint(0, len(triples), (A["TRIPLE_SAMPLE"],), device=device)
            ts = triples[triple_idx]
            h_aligned = adapter(logmap0(hyp_raw[ts[:, 0]], c=CURVATURE_C))
            t_aligned = adapter(logmap0(hyp_raw[ts[:, 2]], c=CURVATURE_C))
            neg_t_idx = torch.randint(0, n_pois, (A["TRIPLE_SAMPLE"],), device=device)
            neg_t_aligned = adapter(logmap0(hyp_raw[neg_t_idx], c=CURVATURE_C))
            pos_score = relation_scorer.score(h_aligned, ts[:, 1], t_aligned)
            neg_score = relation_scorer.score(h_aligned, ts[:, 1], neg_t_aligned)
            triple_loss = F.relu(1.0 - pos_score + neg_score).mean()
            spectral_term = loss_dict.get("spectral", torch.tensor(0.0, device=device))
            total_loss = (A["W_RANKING"] * loss_dict["ranking"] + A["W_RADIUS"] * loss_dict["radius"]
                          + A["W_SPECTRAL"] * spectral_term + A["W_TRIPLE"] * triple_loss)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()
        scheduler.step()
        if epoch % 50 == 0 or epoch == epochs:
            log(f"    [align] epoch {epoch}/{epochs}  loss={epoch_loss / len(loader):.4f}")
    adapter.eval()
    with torch.no_grad():
        raw_aligned = adapter(tangent)
    return raw_aligned


def match_to_llm_stats(projected, target_mean, target_std):
    """`alignment.match_llm_embedding_distribution` with the target statistics precomputed."""
    tm = target_mean.view(1, -1)
    ts = target_std.view(1, -1).clamp_min(1e-5)
    pm = projected.mean(dim=0, keepdim=True)
    ps = projected.std(dim=0, keepdim=True).clamp_min(1e-5)
    return (projected - pm) / ps * ts + tm


def geo_topk(Z, c_val, k=10, chunk=512):
    """Notebook cell 25's `_geo_topk` (arccosh form, chunked)."""
    z2_all = (Z * Z).sum(-1)
    out = []
    for s in range(0, Z.shape[0], chunk):
        q = Z[s:s + chunk]
        q2 = (q * q).sum(-1, keepdim=True)
        sq = (q2 + z2_all.unsqueeze(0) - 2.0 * (q @ Z.t())).clamp_min(0.0)
        den = (1.0 - c_val * q2).clamp_min(1e-15) * (1.0 - c_val * z2_all.unsqueeze(0)).clamp_min(1e-15)
        D = torch.arccosh((1.0 + 2.0 * c_val * sq / den).clamp_min(1.0 + 1e-7)) / c_val ** 0.5
        D[torch.arange(q.shape[0], device=Z.device), torch.arange(s, s + q.shape[0], device=Z.device)] = float("inf")
        out.append(D.topk(k, largest=False).indices)
    return torch.cat(out, 0)


def euc_topk(Z, k=10, chunk=512):
    """Notebook cell 25's `_euc_topk`."""
    out = []
    for s in range(0, Z.shape[0], chunk):
        D = torch.cdist(Z[s:s + chunk], Z, p=2)
        D[torch.arange(D.shape[0], device=Z.device), torch.arange(s, s + D.shape[0], device=Z.device)] = float("inf")
        out.append(D.topk(k, largest=False).indices)
    return torch.cat(out, 0)


def purity_at(nbr, cats):
    nbr = nbr.cpu().numpy()
    return float((cats[nbr] == cats[:, None]).mean())


def d1_gate(z, depths):
    """Stage 5's D1: Spearman with average ranks + monotone mean radius (1% tie tolerance)."""
    r = pd.Series(np.linalg.norm(z, axis=1))
    d = pd.Series(depths)
    rho = float(np.corrcoef(d.rank().to_numpy(), r.rank().to_numpy())[0, 1])
    means = r.groupby(d).mean().sort_index().to_numpy()
    mono = bool(all(b >= a * (1 - D1_TIE_TOL) for a, b in zip(means[:-1], means[1:])))
    return rho, mono, [round(float(x), 4) for x in means]


def main():
    ap = argparse.ArgumentParser(description="Stage 2 of the supervisor's KGE selection")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge several Stage-2 CSVs (e.g. one per Kaggle kernel) and re-decide; "
                         "needs only --out")
    ap.add_argument("--runs", nargs="+", help="kge_bench.py JSONs (run with --save-embs)")
    ap.add_argument("--kg-dir", help="holds poi_poi_triples_<DS>.pt + poi_relation_vocab_<DS>.json")
    ap.add_argument("--dataset", default="LLMGPR")
    ap.add_argument("--meta")
    ap.add_argument("--llm-stats")
    ap.add_argument("--out", required=True, help="output prefix: <out>.json and <out>.csv")
    ap.add_argument("--epochs", type=int, default=ALIGN["EPOCHS"], help="alignment epochs (stage 5: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.merge:
        df = pd.concat([pd.read_csv(p) for p in a.merge], ignore_index=True)
        dup = df["run"].duplicated(keep=False)
        assert not dup.any(), f"the same run appears twice: {sorted(df.loc[dup, 'run'].unique())}"
        df.to_csv(a.out + ".csv", index=False)
        decide(df, a.out, extra=dict(merged_from=a.merge))
        return
    missing = [k for k in ("runs", "kg_dir", "meta", "llm_stats") if getattr(a, k) is None]
    if missing:
        ap.error("missing " + ", ".join("--" + m.replace("_", "-") for m in missing))
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_all = time.time()

    meta = pd.read_csv(a.meta).sort_values("poi_idx")
    cats = meta["category"].fillna("Venue").astype(str).to_numpy()
    depths = np.array([len([x for x in c.split(">") if x.strip()]) for c in cats])
    triples = torch.load(os.path.join(a.kg_dir, f"poi_poi_triples_{a.dataset}.pt"))
    with open(os.path.join(a.kg_dir, f"poi_relation_vocab_{a.dataset}.json")) as f:
        n_rel = len(json.load(f)["relation_to_id"])
    st = np.load(a.llm_stats)
    t_mean = torch.tensor(st["mean"], dtype=torch.float32, device=device)
    t_std = torch.tensor(st["std"], dtype=torch.float32, device=device)
    assert t_mean.numel() == D_MODEL
    g = torch.Generator().manual_seed(a.seed)
    random_top10 = torch.randint(0, len(meta), (len(meta), 10), generator=g)
    purity_random = purity_at(random_top10, cats)

    runs = sorted({p for pat in a.runs for p in glob.glob(pat)})
    rows = []
    for path in runs:
        res = json.load(open(path))
        cfg = res.get("config", {})
        emb_path = path[:-5] + "_poi_embs.npy"
        if not os.path.exists(emb_path):
            print(f"skip {path}: no {os.path.basename(emb_path)} (run kge_bench.py with --save-embs)")
            continue
        z = np.load(emb_path, allow_pickle=False)
        assert z.shape[0] == len(meta), f"{emb_path}: {z.shape[0]} rows vs {len(meta)} POIs"
        name = os.path.basename(path)[:-5]
        print(f"=== {name}  ({cfg.get('model')}, seed {cfg.get('seed')})", flush=True)
        rho, mono, means = d1_gate(z, depths)
        torch.manual_seed(a.seed); np.random.seed(a.seed)               # same start for every variant
        t0 = time.time()
        raw = run_geometry_alignment(z, triples, n_rel, D_MODEL, device, a.epochs)
        aligned = match_to_llm_stats(raw.float(), t_mean, t_std)
        with torch.no_grad():
            hyp_top10 = geo_topk(torch.tensor(z, dtype=torch.float32, device=device), CURVATURE_C)
            al_top10 = euc_topk(aligned)
        overlap = float(np.mean([len(set(h) & set(al)) / 10 for h, al in
                                 zip(hyp_top10.cpu().tolist(), al_top10.cpu().tolist())]))
        row = dict(run=name, model=cfg.get("model"), seed=cfg.get("seed"),
                   test_mrr=res.get("test", {}).get("mrr", res.get("test_mrr")),
                   best_epoch=res.get("best_epoch"),
                   d1_rho=round(rho, 4), d1_monotone=mono, d1_pass=bool(rho > D1_GATE and mono),
                   radius_by_depth=means,
                   overlap_at10=round(overlap, 4),
                   purity_hyp_at10=round(purity_at(hyp_top10, cats), 4),
                   purity_aligned_at10=round(purity_at(al_top10, cats), 4),
                   purity_random=round(purity_random, 4), align_sec=round(time.time() - t0))
        print(f"    D1 {rho:+.4f} monotone={mono}  overlap@10 {row['overlap_at10']}  purity hyp "
              f"{row['purity_hyp_at10']} aligned {row['purity_aligned_at10']} (random {row['purity_random']})"
              f"  test MRR {row['test_mrr']}  ({row['align_sec']}s)", flush=True)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(a.out + ".csv", index=False)
    decide(df, a.out, extra=dict(align_config=ALIGN, llm_stats=a.llm_stats, device=device,
                                 seconds=round(time.time() - t_all)))


def decide(df, out_prefix, extra=None):
    """Seed means per model, the D1 gate, then the supervisor's rule. Shared by the normal run and
    --merge (several Stage-2 CSVs, e.g. models split across Kaggle kernels)."""
    num = ["test_mrr", "d1_rho", "overlap_at10", "purity_hyp_at10", "purity_aligned_at10"]
    agg = df.groupby("model").agg({**{c: "mean" for c in num}, "d1_pass": "all", "seed": "count"})
    agg = agg.rename(columns={"seed": "n_seeds"}).round(4)
    ok = agg[agg["d1_pass"]]
    stage2 = ok["purity_aligned_at10"].idxmax() if len(ok) else None
    stage1 = agg["test_mrr"].astype(float).idxmax() if agg["test_mrr"].notna().any() else None
    decision = dict(
        rule=("supervisor's model_selection_hyperbolic_kge.ipynb: Stage 2 (max purity_aligned@10 "
              "after the alignment adapter) is authoritative over Stage 1 (max test MRR); restricted "
              "to variants passing stage 5's D1 gate (rho > 0.30, monotone, 1% tie tolerance)"),
        stage1_winner=stage1, stage2_winner=stage2, carry_forward=stage2,
        stages_agree=None if stage1 is None else (stage1 == stage2),
        failed_d1_gate=sorted(agg.index[~agg["d1_pass"]].tolist()))
    out = dict(decision=decision, per_model=agg.reset_index().to_dict(orient="records"),
               per_run=df.to_dict(orient="records"), **(extra or {}))
    with open(out_prefix + ".json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("\n" + agg.to_string())
    print(f"\nStage 1 (max test MRR): {stage1}   Stage 2 (max purity_aligned@10, D1-gated): {stage2}"
          f"   -> carry forward: {stage2}" + ("" if stage1 in (None, stage2) else "   (stages DISAGREE)"))
    print(f"wrote {out_prefix}.json")
    return decision

if __name__ == "__main__":
    main()
