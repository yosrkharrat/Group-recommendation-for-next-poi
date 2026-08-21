# HyGro-POI — the LBSN2Vec++ social-network track

Group next-POI recommendation with a diffusion LLM (LLaDA-MoE) and hyperbolic (RotH)
embeddings, on the Foursquare **global-scale check-in dataset with user social networks**
(Yang et al., §5 of the dataset page; the dataset of LBSN2Vec++, TKDE 2020), NYC subset.

What this dataset adds over the earlier TSMC2014 track: a **real friendship graph**. The old
pipeline had to infer user–user ties (affinity z-sum, AUC 0.72); here `FRIEND_OF` edges are
observed. This branch contains only this track — the TSMC2014 pipeline and its history live
on `main`.

**→ Running the fine-tune? The adapted notebook is ready:
[`notebooks/stage5_lbsn_finetune.ipynb`](notebooks/stage5_lbsn_finetune.ipynb).**

```bash
git clone -b lbsn-handoff https://github.com/yosrkharrat/Group-recommendation-for-next-poi.git
cd Group-recommendation-for-next-poi
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
export HF_TOKEN=hf_...                  # free READ token; or a .env beside the notebook
jupyter lab notebooks/stage5_lbsn_finetune.ipynb
```

All data is committed on this branch — nothing else to download besides the model (which the
notebook fetches itself). Do one `RUN_PROFILE = "smoke"` pass (~15 min, any CUDA GPU) before
the real `"full"` run (≥40 GB GPU; the notebook refuses to start `"full"` on less and prints a
measured ETA from step 25). The Acc@t-guided objective is ON (`USE_ACC_AT_T_OBJECTIVE=True`) —
the training log shows `oracle_override=… acc@1/5/10=…` per log step as visible proof.
Background, leakage rules and design notes: [`docs/LBSN_HANDOFF.md`](docs/LBSN_HANDOFF.md).

---

## Where this track stands

| # | Stage | Status | Produces |
|---|---|---|---|
| 0 | Raw dump → house CSVs | ✅ done | `data/lbsn/` — 159,304 check-ins / 1,665 users / 6,103 POIs |
| 1 | Group construction | ✅ done | `data/lbsn/groups/` — 25,847 / 5,782 / 12,204 examples |
| 2 | KG (POI + group + **FRIEND_OF**) | ✅ done | `data/kg_lbsn/kg_triples.pt` — 12,709 entities, 244,080 triples, 12 relations |
| 3 | RotH + depth regulariser (120 epochs) | ✅ done | `data/kg_lbsn/poi_hyperbolic_embs_LBSN_NYC.npy` — **D1 ρ = +0.7096 STRONG** |
| 4 | POI-POI triples for the alignment adapter | ✅ done | `data/kg_lbsn/poi_poi_triples_LBSN_NYC.pt` — 132,394 triples |
| 5 | **LLaDA fine-tune** notebook (Acc@t objective ON, opt-in social prompt) | ✅ built, CPU-verified | `notebooks/stage5_lbsn_finetune.ipynb` — **GPU run is the next step** |

### The headline result

The Poincaré ball has a radial hierarchy on this dataset — mean radius by taxonomy depth:

```
D1 Spearman(depth, radius) = +0.7096   STRONG   (unregularised baseline on TSMC: +0.02 ABSENT)
depth 1: 0.736   depth 2: 0.747   depth 3: 0.974   depth 4: 1.321     fully monotone
```

ρ is tie-capped below the TSMC run's +0.85 because 69% of POIs sit at depth 2 — structural,
not undertraining. The pass bar for this dataset is **monotone radii + STRONG**, which this
clears. Full provenance (config, per-relation MRR, D1) in `data/kg_lbsn/roth_results.json`.

> **Missing control:** the `--depth-weight 0` run for this dataset. Unattended via
> `notebooks/roth_lbsn_kaggle.ipynb` with `RUN_CONTROL = True`.

### The leakage rule (do not soften it)

Only `friendship_old` (the before-period snapshot, 1,506 edges) may appear in the KG, in
training data, or in prompts. `friendship_new_only_LBSN_NYC.csv` (1,138 pairs) is the
friendship-prediction **eval set** — friendships formed during/after the check-in period,
partly an *effect* of the co-visits the model trains on. Its absence from the KG is asserted
at build time; keep it equally absent from fine-tune inputs.

---

## Layout

```
src/                        flat on purpose (scripts import each other directly)
  prepare_lbsn_csvs.py        stage 0  raw dump -> house CSVs      (10 self-checks)
  build_groups.py             stage 1  group construction          (11 self-checks)
  affinity.py                 stage 1  multi-signal user affinity
  build_kg.py                 stage 2  shared KG layer builders    (12 self-checks)
  build_kg_lbsn.py            stage 2  this track's KG + FRIEND_OF (15 self-checks)
  train_roth.py               stage 3  RotH + depth regulariser    (4 self-checks)
  build_poi_poi_triples.py    stage 4  triples for the adapter     (9 self-checks)
  alignment.py                stage 5  ManifoldAwareAdapter (INPUT side)
  objective.py                stage 5  Acc@t objective + HyperbolicScorer
  hyperbolic_group.py                  gyromidpoint, GeometricAttention
  phase0_diagnostics.py                standalone D1 / D2 / D3

notebooks/
  stage5_lbsn_finetune.ipynb    stage 5, the LLaDA fine-tune for THIS dataset — RUN THIS.
                                Acc@t objective ON; opt-in [friends] prompt (USE_SOCIAL_CONTEXT)
  roth_lbsn_kaggle.ipynb        stages 0-4 on a CPU Kaggle kernel, self-locating inputs,
                                expected-vs-actual counts printed at every stage
  stage6b_run2_server.ipynb     the TSMC-configured original stage5 was derived from — kept as
                                the reference for like-for-like comparisons; do not run it here
  archive/preprocessing-global-fsq.ipynb   the original extraction notebook stage 0 reproduces

data/
  lbsn/      stage 0+1 outputs: splits, POI metadata (taxonomy paths), friendship files,
             the 2014 category tree, nyc.mat (the paper's own NYC subset, alt population)
  kg_lbsn/   stages 2-4 outputs + the trained embeddings and checkpoint

docs/LBSN_HANDOFF.md   the fine-tune adaptation checklist + execution notes
```

The `group_examples_*.jsonl` (46 MB) are gitignored — regenerate deterministically in ~10 min:

```bash
python src/build_groups.py --data-dir ./data/lbsn --dataset LBSN_NYC \
       --out-dir ./data/lbsn/groups --no-resplit
```

---

## Reproducing the whole track

Everything is committed, so nothing below is required — it is how the artifacts were made.
Every script has `--self-check` (61 checks across the track; expected state is 0 failures).

```bash
python src/prepare_lbsn_csvs.py --zip lsbn2vec_global.zip           # stage 0 (gdown id in script)
python src/build_groups.py      --data-dir ./data/lbsn --dataset LBSN_NYC \
                                --out-dir ./data/lbsn/groups --no-resplit          # stage 1
python src/build_kg_lbsn.py     --csv-dir ./data/lbsn --groups-dir ./data/lbsn/groups  # stage 2
python src/train_roth.py        --kg-dir ./data/kg_lbsn --data-dir ./data/lbsn \
                                --dataset LBSN_NYC --epochs 120 --log-every 10 --max-eval 4000 \
                                --depth-weight 5.0 --depth-margin 0.3 --root-pull 0.01  # ~5 h CPU
python src/build_poi_poi_triples.py --kg-dir ./data/kg_lbsn \
                                --meta ./data/lbsn/poi_metadata_LBSN_NYC.csv \
                                --out-dir ./data/kg_lbsn --dataset LBSN_NYC \
                                --derive taxonomy --max-per-relation 40000              # stage 4
```

Or run `notebooks/roth_lbsn_kaggle.ipynb` on a CPU-only Kaggle kernel (accelerator `None`) —
self-locating inputs, no pasted paths, and it verifies every stage's counts against the
expected values.

---

## What stage 5 consumes

| Artifact | Used as |
|---|---|
| `data/kg_lbsn/poi_hyperbolic_embs_LBSN_NYC.npy` | `EMB_FILE` — the injected POI embeddings (6103 × 64, poi_idx order) |
| `data/kg_lbsn/poi_poi_triples_LBSN_NYC.pt` + `poi_relation_vocab_LBSN_NYC.json` | §6b `ALIGN_TRIPLES_FILE` / `ALIGN_RELVOCAB_FILE` |
| `data/lbsn/train/val/test_LBSN_NYC.csv` | the check-in task data (TSMC column layout) |
| `data/lbsn/groups/group_examples_{train,val,test}.jsonl` | the group task data (regenerate, see above) |
| `data/lbsn/friendship_old_LBSN_NYC.csv` | optional social context for prompts — old snapshot ONLY |

The adaptation checklist in `docs/LBSN_HANDOFF.md` (the `DATASET` variable, the two hardcoded
`ALIGN_*` filenames, `vocab.pkl`, the split discipline) has been **applied and CPU-verified**
in `notebooks/stage5_lbsn_finetune.ipynb`; the doc remains as provenance for what changed.
