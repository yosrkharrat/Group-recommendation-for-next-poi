# HyGro-POI: group next-POI recommendation with a diffusion LLM and hyperbolic embeddings

When a group of friends picks where to go next, the members often disagree. HyGro-POI models the group's consensus as the **hyperbolic midpoint** of the members' embeddings in a Poincaré ball. When members disagree, that midpoint moves toward the centre of the ball, where the more general categories sit (for example, "food" rather than "ramen"). This replaces hand-designed aggregation rules such as averaging or least-misery. It builds on an individual next-POI recommender: **LLaDA-MoE-7B**, a masked-diffusion LLM fine-tuned with 4-bit QLoRA, which scores all 5,120 POIs in a single denoising step. The group fine-tune is the next stage (status below).

> **Paper:** Z. Hamdani, **Y. Kharrat**, E. Mhamdi, S. Ben Yahia. *HyGro-POI: Hyperbolic Consensus over a Masked Diffusion LLM for Group Next-POI Recommendation.* In preparation for SIGIR 2027.
> Research internship at the University of Southern Denmark (SDU), 2026.

## Key result

The midpoint only means "more general" if the ball is organised by depth: general categories near the centre, specific venues near the edge. The original RotH embeddings had no such structure. A **depth regulariser** added to knowledge-graph training creates it:

| Embeddings | Spearman ρ (radius vs. taxonomy depth) | Mean radius, depth 1 → 5 |
| --- | --- | --- |
| RotH (baseline) | +0.03 (no hierarchy) | 1.256 · 1.253 · 1.258 · 1.256 · 1.273 |
| **RotH + depth regulariser** | **+0.85** | 0.767 · 0.868 · 0.978 · 1.241 · 1.501 |

The effect holds on a second dataset with a real friendship graph (Foursquare LBSN2Vec++): ρ = +0.71, with radii increasing at every depth.

| | |
| --- | --- |
| **Knowledge graph** | 9,445 entities, 175,727 triples (POIs, categories, users, groups) |
| **Group data** | 34,644 / 6,486 / 13,420 train / val / test examples mined from Foursquare NYC check-ins |
| **Reproducibility** | every script has a `--self-check` against synthetic data (96 checks); stages 1–4 run on a CPU-only Kaggle kernel in about 50 minutes |
| **Stack** | PyTorch, Hugging Face, QLoRA, hyperbolic KG embeddings (RotH / RefH / AttH), Kaggle |

---

## Where the project stands

| # | Stage | Status | Produces |
|---|---|---|---|
| 1 | Group construction | ✅ done | `data/groups/` — 34,644 / 6,486 / 13,420 examples |
| 2 | KG construction (POI + group, merged) | ✅ done | `data/kg/kg_triples.pt` — 9,445 entities, 175,727 triples |
| 3 | RotH training with the depth regulariser | ✅ done | `data/kg/poi_hyperbolic_embs_NYC.npy` — **D1 ρ = +0.8485** |
| 4 | POI-POI triples for the alignment adapter | ✅ done | `data/kg/poi_poi_triples_NYC.pt` |
| 5 | **LLaDA fine-tune** (Acc@t objective + adapter + hyperbolic scorer) | ❌ **not started** | — |
| 6 | Evaluation | unchanged from the individual pipeline | |

Stages 1–4 run end-to-end on a **CPU-only Kaggle kernel** via `notebooks/group_pipeline_kaggle.ipynb`
(~50 min, no GPU / HF token / Internet). Stage 5 needs a GPU.

### The headline result so far

The depth regulariser gives the Poincaré ball a **radial hierarchy**, which is the substrate the
whole group-consensus mechanism depends on and which the shipped embeddings did not have:

```
                      D1 Spearman   verdict   radius by taxonomy depth
OLD (shipped)              +0.0301   ABSENT   1.256 1.253 1.258 1.256 1.273   flat
NEW (+depth regulariser)   +0.8485   STRONG   0.767 0.868 0.978 1.241 1.501   monotonic
```

Without a radial ordering, "the hyperbolic midpoint of disagreeing members is a *more general*
category" is a metaphor. With one, it is a measurable coordinate. See `docs/PIPELINE.md` §3.

> **Missing control.** The `--depth-weight 0` run that proves the regulariser *caused* this was
> lost when a scratch directory was cleaned. It is ~40 min on Kaggle and it is the single most
> valuable missing artifact, because it is the entire evidential basis for stage 3.

---

## Layout

```
src/                       all pipeline code, deliberately FLAT
  build_groups.py            stage 1  group construction        (11 self-checks)
  affinity.py                stage 1  multi-signal user affinity
  build_kg.py                stage 2  POI + group KG            (12 self-checks)
  build_kg_lbsn.py           stage 2  for the LBSN2Vec++ social dataset (15 self-checks)
  train_roth.py              stage 3  RotH + depth regulariser  (4 self-checks)
  build_poi_poi_triples.py   stage 4  triples for the adapter   (9 self-checks)
  alignment.py               stage 5  ManifoldAwareAdapter  (supervisor's, INPUT side)
  objective.py               stage 5  Acc@t objective + HyperbolicScorer (29 self-checks)
  hyperbolic_group.py                 gyromidpoint, GeometricAttention (16 self-checks)
  phase0_diagnostics.py               standalone D1 / D2 / D3

notebooks/
  group_pipeline_kaggle.ipynb   stages 1-4, CPU-only, self-locating inputs
  stage6b_run2_server.ipynb     stage 5, the LLaDA fine-tune
  archive/                      superseded evaluation notebooks

data/
  groups/   stage 1 output   (see docs/PIPELINE.md §1 for the schema)
  kg/       stages 2-4 output + the trained embeddings

docs/     PROJECT_DOCUMENTATION.md (individual pipeline), GROUP_REC_PROPOSAL.md, PIPELINE.md
papers/   H-RLPOI (ours), KCGRS (Pervin, DSS 2025), MAC-GPR (Acharya, WWW 2026)
legacy/   superseded: build_group_relations.py, group-kg.ipynb
```

`src/` is flat on purpose: `build_groups.py` imports `affinity.py` directly, and the Kaggle
dataset upload is a flat folder, so sub-packaging would create import friction for no gain.

---

## Running it

**Getting the check-in data.** The five Foursquare files (`train/val/test_NYC.csv`,
`poi_metadata_NYC.csv`, `vocab.pkl`) are gitignored here — TSMC2014 / FSQ-OS Places data, not
ours to redistribute — but they are public in the companion repo:

```bash
BASE=https://raw.githubusercontent.com/eyamhamdi03/finetunning-LLada-MoE-poi-recommendation/main/data
mkdir -p data && for f in train_NYC.csv val_NYC.csv test_NYC.csv poi_metadata_NYC.csv vocab.pkl; do
  curl -sL "$BASE/$f" -o "data/$f"; done
```

Do **not** also take `poi_hyperbolic_embs.npy` from there: that is the pre-regulariser file
(ρ = +0.03, flat radii). The correct one is already committed at
`data/kg/poi_hyperbolic_embs_NYC.npy`.

**Stages 1–4 (Kaggle, CPU).** Attach two datasets — the data (`train/val/test_NYC.csv`,
`poi_metadata_NYC.csv`) and a private dataset containing `src/*.py`. Then run
`notebooks/group_pipeline_kaggle.ipynb`. **You paste no paths**: cell 0 finds both by content.
Every cell reloads its own state, so a kernel restart will not break it.

**Locally**, with the same CSVs in `./data`:

```bash
python src/build_groups.py           --data-dir ./data --out-dir ./data/groups
python src/build_kg.py               --data-dir ./data --groups-dir ./data/groups --out-dir ./data/kg
python src/train_roth.py             --kg-dir ./data/kg --data-dir ./data --out-dir ./data/kg \
                                     --epochs 120 --depth-weight 5.0 --depth-margin 0.3 --root-pull 0.01
python src/build_poi_poi_triples.py  --kg-dir ./data/kg --meta ./data/poi_metadata_NYC.csv \
                                     --out-dir ./data/kg --derive taxonomy
```

Every script has `--self-check`, which runs against a synthetic fixture in seconds and needs no
data. Run them before a long job: **81 checks, 0 failures** is the expected state
(96 including `build_kg_lbsn.py`).

### The LBSN2Vec++ social-network track (new)

Second dataset: the Foursquare **global-scale check-in dataset with user social networks**
(Yang et al., §5 of the dataset page; used by LBSN2Vec++, TKDE 2020). What it adds over
TSMC2014: a **real friendship graph**. The old pipeline had to infer user–user ties (affinity
z-sum, AUC 0.72); here `FRIEND_OF` edges are observed.

**Status: stages 0–4 all done and committed.** KG: 12,709 entities / 244,080 triples /
12 relations. Embeddings: `data/kg_lbsn/poi_hyperbolic_embs_LBSN_NYC.npy` — **D1 ρ = +0.7096
STRONG, radii fully monotone** (0.736 / 0.747 / 0.974 / 1.321 over depths 1–4; ρ is tie-capped
below TSMC's +0.85 because 69% of POIs sit at depth 2). Stage 5 hand-over:
`docs/LBSN_HANDOFF.md`. Still missing: the `--depth-weight 0` control
(`notebooks/roth_lbsn_kaggle.ipynb` with `RUN_CONTROL = True`).

The canonical extraction follows `notebooks/archive/preprocessing-global-fsq.ipynb`
(NYC bounding box over the raw 2.68 GB dump, venues ≥ 10 visits, users ≥ 30 →
**159,304 check-ins, 1,665 users, 6,103 POIs**, real timestamps + coordinates + category
names), reproduced as a script with one deliberate fix: **friendship snapshots are kept
separate**. The notebook merged `friendship_old` + `friendship_new` into one edge list; the
"new" snapshot postdates the check-in period and `(new − old)` is the paper's
friendship-prediction eval set, so only `friendship_old` may enter a KG —
`friendship_new_only_LBSN_NYC.csv` is exported for evaluation and asserted absent from the
triples at build time.

```bash
python src/prepare_lbsn_csvs.py --zip lsbn2vec_global.zip                # stage 0: raw -> house CSVs
python src/build_groups.py      --data-dir ./data/lbsn --dataset LBSN_NYC \
                                --out-dir ./data/lbsn/groups             # stage 1, unchanged
python src/build_kg_lbsn.py     --csv-dir ./data/lbsn --groups-dir ./data/lbsn/groups   # stage 2
python src/train_roth.py        --kg-dir ./data/kg_lbsn --data-dir ./data/lbsn --dataset LBSN_NYC \
                                --epochs 120 --depth-weight 5.0 --depth-margin 0.3 --root-pull 0.01
```

Stage 0 emits the exact house schema (`train/val/test_LBSN_NYC.csv`,
`poi_metadata_LBSN_NYC.csv`), so stage 1 group construction runs **unchanged** — with real
timestamps, the co-location mining is exactly what `build_groups.py` already does. Category
names are mapped to full 2014-Foursquare taxonomy paths via the committed
`data/lbsn/fsq_category_paths_2014.json` (763 nodes, depth 1–4), which restores the
`SUBCATEGORY_OF` spine the depth regulariser needs.

`build_kg_lbsn.py --mat data/lbsn/nyc.mat` (DS tag `MAT_NYC`) builds instead from
`dataset_connected_NYC.mat` — the LBSN2Vec authors' own NYC subset (4,024 users / 8,723 old
friendships, selected for social connectivity, but no timestamps/coordinates/names; the
script reconstructs chronology from the hour-of-week wraps). Keep it for paper-comparable
experiments; the CSV track is the one with full fidelity.

---

## What stage 5 consumes

| Artifact | Used as |
|---|---|
| `data/kg/poi_hyperbolic_embs_NYC.npy` | `EMB_FILE` — the injected POI embeddings |
| `data/kg/poi_poi_triples_NYC.pt` + `poi_relation_vocab_NYC.json` | §6b `ALIGN_TRIPLES_FILE` / `ALIGN_RELVOCAB_FILE` |
| `data/groups/group_examples_{train,val,test}.jsonl` | the group task data |

Two hyperbolic→Euclidean components, easily confused, **complementary not alternative**:

- **`alignment.py` — the INPUT side.** A trained `ManifoldAwareAdapter` replacing the fixed random
  projection when building `W_POI`, preserving neighbourhoods, radius and KG triples.
- **`objective.py::HyperbolicScorer` — the OUTPUT side.** Ranks by geodesic distance rather than a
  dot product. Without it the geometry enters the model and is then discarded at scoring time,
  because an inner product cannot express a geodesic distance.

---

## Next actions

1. **Re-run the `--depth-weight 0` control** on Kaggle (~40 min, unattended). Restores the row
   that makes stage 3 an argument rather than an assertion.
2. **Score-aggregation baselines** — AVG / least-misery / most-pleasure over the *existing*
   checkpoint on the group test set. No training, only inference. This is the number the group
   model has to beat, and it is much better to learn it before the fine-tune than after.
3. ~~Wire `objective.py` into `stage6b_run2_server.ipynb`~~ — done, off by default. Still on the
   individual CSVs, not `data/groups/group_examples_*.jsonl`. Then fine-tune.
4. Add **Hit@10 and NDCG@10** to the evaluation so KCGRS becomes directly citable.

## Open risks

- **Nothing has been tested at the decision layer.** Every result so far is at the embedding
  level. D1 ρ=+0.85 does not by itself imply better recommendations.
- **`established` groups are the easy case by construction** (members already agree), and they
  are 55% of the data; the fully-observed `occasional` slice is only 14%. Report
  heterogeneity-stratified metrics, never a pooled number. And note `regime` is *not* a proxy for
  heterogeneity — stratify on the `heterogeneity` column.
- **Only 21 real group→group transitions exist** in FSQ-NYC at any parameter setting, so "real
  group next-POI" can never be claimed on this dataset. See `docs/PIPELINE.md` §1.
- **KCGRS / MAC-GPR numbers are not directly comparable** — different task (non-sequential),
  different metrics, different group sizes, and possibly a different candidate set. See
  `docs/PIPELINE.md` §5.
