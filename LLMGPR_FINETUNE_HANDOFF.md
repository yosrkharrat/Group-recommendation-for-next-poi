# LLMGPR fine-tune handoff — stage 5 on the `llmgpr-pipeline` branch

Audience: whoever executes the fine-tune on a GPU. Everything upstream of stage 5 is committed
on this branch; nothing below needs re-running unless you want to. Read `LLMGPR_TRACK.md` first
for how this dataset was identified and why Table 1 is read the way it is — this file is only
about **running the fine-tune**.

Notebook: `notebooks/stage5_lbsn_finetune.ipynb` (the filename is historical — it is the LLMGPR
run on this branch).

> This is a separate track from the LBSN pipeline in `docs/LBSN_HANDOFF.md`. Different dataset,
> different group rule, different POI id space. Nothing carries over numerically.

## What exists

`LLMGPR`: **423,376 check-ins / 6,889 users / 14,402 POIs** across New York, Chicago and Los
Angeles, real timestamps, coordinates where the region catalogue covers the venue, and full
Foursquare taxonomy paths. Per-user chronological 70/10/20 already applied
(train 299,451 · val 42,005 · test 81,920 rows).

- **Social layer:** `friendship_old_LLMGPR.csv` — 8,640 before-period edges over 4,872 users.
  `friendship_new_only_LLMGPR.csv` — 5,528 pairs, **EVAL ONLY** (see rule 1).
- **Hyperbolic embeddings:** `data/llmgpr/kg_denoised/poi_hyperbolic_embs_LLMGPR.npy`
  (14,402 × 64, rows in `poi_idx` order). **D1 ρ = +0.3245, STRONG, radii monotone** over
  taxonomy depths 1–4 (1.014 / 1.017 / 1.055 / 1.073). See the risk note below — this is a
  *weak* pass.
- **Alignment triples:** 254,265 triples / **2 relations** (`IS_NEAR_TO` 76,666,
  `FOLLOWED_BY` 177,599).
- **Group task:** LLMGPR's own social-group rule, 71,933 / 16,164 / 33,193 examples.

## File map

| File | What it is |
|---|---|
| `data/llmgpr/{train,val,test}_LLMGPR.csv` | check-in splits, house column layout |
| `data/llmgpr/poi_metadata_LLMGPR.csv` | `poi_idx` 0–14,401, venue hex id, full taxonomy path, lat/lon, locality/region |
| `data/llmgpr/friendship_old_LLMGPR.csv` | 8,640 edges, 0-based user ids — the only friendship file training may touch |
| `data/llmgpr/friendship_new_only_LLMGPR.csv` | 5,528 pairs — **EVAL ONLY, never in training data or prompts** (rule 1) |
| `data/llmgpr/kg_denoised/poi_hyperbolic_embs_LLMGPR.npy` | `EMB_FILE` (14,402 × 64) |
| `data/llmgpr/kg_denoised/poi_poi_triples_LLMGPR.pt` + `poi_relation_vocab_LLMGPR.json` | §6b `ALIGN_TRIPLES_FILE` / `ALIGN_RELVOCAB_FILE` |
| `data/llmgpr/groups_social/` | §9b group task: CSVs + manifest committed, `group_examples_*.jsonl` gitignored (self-built, ~2 min) |
| `data/llmgpr/groups_social/samples/` | 200 records per split + schema README |

Id spaces: `user_id` = rank of the raw numeric id (0–6,888); `poi_idx` = rank of the venue hex id
(0–14,401). Both contiguous, consistent across every artifact above.

## What the fine-tune actually reads — the full KG is NOT needed

Verified by enumerating every read in the notebook and by running §6b's alignment training on
CPU with nothing else present. It reads **seven** files:

```
data/llmgpr/{train,val,test}_LLMGPR.csv        check-in splits
data/llmgpr/poi_metadata_LLMGPR.csv           N_POI, taxonomy paths, poi_cat
data/llmgpr/kg_denoised/poi_hyperbolic_embs_LLMGPR.npy    EMB_FILE  (§2, §7)
data/llmgpr/kg_denoised/poi_poi_triples_LLMGPR.pt         ALIGN_TRIPLES_FILE  (§6b)
data/llmgpr/kg_denoised/poi_relation_vocab_LLMGPR.json    ALIGN_RELVOCAB_FILE (§6b)
```

plus `friendship_{old,new_only}_LLMGPR.csv` when `USE_SOCIAL_CONTEXT=True`, and the
`groups_social/group_examples_*.jsonl` it builds itself for §9b.

`kg_triples.pt`, `kg_entities.json`, `kg_hierarchy.pt`, `kg_relations.json`, `kg_poi_rows.json`,
`kg_manifest.json` and `roth_best.pt` are **not referenced anywhere in the notebook**. The full
knowledge graph is an *intermediate* — `train_roth.py` consumes it to produce the embeddings, and
`build_poi_poi_triples.py` distils its POI→POI subset into the two alignment files. Only those
outputs cross into stage 5, and all of them are committed. Nothing is missing.

(`roth_results.json` appears in prose only, as the place the D1 number *would* be recorded — no
code reads it. See risk 2.)

## Running it

```bash
git clone -b llmgpr-pipeline https://github.com/yosrkharrat/Group-recommendation-for-next-poi.git
cd Group-recommendation-for-next-poi
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
export HF_TOKEN=hf_...        # READ scope; or a .env at the repo root
jupyter lab notebooks/stage5_lbsn_finetune.ipynb
```

**No paths need editing.** `DATA_DIR` is auto-detected by content. Set `RUN_PROFILE = "smoke"`
for the first pass (~15 min, exercises every stage including the adapter, the D1 gate and the
friendship leakage guard), then `"full"`.

`RUN_PROFILE = "full"` requires a **≥ 40 GB GPU** and refuses to start below that — at
`BATCH_SIZE=4` it is ~30 h/epoch. On first run §9b builds the group examples itself
(~2 min wall, ~4.5 GB peak RAM); no manual step.

Verify these lines, then leave it:

```
RUN_PROFILE=full  SUBSAMPLE_FRAC=1.0  VAL_MAX=None  TEST_MAX=None
[hw] 80 GB VRAM -> QUANTIZE=False BATCH_SIZE=16 GRAD_ACCUM=2 GRAD_CKPT=False
POIs: 14402 | emb dim: 64 | condition: hyperbolic
[D1] emb file=poi_hyperbolic_embs_LLMGPR.npy  rho=+0.3245  monotonic=True
POI-POI triples: 254,265  relations: 2
[group-task] examples  train=71933  val=16164  test=33193
[group-prompt] tokens over 2000 examples: mean~2015 max<4085  (MAX_LEN=4096)
```

## The group task (§9b / §11b)

A group here is **LLMGPR/CubeRec's own rule** — a connected component of *friends* (≤ 5
friendship hops) co-present at the same venue in the same 180-min bucket — not the KCGRS
anchor windows the TSMC/LBSN_NYC runs used. That rule feeds the `occasional` compositions and
the co-attendance tie pool; `established` (affinity clique) and `random` are unchanged.

Measured: 4,804 real social groups from 10,603 (venue, 180-min) candidates; affinity validation
AUC **0.894** combined against 6,685 real co-present pairs. Only **308** real group→group
transitions exist, which is why the trainable task is the standard constructed one.

`TRAIN_ON_GROUPS = False` by default, so §11b evaluates the group task on the same trained heads
without changing the individual run. Flip it to fold group examples into `train_ex` — and expect
to halve `BATCH_SIZE`, since group sequences are long.

**Stratify results on `heterogeneity`, not on `regime`** (see `src/build_groups.py`'s docstring).

### `MAX_LEN` is 4096 on this dataset, not 2048

Measured with the real LLaDA tokenizer over all 121,290 group examples: p50 1,959 / p99 3,193 /
**max 3,792** tokens. 2048 truncates 46.5% of them and 3072 still truncates 2.6%; 4096 truncates
none. LLMGPR runs ~2× longer than LBSN_NYC at identical knobs because its categories are full
taxonomy paths and its 14,402 POIs give 5-digit `<poi_i>` tokens.

This matters because HF truncates from the **end**, which would eat `[group summary]`,
`[current time]` and the `[group next POI]` cue — the instruction the model is meant to answer.
The cell's own assert is the guard; if it ever fires, lower `GROUP_HIST_PER_MEMBER`
(measured: 20 → max 3,618, 10 → max 3,041) rather than letting it truncate. Individual §9
prompts are unaffected — `collate` pads to the batch max, not to `MAX_LEN`.

## Three things to know before you trust a result

1. **The embeddings clear the D1 gate by only 0.025.** ρ = +0.3245 against a `> 0.30` floor, with
   radii spanning just 1.014–1.073 — far weaker than LBSN_NYC (+0.63) or TSMC (+0.85). Two
   structural reasons: 72.7% of LLMGPR POIs sit at taxonomy depth 2 (tie-capping ρ), and this KG
   carries only **2** POI-POI relations against LBSN_NYC's 5, because the 10 km region catalogue
   covers ~43% of visited venues so `IS_NEAR_TO` is built over that subset only. Treat +0.32 as
   a baseline to beat. **If the hyperbolic-vs-random ablation comes out flat, suspect this first.**

2. **The GBSR denoising is not evidenced in the repo.** The notebook describes the social layer as
   GBSR-denoised, but `src/denoise_social_gbsr.py` writes
   `friendship_old_denoised_<DS>.csv`, `social_edge_weights_<DS>.csv` and
   `gbsr_denoise_manifest.json`, and **none of those three are committed**. What *is* committed is
   `friendship_old_LLMGPR.csv` — the exact filename `src/prepare_llmgpr_csvs.py` writes for the
   raw before-period snapshot. So either the denoiser's output was renamed over it (in which case
   the manifest is still missing and the threshold/edge-count/mask-separation provenance is lost),
   or the committed graph is pre-denoising and the description is ahead of the artifact.
   **Ask the author to commit `gbsr_denoise_manifest.json` before this is reported as a
   GBSR result.** Same for `kg_denoised/roth_results.json` and `roth_best.pt`, which the LBSN
   track ships and this one does not — the D1 number above was recomputed from the `.npy` with
   `src/train_roth.py`'s own `d1_radial_hierarchy` to stand in.

   **2b. And when GBSR *is* run here, it does nothing.** Smoke run on the real
   `friendship_old_LLMGPR.csv` (10 epochs; BPR converged, train AUC 0.977):

   ```
   mask weight: mean=1.5000  std=1.3e-09  min=1.4999999  max=1.5000
   kept 8,639/8,640 edges (100.0%), pruned 1
   ```

   The mask MLP collapsed to a **constant** — zero variance across all 8,640 edges — so the
   median threshold keeps everything and prunes exactly one edge by floating-point luck. This is
   precisely the outcome `LLMGPR_TRACK.md` §2 said to measure for rather than assume: mean degree
   3.2 here against GBSR's yelp benchmark at 38.6, and a graph that sparse has little redundancy
   to prune. Caveat: 10 epochs, not the default 200 — but a *constant* mask is saturation, not
   undertraining, and val NDCG@20 already peaked at epoch 5 (0.0357) and fell by epoch 10
   (0.0304). Confirm with a full 200-epoch run, then decide whether GBSR belongs in the story at
   all. Right now there is no evidence it changes the social graph.

3. **The dataset cannot be re-derived on this branch.** `prepare_llmgpr_csvs.py` consumes
   `llmgpr_final_{checkins,catalogue,friendships}.csv` from the filtering notebooks, and neither
   those files nor Yang's raw inputs are committed (they are large public research files). The
   committed `data/llmgpr/*.csv` are therefore the source of truth. The stage-0 code is still
   verified by `--self-check`, but a from-zero rebuild needs the raw dump re-downloaded and the
   filtering notebooks re-run — see `LLMGPR_TRACK.md` §6.

## Two rules that protect the results

1. **Friendship leakage.** `friendship_new_only` pairs formed during/after the check-in period and
   are partly an *effect* of the co-visits the model trains on. They are asserted absent from the
   KG and from group construction, and §2 loads them purely to assert disjointness from prompts.
   Keep them out of training inputs and prompts.
2. **Split discipline.** The CSVs already ship the per-user chronological 70/10/20 that the KG's
   train-only relations and the group examples were built against. `RESPLIT = False` — do not
   re-split, or the three stop agreeing on what "train" means.

## Smoke test run before handoff (CPU only — no GPU available here)

All six module self-checks pass on their fixtures: `prepare_llmgpr_csvs` (11 checks),
`build_groups` (both the co-presence and the social fixture), `denoise_social_gbsr` (5 checks,
including that the upstream `detach()` bug stays fixed), `build_kg_lbsn` (14 checks),
`train_roth` (4 checks), `build_poi_poi_triples` (9 checks).

Then on the **real** LLMGPR data:

| Stage | Result |
|---|---|
| 0 · preprocessing | fixture only — inputs not in repo, see risk 3 |
| 1 · group construction | **PASS** — 4,804 social groups → 71,933/16,164/33,193 examples, ~2 min, all causality/clique/recurrence asserts green |
| 2 · GBSR denoising | **RUNS, but is a no-op on this graph** — see risk 2b |
| 2b · KG build | **PASS** — 26,467 entities / 466,589 triples / 12 relations, `FRIEND_OF` 17,280, leakage guard verified 5,528 new-only pairs absent |
| 3 · RotH | not re-run — the trained `.npy` is committed and verified to load and pass the D1 gate |
| 4 · alignment triples | consistent: the committed 254,265-triple file = `IS_NEAR_TO` 76,666 + `FOLLOWED_BY` 177,599, exactly the POI→POI subset of the rebuilt KG |
| 5 · fine-tune, CPU side | **PASS** — §2 loads, emb-row assert, D1 gate, leakage guard with `USE_SOCIAL_CONTEXT=True`; §9b renders all 121,290 group examples, every one gets a hyperbolic consensus line, the rebuilt `collate` handles a mixed group+individual batch with the 10 mask slots flush right, truncation assert passes |

**Not verified:** the GPU fine-tune itself (§6–§11), which needs the model and a ≥ 40 GB card.
That is the one thing this handoff asks you to run.

### Two defects the smoke test found

**a. `src/build_kg.py`'s `prefers_category()` is not reproducible across processes.** Two runs on
identical inputs gave 466,589 and 466,591 triples (`PREFERS_CATEGORY` 10,122 vs 10,124). Cause:
both tie-breaks iterate a **set**, so `str` hash randomisation decides equal-mass ties —
`roots = sorted({p[:1] for p in paths}, key=lambda nd: -mass[nd])` and
`best = max(children, key=lambda c: mass[c])`. Proof, on the real train split:

```
PYTHONHASHSEED=0  rows=10123  sha1=f142ae577a8786bc     # same seed twice ->
PYTHONHASHSEED=0  rows=10123  sha1=f142ae577a8786bc     # identical
PYTHONHASHSEED=1  rows=10124  sha1=8d56dd922c38899c     # different seed ->
PYTHONHASHSEED=2  rows=10124  sha1=cec7ea3c997d91ce     # different KG
PYTHONHASHSEED=3  rows=10125  sha1=3b3cdc03299df3e7
```

**It does not affect the fine-tune's inputs.** The two rebuilds agreed exactly on
`IS_NEAR_TO` (76,666) and `FOLLOWED_BY` (177,599) — and those two are the entire POI→POI subset
that becomes the alignment file (76,666 + 177,599 = 254,265, matching the committed file). Only
`PREFERS_CATEGORY` moved, and it is a USER→CATEGORY relation that `build_poi_poi_triples.py`
drops. So the alignment triples are reproducible; what is not bit-reproducible is the KG that
`train_roth.py` sees, so a *retrained* embedding file would differ slightly and D1 ρ would wobble. Fix is a deterministic
tie-break at both sites (`key=lambda nd: (-mass[nd], nd)`). **Not applied here** — it changes the
KG the committed embeddings were trained on, and `build_kg.py` is shared with the LBSN track, so
it needs a coordinated decision. Until then, pin `PYTHONHASHSEED` when rebuilding.

**b. `ndcg_at_k()` overflows.** `denoise_social_gbsr.py:251` (`item_emb @ user_emb.T`) raises
`divide by zero` / `overflow` / `invalid value` RuntimeWarnings on both the fixture and the real
data. That matmul feeds the val NDCG used for **model selection**, so non-finite values there
silently corrupt which epoch is chosen. Worth a `np.isfinite` guard before trusting any GBSR run.

## Reproducing stages 1–4 from the committed CSVs

```bash
python src/denoise_social_gbsr.py --data-dir ./data/llmgpr --csv-dir ./data/llmgpr \
        --dataset LLMGPR --out-dir ./data/llmgpr                              # stage 2 (GBSR)
python src/build_groups.py --data-dir ./data/llmgpr --dataset LLMGPR \
        --out-dir ./data/llmgpr/groups_social --no-resplit \
        --group-source social --friendship old \
        --friend-old ./data/llmgpr/friendship_old_LLMGPR.csv \
        --profile-top-k 10 --hist-len 90                                      # stage 1 (groups)
python src/build_kg_lbsn.py --csv-dir ./data/llmgpr --dataset LLMGPR \
        --groups-dir ./data/llmgpr/groups_social --out-dir ./data/llmgpr/kg_denoised   # stage 2b
python src/train_roth.py --kg-dir ./data/llmgpr/kg_denoised --data-dir ./data/llmgpr \
        --dataset LLMGPR --epochs 120 --log-every 10 --max-eval 4000 \
        --depth-weight 5.0 --depth-margin 0.3 --root-pull 0.01                # stage 3 (GPU: --device cuda)
python src/build_poi_poi_triples.py --kg-dir ./data/llmgpr/kg_denoised \
        --meta ./data/llmgpr/poi_metadata_LLMGPR.csv --out-dir ./data/llmgpr/kg_denoised \
        --dataset LLMGPR --derive taxonomy --max-per-relation 40000            # stage 4
```
