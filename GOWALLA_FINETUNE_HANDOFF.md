# Gowalla fine-tune handoff — stage 5

Audience: whoever executes the fine-tune on a GPU. Written 2026-09-01. The Gowalla twin of
`LLMGPR_FINETUNE_HANDOFF.md`; read `LLMGPR_GOWALLA.md` first for how the dataset and the
upstream stages were built and what was measured.

Notebook: `notebooks/stage5_lbsn_finetune.ipynb` — the same notebook as the Foursquare arm. It is
parameterised by environment variables, so **no notebook edits are needed**.

## Run it

**On Kaggle/GPU the `gowalla` branch must be pushed first.** §0b stages `src/*.py` from
`REPO_RAW`, which now points at
`raw.githubusercontent.com/yosrkharrat/Group-recommendation-for-next-poi/**gowalla**/src` —
`affinity_blocked.py` and the `--regimes`/`--affinity-backend` flags exist only on this branch, so
a run against `llmgpr-pipeline` would stage the old `build_groups.py` and fail on the unknown
argument (and then OOM on dense affinity if it got past that). Running **locally** from the repo
root is unaffected: `find()` resolves `src/` directly and never fetches.

```bash
export STAGE6B_DATASET=GOWALLA
export STAGE6B_DATA_DIR=./data/gowalla
export STAGE6B_GROUPS_DIR=./data/gowalla/groups_social
jupyter lab notebooks/stage5_lbsn_finetune.ipynb
```

`_autodetect_data_dir` finds `./data/gowalla` on its own (it searches by content for
`train_GOWALLA.csv` + `poi_metadata_GOWALLA.csv`), and `find()` locates the embedding and
alignment files recursively under it, so all three exports are belt-and-braces: the groups
directory is now `groups_social`, which is also the notebook's default.

## One preparation step (5 seconds)

Five of the eight stage-5 inputs are **not committed** — 83 MB of derived CSVs that regenerate
byte-identically (verified by md5 on all six files) from the committed 13 MB
`gowalla_final_checkins.csv.gz`:

```bash
python src/prepare_gowalla_csvs.py        # ~5 s; writes train/val/test/poi_metadata/friendship/users
```

Nothing else is needed: the 1.7 GB raw Gowalla dump is **not** required, unlike the Foursquare
arm whose dataset "cannot be re-derived on this branch" (`LLMGPR_FINETUNE_HANDOFF.md` risk 3).
Everything `prepare_gowalla_csvs.py` reads is committed.

## The eight inputs stage 5 reads

| file | notebook symbol | state |
|---|---|---|
| `data/gowalla/train_GOWALLA.csv` | §2 splits (646,749 rows) | regenerate (5 s) |
| `data/gowalla/val_GOWALLA.csv` | §2 (87,803) | regenerate |
| `data/gowalla/test_GOWALLA.csv` | §2 (168,493) | regenerate |
| `data/gowalla/poi_metadata_GOWALLA.csv` | §2, `N_POI` = 47,783 | regenerate |
| `data/gowalla/friendship_old_GOWALLA.csv` | §9b group build (117,949 edges) | regenerate |
| `data/gowalla/kg_raw/poi_hyperbolic_embs_GOWALLA.npy` | `EMB_FILE` (47,783 × 64) | **committed** |
| `data/gowalla/kg_raw/poi_poi_triples_GOWALLA.pt` | `ALIGN_TRIPLES_FILE` (1,051,826 × 3) | **committed** |
| `data/gowalla/kg_raw/poi_relation_vocab_GOWALLA.json` | `ALIGN_RELVOCAB_FILE` (2 relations) | **committed** |

Group examples (`group_examples_{train,val,test}.jsonl`, ~310 MB) are gitignored and **§9b builds
them itself** (~25 min) if absent, so there is no manual step. 200-record samples plus schema
notes are committed at `data/gowalla/groups_social/samples/`. Sizes: train 84,205 · val 18,964 ·
test 42,694 (established 45,026 / 10,833 / 25,929 · occasional 35,119 / 7,208 / 14,554 ·
random 4,060 / 923 / 2,211).

## What differs from the Foursquare arm

**Stages 0–4 are now at full parity.** Diffing the two `groups_manifest.json` configs shows zero
substantive divergences: all three KCGRS regimes, window 180, hops 5, sizes 2–5,
`min_companion_repeat` 2, `max_venue_visitors` 50, `affinity_percentile` 99, `hist_len` 90,
`profile_top_k` 10, seed 42. The alignment stage matches the FSQ arm's *actual* provenance
(`derived: none`, `max_per_relation: null` → the two flat relations), not its handoff's reproduce
command, which contradicts its own committed artifact. Four things remain genuinely different,
all forced by the data or the hardware:

1. **`established` needs `src/affinity_blocked.py`.** `affinity.py`'s five dense `[n,n]` float64
   matrices are ~8 GB each at 31,667 users. The blocked backend computes the identical z-summed
   affinity and the identical exact percentile cut (5.660150 over all 501,383,611 pairs →
   5,013,928 edges, exactly 1.00%), and its self-check asserts threshold, neighbour sets and every
   `G[i,j]` match the dense path bit-for-bit. `--affinity-backend auto` keeps the dense path below
   12,000 users, so FSQ reproduction is untouched. One diagnostic is skipped in blocked mode:
   `validate()`'s per-component AUC table needs the dense matrices; the manifest records it as
   skipped rather than faking a number.
2. **The KG dir is `kg_raw`, not `kg_denoised`** — and that is the finding, not an oversight.
   GBSR is a measured no-op on this graph at every bottleneck strength tested; the denoised
   control is at `groups_social_denoised/` and every count moves by ≤ 0.21%
   (`LLMGPR_GOWALLA.md` §5). Cite it as the control.
3. **RotH hyperparameters — the one open parity gap.** See below.
4. **D1 ρ = +0.8683 STRONG** against FSQ's +0.3245, and `IS_NEAR_TO` covers 100% of POIs here
   (FSQ: 43%). So if the hyperbolic-vs-random ablation comes out flat, the weak-hierarchy
   explanation the FSQ handoff offered does **not** apply — look elsewhere.

## Open items, stated plainly

### a. RotH hyperparameters are not yet at parity

The FSQ/LBSN arm used `--epochs 120 --batch-size 512 --n-neg 128`; the committed Gowalla
embeddings used `--epochs 50 --batch-size 4096 --n-neg 32`. This was forced by wall clock and is
**measured, not estimated**: FSQ's exact settings run at **15.3 min/epoch = 30.7 h** for 120
epochs on an M-series laptop (the batch-4096 variant is 3.2 min/epoch). The FSQ arm itself trained
on Kaggle CUDA, where the exact settings are ~2–3 h, so **running the command below on a GPU is
both faithful and cheap** and is the recommended way to close this gap:

```bash
python src/train_roth.py --kg-dir ./data/gowalla/kg_raw --data-dir ./data/gowalla \
    --dataset GOWALLA --out-dir ./data/gowalla/kg_parity --epochs 120 \
    --batch-size 512 --n-neg 128 --log-every 10 --max-eval 4000 \
    --depth-weight 5.0 --depth-margin 0.3 --root-pull 0.01 --device cuda
```

**Decision taken (2026-09-01): keep the committed embeddings and do NOT run the parity job.**
This is a parameter-parity gap, not a correctness one — RotH is fully trained (50/50 epochs),
D1 ρ = +0.8683 STRONG, and stage 5 can consume it as-is. The 120/512/128 run is optional, belongs
on the GPU box if anyone wants strict comparability with the Foursquare numbers, and is **not** a
prerequisite for the fine-tune. Do not treat its absence as an unfinished stage.

### b. The depth weight may be mistuned

The committed embeddings were trained at `--depth-weight 5.0`, where the depth term is **55.9% of
the objective** at epoch 50. The matched probe (`data/gowalla/roth_depth_weight_probe.json`) says
`--depth-weight 1.0` is likely a better operating point: at equal budget it gives 2.3× the
`HAS_CATEGORY` MRR (0.0861 vs 0.0378) while holding D1 STRONG (+0.8055), and at `dw = 5` more
training makes that relation *worse* (0.0378 at 6 epochs → 0.0105 at 50). A 50-epoch `dw = 1` arm
was started and did not finish, so **it is not committed and the numbers above are the 6-epoch
matched ones, not a 50-epoch result.** To produce the candidate:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_roth.py --kg-dir ./data/gowalla/kg_raw \
    --data-dir ./data/gowalla --dataset GOWALLA --out-dir ./data/gowalla/kg_raw_dw1 \
    --epochs 50 --batch-size 4096 --n-neg 32 --log-every 10 --max-eval 4000 \
    --depth-weight 1.0 --depth-margin 0.3 --root-pull 0.01 --device mps   # --device cuda on a GPU box
```

The committed `dw = 5` embeddings are valid and clear the D1 gate comfortably; this is a possible
improvement, not a blocker. Do not swap them mid-experiment without re-running the ablation.

## Reproducing stages 0–4 from scratch

```bash
python src/prepare_gowalla_csvs.py                                            # stage 0, 5 s
python src/build_groups.py --data-dir ./data/gowalla --dataset GOWALLA \
    --out-dir ./data/gowalla/groups_social_raw --no-resplit \
    --group-source social --friendship old \
    --friend-old ./data/gowalla/friendship_old_GOWALLA.csv \
    --regimes occasional random --profile-top-k 10 --hist-len 90            # stage 1, ~10 min
python src/denoise_social_gbsr.py --data-dir ./data/gowalla --csv-dir ./data/gowalla \
    --dataset GOWALLA --out-dir ./data/gowalla \
    --batch-size 8192 --hsic-sample 1024 --eval-every 2 --early-stop 5      # stage 2, ~1 h (a no-op)
PYTHONHASHSEED=0 python src/build_kg_lbsn.py --csv-dir ./data/gowalla --dataset GOWALLA \
    --taxonomy ./data/gowalla/gowalla_category_paths.json \
    --groups-dir ./data/gowalla/groups_social_raw \
    --friendship-old ./data/gowalla/friendship_old_GOWALLA.csv \
    --out-dir ./data/gowalla/kg_raw                                          # stage 2b, ~3 min
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_roth.py --kg-dir ./data/gowalla/kg_raw \
    --data-dir ./data/gowalla --dataset GOWALLA --out-dir ./data/gowalla/kg_raw \
    --epochs 50 --batch-size 4096 --n-neg 32 --log-every 10 --max-eval 4000 \
    --depth-weight 5.0 --depth-margin 0.3 --root-pull 0.01 --device mps      # stage 3, ~4.5 h
PYTHONHASHSEED=0 python src/build_poi_poi_triples.py --kg-dir ./data/gowalla/kg_raw \
    --meta ./data/gowalla/poi_metadata_GOWALLA.csv --out-dir ./data/gowalla/kg_raw \
    --dataset GOWALLA --derive taxonomy --max-per-relation 40000              # stage 4, ~1 min
```

`PYTHONHASHSEED=0` on the KG builds: `build_kg.py`'s modal-category tie-break iterates a dict,
so the KG is not bit-reproducible without it (`LLMGPR_FINETUNE_HANDOFF.md` known issue a — still
unfixed by decision, shared with the FSQ track).

## Self-checks

Every stage script has one, and all pass: `prepare_gowalla_csvs` (fixture), `build_groups`
(co-presence + social), `denoise_social_gbsr` (5 checks, including that the upstream `detach()`
bug stays fixed and that the mask separates planted signal from noise), `build_kg_lbsn` (14),
`train_roth` (4), `build_poi_poi_triples` (9).
