# Gowalla fine-tune handoff — stage 5

Audience: whoever executes the fine-tune on a GPU. Written 2026-09-01. The Gowalla twin of
`LLMGPR_FINETUNE_HANDOFF.md`; read `LLMGPR_GOWALLA.md` first for how the dataset and the
upstream stages were built and what was measured.

Notebook: `notebooks/stage5_lbsn_finetune.ipynb` — the same notebook as the Foursquare arm. It is
parameterised by environment variables, so **no notebook edits are needed**.

## Run it

```bash
export STAGE6B_DATASET=GOWALLA
export STAGE6B_DATA_DIR=./data/gowalla
export STAGE6B_GROUPS_DIR=./data/gowalla/groups_social_raw
jupyter lab notebooks/stage5_lbsn_finetune.ipynb
```

`_autodetect_data_dir` finds `./data/gowalla` on its own (it searches by content for
`train_GOWALLA.csv` + `poi_metadata_GOWALLA.csv`), and `find()` locates the embedding and
alignment files recursively under it, so `STAGE6B_DATA_DIR` is belt-and-braces. **`STAGE6B_GROUPS_DIR`
is not** — the default is `<DATA_DIR>/groups_social`, and this arm's directory is
`groups_social_raw`. Set it.

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
| `data/gowalla/kg_raw/poi_poi_triples_GOWALLA.pt` | `ALIGN_TRIPLES_FILE` (160,000 × 3) | **committed** |
| `data/gowalla/kg_raw/poi_relation_vocab_GOWALLA.json` | `ALIGN_RELVOCAB_FILE` (4 relations) | **committed** |

Group examples (`group_examples_{train,val,test}.jsonl`, 146 MB) are gitignored and **§9b builds
them itself** (~10 min) if absent, so there is no manual step. 200-record samples plus schema
notes are committed at `data/gowalla/groups_social_raw/samples/`.

## What differs from the Foursquare arm

1. **The regime set.** `established` needs the dense `[n,n]` affinity matrices — ~8 GB each at
   31,667 users. §9b now picks `occasional random` for `GOWALLA` automatically
   (`_DEFAULT_REGIMES`, overridable via `STAGE6B_GROUP_REGIMES`). **Before this patch the
   auto-build would have OOM'd.** If you build groups by hand, pass
   `--regimes occasional random`.
2. **The KG dir is `kg_raw`, not `kg_denoised`** — and that is the finding, not an oversight.
   GBSR is a measured no-op on this graph at every bottleneck strength tested; the denoised arm
   is built and diffed at `groups_social_denoised/` and every count moves by ≤ 0.21%
   (`LLMGPR_GOWALLA.md` §5). Cite it as the control.
3. **4 alignment relations, not 2** — `FOLLOWED_BY`, `IS_NEAR_TO`, `SAME_TAXONOMY_L2`,
   `SAME_TAXONOMY_L3` at 40,000 each (capped). The FSQ arm had only the two flat ones, which its
   handoff flagged as leaving "the TransE term with no hierarchical signal of its own". Fixed here.
4. **D1 ρ = +0.8683 STRONG** against FSQ's +0.3245. `IS_NEAR_TO` covers 100% of POIs here
   (FSQ: 43%), so if the hyperbolic-vs-random ablation comes out flat, the weak-hierarchy
   explanation the FSQ handoff offered does **not** apply — look elsewhere.

## Open item, stated plainly

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
