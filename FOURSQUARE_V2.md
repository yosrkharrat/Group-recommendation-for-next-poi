# Foursquare run guide — `foursquare-v3`

Run from the repository root. Commands use Bash on the GPU server.
Use CUDA-enabled PyTorch; the full fine-tune requires at least 40 GB VRAM.

**Status:** CPU artifacts rebuilt and checked. KGE comparison, embedding retraining and
fine-tuning under the corrected windows are pending. Existing embeddings are stale.

## 1. Dependencies and data preparation

Use the committed CSVs and matched groups. Exclusion windows: three-city **±24 hours**;
whole-history **±1 hour**, read automatically from their manifests.

```bash
python -m pip install -r requirements.txt jupyterlab
python -m unittest discover -s tests -p test_group_outing_guard.py -v
python src/build_kg_lbsn.py --csv-dir data/llmgpr --dataset LLMGPR --out-dir data/llmgpr/kg_poi --poi-only --drop-group-outings data/groups_llmgpr_match/LLMGPR_3CITY data/groups_llmgpr_match/LLMGPR
python src/build_poi_poi_triples.py --kg-dir data/llmgpr/kg_poi --meta data/llmgpr/poi_metadata_LLMGPR.csv --out-dir data/llmgpr/kg_poi --dataset LLMGPR
python src/build_social_graph.py
python src/build_group_examples_llmgpr_match.py --guard-sets data/groups_llmgpr_match/LLMGPR_3CITY data/groups_llmgpr_match/LLMGPR --train-sets data/groups_llmgpr_match/LLMGPR_3CITY
```

Expected: **2,434** train check-ins excluded; **398,761** POI-KG triples;
**10,792** social edges; **6,343 / 1,247 / 1,251** group train/validation/test examples.

The committed graph, alignment triples and social graph are already the output of these commands,
and the stage-5 checks accept them on Windows, Linux and macOS (text files are fingerprinted with
normalised line endings). Re-running them is safe, but the graph builder is not bit-reproducible
across machines (same counts, a few dozen different triples), so run steps 2 to 4 on the same copy
of `data/llmgpr/kg_poi/` and do not `git checkout` those files in between.

## 2. KGE comparison — GPU

```bash
mkdir -p outputs/foursquare_kge
for model in roth refh atth murp; do
  for seed in 42 43 44; do
    python src/kge_bench.py --kg-dir data/llmgpr/kg_poi --data-dir data/llmgpr --dataset LLMGPR --model "$model" --seed "$seed" --split-seed 42 --epochs 150 --patience 10 --device cuda --save-embs --out "outputs/foursquare_kge/${model}_s${seed}.json"
  done
done
python src/kge_alignment_eval.py --runs 'outputs/foursquare_kge/*_s4?.json' --kg-dir data/llmgpr/kg_poi --dataset LLMGPR --meta data/llmgpr/poi_metadata_LLMGPR.csv --llm-stats data/llm/llada_moe_embed_stats.npz --device cuda --out outputs/foursquare_kge/stage2
```

## 3. Install the selected embeddings

Uses the existing Stage-2 rule: winning model, then its highest-purity passing seed.
The embedding and its provenance sidecar must travel together.

```bash
python - <<'PY'
import json, shutil
from pathlib import Path
root = Path("outputs/foursquare_kge")
result = json.loads((root / "stage2.json").read_text())
model = result["decision"]["carry_forward"]
assert model is not None, "No model passed selection"
run = max((r for r in result["per_run"] if r["model"] == model and r["d1_pass"]),
          key=lambda r: r["purity_aligned_at10"])["run"]
source = root / f"{run}_poi_embs.npy"
target = Path(f"data/llmgpr/kg_poi/poi_hyperbolic_embs_LLMGPR_{model}.npy")
shutil.copy2(source, target)
shutil.copy2(str(source) + ".manifest.json", str(target) + ".manifest.json")
(root / "selected_model.txt").write_text(model)
print("Selected:", run, "→", target)
PY
```

## 4. Fine-tune — smoke first, then full

```bash
export STAGE6B_DATASET=LLMGPR
export STAGE6B_KGE_MODEL="$(cat outputs/foursquare_kge/selected_model.txt)"
export STAGE6B_USER_TOKENS=1
export STAGE6B_SOCIAL_SMOOTHING=copresence
export STAGE6B_GROUP_OVERLAP=remove
export STAGE6B_GROUP_SET=llmgpr_match
export STAGE6B_TRAIN_ON_GROUPS=1
export STAGE6B_GROUP_TRAIN_FRAC=all
export STAGE6B_RUN_PROFILE=smoke
export STAGE6B_OUT_DIR=./outputs/foursquare_window_smoke
jupyter lab notebooks/stage5_lbsn_finetune.ipynb
```

Run all cells. Batch size is left to the notebook: on a GPU of 40 GB or more it uses
`BATCH_SIZE=16`, `GRAD_ACCUM=2`. Only if the long group prompts (up to about 3,900 tokens) run
out of memory, set e.g. `STAGE6B_BATCH_SIZE=8` and `STAGE6B_GRAD_ACCUM=4`, keeping the product
at 32. `STAGE6B_BATCH_SIZE=1` is only for a 16 GB card, where the `full` profile is refused anyway;
at batch 1 the full run takes 16 times as many steps per epoch.

After smoke passes, stop Jupyter and relaunch:

```bash
export STAGE6B_RUN_PROFILE=full
export STAGE6B_OUT_DIR=./outputs/foursquare_window_full
jupyter lab notebooks/stage5_lbsn_finetune.ipynb
```

Use new output directories; do not resume old-graph checkpoints.
Save the run directory, selected KGE results, embedding and sidecar.
User tokens are regenerated automatically; no separate user-KGE training is needed.

---

# `foursquare-v3`: social graph, user tokens and group training (Foursquare / LLMGPR)

*Branch `foursquare-v3` continues `foursquare-v2` from commit `3c0fa02`: the same pipeline plus the
matched-group examples, training on groups and the two-way overlap guard. The file keeps its name.*

Branched from `llmgpr-gowalla` on 2026-11-19. Stage 5 defaults to Foursquare here
(`STAGE6B_DATASET=LLMGPR`).

Design decisions (2026-11-19):

1. The social graph is a weighted user–user **adjacency**, not a second embedding. Members
   already live in the POI space through their signatures.
2. Co-presence comes from train check-ins with no friendship requirement, and `FRIEND_OF` uses old
   friendships only. The LLMGPR-matched groups stay a reproduction set for the comparison tables.
3. Affinity (`SIMILAR_TO`) is left out of v1, since it is partly circular. It is an ablation later.
4. Users enter the LLM as **user tokens**. A `<grp>` consensus token, if kept, is an ablation built
   from the same vectors.
5. Main results use one group rule across datasets (still to be fixed; see below).

## What is new

| Piece | Where | Status |
|---|---|---|
| Weighted user–user social graph | `src/build_social_graph.py` → `data/social_graph/LLMGPR/` | built, audited |
| User tokens + social smoothing | `src/user_tokens.py`, stage 5 section 6c | built, CPU-tested; opt-in |
| Three-city reproduction groups | `data/groups_llmgpr_match/LLMGPR_3CITY/` | built (match 0.836) |
| Extended POI catalogue | `data/llmgpr/poi_metadata_LLMGPR.csv` (15,315 POIs) | built |
| POI-only KG + KGE comparison | `data/llmgpr/kg_poi/` | **MuRP** wins on the new KG (2026-11-20) |

### Catalogue extension for the LLMGPR comparison (option B)
The first LLMGPR-matched Foursquare groups were built from whole check-in histories. 54% of their
test targets were venues outside our 14,402-POI catalogue, so the model could not output them, and
75% of those venues were outside NY/LA/Chicago. Extending the catalogue to every venue of those
histories would mean 430k POIs, which is too large. Instead:

1. **Groups rebuilt on three-city check-ins only** (what LLMGPR §4.1 says its data contains):
   `python src/build_llmgpr_match_groups.py --datasets LLMGPR_3CITY`. The best three-city reading
   in the sweep is anchor + friends within ±1 day, with ≥2 group check-ins. That gives 1,251 groups,
   9,575 group check-ins, 7.65 check-ins and 3.33 members per group (LLMGPR: 1,715 / 12,594 / 7.34
   / 3.72), match 0.836. The whole-history set (0.895) stays in `data/groups_llmgpr_match/LLMGPR/`
   for the record.
2. **Catalogue extended** (`python src/extend_llmgpr_catalogue.py`). It appends the 913 venues those
   groups use but the catalogue lacked: ids 14402–15314, so old ids are unchanged, and categories
   and localities follow the same conventions. It also fills coordinates for every POI (43% → 100%)
   from the raw venue file, which agrees exactly where both exist. Every group test target is now a
   POI the model can output.
3. **POI-only KG** (`python src/build_kg_lbsn.py --csv-dir ./data/llmgpr --dataset LLMGPR --out-dir
   ./data/llmgpr/kg_poi --poi-only`, then `build_poi_poi_triples.py` for the alignment triples).
   It has 15,695 entities and 400,163 triples (IS_NEAR_TO 191,576 now that every POI has
   coordinates). On the old catalogue the same builder reproduces the earlier POI-only KG exactly
   (14,774 / 283,420).
4. **RefH retrained on it** with the shipped RefH's exact config (dim 64, 150 epochs, seed 42, on a
   CUDA GPU): MRR 0.1525, **D1 +0.678 STRONG and monotone** (the shipped RefH had +0.320).
   `data/llmgpr/kg_denoised/` (14,402-row embeddings of the old user/group KG) is removed from this
   branch, since its files would also clash with these names in stage 5's `find()`.
5. **Four-model comparison on this KG** (2026-11-20, the supervisor's two-stage selection with MuRP
   added at the reviewer's suggestion). `src/kge_bench.py` trains RotH / RefH / AttH / MuRP on one
   leak-free split, seeds 42–44. `src/kge_alignment_eval.py` then runs Stage 2: the stage-5 alignment
   adapter and the per-dimension affine to LLaDA-MoE's embedding statistics, scored by leaf-category
   purity@10 (random = 0.014). Three-seed means:

   | Model | Test MRR | D1 | purity_aligned@10 |
   |---|---|---|---|
   | **MuRP** | **0.2305** | +0.498 | **0.0927** |
   | RefH | 0.2023 | +0.454 | 0.0654 |
   | RotH | 0.1774 | +0.487 | 0.0670 |
   | AttH | 0.1768 | +0.386 | 0.0605 |

   All pass the D1 gate. MuRP wins both stages, so stage 5 now uses it (`KGE_WINNER["LLMGPR"] =
   "murp"`). The shipped file is seed 43, the best Stage-2 seed: MRR 0.2189, D1 +0.488, purity 0.111.
   MuRP seed 42 trained to epoch 144 and lands at purity 0.061: longer training buys MRR (0.252) and
   costs category structure. The previous RefH file (150 epochs) scores purity 0.068, the same as the
   benchmark RefH. All per-run results are in `data/llmgpr/kg_poi/kge_comparison/`;
   `STAGE6B_KGE_MODEL=refh` re-runs the old embedding.
6. **KG rebuilt without the held-out group outings, MuRP retrained on it** (2026-11-22). The graph
   in item 3 drew its `FOLLOWED_BY` edges from every individual train check-in, including the
   2,220 that are a held-out group outing. `build_kg_lbsn.py --drop-group-outings
   ./data/groups_llmgpr_match/LLMGPR_3CITY ./data/groups_llmgpr_match/LLMGPR` (guard ±120 min, the
   stage-5 rule) drops them first: same 15,695 entities and ids, 398,850 triples (`FOLLOWED_BY`
   177,599 → 176,286). `data/llmgpr/kg_poi/` now holds this graph. The four-model comparison was
   re-run on it unchanged (same job, seeds 42–44). Three-seed means, before → after:

   | Model | Test MRR | D1 | purity_aligned@10 |
   |---|---|---|---|
   | **MuRP** | 0.2305 → **0.2283** | +0.498 → +0.498 | 0.0927 → **0.0957** |
   | RefH | 0.2023 → 0.1987 | +0.454 → +0.454 | 0.0654 → 0.0653 |
   | RotH | 0.1774 → 0.1809 | +0.487 → +0.501 | 0.0670 → 0.0666 |
   | AttH | 0.1768 → 0.1749 | +0.386 → +0.389 | 0.0605 → 0.0603 |

   Every shift is within the seed-to-seed spread; all twelve runs pass the D1 gate and MuRP still
   wins both stages. The shipped `poi_hyperbolic_embs_LLMGPR_murp.npy` is now **seed 44 of this
   run**, its best Stage-2 seed (MRR 0.2194, D1 +0.492, purity 0.107; seed 43 falls to purity 0.077
   here). Stage-5 numbers produced with the earlier seed-43 file are not comparable with later
   ones. Per-run results: `data/llmgpr/kg_poi/kge_comparison/`; the item-5 tables are kept in
   `data/llmgpr/kg_poi/kge_comparison_before_outing_removal/`. The RefH files in `kg_poi/`
   (`*_refh.npy`, `refh_best.pt`) are still the ones trained on the item-3 graph.

The new 913 POIs have no individual check-ins in the 10 km pipeline data. They are learned through
the KG (category, location, proximity) and the group sequences.

### Social graph
Two relations, both from **train-split** check-ins only:

- `FRIEND_OF`: 8,640 pairs from `friendship_old` only, weight 1. `friendship_new_only` postdates
  the check-ins and is asserted absent.
- `CO_PRESENT`: 2,156 pairs met at the same venue within 60 min at least twice. There is no
  friendship requirement. Each meeting is weighted by venue rarity (idf). That weighting was
  chosen on validation data: within-edge AUC 0.831 vs 0.771 for a plain count.

The two relations barely overlap (189 pairs). The validation, test and masked outings of both
reproduction sets (`data/groups_llmgpr_match/LLMGPR_3CITY`, which the comparison evaluates on, and
the whole-history `LLMGPR`) were 2,220 of our train check-ins. They are removed before co-presence
is computed, and the build asserts that none remain. (Until 2026-11-20 only the whole-history set
was guarded, which left 830 check-ins of three-city evaluation outings in.)
Rebuild: `python src/build_social_graph.py` (~40 s; `--self-check` first).

### User tokens
Each user becomes one `<user_i>` token in the POI token space:

1. The gyromidpoint of their top-10 train POIs.
2. Optionally pulled towards their ties (one hop, neighbours' mass `beta`).
3. The supervisor's trained alignment adapter and the **same** affine as `W_POI`.

The notebook asserts that POI points pushed through the user path reproduce `W_POI`. Prompts
gain `[user] <user_i>` and `[member N] <user_i>`.

Offline check (`python src/user_tokens.py --diagnose`, no LLM). On the MuRP embeddings, every
relation moves users closer to their next new POI and shuffled ties move them away, so the gain is
social. **Co-presence** gains most per moved user (+30% MRR at beta 1); friendship helps a little;
`both` has the best all-user MRR. On RefH, friendship hurt, so the stage 5 ablation decides.
Recommended: `STAGE6B_SOCIAL_SMOOTHING=copresence`.

### Group-outing overlap (`STAGE6B_GROUP_OVERLAP`)
A group check-in is built from its members' own check-ins. A group's held-out outing is therefore
often also a member's individual **train** check-in: 1,401 train rows are a val/test outing of a
three-city group (same user, same venue, within 120 min). Without a guard, the model trains on the
answer for about half the group evaluation. LLMGPR's protocol, as described, splits user and group
sequences separately, so it does not separate the two either.

- `remove` (default for LLMGPR): stage 5 drops those rows from `train_df` before anything is built
  from it (targets, histories, profiles, friend favourites, user tokens, the visited sets of
  section 11c). The rule and sets are the social graph's: val/test/masked outings of
  `LLMGPR_3CITY` and `LLMGPR`, ±120 min, which is 2,220 of 299,451 rows. An assert checks that none
  are left.

  It also covers the other direction, which matters once groups are trained on: 711 of the 6,413
  three-city group **train** outings are a member's individual val or test check-in. Group training
  data is the scarce side, so those outings stay, and the matching individual rows leave the
  individual evaluation instead: 302 of 42,005 val rows and 247 of 81,920 test rows
  (`GROUP_TRAIN_SETS`, same ±120 min rule, asserted). This is done whether or not groups are
  trained on, so the individual test set is identical in both arms of that ablation.
- `keep`: all three splits as shipped, i.e. LLMGPR's protocol as described.

`probe` subsamples train by a seeded hash of (user, check-in time), not by position. The `remove`
sample is then exactly the `keep` sample minus the removed rows (14,482 vs 14,367 examples), so the
two runs differ only by the overlap.

### Group examples from the matched groups (`STAGE6B_GROUP_SET`)
`src/build_group_examples_llmgpr_match.py` turns the three-city group files into the group examples
stage 5 reads (`<group dir>/examples_<overlap>/group_examples_{train,val,test}.jsonl` + a manifest).
Stage 5 builds them itself when they are missing or were built with other settings (about 15 s).

- **One example per group outing.** The target is the outing's venue, mapped to the pipeline
  `poi_idx` by `venue_id`. The split is the group file's leave-one-out: the last outing is test,
  the second last is validation, the rest is training. `masked` rows are never used.
- **Members are the users present at that outing**, so every example is a real joint outing of
  friends. 93% of outings have two people present; the largest has eight.
- **Each member** gets a causal profile and their 30 most recent check-ins before the outing.
- **The group's own earlier outings** are added as a `[group history]` line in the prompt (up to
  30). This is LLMGPR's group sequence.
- **`remove`** additionally keeps a member's own check-in at the target venue, within 120 min
  before the outing, out of that example's history and profile. `keep` removes nothing.

| | train | val | test |
|---|---|---|---|
| `remove` | 6,351 | 1,248 | 1,251 |
| `keep` | 6,369 | 1,248 | 1,251 |

A few outings are skipped because no member present has any earlier check-in. The script's
self-check re-reads the written files against the raw inputs: every history item is before its
outing, no target copy sits inside the guard, the group history is exactly the group's earlier
outings, and one validation and one test example exist per group.

Prompt length with the LLaDA tokenizer (`remove`): median about 1,800 tokens, 99th percentile
about 2,800, maximum 3,934 against the 4,085 cap. Groups with six members shown get 20 check-ins
per member instead of 30: at 30 one test prompt reached 4,283 tokens and would have lost its tail.

Stage 5 then:
- trains on users **and** groups by default on this set (`STAGE6B_TRAIN_ON_GROUPS=1`, all group
  training examples; `STAGE6B_GROUP_TRAIN_FRAC` takes `all` or a fraction of the individual count);
- evaluates section 11b on the matched validation and test examples;
- in section 11c (LLMGPR's protocol) keeps every test example, since each is already its group's
  last outing, and excludes the group's earlier outings from the 500 candidates as well as the
  members' histories.

`STAGE6B_GROUP_SET=social` restores the older 180-min / 5-hop set; its prompts are unchanged.

## Running stage 5

```bash
STAGE6B_USER_TOKENS=1 STAGE6B_SOCIAL_SMOOTHING=copresence   # + STAGE6B_SOCIAL_BETA (default 1.0)
STAGE6B_GROUP_OVERLAP=remove                          # remove | keep (LLMGPR's protocol)
STAGE6B_GROUP_SET=llmgpr_match                        # llmgpr_match | social
STAGE6B_TRAIN_ON_GROUPS=1                             # 1 | 0 (0 = the model never sees a group prompt)
STAGE6B_RUN_PROFILE=smoke                             # full | probe | smoke (full needs >= 40 GB)
```

The ablation grid is `STAGE6B_TRAIN_ON_GROUPS` = `0` / `1`, `STAGE6B_USER_TOKENS=0`, then `1` with
`STAGE6B_SOCIAL_SMOOTHING` = `none` / `friends` / `copresence` / `both`. The comparison with LLMGPR
reports `STAGE6B_GROUP_OVERLAP` = `keep` and `remove`.

## Status (2026-11-22)

- **Smoke run passed end to end on a 16 GB Kaggle GPU** (this branch at `94584be`: leak-free KG,
  MuRP seed 44, matched groups, `TRAIN_ON_GROUPS=1`, user tokens with `copresence` smoothing,
  `BATCH_SIZE=1`, `GRAD_ACCUM=8`). It trains on 300 examples for one epoch, so its numbers only
  show that every stage runs. About 35 min of training, then about 3 h for section 11c, which
  scores all 6,646 individual users in every profile.
- **The `full` profile has not been run with group training.** It needs a GPU of 40 GB or more
  (the notebook stops otherwise). Group prompts reach about 3,900 tokens against 530 for
  individual ones, so the default `BATCH_SIZE` may not fit: lower it with `STAGE6B_BATCH_SIZE`
  and raise `STAGE6B_GRAD_ACCUM` to keep the effective batch.

## Not done yet

- **Ablations and baselines** (the grid above, and the baselines under the same protocol).
- **Time order.** 698 of the 1,251 group test outings fall before the end of some member's
  individual training history; this follows from the per-user chronological split and is to be
  reported, with the 553 remaining outings as a robustness subset.
- **Gowalla and Weeplace social graphs** wait until Foursquare shows a gain.
