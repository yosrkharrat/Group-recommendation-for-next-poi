# `foursquare-v2`: social graph + user tokens (Foursquare / LLMGPR)

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

The new 913 POIs have no individual check-ins in the 10 km pipeline data. They are learned through
the KG (category, location, proximity) and the group sequences.

### Social graph
Two relations, both from **train-split** check-ins only:

- `FRIEND_OF`: 8,640 pairs from `friendship_old` only, weight 1. `friendship_new_only` postdates
  the check-ins and is asserted absent.
- `CO_PRESENT`: 2,203 pairs met at the same venue within 60 min at least twice. There is no
  friendship requirement. Each meeting is weighted by venue rarity (idf). That weighting was
  chosen on validation data: within-edge AUC 0.830 vs 0.771 for a plain count.

The two relations barely overlap (201 pairs). The validation and test outings of the LLMGPR
reproduction set (`data/groups_llmgpr_match/LLMGPR`) were 1,134 of our train check-ins. They are
removed before co-presence is computed, and the build asserts that none remain.
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
social. **Co-presence** gains most per moved user (+25% MRR at beta 1); friendship helps a little;
`both` has the best all-user MRR. On RefH, friendship hurt, so the stage 5 ablation decides.
Recommended: `STAGE6B_SOCIAL_SMOOTHING=copresence`.

## Running stage 5

```bash
STAGE6B_USER_TOKENS=1 STAGE6B_SOCIAL_SMOOTHING=copresence   # + STAGE6B_SOCIAL_BETA (default 1.0)
STAGE6B_RUN_PROFILE=smoke                             # full | probe | smoke (full needs >= 40 GB)
```

The ablation grid is `STAGE6B_USER_TOKENS=0`, then `1` with `STAGE6B_SOCIAL_SMOOTHING` = `none` /
`friends` / `copresence` / `both`.

## Not done yet

- **No GPU run on the extended catalogue yet.** The first Kaggle smoke run (old catalogue, user
  tokens + smoothing `both`) completed on 2026-11-19.
- **Stage 5 does not yet evaluate on the three-city reproduction groups** (it still uses
  `groups_social`). Join them on `venue_id`, because their `poi_idx` is the house8 id space.
- **The group model is not trained on groups** (`TRAIN_ON_GROUPS=False`). The main-results group
  rule is undecided: one strict rule gives about 210 Foursquare groups on this subset. That
  question is with the supervisor.
- **Gowalla and Weeplace social graphs** wait until Foursquare shows a gain.
