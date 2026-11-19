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
| RefH embeddings (Foursquare's winning KGE) | `data/llmgpr/kg_denoised/poi_hyperbolic_embs_LLMGPR_refh.npy` | copied from `llmgpr-pipeline` @ `d8b3f48`; same KG, byte-identical inputs |

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

Offline check (`python src/user_tokens.py --diagnose`, no LLM): pulling users towards their real
ties moves them +35% MRR closer to their next new POI at beta 2. Shuffled ties give −12%. The
gain is largest for low-history users.

## Running stage 5

```bash
STAGE6B_USER_TOKENS=1 STAGE6B_SOCIAL_SMOOTHING=both   # + STAGE6B_SOCIAL_BETA (default 2.0)
STAGE6B_RUN_PROFILE=smoke                             # full | probe | smoke (full needs >= 40 GB)
```

The ablation grid is `STAGE6B_USER_TOKENS=0`, then `1` with `STAGE6B_SOCIAL_SMOOTHING` = `none` /
`friends` / `copresence` / `both`.

## Not done yet

- **No GPU run of the user tokens yet.** The first Kaggle smoke run is pending.
- **The group model is not trained on groups** (`TRAIN_ON_GROUPS=False`). The main-results group
  rule is undecided: one strict rule gives about 210 Foursquare groups on this subset. That
  question is with the supervisor.
- **Gowalla and Weeplace social graphs** wait until Foursquare shows a gain.
