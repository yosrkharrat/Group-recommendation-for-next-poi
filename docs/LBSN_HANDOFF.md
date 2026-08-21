# LBSN track handoff — for the fine-tune adaptation and execution

Audience: whoever adapts `notebooks/stage6b_run2_server.ipynb` to the new dataset (prompts,
config), and whoever executes the fine-tune on a GPU. Everything upstream of stage 5 is done,
verified, and committed; nothing below requires re-running anything unless you want to.

## What exists, and the two headline numbers

The LBSN2Vec++ social dataset (Foursquare global check-ins + real friendships), NYC, canonical
population `LBSN_NYC`: **159,304 check-ins / 1,665 users / 6,103 POIs**, real timestamps,
coordinates, and category names, per-user chronological 70/10/20 splits.

- **KG (stage 2):** 12,709 entities / 244,080 triples / 12 relations — everything the TSMC KG
  had **plus `FRIEND_OF`** (3,012 directed edges from the *before-period* friendship snapshot).
- **Hyperbolic embeddings (stage 3):** `data/kg_lbsn/poi_hyperbolic_embs_LBSN_NYC.npy`
  (6103 × 64, rows in `poi_idx` order). **D1 ρ = +0.7096 STRONG, radii fully monotone**
  over taxonomy depths 1–4 (0.736 / 0.747 / 0.974 / 1.321). ρ is tie-capped below TSMC's
  +0.85 because 69% of POIs sit at depth 2 — expected, not a deficiency. Full provenance
  (config, per-relation MRR, D1) in `data/kg_lbsn/roth_results.json`.

## File map

| File | What it is |
|---|---|
| `data/lbsn/train/val/test_LBSN_NYC.csv` | check-in splits, exact TSMC column layout (see caveat 2) |
| `data/lbsn/poi_metadata_LBSN_NYC.csv` | poi_idx 0–6102, venue hex id, **full taxonomy path**, lat/lon |
| `data/lbsn/friendship_old_LBSN_NYC.csv` | 1,506 edges, 0-based user ids — the only friendship file training may touch |
| `data/lbsn/friendship_new_only_LBSN_NYC.csv` | 1,138 pairs — **EVAL ONLY, never in training data or prompts** (see caveat 1) |
| `data/lbsn/users_LBSN_NYC.csv` | user_id → raw anonymised id, for joins back to the dump |
| `data/kg_lbsn/poi_hyperbolic_embs_LBSN_NYC.npy` | `EMB_FILE` (6103 × 64) |
| `data/kg_lbsn/poi_poi_triples_LBSN_NYC.pt` + `poi_relation_vocab_LBSN_NYC.json` | §6b `ALIGN_TRIPLES_FILE` / `ALIGN_RELVOCAB_FILE` (132,394 triples, 5 relations) |
| `data/kg_lbsn/roth_best.pt`, `roth_results.json` | checkpoint + provenance |
| `data/lbsn/groups/` | stage-1 outputs; the `group_examples_*.jsonl` are gitignored — regenerate in ~10 min (command in `.gitignore`) |

Id spaces: `user_id` = rank of the raw anonymised id (0–1664); `poi_idx` = rank of the venue
hex id (0–6102). Both contiguous, both recorded in the exported CSVs. Every artifact above
uses these spaces consistently.

## Adapting `stage6b_run2_server.ipynb` — the exact touchpoints

1. `DATASET = "LBSN_NYC"` — this alone repoints `train_{DATASET}.csv`,
   `poi_metadata_{DATASET}.csv`, and `EMB_FILE = poi_hyperbolic_embs_{DATASET}.npy`.
   `N_POI` is derived from the metadata (→ 6,103) and the emb-rows assert will pass.
2. **Two hardcoded names** don't follow `DATASET`:
   `ALIGN_TRIPLES_FILE = "poi_poi_triples_NYC.pt"` and
   `ALIGN_RELVOCAB_FILE = "poi_relation_vocab_NYC.json"` → change to `_LBSN_NYC`.
   Note `ALIGN_NUM_RELATIONS` will become 5 (was 6) — it is read from the vocab, so no edit.
3. **`vocab.pkl` is loaded and there is no LBSN_NYC vocab.pkl.** The old file belongs to the
   5,120-POI token space. The `<poi_0>`…`<poi_6102>` tokens themselves are generated from
   `N_POI` in-notebook; whatever else vocab.pkl feeds (inspect its use before deciding) must
   be regenerated for 6,103 POIs or dropped. This is the one genuinely open adaptation item.
4. `venue_category_id` **is an empty column** in the new CSVs (the raw dump has no category
   ids). `venue_category_name` is real and populated — if the old prompts used the id, switch
   them to the name; the name is also better prompt material.
5. Prompt opportunity, not obligation: the new dataset's whole point is the social layer.
   `friendship_old_LBSN_NYC.csv` can inject "friends of this user" context into prompts.
   If you do: friends from `friendship_old` ONLY (see caveat 1).

## Two rules that protect the results

1. **Friendship leakage.** `friendship_new_only` pairs are friendships formed *during/after*
   the check-in period — they are the eval set of the friendship-prediction task and partly an
   *effect* of the very co-visits the model trains on. They are asserted absent from the KG at
   build time; keep them equally absent from fine-tune inputs and prompts.
2. **Split discipline.** The CSVs are already split per-user chronologically (same
   `resplit_per_user` as the whole pipeline). Do not re-split; use the files as they are so
   the group examples, the KG's train-only relations, and the fine-tune all agree on what
   "train" means.

## Executing the fine-tune (supervisor)

Same flow as the TSMC run of `stage6b_run2_server.ipynb`: attach the data files + the two
align files + `EMB_FILE` where the notebook's `find()` can see them, GPU required, and do a
`SMOKE_TEST = True` pass first — it exists precisely to catch config drift in minutes instead
of hours. The D1 gate inside the notebook re-checks the embeddings at load time; with this
file it will report ρ = +0.71, monotonic — that is the expected PASS state for LBSN_NYC
(the ≥ +0.8 seen on TSMC is not the bar here; monotone + STRONG is).

## Before/alongside the fine-tune (recommended order)

1. **Score-aggregation group baselines** (AVG / least-misery / most-pleasure) on the existing
   checkpoint — no training, and it is the number the fine-tune must beat.
2. The `--depth-weight 0` **control run** for LBSN_NYC is still missing:
   `notebooks/roth_lbsn_kaggle.ipynb` with `RUN_CONTROL = True` produces it unattended.

## Reproducing anything from zero

```bash
python src/prepare_lbsn_csvs.py --zip lsbn2vec_global.zip          # stage 0 (zip: gdown id in the script)
python src/build_groups.py  --data-dir ./data/lbsn --dataset LBSN_NYC \
                            --out-dir ./data/lbsn/groups --no-resplit   # stage 1
python src/build_kg_lbsn.py --csv-dir ./data/lbsn --groups-dir ./data/lbsn/groups  # stage 2
python src/train_roth.py    --kg-dir ./data/kg_lbsn --data-dir ./data/lbsn \
                            --dataset LBSN_NYC --epochs 120 --log-every 10 --max-eval 4000 \
                            --depth-weight 5.0 --depth-margin 0.3 --root-pull 0.01  # stage 3, ~5 h CPU
python src/build_poi_poi_triples.py --kg-dir ./data/kg_lbsn \
                            --meta ./data/lbsn/poi_metadata_LBSN_NYC.csv \
                            --out-dir ./data/kg_lbsn --dataset LBSN_NYC --derive taxonomy \
                            --max-per-relation 40000                # stage 4
```

Or run `notebooks/roth_lbsn_kaggle.ipynb` on a CPU Kaggle kernel — it does all of the above
with self-locating inputs and prints expected-vs-actual counts at every stage. Every script
has `--self-check` (61 checks total across the track; expected state is 0 failures).
