# Comparison-protocol evaluation — run sheet

**One setting.** Open `notebooks/stage5_lbsn_finetune.ipynb` on the dataset's branch, and in
the config cell (§1) set

```python
EVAL_CKPT = "<the trained checkpoint>"
```

then run the notebook top to bottom. That is all. `EVAL_CKPT` is either

* a local folder holding `adapter_model.safetensors` + `poi_head.pt` — the run's `best/` (lowest
  validation loss) or `latest/` folder, e.g. `"/kaggle/input/<dataset>/best"`, or
* a Hugging Face repo id, e.g. `"yosrr12/llada-moe-run2-LLMGPR-hyperbolic-ckpt"` — its `best/`
  subfolder is downloaded (`EVAL_CKPT_TAG = "latest"` for the other one). A private repo needs
  `HF_TOKEN` in the environment.

It can also be passed without editing the notebook: `STAGE6B_EVAL_CKPT=<path or repo id>`.

With `EVAL_CKPT` set, the notebook derives everything else: the checkpoint is loaded through the
manual-resume path in §10, the training loop is skipped (resume epoch > `EPOCHS`), the ≥40 GB
GPU guard is bypassed (evaluation runs on a 16 GB card), the full validation and test splits are
used, and sections **11**, **11b** and **11c** run. Leave `EVAL_CKPT = None` to train as before.

| branch | dataset | what runs |
|---|---|---|
| `llmgpr-pipeline` | Foursquare | already trained → `EVAL_CKPT` = that checkpoint, run all |
| `llmgpr-gowalla` | Gowalla | not trained yet → train as usual; §11c runs at the end by itself |
| `llmgpr-Weeplace` | Weeplace | same as Gowalla |

Everything needed is committed on each of those branches (`src/eval_protocol.py`, the notebook's
§11c and `EVAL_CKPT` switch, `notebooks/eval_comparison_protocol.py`). The Weeplace notebook
fetches `src/` from `llmgpr-pipeline`, where the module is also committed.

---

## What comes out

§11c prints two lines per task:

```
[11c] individual comparison protocol           : HR@5=…  NDCG@5=…  HR@10=…  NDCG@10=…  (n=…)
[11c] individual full catalogue, same examples : HR@5=…  NDCG@5=…  HR@10=…  NDCG@10=…  (n=…)
[11c] group      comparison protocol           : …
[11c] group      full catalogue, same examples : …
```

* **`comparison protocol`** — the cells for the paper's comparison tables (beside LLMGPR's
  published rows).
* **`full catalogue, same examples`** — a control on exactly those examples; it is lower by
  construction (the stricter protocol) and does **not** go in those tables.

Files to send back, from `OUT_DIR` (default `./outputs`):
`comparison_protocol_results_<DATASET>_<condition>.json`,
`comparison_protocol_candidates_<DATASET>.npz`, and the printed output of §11 and §11c. The json
carries the candidate-set statistics (`frac_examples_filled`, `n_anchor_no_coord`) the paper has
to disclose.

Cost: two forward passes over ~1,000 individual and ~30,000 group examples — well under one
training epoch on the same GPU.

## The one check that matters

§11 is the **control**. On the Foursquare checkpoint it must reproduce the full-catalogue numbers
already reported for it (individual HR@5 ≈ 0.467 / HR@10 ≈ 0.537; group HR@5 ≈ 0.317 /
HR@10 ≈ 0.390 — §11 prints them as `Acc@k`, the same quantity). If it does not, the session is
not scoring with the weights the heads were trained against — most likely §6b's in-session
alignment produced a different frozen `W_POI` — and **§11c's numbers must not be reported**.
Send the §11 output either way.

---

## What the protocol does, exactly (so the numbers can be defended)

| LLMGPR §4.1 says | what runs |
|---|---|
| leave-one-out: last check-in = test, second-last = val | test example = each user's **last** check-in; for the group task, **one example per member set, the chronologically last**. The checkpoint is **not** retrained on all-but-two: it is the 70%-trained one, so it saw *less* data than LLMGPR's — the conservative direction, stated in the paper. |
| max sequence length 200 | no-op: the prompt budget already caps history far below 200. |
| ground truth + 500 **unvisited** POIs **nearest to it** within the **same region** | unvisited = not in the example's own input history (user's prefix / group's joint history); nearest = to the **ground-truth POI**, great-circle distance; region = the city (`locality`; TSMC NYC is one metropolitan region). Sets are seeded → deterministic and saved. |
| HR@k, NDCG@k | exactly those, k = 5 and 10, one relevant item per example; strict-greater rank, the same tie rule as §11. |

Every decision above, with its reasoning, is in the docstring of `src/eval_protocol.py`;
`python src/eval_protocol.py --self-check` must print `ALL PASS (28/28)` (§11c runs it itself
and stops if it does not).

## Known hazards

- **Coordinates.** On the 3-city Foursquare export (`LLMGPR`, 14,402 POIs) only **43%** of
  visited POIs have lat/lon, so **~56%** of candidate sets fall back to seeded-random same-region
  negatives — not "nearest". §11c warns and the json records the fraction; a column built like
  that must not be labelled as LLMGPR's protocol in the paper. TSMC FSQ-NYC (5,120) and Gowalla
  have 100% coverage and never take that path.
- **Config must match the training run** (`SCORING_MODE`, `N_MASKS`, `EMB_FILE`,
  `USE_CURVATURE_ALIGNMENT`, `HIST_LEN`, `PROFILE_TOP_K`, `MAX_LEN`, groups dir). A mismatch
  fails loudly when the heads load — that is intended, not something to work around.
- **A stale `OUT_DIR/ckpt_latest`** from an earlier session cannot be picked up instead of
  `EVAL_CKPT`: the switch forces `RESUME = False`.

## Still to confirm for the paper's implementation-details section

1. Catalogue of the reported Foursquare run: TSMC FSQ-NYC (5,120 POIs) or the 3-city export
   restricted to New York (14,402)?
2. Training fraction: full split, or the 5% fast-probe subsample?
3. `MAX_LEN`: 1,024 or 4,096?
