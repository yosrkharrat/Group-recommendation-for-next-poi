# Samples — Gowalla *social* groups (raw arm)

The first 200 records of each split, committed so the record schema is readable without the
~146 MB of full files. The full `group_examples_*.jsonl` are gitignored and regenerate
deterministically (seed 42, ~10 min wall):

    python src/build_groups.py --data-dir ./data/gowalla --dataset GOWALLA \
        --out-dir ./data/gowalla/groups_social_raw --no-resplit \
        --group-source social --friendship old \
        --friend-old ./data/gowalla/friendship_old_GOWALLA.csv \
        --regimes occasional random --profile-top-k 10 --hist-len 90

Full sizes: train 39,317 · val 8,027 · test 16,970 examples.

## Two differences from the Foursquare (`data/llmgpr/groups_social/`) arm — both deliberate

1. **`--regimes occasional random`, no `established`.** That regime needs the dense `[n,n]`
   affinity matrices, which are ~8 GB each at Gowalla's 31,667 users. `build_groups.py` now
   skips building them unless `established` is requested. Restoring it needs a sparse/blocked
   rewrite of `affinity.py`, not a bigger machine.

2. **The friendship graph here is NOT denoised, and that is the finding, not an omission.**
   GBSR is a measured no-op on this graph (mask std 2.2e-05, 117,933 of 117,949 edges pinned at
   the sigmoid ceiling; no bottleneck strength changes it — see `LLMGPR_GOWALLA.md` §5). The
   denoised arm is built and diffed anyway at `../../groups_social_denoised/`: every count moves
   by ≤ 0.21%. Use this raw arm; cite the denoised one as the control.

   *(The Foursquare samples README describes its graph as "the GBSR-denoised
   `friendship_old_LLMGPR.csv`". That claim is false — see `LLMGPR_TRACK.md` §2 — and is not
   repeated here.)*

Gowalla has only one friendship snapshot, so `friendship_new_only_GOWALLA.csv` is empty by
construction and no before/after leakage rule applies (or can be enforced) on this dataset.
