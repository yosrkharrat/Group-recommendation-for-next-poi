# Samples — Gowalla *social* groups (FULL PARITY with the Foursquare arm)

The first 200 records of each split. The full `group_examples_*.jsonl` are gitignored and
regenerate deterministically (seed 42, ~25 min):

    python src/build_groups.py --data-dir ./data/gowalla --dataset GOWALLA \
        --out-dir ./data/gowalla/groups_social --no-resplit \
        --group-source social --friendship old \
        --friend-old ./data/gowalla/friendship_old_GOWALLA.csv \
        --profile-top-k 10 --hist-len 90

That is the Foursquare command with only the paths changed — **all three KCGRS regimes**, and
every other knob at the value the FSQ arm used (verified by diffing the two `groups_manifest.json`
configs: zero substantive divergences).

Full sizes: train 84,205 · val 18,964 · test 42,694 examples
(established 45,026 / 10,833 / 25,929 · occasional 35,119 / 7,208 / 14,554 · random 4,060 / 923 / 2,211).

## `established` needed a new affinity backend, not a relaxed rule

`affinity.py` allocates five dense `[n,n]` float64 matrices plus `triu_indices` — ~0.5 GB each at
Foursquare's 7,849 users, **~8 GB each at Gowalla's 31,667**. `src/affinity_blocked.py` computes
the identical z-summed affinity and the identical exact percentile cut blockwise:

    exact cut 5.660150 over all 501,383,611 pairs -> 5,013,928 edges (1.00%),
    31,667/31,667 users with >=1 neighbour, mean degree 316.7

Its `--self-check` asserts that the threshold, every neighbour set and every `G[i,j]` match the
dense path bit-for-bit at percentiles 99/95/90, so this is an exact reimplementation and not an
approximation. `build_groups.py --affinity-backend auto` keeps the dense path below 12,000 users,
so Foursquare reproduction is untouched.

One diagnostic is not produced in blocked mode: `validate()`'s per-component AUC-vs-co-presence
table ranks every pair of every component and so needs the dense matrices this backend exists to
avoid. It feeds no output and the manifest records it as skipped rather than faking a number.

## `../groups_social_denoised/` and `../groups_social_raw/`

Both are **superseded by this directory** and kept only as the record of two measurements:
`groups_social_raw` is the same build without `established` (the earlier divergence), and
`groups_social_denoised` is the GBSR-denoised control that showed every count moving by ≤ 0.21%
(GBSR is a measured no-op here — `LLMGPR_GOWALLA.md` §5). Stage 5 should point at
**`groups_social`**.
