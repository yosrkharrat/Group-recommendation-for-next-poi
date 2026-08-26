# Samples — LLMGPR *social* groups

The first 200 records of each split, committed so the record schema is readable without
cloning ~300 MB. The full `group_examples_*.jsonl` are gitignored and regenerate
deterministically (~2 min wall, ~4.5 GB peak RAM):

    python src/build_groups.py --data-dir ./data/llmgpr --dataset LLMGPR \
        --out-dir ./data/llmgpr/groups_social --no-resplit \
        --group-source social --friendship old \
        --friend-old ./data/llmgpr/friendship_old_LLMGPR.csv \
        --profile-top-k 10 --hist-len 90

`notebooks/stage5_lbsn_finetune.ipynb` §9b runs exactly this command itself if the files are
absent, so the fine-tune needs no manual step.

Full sizes: train 71,933 · val 16,164 · test 33,193 examples.

## How these differ from `data/groups/samples/` and `data/lbsn/groups/`

Same record schema, **different group source**. `--group-source social` replaces the KCGRS
anchor-window miner with LLMGPR/CubeRec's own rule — a connected component of *friends*
(≤ 5 friendship hops) co-present at the same venue in the same 180-min bucket — over the
GBSR-denoised `friendship_old_LLMGPR.csv` **only**. `friendship_new_only` never enters, since
those edges postdate the check-ins and are partly an *effect* of the co-visits.

That rule feeds the `occasional` compositions and the co-attendance tie pool; the
`established` (affinity-clique) and `random` regimes are built exactly as on the other tracks.

Measured on this build: 4,804 real social groups (sizes 2:3,975 3:552 4:158 5:63 6:28 7:20 8:8)
from 10,603 (venue, 180-min) candidates over a 4,872-user / 8,640-edge friendship graph.
Affinity validation AUC 0.894 combined (rhythm 0.790, taste 0.780, territory 0.762,
far-co-visit 0.579) against 6,685 real co-present pairs. Only **308** real group→group
transitions exist, which is why the trainable task is the standard constructed one — see
`src/build_groups.py`'s module docstring.

**POI id space:** `poi_idx` 0–14,401, from `data/llmgpr/poi_metadata_LLMGPR.csv`. These records
are **not** interchangeable with `data/groups/` (TSMC, 5,120 POIs) or `data/lbsn/groups/`
(LBSN_NYC, 6,103 POIs). §9b loads from this directory by explicit path, never by filename
search, for exactly that reason.

One record:

```json
{
  "example_id": "train_0000000",   // synthetic group EXAMPLE id -- a separate namespace
                                   // from ephemeral_groups.csv's integer group_id
  "anchor": 0,                     // whose real next-POI event this is built from
  "members": [0, 1697],            // anchor first, then companions
  "size": 2,
  "regime": "established",         // established | occasional | random
  "hist": [436, 3597, ...],        // JOINT history, poi_idx, time-ordered
  "hist_hours": [17, 17, ...],
  "hist_owner": [1697, 0, ...],    // which member contributed each history item
  "member_profiles": [...],        // per-member causal profile at time t (top-10 POIs here)
  "target": 256,                   // ground truth next poi_idx  -> POI_ID_START + 256
  "t_hour": 4,
  "t_dow": "Saturday",
  "heterogeneity": 0.3701,         // STRATIFY ON THIS, not on `regime`
  "split": "train"
}
```

§9b splits the joint `hist`/`hist_owner` back into per-member histories before rendering, so a
member's "recent check-ins" line shows their own trail and not the whole group's.
