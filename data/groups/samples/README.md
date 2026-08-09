# Samples

The first 200 records of each split, committed so the record schema is readable without
cloning 66 MB. The full files are gitignored and regenerate deterministically:

    python src/build_groups.py --data-dir ./data --out-dir ./data/groups --seed 42

Full sizes: train 34,644 · val 6,486 · test 13,420 examples.

One record:

```json
{
  "example_id": "train_0000000",   // synthetic group EXAMPLE id -- a separate namespace
                                   // from ephemeral_groups.csv's integer group_id
  "anchor": 1,                     // whose real next-POI event this is built from
  "members": [1, 431],             // anchor first, then companions
  "size": 2,
  "regime": "established",         // established | occasional | random
  "hist": [436, 3597, ...],        // joint history, poi_idx, time-ordered
  "hist_hours": [17, 17, ...],
  "hist_owner": [431, 1, ...],     // which member contributed each history item
  "member_profiles": [...],        // per-member causal profile at time t
  "target": 256,                   // ground truth next poi_idx  -> POI_ID_START + 256
  "t_hour": 4,
  "t_dow": "Saturday",
  "heterogeneity": 0.3701,         // STRATIFY ON THIS, not on `regime`
  "split": "train"
}
```
