# LLMGPR on Gowalla — dataset identification and Table 1 recovery

Audience: a fresh session picking this up cold. Everything below was established on
2026-08-31, on the `gowalla` branch (cut from `llmgpr-pipeline`). Companion to
`LLMGPR_TRACK.md`, which did the same job for the Foursquare column; conventions, bounding
boxes and city centres are inherited from there and not re-derived.

**Goal.** Reproduce the Gowalla column of LLMGPR (CIKM'25, `papers/3746252.3761018.pdf`)
Table 1 — "same number of users, check-ins, POIs after filtering" — and emit the working
dataset for the pipeline stages (groups, KG, fine-tune) that already run on Foursquare.

Their column:

| | users | groups | POIs | cats | user ck | group ck | ck/user | ck/group | users/group |
|---|---|---|---|---|---|---|---|---|---|
| Gowalla | 31,751 | 2,186 | 81,123 | 537 | 862,502 | 7,738 | 27.16 | 3.54 | 2.95 |

---

## Status board

| Question | Status | Answer |
|---|---|---|
| Which Gowalla dump do they use? | ✅ **settled** | The full crawl behind their citation [24] (Liu et al. CIKM'13) — 36.0M check-ins / 319k users / 2.84M spots **with categories** + friendship graph, once at `yongliu.org/datasets`, mirrored at **figshare 22126586** (md5-verified). SNAP `loc-gowalla` is ruled out: no categories, 6.4M check-ins, window ends Oct 2010 |
| Is their ≥10 filter real here? | ✅ **settled — no** | Their mean 27.16 ck/user is **below** the unfiltered in-region mean everywhere; any activity cut raises it. The rule as written tops out at **match 0.74** with 0.37× their users (11.6k) at 65–75 ck/user |
| Are users/POIs/check-ins recoverable? | ✅ **yes — 0.995** | Best grid row: R=20 km + window ≤2011-04 + 500-cap → 32,143 / 81,240 / 863,873 (all ≤1.2% off). Three unstated knobs, so reported as a fit, not adopted |
| What is adopted? | ✅ | **R=15 km collection, day-dedup, NO user filter; #POIs = km≤28 catalogue** → users 0.997× / POIs 1.001× / ck 1.047× (match3 0.984, one fitted radius per role) |
| Is `#POIs` a catalogue again? | ✅ **yes** | 81,123 ≈ the km≤28 catalogue (81,240, 1.001×); visited-venue counts are 47.8k (R=15) or 108–110k (boxes) — nothing else lands |
| Is `#categories` recoverable? | ❌ **hard-bounded no** | Their 537 exceeds every in-region basis this dump can produce (ids 341, raw strings 344, taxonomy 266) and undershoots every whole-history basis (637–667; global vocab 667). Carried as the residual column |
| Where did their column come from? | ✅ as far as it goes | No earlier paper carries it ([28]/MAC/DCLR/PTIA are two-dataset papers; CubeRec's Gowalla differs) — it is their own undocumented pipeline; columns provably from different stages, like Foursquare's (§1.4a excess-budget: mean 10.63 ck/POI vs a ≥10 floor) |
| Group construction | ❌ not started | Friendship coverage is far better than Foursquare's: 235,898 directed edges over 26,779 of the 31,667 adopted users (84.6%, mean degree 8.8, 100% reciprocal) vs FSQ's 28% / 3.2 |

---

## 1. The dump, identified

LLMGPR cites Gowalla as [24] = X. Liu, Y. Liu, K. Aberer, C. Miao, CIKM 2013 — the paper whose
crawl was distributed from `yongliu.org/datasets` (site now domain-squatted). The figshare
mirror (article 22126586) carries six files, md5s matching figshare's own manifest:

```
gowalla_checkins.csv            1.29 GB   userid, placeid, datetime        36,001,959 rows
gowalla_spots_subset1.csv        343 MB   id, lat/lng, spot_categories      2,724,891 spots (100% catted)
gowalla_spots_subset2.csv        8.2 MB   id, lat/lng, name, city_state       120,997 spots (no categories)
gowalla_friendship.csv            67 MB   userid1, userid2                  4,418,339 directed, 100% reciprocal
gowalla_category_structure.json   99 KB   7 mains, 266 distinct category ids
gowalla_userinfo.csv              17 MB   per-user profile counters
```

Global measurements: 36,001,959 check-ins / 319,063 users (112.8 each), dates
**2009-01-21 .. 2011-08-16** (10 months past SNAP's window), 667 distinct category ids across
all spots (the structure file's 266 plus legacy ids still attached to spots). Check-in ids are
the crawl's native user/spot ids; friendship attaches directly; subset1 ∩ subset2 = ∅.

The 3-city extract (Foursquare-track boxes, pad up to 0.4°): 151,642 spots (96.8% catted),
2,419,169 check-ins / 44,733 users; plain boxes: 109,691 spots, 2,003,626 / 42,514
(47.1 ck/user unfiltered — remember that number).

## 2. What their column cannot be

**(a) Not the documented rule.** *"users and POIs with less than 10 interactions are removed"*
keeps 18,608 users in the boxes (0.59×) and 11,627 at R=15 (0.37×), at 65–103 ck/user against
their 27.16. Structurally: their mean is *below* the unfiltered mean, and monotone activity cuts
only raise means — so **no threshold, on any basis (raw / distinct-POIs / global), reproduces the
user column**. Best documented-rule row over the entire grid: 0.74.

**(b) Not whole-history accounting** (Foursquare's winning reading). The ≥10 in-region users
visit ~1.2M venues worldwide (15× their #POIs) at ~490 ck/user (18× their 27.16). Every
whole-history variant scores ≤ 0.51.

**(c) Not the SNAP window.** Clipping to ≤2010-10 drops 3-city users to ≤24,870 at every radius
— the column needs the crawl's full 2011 tail.

**(d) `#categories` = 537 is unreachable, both ways.** In-region: 341 ids / 344 raw
`spot_categories` strings at the widest pad; per-city sums start at 739. Whole-history: 637–667.
This is the same class of bound that killed Yang §3 for Foursquare (429 < 436), except here it
brackets 537 from both sides. Their category data is not this dump's `spot_categories`, or the
cell was computed at a stage nobody reports.

## 3. What it is — the recovered reading

The `llmgpr-gowalla-recovery.ipynb` grid (region × window × dedup × threshold × bases × cap ×
catalogue) finds a tight, consistent family:

```
                                              users            POIs        ck      ck/u  m3
best fit    R=20, <=2011-04, cap 500       32,143 1.012x   81,240 1.001x  863,873 26.9  0.995
            R=17, Tu=5 global, cap 500     32,115 1.011x   81,240 1.001x  867,162 27.0  0.994
adopted     R=15, day-dedup, no filter     31,667 0.997x   81,240 1.001x  903,045 28.5  0.984
```

- **`#users` and `#check-ins` come from a ~15 km metro radius with NO activity filter** —
  R=15 alone puts users at 0.997×. (Centres are Yang's Foursquare city centres; read the
  finding as "a ~15 km radius", not those exact coordinates.)
- **`#POIs` is a region catalogue again**: every venue within ~28 km, visited or not
  (81,240 vs their 81,123). The region enters nowhere else — the same decomposition §1.4 proved
  for Foursquare's 10 km catalogue.
- The residual +4.7% on check-ins closes exactly under any one of: a 500-cap, a 2011-04
  snapshot, or R→20 with the other knobs — all unstated; we spend one fitted radius per column
  role and refuse the rest, per the per-city-radii lesson (`LLMGPR_TRACK.md` §1.4).
- The paper's stated **200-cap makes things worse here** (0.75× on check-ins): their Gowalla
  tail is far heavier than Foursquare's.

**Decision.** Emit the adopted build as the working dataset, report the documented-rule core
alongside (the same two-arm posture as the Foursquare track):

```
adopted   R=15, day-dedup, unfiltered   31,667 users / 903,045 ck / 81,240-POI catalogue / 326 cats
core      >=10/>=10 at R=15, in/in      11,627 users / 757,936 ck / 18,843 POIs visited
```

## 4. Artifacts

| File | What it is |
|---|---|
| `llmgpr-gowalla-recovery.ipynb` | end-to-end: download+md5, universe, scan, eliminations, grid, emit — executed in-repo, no Kaggle needed (raw dump is 1.7 GB) |
| `data/gowalla/gowalla_final_checkins.csv.gz` | 903,045 deduped R=15 check-ins: userid, placeid, utc_time, city, category_id/name, km |
| `data/gowalla/gowalla_final_catalogue.csv` | the km≤28 catalogue: 81,240 venues, lat/lng, category, city, km |
| `data/gowalla/gowalla_final_users.csv` | 31,667 users: in-region (deduped + raw) and global counts |
| `data/gowalla/gowalla_final_friendships.csv` | 235,898 directed edges, both ends retained, `source=crawl` (single snapshot — **no old/new split exists**, so the Foursquare leakage rule cannot apply) |
| `data/gowalla/gowalla_final_stats.csv` | the ours/theirs table incl. match scores |
| `data/gowalla_raw/` (gitignored) | the six figshare files + `_*.parquet` scan caches |

Downstream feasibility, measured on the adopted build: 21,632 users (68.3%) have ≥3 in-region
check-ins (the leave-one-out floor); friendship mean degree 8.8 over 84.6% of users — the
GBSR/social pre-check that failed on Foursquare (degree 3.2, 28%) starts from much better ground
here.

## 5. The pipeline on Gowalla (2026-08-31, second commit on this branch)

Stages 0–4 of the `llmgpr-pipeline` flow, run on the adopted build. Two arms, mirroring the
FSQ track's control/denoised split — and this time the denoised arm is real, because the
`build_kg_lbsn.py` hardcode that silently kept GBSR's output out of the FSQ KG now has a
`--friendship-old` override.

### Stage 0 — `src/prepare_gowalla_csvs.py`

`gowalla_final_*` → house CSVs (`--dataset GOWALLA`), same schema as `prepare_llmgpr_csvs.py`
so every downstream stage runs unchanged. Gowalla-specific decisions (full docstring in the
script): taxonomy from `gowalla_category_structure.json` (7>134>128 tree, exported as
`gowalla_category_paths.json` for the KG; 85.9% of named categories resolve, depth histogram
1:7,926 / 2:29,219 / 3:10,638), friendship = ONE crawl snapshot written to `friendship_old_*`
with `friendship_new_only_*` EMPTY (no before/after split exists; leakage rule inapplicable),
`timezone_offset` = city standard time (UTC is authoritative). Splits: 646,749 / 87,803 /
168,493 over 31,667 users / 47,783 POIs; lat/lon coverage **100.0%** (FSQ managed 43%), so
`IS_NEAR_TO` covers every POI.

### Stage 1 — groups (`build_groups.py --group-source social`, raw arm)

```
real social groups (LLMGPR/CubeRec rule)   36,621   sizes 2:31,120 .. 8:42, mean 2.22
distinct member-sets                       18,362   1.99 pooled check-ins each
  >= 2 pooled events                        2,987
  >= 3 pooled events                        1,782      <- LLMGPR's 2,186 sits between these
real group->group transitions              90,152
trainable examples (occasional+random)     39,317 train / 8,041 val / 16,970 test
```

Two findings worth the trip:

- **The recurrence-filter hypothesis confirms on a second dataset.** LLMGPR's 2,186 groups is
  bracketed by our ≥2 (2,987) and ≥3 (1,782) recurring member-sets, exactly the FSQ §4.0
  pattern (their 1,715 vs our ≤1,487 there). Their group count is a recurrence-filtered subset
  of what the cited procedure yields; the filter is still documented nowhere.
- **Real group transitions are TRAINABLE on Gowalla: 90,152.** On FSQ-NYC the same measurement
  gave 17–62, and the entire regime construction exists because of that scarcity. Gowalla's
  denser co-presence + real friendships change the design space: the natural task (group at A →
  predict B) has data here. Flagged for the evaluation plan.

One scale patch, measured not asserted: the dense `[n,n]` affinity machinery (five ~8 GB
matrices at 31.7k users) is now built **only when `established` ∈ `--regimes`** — it is that
regime's only consumer. Gowalla runs `--regimes occasional random`; adding `established` back
needs a sparse/blocked `affinity.py` rewrite, not a bigger machine.

### Stage 2 — GBSR (`denoise_social_gbsr.py`)

The val-NDCG overflow flagged in `LLMGPR_FINETUNE_HANDOFF.md` (known issue b) is fixed: scores
are computed in float64 and non-finite values floored, so a diverged epoch can never win model
selection. Run config: defaults except `--batch-size 4096` (full-graph propagation per batch ×
457 batches at 1024 was ~4 min/epoch on CPU; sparse ops have no MPS support). Input: 117,949
undirected edges over 26,779 users, mean degree 8.8 — 2.8× denser than the FSQ graph GBSR
no-op'd on, so the mask has redundancy to work with this time.

### Stages 2b/3 — KG + RotH (raw arm)

`build_kg_lbsn.py` (PYTHONHASHSEED=0; the `prefers_category` tie-break issue is unchanged):
**116,401 entities / 2,125,760 triples / 12 relations**, group layer included (MEMBER_OF
36,621 sets, CO_ATTENDED 63,616, OCCURRED_AT + GROUP_PREFERS 36,621 each), hierarchy 187,791
pairs, leakage guard green (trivially — the new-only set is empty by construction).

`train_roth.py`: fixed a genuine device bug (`ball_points` created its reference curvature on
CPU — every non-CPU run crashed at the first `depth_loss`; self-check still passes). Run
config vs the FSQ handoff: `--batch-size 4096 --n-neg 32 --epochs 50` on MPS — measured
necessity, not preference: the handoff's 512/128/120 is 8+ min/epoch ≈ 20+ hours on this
machine (the FSQ arm trained on Kaggle CUDA). Deviations are in `roth_results.json`'s config
dump.

### Commands (repro)

```bash
python src/prepare_gowalla_csvs.py
python src/denoise_social_gbsr.py --data-dir ./data/gowalla --csv-dir ./data/gowalla \
       --dataset GOWALLA --out-dir ./data/gowalla --batch-size 4096
# raw / control arm
python src/build_groups.py --data-dir ./data/gowalla --dataset GOWALLA \
       --out-dir ./data/gowalla/groups_social_raw --no-resplit --group-source social \
       --friendship old --friend-old ./data/gowalla/friendship_old_GOWALLA.csv \
       --regimes occasional random --profile-top-k 10 --hist-len 90
PYTHONHASHSEED=0 python src/build_kg_lbsn.py --csv-dir ./data/gowalla --dataset GOWALLA \
       --taxonomy ./data/gowalla/gowalla_category_paths.json \
       --groups-dir ./data/gowalla/groups_social_raw \
       --friendship-old ./data/gowalla/friendship_old_GOWALLA.csv --out-dir ./data/gowalla/kg_raw
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_roth.py --kg-dir ./data/gowalla/kg_raw \
       --data-dir ./data/gowalla --dataset GOWALLA --out-dir ./data/gowalla/kg_raw \
       --epochs 50 --batch-size 4096 --n-neg 32 --log-every 5 --max-eval 4000 \
       --depth-weight 5.0 --depth-margin 0.3 --root-pull 0.01 --device mps
# denoised arm: rerun the last three with --friend-old/--friendship-old
#   ./data/gowalla/friendship_old_denoised_GOWALLA.csv, out-dirs *_denoised
```

## 6. Next steps

1. When GBSR lands: build `groups_social_denoised` + `kg_denoised` + RotH on the denoised
   graph (commands above), then compare arms — mask-weight separation first (`gbsr_denoise_manifest.json`);
   if the mask collapses to a constant here too, the FSQ no-op generalizes and that is the result.
2. `build_poi_poi_triples.py` on whichever KG wins → the stage-5 alignment file.
3. The 500-candidate evaluator (`gowalla_final_catalogue.csv` has coords + city).
4. The 90,152 real transitions deserve a real-group eval arm — FSQ never had this option.

**Do not** re-open the ≥10 reading for Gowalla — §2(a) is a structural bound, not a tuning miss.
