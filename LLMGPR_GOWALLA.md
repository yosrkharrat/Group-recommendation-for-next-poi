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

## 5. Next steps

0. Wire `src/prepare_llmgpr_csvs.py` (or a Gowalla twin) to the `gowalla_final_*` files — the
   blockers are Gowalla-specific: category names are Gowalla's own vocabulary (no
   `fsq_category_paths_2014.json` mapping), and the friendship file has no old/new split.
1. Group construction (their 2,186 groups / 3.54 ck/group / 2.95 users/group) — same §4 rebuild
   as Foursquare: rolling window + cliques + recurring member-sets, on the friendship graph.
   CubeRec's own Gowalla yields 2.66 int./group at 2.31 users/group — their 3.54/2.95 is close
   to the cited method's range here, unlike Foursquare's 7.34.
2. The 500-candidate evaluator can reuse `gowalla_final_catalogue.csv` directly (coords + city).

**Do not** re-open the ≥10 reading for Gowalla — §2(a) is a structural bound, not a tuning miss.
