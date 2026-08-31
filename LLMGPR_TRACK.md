# LLMGPR reproduction track — dataset identification and GBSR

Audience: a fresh session picking this up cold. Everything below was established on
2026-08-24; nothing here needs re-deriving.

> **This is a fresh, standalone track.** It shares the repo with the HyGro-POI / LBSN pipeline
> (`README.md`, `docs/`, `data/kg*`, `notebooks/`) but nothing here builds on it — different
> dataset, different splits, different evaluation protocol. Do not assume any artifact,
> statistic, or convention from that pipeline carries over. All files for this track live at
> the repo root plus `src/probe_section3.py` and `src/to_gbsr.py`.

> **Gowalla (2026-08-31, `gowalla` branch):** the same Table-1 recovery for their Gowalla
> column lives in `LLMGPR_GOWALLA.md` + `llmgpr-gowalla-recovery.ipynb`, with the working
> dataset committed at `data/gowalla/`. Headline: dump identified (the Liu CIKM'13 crawl,
> figshare 22126586); users/POIs/check-ins recovered at 0.997×/1.001×/1.047× — but only with
> **no ≥10 user cut** (their stated rule tops out at 0.74 there) and `#POIs` as a **km≤28
> catalogue**; `#categories` (537) is hard-bounded unreachable from that dump (341 in-region,
> 637–667 whole-history). Same multi-stage-table pathology as §1.4/§1.9b below.

**Goal.** Follow LLMGPR (CIKM'25, `papers/3746252.3761018.pdf`) — its data split, its
leave-one-out evaluation, its group-construction rule — on Foursquare NYC / LA / Chicago,
matching its Table 1 statistics. Then denoise with GBSR (KDD'24, `yimutianyang/KDD24-GBSR`).

---

## Status board

| Question | Status | Answer |
|---|---|---|
| Which Foursquare dump does LLMGPR use? | ✅ **settled** | Yang §5 **raw** files (`raw_POIs.txt` + `raw_Checkins_anonymized.txt`) |
| Why did our POI counts never match? | ✅ **settled** | Their `#POIs` is a region catalogue, not a post-filter count — provable from their own table (§1.4) |
| Can the catalogue reading match it? | ✅ **settled** | Yes — **match 0.978**: R=10 km catalogue + users ≥60 on full histories, no POI filter (§1.4) |
| Can §5 friendships attach to the check-ins? | ✅ **settled** | Yes — raw and filtered share one user-id space |
| What is their exact user/POI filter? | ✅ **settled** | users ≥60 check-ins on full histories; **no POI filter** — `#POIs` is a 10 km catalogue → match 0.978 |
| Can the paper's literal ≥10 be right? | ✅ **yes — adopted** | **0.938** on the filtered file with pad 0.4° + 200-cap (§1.9d). 3 of 4 columns within 1.5%. Residual: `#users` vs `ck/user` cannot both be satisfied |
| Do filter boundaries explain the gap? | ✅ **settled — no** | `>`, distinct-partner counts and POI-first ordering are all **worse** (§1.9d). The permissive reading is correct |
| Where did their Table 1 come from? | ✅ **settled** | Copied from Long et al. WWW'24 [28], which cites Yang WWW'19 = §5 — confirming our dump (§1.9b) |
| Is their table even attainable? | ✅ **settled** | 3-city yes (off 1.9 pts); **arXiv v1 NYC is impossible** |
| Does GBSR denoise groups? | ✅ **settled** | **No** — it denoises the user–user social graph |
| Group construction rebuild | ❌ **not started — this is now the critical path** | 2,196 eligible users, 3,555 edges |

---

## 1. The dataset investigation

### 1.1 What we were doing wrong

`group-recommendation-using-fsq-section3.ipynb` (the starting notebook) extracted from **§5's
filtered file** `dataset_WWW_Checkins_anonymized.txt`. That cannot reach their table, for a
reason that is arithmetic rather than tuning:

~~LLMGPR §4.1 removes POIs with fewer than 10 interactions, so 80,962 POIs require
≥ 809,620 check-ins. That extraction yields 592,341 across the three cities before any filter.
No filter setting closes that gap.~~

> ⚠️ **This argument is VOID and the filtered file is NOT eliminated.** It assumes `#POIs` is a
> post-≥10-filter count — which §1.4 later disproved from LLMGPR's own Weeplace tables (identical
> POI count against check-in totals differing by 300k). Worse, the citation trail in §1.9b points
> *at* the filtered file: [28] cites Foursquare as Yang WWW'19, whose own dataset is
> `dataset_WWW_Checkins`, and LLMGPR needs the friendship users for group construction anyway.
> Every sweep from §1.6 onward ran on the **raw** file on the strength of this void argument.
> `llmgpr-filtered-file-literal10.ipynb` runs the sweep that was skipped.

Where the two files actually start, three-city boxes, before any filter:

```
                  filtered        raw       theirs    filt      raw
POIs               102,541    237,728       80,962   1.27x    2.94x
users               14,401    152,480        7,507   1.92x   20.31x
check-ins          592,341  2,227,756    1,214,631   0.49x    1.83x
```

The filtered file's only shortfall is check-ins, and whole-history counting supplies exactly
that: it holds 22,809,624 check-ins over 114,324 users globally — **199.5 per user**, against
their reported 161.80, where the raw dump sits at 14.6.

Two incidental bugs in that notebook, for the record:

- The k-core was switched off: `MIN_USER_CHECKINS = 1` / `MIN_POI_CHECKINS = 1` tested with
  `>`, keeping ≥2, never the paper's ≥10. `>` is also an off-by-one — "less than 10 removed"
  means keep ≥10.
- `FRIENDSHIP = "union"` pulls in `friendship_new`, which postdates the check-in window.
  Leakage. **Only `friendship_old` may ever be used for construction or training.**

### 1.2 Ruling out §3

LLMGPR cites its Foursquare as `[4]` = Chen et al., IMWUT'20, whose dataset section reads
verbatim: *"33,278,683 check-in records of 266,909 users at 3,680,126 unique POIs between April
2012 and September 2013 in the most checked 415 cities worldwide"* — that is Yang's **§3**
("Global-scale Check-in Dataset", 739.8 MB).

Measured on §3 (`notebook output.ipynb`):

```
3 cities, unfiltered   975,405 check-ins / 118,573 POIs / 26,912 users
3 cities, >=10 core      9,490 users / 19,474 POIs / 386 cats / 648,608 check-ins
LLMGPR target            7,507 users / 80,962 POIs / 436 cats / 1,214,631 check-ins
```

**§3 is ruled out by a hard bound: it contains 429 distinct categories in the entire global
dump, and LLMGPR reports 436** — identically in the NYC-only arXiv v1 and the 3-city
camera-ready. You cannot observe 436 categories in a vocabulary of 429.

### 1.3 The identification

§5's **raw** files (`output-2.ipynb`):

```
raw_POIs.txt global vocabulary        519 categories
raw_POIs.txt over NY + CHI + LA       436 categories   <-- LLMGPR reports 436, exactly
3 cities, unfiltered            2,227,756 check-ins / 237,728 POIs / 152,480 users
3 cities, >=10 core                30,907 users / 37,113 POIs / 1,380,405 check-ins
```

The category match is exact and the bounding boxes are confirmed correct. **Their POI metadata
is `raw_POIs.txt` restricted to these three cities.**

### 1.4 Their `#POIs` column is not a post-filter count — provable three ways

The column that never matched under *any* interaction filter, for a reason that no longer depends
on our extraction being right about anything. It is reachable under a different reading of what
the column counts — see the end of this section — but not as the output of their stated rule.

**(a) From their own table, with no external data.** §4.1 states verbatim: *"users and POIs with
less than 10 interactions are removed."* If every retained POI carries ≥10 check-ins then
`#check-ins ≥ 10 × #POIs`, and the slack — the **excess budget** — is the total supply of
check-ins available above the floor, for all venues combined:

```
dataset        #POIs   #check-ins   mean deg   10 x POIs     excess   excess/POI
Foursquare    80,962    1,214,631      15.00     809,620    405,011        5.00
Weeplace      44,194      623,654      14.11     441,940    181,714        4.11
Gowalla       81,123      862,502      10.63     811,230     51,272        0.63
```

Gowalla is the tell. A floor of 10 with a mean of **10.63** requires essentially every venue to
sit *exactly* at 10. Measured against real data (`output5.ipynb`, top venues of the §5-raw
3-city extract), its entire excess budget — 51,272 check-ins for all 81,123 venues — is consumed
by **the top three airports alone** (19,007 + 17,831 + 14,540 = 51,348). Real LBSN venue degrees
are power-law. All three datasets land at
10.6–15.0, just above the floor, which is the signature of a POI count taken at a *different
pipeline stage* than the check-in count.

**(b) From the monotonicity of filtering.** Filters only ever *remove* check-ins, so no subset
can have more POIs clearing ≥10 than the unfiltered whole does. That whole is now measured
(`output5.ipynb`, survivor curve on the unfiltered §5-raw extract):

```
                 POIs >=1    POIs >=10   mean deg    theirs   theirs / ceiling
3 cities          237,728       42,167      41.18    80,962       1.92x   IMPOSSIBLE
New York          113,326       20,939      44.69    63,445       3.03x   IMPOSSIBLE
```

§3 corroborates independently: there, 63,445 would need 91.3% of NYC's 69,520 ever-visited
venues to clear ≥10, against a measured 17.2%.

Read the curve the other way and it dates their number to a *different* threshold in each of
their two papers — which is (c) below, measured rather than argued:

```
their #POIs, as a post-filter count, implies a threshold of
  3 cities (camera-ready)   between k=4 (87,938) and k=5 (74,203)
  New York (arXiv v1)       between k=2 (68,390) and k=3 (51,525)
```

**(c) From their two versions disagreeing.** LA + Chicago add **+110% POIs / +91% check-ins** on
top of NYC in the real dump, but only **+28% / +31%** between their arXiv v1 (NYC) and
camera-ready (3-city) tables. Their two LA/Chicago slices are ~4× under-weight relative to
reality — so the two tables do not sit on one preprocessing rule either, which the implied
thresholds in (b) confirm independently.

**What the column is — recovered.** Read `#POIs` as the **POI catalogue of the region** (every
venue in it, not only those the retained users visited) and Table 1 comes back. `output6.ipynb`
crossed radius × per-city radii × four accounting conventions (user threshold measured in-region
or on the full history; check-ins counted either way):

```
mode        best match     what it means
full/full        0.978     region fixes the CATALOGUE ONLY; users and check-ins are full-history
in/full          0.905     users selected in-region, counted on full histories
in/in            0.851     everything clipped to the region (this was output5's test D)
full/in          0.837
```

**Winner: uniform R = 10 km around Yang's city centres, users ≥60 check-ins, no POI filter at
all — match 0.978.**

```
                   ours        theirs    ratio
users             7,849         7,507    1.05x
POIs (catalogue) 82,188        80,962    1.02x
categories          432           436    0.99x
check-ins     1,191,781     1,214,631    0.98x
check-ins/user    151.8         161.8    0.94x
```

The decomposition is the finding. Those users and check-ins are **exactly** output42's
`Tu=60, Tp=1` row, which scored 0.856. The only thing that changed is how `#POIs` is counted —
visited venues (162,805) → 10 km catalogue (82,188) — and that single reinterpretation is the
whole 0.856 → 0.978 lift. So:

- their user cut is **≈60 check-ins**, not 10;
- there is **no POI-side interaction filter at all**, at any threshold;
- `#POIs` is a **region catalogue**, and the region enters *nowhere else* — not user selection,
  not check-in counting.

Their §4.1 sentence describes neither knob. The numbers are still theirs.

**The per-city refinement is a fit, not evidence — do not report it.** A grid over per-city radii
reaches 0.981 at NY=12 / LA=12 / Chicago=5 km, with the catalogue 31 venues from theirs. That
looks stunning and means nothing: in `full/full` the radius touches **only** `#POIs` and
`#categories` — `#users` and `#check-ins` are computed from full histories and are invariant to
it (visible in output6's own top-15, where every row at Tu=60 reads 7,849 / 1,191,781). So the
grid fits three free radii against essentially one scalar. It also truncates Chicago at 5 km,
which guts the very catalogue the 500-candidate sampler is built on. **Use the uniform R = 10 km:
one free parameter, 1.5% off.**

**Residual, and the honest caveats.** Users run 1.05× high and check-ins 0.98× low, i.e. their
cut sits slightly above 60 — the notebook's Tu grid has been refined to step through 55–80 to
close it, since Tu moves two columns and is legitimate to tune. Two things do not close:
`#categories` reaches 436 only at R ≥ 40 km (R = 10 gives 432, a 0.9% gap), so the category
column mildly favours a wider region than the POI column — worth one footnote sentence, not a
fork. And the arXiv v1 NYC table remains unattainable (§1.8) regardless of reading.

**Decision taken:** build on the recovered reading — users ≥60 check-ins on full 3-city
histories, `#POIs` as the uniform 10 km catalogue — and state radius and accounting explicitly
in the paper, noting that §4.1 describes neither. Report the ≥10-core numbers alongside so the
table stays honest. **No interaction filter on POIs.**

### 1.9b The table is inherited, and the citation chain confirms our dump

Chasing the two references in the filter sentence settled more than the threshold question.

**Their Foursquare table is not theirs.** Reference [28] — Long et al., *Physical Trajectory
Inference Attack and Defense in Decentralized POI Recommendation*, WWW'24, **same first author** —
reports the identical Foursquare statistics:

```
                     [28] WWW'24    LLMGPR CIKM'25
#users                    7,507            7,507    identical
#POIs                    80,962           80,962    identical
#categories                 436              436    identical
#check-ins            1,214,631        1,214,631    identical
#check-ins per user        161.80          162.80    differs  <- 1,214,631/7,507 = 161.80
```

LLMGPR inherited the table and mistyped one cell on the way in. Their dataset citations are
shuffled too ([28] cites Weeplace as Liu 2013; LLMGPR cites that as its *Gowalla* source).

**Our dump identification is confirmed by their own citation trail.** [28] cites Foursquare as
its reference [32] = **Yang, Qu, Yang, Cudré-Mauroux, WWW 2019, "Revisiting user mobility and
social relationships in LBSNs"** — that *is* the §5 release (`dataset_WWW2019`). We identified it
independently through the 436-category match; the provenance chain now agrees. Note LLMGPR's own
Foursquare citation ([4], Chen et al. IMWUT'20) points at §3 instead, which we ruled out — so
their citation is wrong and [28]'s is right.

**The ≥10 rule traces to the earliest paper in the chain, where it is stated precisely and the
numbers are self-consistent.** DCLR (arXiv 2204.06516, 2022): *"we remove users with less than 10
check-ins for both datasets, as well as POIs that have less than 10 visits."* Its Weeplace shows
886,408 check-ins over 35,675 POIs = **24.85 per POI** — entirely plausible for a real ≥10 cut.
So the convention is genuine and precisely specified; the supervisor's reading of intent is right.

**But the check-in and POI columns are carried independently, and this proves the catalogue
reading with none of our data.** Their Weeplace across the same two papers:

```
                     [28] WWW'24    LLMGPR CIKM'25
#users                    4,560            4,560    identical
#POIs                    44,194           44,194    identical
#categories                 625              625    identical
#check-ins              923,600          623,654    DIFFERS by 299,946
mean check-ins per POI     20.90            14.11
```

Same dataset, same users, byte-identical POI count — against check-in totals differing by 1.48×.
**If `#POIs` were the output of a ≥10 filter applied to those check-ins, a 300k change in the
check-in set could not leave the POI count unchanged.** It did. So `#POIs` travels independently
of the check-in column, which is exactly §1.4's conclusion, now provable from two of their own
published tables.

Also: **arXiv v1 (Nov 2024) contains no filter sentence at all** — verified against the source.
The ≥10 claim appears only in the camera-ready, describing numbers that were already fixed in v1
and inherited from [28] before that. It is a post-hoc description of a pipeline built elsewhere.

**No paper in the chain releases code or data**, so the dataset cannot simply be downloaded.

**Consequence for the build (recommendation changed).** Matching their Table 1 is no longer a
meaningful target: it is an inherited table whose columns provably come from different stages, on
a dump we have correctly identified but through a pipeline nobody documents. The defensible move
is to apply the **documented** rule — users with <10 check-ins removed, POIs with <10 visits
removed, per DCLR — report our own statistics, and carry the provenance finding as a
reproducibility note. That also satisfies fidelity to the stated method. Our numbers under that
rule are the ≥10 core: **30,907 users / 37,113 POIs / 1,380,405 check-ins**.

### 1.9c The literal ≥10 works on the filtered file — 0.913

`output7.ipynb` ran their rule **exactly as written** (users and POIs with <10 interactions
removed) on the file the citations point to. Every base number came out as predicted:

```
filtered file, whole      22,809,624 check-ins / 114,324 users = 199.5 per user
filtered file, 3 cities      592,341 check-ins /  14,401 users /   102,541 POIs
```

**Best literal configuration: R = 25 km, threshold counted in-region, check-ins counted over
whole histories, `#POIs` = venues the retained users visited — match 0.913.**

```
                   ours        theirs    ratio
POIs             77,897        80,962    0.96x
check-ins     1,267,305     1,214,631    1.04x
categories          422           436    0.97x
users             5,733         7,507    0.76x   <-- the whole residual
check-ins/user    221.1         161.8    1.37x   <-- its mirror
```

Three of four columns essentially land. Against **~0.62** for the same rule on the raw file, the
filtered file is decisively the right base — confirming §1.1's elimination was not just void but
backwards.

**One stated mechanism is still unapplied, and it points the right way.** Both papers say *"the
maximum sequence length is set to 200"*. These users average 221 check-ins over their whole
histories; capping each sequence at 200 pulls the mean toward their 161.80 — the exact direction
and roughly the magnitude of the miss. If Table 1's check-in total is the post-truncation figure,
that is the residual. `llmgpr-filtered-file-literal10.ipynb` has since been patched with a
`cap ∈ {none, 200}` axis (and a memoised sweep, since it grew to ~7k configurations). **Re-run it.**

**Boundary conventions are not the residual.** Checked by direction: `>10` instead of `>=10`,
"interactions" as distinct partners instead of check-ins, POI-filter-first, and iterating to a
fixed point all move `#users` **down**, and we need it up 31%. Only the region can move it up.
A fixture run of `llmgpr-boundaries-and-region.ipynb` puts the `>=T` → `>T` flip at **−0.8% of
users** — single digits, wrong direction, as predicted. Swept anyway, since categories are still
0.97× and boundaries matter at the margin once the big columns land.

**The axis never tested: growing the region.** Every region sweep so far ran *inward* — a
shrinking radius from a fixed bounding box. The box was validated against §3's NYC POI count,
never against how many filtered-file users a metro should hold. Growing it raises `#users`
(wanted) but also check-ins (already 1.04×) and POIs (already 0.96×) — so the **collection region
and the catalogue region cannot be the same object**. That is how [28] builds it anyway:
candidates come from k-means clusters inside a city, which is not the data footprint.
`llmgpr-boundaries-and-region.ipynb` parameterises the two separately, over bbox padding
0.0–0.4° with the venue universe re-scanned from `raw_POIs.txt` (a wider box contains venues our
POI table does not).

Note the neighbour at Tu=5 scoring 0.921: 7,313 users (0.97×) / 78,253 POIs (0.97×) /
1,564,334 check-ins (1.29×). There, users *and* POIs both land and only the check-in column is
long — by about the factor the 200-cap would remove. Worth watching in the re-run.

**Where the decision now sits.**

| build | match | fidelity to their stated method |
|---|---|---|
| raw + Tu=60 + 10 km catalogue | 0.978 | contradicts §4.1 (60, not 10) |
| **filtered + literal ≥10/≥10** | **0.913** | **exactly as written** |
| raw + literal ≥10/≥10 | ~0.62 | as written, wrong file |

The gap is now 6.5 points. **Recommendation: take the filtered file with the literal ≥10.** It
follows the published method, needs no footnote defending a threshold of 60, and its residual sits
in one column with a named, stated cause still to be tested.

### 1.9d Their rule as written reaches 0.938 — and boundaries are settled

`output8.ipynb` grew the collection region (bbox padding, venue universe re-scanned from
`raw_POIs.txt`), decoupled it from the catalogue region, applied the papers' stated 200-cap, and
swept every boundary convention. Base numbers: padded universe 323,809 venues / 712,973 check-ins
/ 14,854 users (pad 0.0 reproduces 237,728 exactly).

**Best configuration under their rule exactly as written — match 0.938.** Collection pad 0.4°,
≥10/≥10, threshold counted in-region, check-ins over whole histories, catalogue = 10 km radius,
user-filter first, single pass, 200-cap:

```
                   ours        theirs    ratio
POIs             82,188        80,962    1.015x
categories          432           436    0.991x
users             6,889         7,507    0.918x
check-ins     1,044,080     1,214,631    0.860x
check-ins/user    151.6         161.8    0.937x
```

**Boundary conventions are closed — every variant is worse, so the permissive reading we already
used is the right one:**

```
bnd     >=T -> >T             0.938 -> 0.923    users 6,889 -> 5,684
metric  check-ins -> distinct 0.938 -> 0.923    users 6,889 -> 5,684
order   user-first -> poi     0.938 -> 0.927    users 6,889 -> 5,499
```

*(Correction to §1.9c's estimate: the `>=`/`>` flip was projected at under 1% from a fixture. On
real data it costs 1,205 users — 17.5%. Direction was right, magnitude was not.)*

**Both new axes paid off, isolated:**

```
collection pad 0.0 -> 0.4    0.922 -> 0.938   users 6,396 -> 6,889
200-cap off -> on            0.927 -> 0.938   ck/user 220.6 -> 151.6
```

The cap does exactly what §1.9c predicted for `ck/user`.

**The row worth keeping.** Same rule, iterated, *no* cap: 5,499 users / **82,188 POIs** / 432
categories / **1,212,882 check-ins** — check-ins at **0.9986×** of their 1,214,631, POIs 1.015×,
categories 0.991×. Only `#users` misses, at 0.73×.

So **three of their four columns are reproducible to within 1.5% under their stated rule.** The
irreducible tension is `#users` against `#check-ins-per-user`: their 161.80 sits between our
in-region basis (~100) and whole-history basis (~220), and the 200-cap lands at 151.6 — close,
but no single configuration satisfies both columns at once. That is the residual, and it is
consistent with §1.9b: the columns were assembled from different stages of a pipeline nobody
documents.

**Decision. Build on the filtered file with their rule as written.**

| build | match | fidelity |
|---|---|---|
| raw + Tu=60 + 10 km catalogue | 0.978 | contradicts §4.1 |
| **filtered + literal ≥10/≥10 + pad 0.4 + 200-cap** | **0.938** | **exactly as written** |
| filtered + literal ≥10/≥10 (output7) | 0.913 | as written |
| raw + literal ≥10/≥10 | ~0.62 | as written, wrong file |

Four points apart, and the literal build needs no footnote defending an unstated threshold of 60.
Best over *all* configurations is 0.977, but it requires `>5 distinct POIs` — not their rule, and
not worth the fidelity cost.

### 1.9 Is the literal ≥10 recoverable? (supervisor challenge, open)

Their §4.1 sentence is unambiguous and it is the **only** filtering statement in the paper —
every mention of remove/filter/fewer-than/threshold across all 11 pages was checked, and
Definition 1 fixes a "sequence" as a user's whole chronological history, so the ≥10 is on total
check-ins per user. POIs *are* covered by the same sentence at the same threshold.

Two parts to the challenge, with different answers.

**The POI half is settled: no.** A POI cut at 10 or 20 moves *away* from their count, monotonically.

```
POIs with >=k check-ins, unfiltered 3-city        vs their 80,962
  >=1   237,728   2.94x        >=10    42,167   0.52x
  >=4    87,938   1.09x        >=20    22,217   0.27x
  >=5    74,203   0.92x        >=30    14,411   0.18x
```

**The user half stands on their own numbers, not our choice.** Two columns of their table give
independent estimates of the cut, and they agree:

```
their #users  7,507      -> implies Tu ~ 63
their ck/user 162.80     -> implies Tu ~ 66      (region-independent, POI-independent)
at the stated Tu=10      -> 38,500 users (5.1x theirs), 49.4 ck/user (0.30x theirs)
```

**But ≥10 can still be literally true against a different baseline.** If their starting pool was
already activity-selected, their "10" and our "60" are one cut measured from two places. The
prime candidate is not a guess: Yang's §5 ships `dataset_WWW_Checkins_anonymized.txt` — the
headline file most people download — and §1.5 proved it *is* "raw check-ins of the friendship
users". It is therefore pre-selected, and its global mean is **199.5 check-ins per user** against
their 162.80 (0.82×), where the raw dump is off by 3.3×. The window needs no tuning:

```
14,401  three-city users in the filtered file      (>=10 on a GLOBAL basis keeps ~all)
 7,507  <-- THEIRS, between the two
 5,116  those clearing >=10 measured IN-REGION     (rule E, §1.6)
```

`llmgpr-ten-threshold-test.ipynb` sweeps pool × threshold-basis × reporting-basis × radius with
**the threshold pinned at 10**, and supplies the one measurement every sweep so far lacked: each
user's **global** check-in count, from both files. Verdict thresholds are wired in — ≥0.95 means
the paper's rule is vindicated and we should switch to it.

It also re-opens §1.8. That impossibility proof assumed check-ins are counted **in-region**; it
does not apply if they are counted over whole histories, so the arXiv NYC table may be back in
play. The notebook re-runs that bound on all three bases.

### 1.10 Provenance audit — what is documented vs what we chose

Written for the paper. Everything in the first table is quotable; everything in the second is our
inference and must be stated as such.

**Documented, verbatim.**

| decision | quote | source |
|---|---|---|
| user cut ≥10, POI cut ≥10 | *"users and POIs with less than 10 interactions are removed"* | LLMGPR §4.1 |
| "interactions" = check-ins / visits | *"we remove users with less than 10 check-ins for both datasets, as well as POIs that have less than 10 visits"* | DCLR 2022 |
| boundary keeps ≥10 | *"less than 10 … are removed"* | all three papers |
| cities | *"the cities of New York, Los Angeles, and Chicago"* | LLMGPR §4.1, [28], DCLR |
| source dump | Foursquare = Yang et al. WWW'19 → §5 release | [28] ref [32] |
| sequence = whole history | *"By sorting all check-ins of the user or the group chronologically"* | LLMGPR Def. 1 |
| max sequence length 200 | *"the maximum sequence length is set to 200"* | LLMGPR §4.1, [28] |
| leave-one-out | *"the last check-in POI is for testing, the second last POI is for validation, and all others are for training"* | LLMGPR §4.1 |
| 500 candidates | *"we only pair it with 500 unvisited and nearest POIs within the same region"* | LLMGPR §4.1 |
| group rule | *"if a set of users who are connected on the social network visit the same venue at the same time"* | LLMGPR §4.1, via [3] = CubeRec |
| cold-start | *"200 randomly selected group-level check-in sequences … with fewer than 10 check-ins each"* | LLMGPR §4.6 |

**Undocumented — ours, with the measured cost of each choice.**

| decision | our choice | worth |
|---|---|---|
| which §5 file | **filtered** (`dataset_WWW_Checkins`) | 0.62 → 0.913 — the single biggest lever |
| what `#POIs` counts | region **catalogue** | 0.856 → 0.978 on raw (§1.4) |
| check-in reporting basis | whole history, **capped at 200** | ck/user 220.6 → 151.6 |
| collection region geometry | bbox **+ 0.4°** | +0.016 |
| catalogue region | **10 km** radius from city centre | sets the POI column |
| threshold basis | **in-region** (not whole-history) | large |
| single pass vs iterated | **single** | small |
| filter order | **users first** | +0.011 |
| friendship snapshot | **`friendship_old` only** | avoids leakage; `new` postdates the window |
| co-presence time window | *nothing chosen yet* | **undocumented at every level — see §4** |
| clique vs connected component | *nothing chosen yet* | drives group size |
| group recurrence filter | *nothing chosen yet* | drives `#groups` (§4) |
| "same region" for candidates | *nothing chosen yet* | evaluator |

Nine of our thirteen choices are load-bearing and unstated in any paper in the chain. That is the
honest framing for the write-up: **the stated rule is reproducible to 0.938 only after fixing nine
undocumented degrees of freedom**, and that sentence is itself the reproducibility contribution.

### 1.5 The id-space result

Raw and filtered check-ins **share one user-id space**: 14,401 filtered 3-city users, literal
id intersection 14,401, and a `(venue_id, utc_time)` fingerprint match put 14,400 of them on
the *same* raw id at vote purity 1.00. **Friendships attach to raw check-ins directly — no id
map needed anywhere.**

A structural confirmation fell out: raw check-ins restricted to friendship users give 592,340
check-ins over 102,541 POIs; the filtered file over the same boxes gives 592,341 over 102,541.
**`dataset_WWW_Checkins` *is* "raw check-ins of the friendship users".**

### 1.6 Hypothesis tested and rejected

*Do they restrict to socially-connected users?* **No.** Only 14,400 friendship users are in the
three cities, and all their check-ins total 592,340 — 49% of the needed 1,214,631, before any
filter. Rules E and F (friends + core) scored 0.48.

Rule sweep results (`output3.ipynb`, `match` = mean of min/max ratio across users / POIs /
categories / check-ins):

```
rule                    users      POIs   check-ins  match
A no filter           152,480   237,728   2,227,756   0.48
B >=10 core            30,907    37,113   1,380,405   0.62
C >=10 users only      38,500   218,103   1,901,636   0.55
D friends only         14,400   102,541     592,340   0.69
E friends + core        5,116    11,800     349,519   0.48
TARGET                  7,507    80,962   1,214,631   1.00
```

### 1.7 The filter, resolved

In New York they keep 79% of check-ins and 56% of POIs while dropping 93% of users. A k-core
sheds check-ins roughly in proportion to users; this does not. It is an **activity cut on
users**, and 10 was the only threshold ever tried. `llmgpr-threshold-sweep.ipynb`
(→ `output42.ipynb`) swept 192 rules over user threshold × POI threshold × single-pass/iterated.

**Winner of that sweep: users ≥60 check-ins, POIs ≥3, single-pass — match 0.923.**
**Superseded by §1.4:** the POI-side filter turns out to be spurious. Keep `Tu = 60`; drop `Tp`
entirely and count `#POIs` as the region catalogue instead, which reaches 0.978.

```
                  ours        theirs    ratio
users            7,849         7,507    1.05x
POIs            95,711        80,962    1.18x
categories         425           436    0.97x
check-ins    1,111,794     1,214,631    0.92x
```

Runners-up cluster tightly around it (Tu=60/Tp=2 iterated 0.921, Tu=50/Tp=3 iterated 0.917), so
the **user** threshold is robust at 60 — not 10. What the sweep could not see is that `Tp` was
standing in for a mis-read column: once `#POIs` is read as a catalogue, `Tp = 1` (no POI filter)
wins outright. §1.4 has the resolved form.

### 1.8 Their NYC table is not attainable, and this is provable

For any *k* users, the top *k* by activity hold the **maximum possible** share of check-ins. So
if their reported users hold more than the top-*k* share, no subset whatsoever reproduces it:

```
            their k users must hold   the TOP k actually hold    gap
3 cities              54.5%                    52.6%           +1.9 pts   attainable (bbox noise)
New York              79.3%                    59.9%          +19.4 pts   IMPOSSIBLE
```

**Trust the CIKM 3-city table; treat the arXiv v1 NYC table as unreliable.** No 6,078-user
subset of NYC can hold 79.3% of its check-ins. Either their NYC region is far larger than the
standard bbox, or that table is wrong. Do not tune against it.

## 2. GBSR is not a group denoiser

Read the repo directly and pulled its data files. There is no group anywhere in it. Datasets are
`douban_book`, `epinions`, `yelp` — none LBSN, none with groups.

```
traindata.npy      defaultdict{int user -> list[int item]}    19,539 users / 367,645 pairs
testdata.npy       defaultdict{int user -> list[int item]}    19,539 users /  83,239 pairs
user_users_d.npy   defaultdict{int user -> set[int user]}     18,862 users / 727,384 directed
                                                              (100% reciprocal = stored undirected)
```

No timestamps, no sequences, no group membership. `models/GBSR.py::graph_learner` builds a
Gumbel-sigmoid mask over `self.social_index` — **social edges only** — and the loss is
BPR + L2 + β·HSIC between embeddings from the original vs masked graph.

**Where it actually fits.** Our group rule is co-presence **AND a social edge**, so noisy edges
manufacture spurious groups. Run GBSR on the social graph *first*, then induce groups from the
cleaned graph. Group quality inherits the denoising — a cleaner story than bolting a denoiser
onto groups it was never built for.

### Three traps

1. **The torch port's graph learner never trains.** `torch_version/GBSR.py` does
   `weights = weights.detach()` before building `masked_Graph`, severing every gradient path to
   `linear_1`/`linear_2`. They stay at initialization, so "preference-guided refinement"
   degenerates into a frozen random MLP plus Gumbel noise. The README's hedge that PyTorch
   underperforms is consistent. **Use the TensorFlow version, or remove the `detach()`.**
2. **Model selection happens on the test set.** `rec_dataset.py` loads `valdata` and `testdata`
   from the same `testdata.npy`, and `run_GBSR.py` tracks best NDCG@20 on `testdata` every
   epoch. No validation split.
3. **GBSR is non-sequential top-N.** It is not a next-POI model and cannot serve the Acc@t
   leave-one-out protocol. It is a social-graph denoiser; its recommender is a side effect.

### Social coverage is the binding constraint — and it is thin

Measured on the winning dataset (`output42.ipynb`):

```
users with >=1 friendship_old edge   2,196 / 7,849   = 28.0%     mean degree 3.2
GBSR's own yelp, for scale          18,862 / 19,539  = 96.5%     mean degree 38.6
```

Only those **2,196 users can ever belong to a group**, since the rule requires co-presence
**and** an edge. LLMGPR reports 1,715 groups averaging 3.72 members ≈ 6,380 memberships — about
2.9 groups per eligible user. Demanding but not absurd *if* co-presence among friends is
frequent. Group construction must now measure exactly that.

**Before investing in GBSR, check whether it has anything to do.** A social graph with mean
degree 3.2 against yelp's 38.6 carries far less redundancy, and GBSR's whole premise is pruning
redundant edges. On a graph this sparse, masking edges may simply destroy signal. A cheap
pre-check: run the plain LightGCN-S baseline with and without the social branch — if the social
branch barely moves the metric, denoising it cannot matter either.

---

## 3. Artifacts

All at repo root unless noted. Every notebook was executed end-to-end against synthetic
fixtures before being handed over; the four `output*.ipynb` files are real Kaggle runs.

| File | What it does | State |
|---|---|---|
| `section3-llmgpr-match.ipynb` | §3 stats + §3↔§5 id probe | ✅ run → `notebook output.ipynb` |
| `raw-checkins-llmgpr-test.ipynb` | §5 raw category test + stats | ✅ run → `output-2.ipynb` |
| `llmgpr-rule-sweep.ipynb` | 6 filter rules, id-space test, emit | ✅ run → `output3.ipynb` |
| `llmgpr-threshold-sweep.ipynb` | 192-rule threshold grid | ✅ run → `output42.ipynb` |
| `llmgpr-poi-column-proof.ipynb` | closes the `#POIs` column: excess budget, survivor curve, radius sweep; emits POI coordinates | ✅ run → `output5.ipynb` |
| `llmgpr-catalogue-recovery.ipynb` | radius × per-city radii × 4 accounting conventions | ✅ run → `output6.ipynb` (**0.978**); since patched: finer Tu grid, emit no longer ships a 5 km Chicago |
| `llmgpr-boundaries-and-region.ipynb` | boundaries × grown collection region × 200-cap, catalogue decoupled | ✅ run → `output8.ipynb` (**0.938** as written; boundaries all worse) |
| `llmgpr-ten-threshold-test.ipynb` | can the literal ≥10 work on a pre-selected pool? adds per-user **global** counts; re-tests §1.8 | ⏳ built, tested, NOT RUN — superseded in priority by the row below |
| `llmgpr-filtered-file-literal10.ipynb` | their rule literally (≥10/≥10) on the FILTERED file | ✅ run → `output7.ipynb` (**0.913**); since patched: `cap ∈ {none,200}` axis + memoised sweep — **re-run** |
| `src/to_gbsr.py` | our data → GBSR's three `.npy` files | ✅ validated vs GBSR's yelp |
| `src/probe_section3.py` | CLI form of the §3 probe | ✅ smoke-tested |
| `group-recommendation-using-fsq-section3.ipynb` | the original; **superseded** | ⚠️ wrong dump |

### The working dataset

**Current build — `output6.ipynb`** (the recovered reading, §1.4). Re-run
`llmgpr-catalogue-recovery.ipynb` once after the patch so the emit uses the uniform 10 km
catalogue rather than output6's 5 km Chicago:

| File | Contents |
|---|---|
| `llmgpr_cat_checkins.parquet` | 7,849 users / 1,191,781 check-ins (full 3-city histories, Tu ≥ 60) |
| `llmgpr_cat_catalogue.parquet` | the ranking catalogue: 82,188 venues with lat/lon/category |
| `llmgpr_pois_xy.parquet` | all 237,728 venues with coordinates (`output5.ipynb`) |
| `llmgpr_final_friendship_old.parquet` | 3,555 edges over 2,196 users (`output42.ipynb`, still valid — same 7,849 users) |

Superseded: `llmgpr_final_checkins.parquet` (7,849 / 95,711 / 1,111,794) applied the spurious
`Tp = 3` filter. Its user set is the same; only the check-ins and POI set differ.

**Download these from that Kaggle run and commit them** — they are the input to everything
downstream, and regenerating costs a full re-run. Nothing else needs re-deriving.

To rebuild anyway: run `llmgpr-threshold-sweep.ipynb` with the `output3` notebook attached via
**+ Add Input → Your Work** (minutes), or with Internet on and nothing attached (~25 min).

### Downloading the dumps

`gdown` **fails** on §3: it is a legacy Drive id requiring a `resourcekey`, and gdown's fuzzy
parser normalizes the URL to `uc?id=...` and drops it → 403 HTML → `FileURLRetrievalError`.
Use `drive.usercontent.google.com`, which honours it:

```bash
# Section 3 — dataset_TIST2015.zip, 775,746,915 bytes
curl -L --fail -o tist2015.zip "https://drive.usercontent.google.com/download?id=0BwrgZ-IdrTotZ0U0ZER2ejI3VVk&export=download&resourcekey=0-rlHp_JcRyFAxN7v5OAGldw&confirm=t"
# Section 5 — dataset_WWW2019.zip, 2,684,000,558 bytes
curl -L --fail -o dataset_WWW2019.zip "https://drive.usercontent.google.com/download?id=1PNk3zY8NjLcDiAbzjABzY5FiPAFHq6T8&export=download&confirm=t"
```

Kaggle disk is 20 GB. §5 fully extracted is ~7.8 GB plus the 2.5 GB zip — delete the zip after
extraction. All notebooks here do that already.

### Bounding boxes (confirmed correct — do not widen)

```python
"New York":    lon -74.3  .. -73.6,  lat 40.4 .. 41.0
"Chicago":     lon -88.0  .. -87.5,  lat 41.6 .. 42.1
"Los Angeles": lon -118.7 .. -117.6, lat 33.6 .. 34.4
```

Yang's own city centres, from `dataset_TIST2015_Cities.txt`: NY (40.707864, −73.905237),
Chicago (41.826546, −87.641298), LA (34.000002, −118.250001). At R = 25 km these reproduced
§3's NYC POI count to within 1.9%.

---

## 4. Group construction must be rebuilt

Independent of which dataset wins. The original run produced 20,676 co-presence events over
**17,701 distinct member-sets — 1.17 check-ins per group against LLMGPR's 7.34**. Leave-one-out
needs ≥3 per group; with only 2,975 spare events beyond one-each, at most 1,487 sets (8.4%)
could ever qualify. We generate *more* raw events than they do (20,676 vs 12,594) — what we lack
is **recurrence**, not volume.

### 4.0 Our 1.17 is what the cited procedure actually produces

LLMGPR cites [3] = **CubeRec** (SIGIR'22) for the group rule. CubeRec built synthetic groups the
same way on two datasets, and reports:

```
                groups   group-item int.   int./group   avg size
CubeRec Yelp    24,103            26,883         1.12       4.45
CubeRec Gowalla 78,453           208,336         2.66       2.31
LLMGPR           1,715            12,594         7.34       3.72
ours            17,701            20,676         1.17       2.76
```

**Our 1.17 sits squarely in the cited method's own range; LLMGPR's 7.34 is 6.6× what that method
yields on Yelp.** And they report only 1,715 groups where the procedure produces tens of
thousands. Fewer groups with far higher recurrence means **they kept only recurring member-sets** —
a filter documented nowhere. Our 17,701 sets can yield at most 1,487 with ≥3 pooled check-ins,
against their 1,715: close enough that a recurrence filter is very likely the explanation.

Two consequences. First, our low recurrence is **not a bug to tune away** — it is what the
published procedure gives, and that is a reportable result. Second, **CubeRec does not specify the
co-presence time window either** (checked directly: it states only "visit the same venue at the
same time"). So the window is a free parameter at every level of the chain, and whatever we pick
must be justified on its own terms and reported.

Causes and fixes:

| Problem | Fix |
|---|---|
| `dt.floor("720min")` — 12h buckets; check-ins a minute apart split, 11h59m apart merge | rolling time window, sorted by time |
| `nx.connected_components` chains strangers through a hub venue | cliques |
| one row per event; groups never recur | persistent member-sets with **pooled sequences** |
| `#groups` means different things in cells 21 and 22 | groups = distinct member-sets; events = group check-ins |
| `friendship_new` in the union | `friendship_old` only |

---

## 5. Reference numbers

LLMGPR Table 1 (CIKM'25 camera-ready, 3 cities):

| | users | groups | POIs | cats | user ck | group ck | ck/user | ck/group | users/group |
|---|---|---|---|---|---|---|---|---|---|
| Foursquare | 7,507 | 1,715 | 80,962 | 436 | 1,214,631 | 12,594 | 162.80 | 7.34 | 3.72 |

arXiv v1 of the same paper is **NYC only**: 6,078 / 1,557 / 63,445 / 436 / 923,856 / 10,899 /
152 / 7 / 3.67. LA + Chicago add only ~24% over NYC alone — worth a footnote when citing.

**Protocol details recovered from [28]** (§1.9b), which LLMGPR omits:

- *"we divide each city into 5 regions with k-means clustering"* — so the "same region" in
  *"500 nearest unvisited POIs within the same region"* is a **k-means cluster of POIs inside a
  city**, not a bounding box or a radius. Build the sampler that way.
- [28] compares against the **200** nearest candidates; LLMGPR raised it to 500.
- Leave-one-out is applied **per sequence**, maximum sequence length 200.

**Metric incomparability — build the evaluator to their protocol from the start.** LLMGPR ranks
each ground-truth POI against **500 nearest unvisited POIs in the same region**, reporting
HR@5/10 and NDCG@5/10. Full-catalogue ranking over ~80k POIs produces numbers an order of
magnitude smaller. The two can never appear in one table. Since this track is starting clean,
implement the 500-candidate sampler as the primary evaluator rather than retrofitting it.

---

## 6. Immediate next steps

Table 1 is recovered at 0.978 (§1.4). One question re-opened by supervisor review: whether the
paper's literal ≥10 works on a pre-selected pool (§1.9) — that is step 0 and it may change which
build we use. Group construction is the critical path behind it.

0. **Re-run `llmgpr-filtered-file-literal10.ipynb`** with the new `cap` axis (§1.9c). The
   literal ≥10 already reaches 0.913 on the filtered file; the 200-cap on sequences is the last
   stated mechanism that could close the `#users`/`ck-per-user` residual. If it clears ~0.95,
   adopt that build and the threshold-of-60 problem disappears entirely. Emits
   `llmgpr_filt_checkins.parquet` + `llmgpr_filt_catalogue.parquet`, which then replace the
   `llmgpr_cat_*` files as the working dataset.
0c. **Then `llmgpr-ten-threshold-test.ipynb`** (§1.9). If the literal ≥10 reaches ≥0.95 on
   the filtered release, switch to it — following their stated method exactly is worth more than
   a 0.978 that needs a footnote. If it does not, we keep Tu=60 and justify it with the
   two-independent-estimates argument, which is a stronger defence than "it fit best".
0b. **Re-run `llmgpr-catalogue-recovery.ipynb`** (minutes; attach output3 + output5). Only to
   re-emit, not to re-decide: the patched version steps Tu through 55–80 to close the 1.05× user
   residual, and its emit no longer ships output6's 5 km Chicago. Take
   `llmgpr_cat_checkins.parquet` and `llmgpr_cat_catalogue.parquet` from it. **Do not re-open the
   reading** — §1.4 is settled, and the per-city radii there are a fit, not a finding.
1. **Rebuild group construction** per §4, on `llmgpr_cat_checkins.parquet`
   (7,849 users / 1,191,781 check-ins) plus
   `llmgpr_final_friendship_old.parquet` (3,555 edges, 2,196 users). The measurement that
   matters: **how many member-sets reach ≥3 pooled check-ins**, since that is what
   leave-one-out needs. LLMGPR claims 1,715 groups at 7.34 check-ins each; the previous attempt
   managed 1.17. If the gap persists on the correct dataset with a rolling window and cliques,
   that is the finding to report.
2. **Then decide on GBSR** using the degree-3.2 pre-check in §2. Convert with `src/to_gbsr.py`;
   prefer the TensorFlow implementation, or patch the `detach()`.
3. **Build the evaluator to their 500-candidate protocol from the start** (§5). It ranks against
   the 500 *nearest* unvisited POIs, so it needs venue coordinates and a catalogue to draw
   candidates from — `llmgpr_cat_catalogue.parquet` (step 0), or `llmgpr_pois_xy.parquet`
   (`output5.ipynb`) for the unrestricted set.
4. Baselines a reviewer will demand: groups — MF-AVG, AGREE, GroupIM, CubeRec, HHGR, MICL;
   individuals — LLM4POI, LLMMove, Diff-POI, STAN, DRAN, META-ID, GeoMF.

**Do not tune against the arXiv v1 NYC table** (§1.8) — it is provably unattainable.
