# Group-rule sweep: which reading of LLMGPR's group rule should we use?

*Run 2026-11-19 on branch `llmgpr-gowalla`. Script: `src/group_rule_sweep.py`.
Results: `data/group_rule_sweep/{LLMGPR,GOWALLA}_sweep.{csv,json}`.*

## The question

LLMGPR (CIKM'25 §4.1, citing CubeRec) defines a group in one sentence: *"if a set of users who
are connected on the social network visit the same venue at the same time, then they are
regarded as members of a group."* Their paper weights nothing. Friendship is yes/no, members
are an unordered set, and the only weight is a single α = 0.7 on the aggregated member vector.
Three things in that sentence are never defined, by LLMGPR or by CubeRec:

| Undefined | Our pipeline today | Where the value came from |
|---|---|---|
| "same time" | 180-min floor buckets | commit `aa6679d` (26 Aug), ported from `group-recommendation-using-fsq-section3.ipynb`, which used 720 min. No sweep chose it. |
| "connected" | reachable within **5** friendship hops | same notebook (`MAX_HOPS = 5`) |
| what a "group" is | every co-presence event; no recurrence filter | — |

We believed those values were chosen to match LLMGPR's group count. This sweep tests that, and
measures every combination against their Table 1.

## What was measured

| Axis | Values |
|---|---|
| co-presence | `bucket180`, `bucket60` (floor of local time, the pipeline's method); `rolling60` (anchor windows [t, t+60], maximal sets, as in `build_ephemeral_groups`) |
| friend rule | `hops5_cc` (today's rule); `hops1_cc` (direct friends, connected components); `clique` (**every pair** must be direct friends) |
| recurrence | a member-set counts as a group if it was co-present ≥ 1, 2 or 3 times |
| scope | `all` check-ins (comparable to Table 1); `train` (the 70% the pipeline mines groups from) |

Sizes 2–8, as in the pipeline. **Check:** `train / bucket180 / hops5_cc / ≥1` reproduces the
committed pipeline counts exactly (Foursquare 4,804, Gowalla 36,621), and the script asserts it.

**Gowalla is the fair comparison.** Our Gowalla matches theirs (31,667 users / 903k check-ins vs
31,751 / 862k). Our Foursquare pipeline arm (`data/llmgpr`) is a **subset**: 6,889 users /
423k check-ins vs their 7,507 / 1.21M. So Foursquare counts undershoot theirs by construction.

## Results (scope = all check-ins)

**Gowalla.** LLMGPR: **2,186 groups**, 7,738 group check-ins, 3.54 per group, 2.95 users per group.

| co-presence | friend rule | ≥ events | groups | × LLMGPR | group ck | ck/group | users/group | % pairs |
|---|---|---|---|---|---|---|---|---|
| bucket180 | hops5 (today) | 1 | 29,127 | 13.3× | 57,380 | 1.97 | 2.47 | 71% |
| bucket180 | hops5 (today) | 3 | 2,485 | 1.14× | 29,131 | 11.7 | 2.11 | 91% |
| rolling60 | hops5 | 1 | 28,738 | 13.1× | 57,595 | 2.00 | 2.47 | 71% |
| rolling60 | clique | 1 | 6,450 | 2.95× | 36,503 | 5.66 | 2.19 | 84% |
| rolling60 | clique | 2 | 3,471 | 1.59× | 33,524 | 9.66 | 2.13 | 88% |
| **rolling60** | **clique** | **3** | **2,392** | **1.09×** | 31,366 | 13.1 | 2.11 | 91% |
| bucket60 | clique | 3 | 2,083 | 0.95× | 26,530 | 12.7 | 2.09 | 93% |

**Foursquare (pipeline subset).** LLMGPR: **1,715 groups**, 12,594 group check-ins, 7.34 per group, 3.72 users per group.

| co-presence | friend rule | ≥ events | groups | × LLMGPR | group ck | ck/group | users/group | % pairs |
|---|---|---|---|---|---|---|---|---|
| bucket180 | hops5 (today) | 1 | 5,209 | 3.04× | 6,839 | 1.31 | 2.35 | 77% |
| bucket180 | hops5 (today) | 3 | 229 | 0.13× | 1,693 | 7.39 | 2.00 | 100% |
| rolling60 | clique | 1 | 630 | 0.37× | 2,187 | 3.47 | 2.04 | 96% |
| **rolling60** | **clique** | **3** | **210** | **0.12×** | 1,658 | 7.90 | 2.01 | 99% |

Full grids (54 rows per dataset, both scopes, plus median check-ins per group and distinct
venues per group) are in the CSVs.

## Findings

1. **The 5-hop rule doesn't match LLMGPR's group count; it inflates it.** Without a recurrence
   filter it gives 13× their Gowalla count and 3× their Foursquare count. Every group it adds
   is a one-off. On Gowalla, 5 hops turns 6,450 groups into 28,738, yet with ≥3 recurrences
   the two rules agree almost exactly (2,400 vs 2,392). The extra reach adds nothing that recurs.

2. **The group count is matched by the recurrence filter, not by hops or window.** On Gowalla,
   ≥3 recurrences lands at 0.95–1.14× of their 2,186 under **every** friend rule and window.
   This confirms on the full grid what `LLMGPR_GOWALLA.md` found for one setting: their group
   count is a recurrence-filtered subset, and the filter is not documented.

3. **No reading reproduces their group size or their check-ins per group.** Once groups must
   recur, 88–100% of them are pairs (mean size 2.0–2.1). LLMGPR reports 2.95 (Gowalla) and
   3.72 (Foursquare). On Gowalla ours average 9–13 check-ins per group against their 3.54. That
   isn't a matter of counting distinct venues instead of visits: distinct venues per group are
   still 8–11. Their construction contains a step the paper doesn't describe, and this family
   of rules cannot recover it. **We stop trying to match their table exactly and report the gap.**

4. **Foursquare can't reach their counts on the pipeline subset.** Direct-friend groups top
   out at 630 of their 1,715 even without a recurrence filter. With ≥3 recurrences they number
   210, which is too few to train on. Foursquare groups need the full recovered dataset
   (`llmgpr_cat_checkins.parquet`, 7,849 users / 1.19M check-ins), not the 423k subset.
   *Open question: why the pipeline's Foursquare arm is a subset.*

5. **Window choice barely matters once groups recur.** At ≥3 recurrences, bucket60, bucket180
   and rolling60 differ by under 15% on both datasets.

## Recommendation

> **Superseded in part by Part 2 below.** For matching LLMGPR's Gowalla, "same time" = the
> same minute fits far better than any 60/180-min window. The table below is still the
> recommendation for groups that recur often enough to train on.

| Choice | Pick | Why |
|---|---|---|
| co-presence | **rolling 60 min** | Floor buckets split check-ins a minute apart across a boundary (the self-check plants exactly that). Rolling windows don't, and the pipeline already has this builder (`build_ephemeral_groups`). |
| friend rule | **clique (every pair are direct friends)** | The cleanest reading of "connected", and nobody joins a group through a friend-of-a-friend. Counts are within a few % of `hops1_cc`. |
| recurrence | **≥ 3 events per member-set** | Matches their group count on Gowalla (1.09×), and leave-one-out needs ≥ 3 check-ins per group (train / val / test). |

**Gowalla under this rule:** 2,392 groups, 31,366 group check-ins, 3,685 users (all scope);
1,731 groups on the train split.
**Foursquare:** rebuild on the full dataset first (finding 4).

## Not done yet

- Applying the rule in `build_groups.py`. It currently feeds the `occasional` regime and the
  co-attendance tie graph, so changing it moves the KG and the examples. That needs your go-ahead.
- Weighted social edges (step 2): friendship (yes/no) + co-attendance strength (`weight_idf`) +
  optional affinity, as separate relations.
- Re-running this sweep on the full Foursquare dataset.

---

# Part 2: what LLMGPR actually uses (research + probes, 2026-11-19)

Prompted by: *"they must use something logical in their group construction; check what data
version they used."* Below are the literature trail and three further probes.

## 2.1 Literature trail

| Source | What it says about groups |
|---|---|
| LLMGPR arXiv v1 (Nov 2024, 2411.13415v1) | NYC only, 2 datasets. Same sentence, no window. FSQ 6,078 users / **1,557 groups** / 10,899 group ck / 3.67 users per group; Weeplace 3,795 / 867 / 10,404 / 4.58. Weeplace cited as Liu 2013. |
| LLMGPR arXiv v2 = CIKM camera-ready (Aug 2025) | 3 cities, 3 datasets. Same sentence; **no window, size or recurrence filter**; no data or code statement. Weeplace now cited as Long et al. 2023b (their own decentralised-POI paper). |
| Code release | **None.** No GitHub; Papers with Code: "no code available". Same as every paper in the chain (see `LLMGPR_TRACK.md`). |
| CubeRec (SIGIR'22, their [3], co-author Jing Long) | Same sentence verbatim, citing **[32] GroupIM** and **[41] Yin et al. SIGR (ICDE'19)**. No window. Gowalla: 78,453 groups, 2.66 items per group, size 2.31. |
| Yin et al. SIGR (ICDE'19) / CAGR (TKDE, arXiv 2010.00813) | Origin of the sentence ("visit the same restaurant at the same time"), Yelp LA. **No window.** Yelp 24,103 groups, size 4.45, 1.12 interactions per group. |
| **GroupIM (SIGIR'20, arXiv 2006.03736)** | **The only paper in the chain with an explicit rule:** *"check-ins at the same POI within 15 minutes by an individual and her subset of friends … constitutes a single group interaction"*; users and items with ≥5 check-ins. Uses **Weeplaces and Gowalla from yongliu.org**, the same source as LLMGPR's. Gowalla (restaurants, all US): 44,565 groups, **2.8 users and 3.47 items per group**, close to LLMGPR's Gowalla 2.95 / 3.54. Repo (CrowdDynamicsLab/GroupIM) ships Weeplaces group IDs only: no timestamps or sequences, so LLMGPR could not have reused them, and the counts differ (22,733 groups). |

So the rule's lineage is **SIGR (no window) → GroupIM (15 min, ego-centric) → CubeRec → LLMGPR**,
and no paper after GroupIM states a window.

## 2.2 Probes

Scripts: `src/group_rule_sweep.py` (now also `ego<W>` = GroupIM's rule and `bucket1` = same
minute), `src/group_identity_probe.py` (what counts as one group; minutes or `--seconds`),
`src/group_fullgraph_probe.py` (paths through users outside the dataset). Results under
`data/group_rule_sweep/{groupim_rule,identity,identity_tight,identity_seconds,minute,fullgraph}/`.

**(a) GroupIM's exact rule (15 min, anchor + friends) doesn't change the picture.** Gowalla:
5,633 groups at ≥1 check-in, 2,003 at ≥3; size 2.10–2.22. Same behaviour as the clique rule.

**(b) What counts as one group.** We tried an exact member-set, an anchor person plus every friend
ever co-present with them, and circles of friends (connected components). On Gowalla at 15–60
min every reading gives **4–9× their group check-ins**, while individual check-ins match (1.05×).
Anchors and circles push size up to 2.6–3.9 but inflate check-ins further. Foursquare (subset)
stays at ≤0.71× their group count under every reading, including the old+new friendship union
(+30–50% groups).

**(c) "Same time" = the same minute fits their Gowalla.** Same POI, same calendar minute, direct
friends, no recurrence filter:

| Gowalla | ours | LLMGPR |
|---|---|---|
| groups | 2,030 (0.93×) | 2,186 |
| group check-ins | 7,010 (0.91×) | 7,738 |
| check-ins per group | 3.45 | 3.54 |
| users per group | **2.06** | **2.95** |

Allowing a shared friend anywhere in the **full** 407k-user crawl graph gives 2,238 (1.02×) /
7,342 (0.95×) / 3.28 / 2.06. Exact-second equality is far too strict (108 groups); ±30 s is about
the same as the same minute (2,101 / 9,035).

**(d) Group size is the one statistic nothing reproduces.** Within one minute, 95% of co-present
sets are pairs, so no friendship reading (direct, shared friend, same component of the full
graph) can lift size above about 2.06. Size rises only with long windows or merged identities,
and those overshoot group check-ins by 4–9×. **No single construction matches all four Gowalla
statistics.**

**(e) The minute reading fails on Foursquare.** Same minute gives 119 groups on our subset;
tripling for full-data scale still gives about 360, against 1,715. Foursquare's own numbers
(7.34 ck/group) fit only long windows with ≥3 recurrences (§ Part 1). **On our data, no one
rule fits both datasets.** Either LLMGPR's construction differed per dataset, or our Foursquare
input differs from theirs (it's a 423k-check-in subset; see Part 1, finding 4).

## 2.5 Excluding pairs (groups of 3+ only)

Mean size 2.06 is too low to call these "groups" convincingly. `--min-size 3` (results in
`data/group_rule_sweep/min_size3/`, scope = all check-ins) shows how much real co-presence of 3+
friends exists:

| | Gowalla, ≥1 check-in | Gowalla, ≥3 check-ins | Foursquare subset, ≥1 | Foursquare subset, ≥3 |
|---|---|---|---|---|
| same minute, direct friends | 110 groups | 10 | 2 | 0 |
| GroupIM (15 min, anchor + friends) | 1,013 | 172 | 15 | 0 |
| rolling 60 min, clique | 1,015 (size 3.2) | 217 | 25 | 2 |
| 180-min bucket, direct friends | 1,339 (size 3.4) | 230 | 61 | 0 |

**Real co-presence of three or more friends is rare in both datasets.** On Gowalla it gives
about 1,000–1,300 groups seen once and about 220 that recur; on our Foursquare subset, almost
none. Larger groups therefore can't come from observed co-presence alone. They have to come
from a constructed regime (the pipeline's `established` affinity cliques already build groups of
any requested size, e.g. 3–5) or from merging the real groups (anchor / circle identities in
2.2 (b), sizes 2.6–3.9, at the cost of 4–9× their group check-ins).

## 2.6 Clues inside LLMGPR's own tables

Papers that cite LLMGPR (MAC-GPR WWW'26 in `papers/`, plus a web search) either copy its numbers
or build their own groups (MAC-GPR: one-hour slots); none describes or reuses LLMGPR's groups.
Arithmetic on LLMGPR's two tables, however, shows two regularities that a co-presence rule
wouldn't produce:

| table | groups | group ck | ck / group | remainder | memberships (groups × size) | ÷ users |
|---|---|---|---|---|---|---|
| v1 Foursquare (NYC) | 1,557 | 10,899 | **7.0000** | **0** | 5,714 | **0.94** |
| v1 Weeplace (NYC) | 867 | 10,404 | **12.0000** | **0** | 3,971 | **1.05** |
| v2 Foursquare | 1,715 | 12,594 | 7.3434 | 589 | 6,380 | 0.85 |
| v2 Weeplace | 923 | 11,974 | 12.9729 | 898 | 4,034 | 0.88 |
| v2 Gowalla | 2,186 | 7,738 | 3.5398 | 1,180 | 6,449 | 0.20 |

1. **v1's group check-ins are exact multiples of the group count in both datasets** (odds about
   1 in 10^6 by chance). Either every v1 group had exactly 7 (FSQ) or 12 (Weeplace) check-ins,
   which means fixed-length group sequences, or the column was computed from a rounded mean
   rather than counted. v2's values are not round, so v2 was re-measured.
2. **On Foursquare and Weeplace, group memberships ≈ the number of users (0.85–1.05×); on
   Gowalla, 0.20×.** On our Foursquare data only 28% of users have any friendship edge. A
   friends-at-the-same-place rule therefore can't put close to one membership per user unless
   their social graph is much denser than ours or users sit in many groups. A ratio near 1 looks
   like **users divided into groups** (a partition, e.g. clustering), not co-presence.
   Gowalla's 0.20 is consistent with a co-presence rule, which matches what §2.2 (c) found
   (same-minute fits Gowalla only).
3. **Unconfirmed lead.** v2 cites Weeplace as Long et al. 2023 (Jing Long's decentralised-POI
   work). DCLR (arXiv 2204.06516, same first author) *"identif[ies] user groups"*: it assigns
   each user neighbours by category-distribution similarity and by geographic distance between
   visit centroids. If LLMGPR's Foursquare/Weeplace groups came from that pipeline rather than
   from co-presence, it would explain the near-partition and the larger sizes (3.7–4.4). Nothing
   in LLMGPR says so. This is a hypothesis to put to the authors, not a finding.

**Bottom line:** the evidence suggests LLMGPR's groups were built differently per dataset:
Gowalla consistent with same-minute co-presence; Foursquare/Weeplace consistent with something
that assigns nearly every user to a group. Only the authors can confirm it.

## 2.7 Closest match per dataset (decision 2026-11-19)

Decision: for each dataset, use the reading of LLMGPR's rule that comes closest to its own
Table 1, and state it in the paper. Script `src/group_match_search.py`: 264 configurations per
dataset (11 windows × 4 group identities × min check-ins 1/2/3 × min size 2/3). It scores
**match** = mean of min(ours/theirs, theirs/ours) over #groups, #group check-ins, check-ins per
group and users per group (1.0 = exact). Ranked results are in `data/group_rule_sweep/match/`.

| dataset | best reading | match | groups | group ck | ck/group | users/group |
|---|---|---|---|---|---|---|
| **Weeplace** | anchor + friends within **60 s**; group = the anchor with every friend ever co-present; **≥3** group check-ins | **0.943** | 920 / 923 | 12,775 / 11,974 | 13.89 / 12.97 | 3.96 / 4.37 |
| **Gowalla** (a) | **same calendar minute**, exact member-set, no filters | 0.880 | 2,048 / 2,186 | 7,045 / 7,738 | 3.44 / 3.54 | 2.06 / 2.95 |
| **Gowalla** (b) | anchor ± **3 h**, exact member-set, **≥3 members** | 0.876 | 2,174 / 2,186 | 9,243 / 7,738 | 4.25 / 3.54 | 3.51 / 2.95 |
| Foursquare (subset, old friendship) | anchor ± 1 day, anchor identity, ≥2 ck | 0.698 | 923 / 1,715 | 6,551 / 12,594 | 7.10 / 7.34 | 2.85 / 3.72 |
| Foursquare (subset, **old+new** friendship) | anchor ± 1 day, anchor identity, ≥2 ck | 0.806 | 1,202 / 1,715 | 8,510 / 12,594 | 7.08 / 7.34 | 3.28 / 3.72 |

(ours / theirs.)

- **The rules really are dataset-specific.** Weeplace's best rule scores 0.60 on Gowalla.
- **Gowalla:** the two readings tie (0.880 vs 0.876). **(b) is recommended**: it matches the group
  count to 0.995, and its groups have ≥3 members (mean 3.51). Reading (a)'s groups average
  2.06 members, mostly pairs, which the user judged too small.
- **Weeplace caveat:** our Weeplace has 845,540 check-ins against their 623,654 (1.36×), so
  absolute counts there may be inflated. Users match (4,566 vs 4,560).
- **Foursquare is undecidable on the subset.** Group and check-in counts are capped at about 0.7
  by data volume (423k vs 1.21M check-ins). Check-ins per group (0.97) and size (0.88) already
  fit with **old+new friendship** and a day-scale window, which suggests LLMGPR used both
  snapshots. (We excluded "new" to avoid leakage; for reproducing *their* groups it belongs in.)
  Next: rerun on the full dataset.

Reproduce:
```bash
python src/group_match_search.py --dataset GOWALLA --data-dir ./data/gowalla
mkdir -p /tmp/weeplace && for f in train val test friendship_old poi_metadata; do
  git show llmgpr-Weeplace:data/Weeplace/${f}_WEEPLACE.csv > /tmp/weeplace/${f}_WEEPLACE.csv; done
python src/group_match_search.py --dataset WEEPLACE --data-dir /tmp/weeplace
python src/group_match_search.py --dataset LLMGPR --data-dir ./data/llmgpr \
  --friend-files ./data/llmgpr/friendship_old_LLMGPR.csv ./data/llmgpr/friendship_new_only_LLMGPR.csv \
  --out-dir ./data/group_rule_sweep/match_fsq_oldnew
```

## 2.8 Foursquare: the data version matters (2026-11-19)

Full Foursquare data came from Kaggle (`yosrkharrat/output42`, `yosrkharrat/output8`; output6's
files are gone). Downloads are in `data/llmgpr_full/` (untracked, about 2.5 GB; the raw
`dataset_WWW2019/` folder is 2.1 GB).

**Our two Foursquare builds have different users** (only 3,036 shared), and their social graphs
differ sharply:

| build | users | Table-1 fit (individual columns) | friendship_old edges | users with a friend |
|---|---|---|---|---|
| `output42` (the 0.978 fit) | 7,849 | 0.978 | 3,555 | 2,196 (28%) |
| `output8` (the pipeline's set) | 6,889 | 0.938 | 8,640 | 4,872 (71%) |

LLMGPR's groups put nearly every user in a group (§2.6), which a 28%-friend user set can't do.
On `output42` with 1.11M check-ins the best match is only **0.647**, lower than on the subset,
because friendship is the bottleneck. So **LLMGPR's Foursquare user set looks like `output8`'s,
not `output42`'s**, even though `output42` fits the individual columns slightly better.

`src/prepare_llmgpr_fsq_output8_full.py` rebuilds the output8 users from the raw files
(friendship id map verified: 8,640/8,640 edges identical to the pipeline's):

| check-in scope (output8 users, old+new friendship) | check-ins | best match | reading |
|---|---|---|---|
| pipeline, 10 km of centres | 423k | 0.806 | ±1 day, anchor, ≥2 |
| three-city boxes | 567k | 0.836 | ±1 day, anchor, ≥2 |
| **whole history** | **1.52M** | **0.895** | **±1 h, anchor, ≥1** |

Best Foursquare reading: **anchor + friends within ±1 hour; group = the anchor with every friend
ever co-present; whole check-in histories; old+new friendship** → 1,817 / 13,258 / 7.30 / 2.58
against 1,715 / 12,594 / 7.34 / 3.72. Users per group (0.69) is the only weak statistic.

## 2.9 Summary: closest reading per dataset

| dataset | window | group identity | filter | data | match |
|---|---|---|---|---|---|
| Weeplace | ±60 s | anchor + all co-present friends | ≥3 group check-ins | llmgpr-Weeplace branch CSVs | **0.943** |
| Foursquare | ±1 h | anchor + all co-present friends | none | output8 users, whole history, old+new friends | **0.895** |
| Gowalla | ±3 h | exact member-set | ≥3 members | data/gowalla | **0.876** |

Two of the three best readings use **the same group identity: an individual plus every friend
they were co-present with**, which is GroupIM's "an individual and her subset of friends". The
windows differ per dataset. Group size is the hardest statistic everywhere: we come in under
theirs on Foursquare and Weeplace.

## 2.10 Weeplace: why we have 1.36× their check-ins (2026-11-19)

Our Weeplace (the `llmgpr-Weeplace` branch, `src/prepare_weeplace_csvs.py` there) keeps raw
check-ins in the three cities, then applies the literal ≥10 filter (users first, then POIs, one
pass): **4,566 users (1.00×), 845,540 check-ins (1.356×), 12,931 visited POIs**. LLMGPR reports
4,560 / 623,654. The earlier paper by the same authors ([28], WWW'24) reports the *same* users and
POIs with **923,600** check-ins, so the check-in column changed between their own papers.

Tested on our build (target 623,654):

| reduction | check-ins | ratio |
|---|---|---|
| none | 845,540 | 1.356 |
| drop exact duplicates (user, POI, second) | 845,540 | 1.356 |
| one per user-POI-hour | 825,042 | 1.323 |
| one per user-POI-day | 775,534 | 1.244 |
| drop consecutive repeats of the same POI | 769,735 | 1.234 |
| drop same-POI repeats within 10 / 30 / 60 min | 828k / 826k / 824k | 1.32–1.33 |
| per-user cap 200 (their "maximum sequence length") | 441,011 | 0.707 |
| per-user cap 250 / 300 | 500,538 / 549,890 | 0.80 / 0.88 |
| venues within 8 / 10 km of Yang's city centres | 322,740 / 712,508 | 0.52 / 1.14 |

**No natural rule reproduces 623,654.** A per-user cap would need about 400; a radius would need
about 9.5 km on a knife-edge (8 km → 0.52×, 10 km → 1.14×). Both would be curve-fitting. Users match
exactly, and the check-in column moved by 300k between their own two papers, so this looks like
an unreported step on their side. **We keep our build and report the 1.36× ratio.** Effect on
groups: our Weeplace has more check-ins to co-occur, so group counts may be inflated relative to
theirs. The eight-number score (§2.11) charges for it.

## 2.11 Eight-number score + counting fix (2026-11-19)

**Fix.** `member_set` / `circle` groups used to count each member's view of one outing as a
separate group check-in for windows over a minute (`member_set_outings` in
`src/group_match_search.py` now merges them; planted-case check). Anchor-identity results are
unaffected. Every search and probe was rerun; the earlier §2.2(b) and §2.7 numbers for member-set
and circle readings at windows > 1 min are superseded by the CSVs.

**Score.** `match8` = mean ratio over all eight Table-1 numbers: users, POIs (catalogue:
FSQ 82,188, Gowalla 81,240, Weeplace ≈45,520 from the branch's documented 1.03×), user check-ins,
check-ins per user, and the four group numbers. A group match can no longer hide a data mismatch.

| dataset / data variant | best rule (by match8) | match8 | group match | groups | group ck | ck/group | users/group | users | user ck |
|---|---|---|---|---|---|---|---|---|---|
| **Gowalla** | ±30 s, exact member-set, no filters | **0.941** | 0.906 | 2,101 | 7,434 | 3.54 | 2.07 | 1.00× | 1.05× |
| Gowalla, groups ≥3 members | ±3 h, exact member-set, ≥3 members | 0.856 | 0.737 | 2,174 | 4,297 | 1.98 | 3.51 | 1.00× | 1.05× |
| **Weeplace** | ±60 s, anchor, ≥3 group ck | **0.902** | 0.943 | 920 | 12,775 | 13.89 | 3.96 | 1.00× | 1.36× |
| **FSQ: output8 users, whole history, old+new** | ±1 h, anchor | **0.878** | 0.895 | 1,817 | 13,258 | 7.30 | 2.58 | 0.92× | 1.25× |
| FSQ: output8, whole history, old only | ±3 h, anchor | 0.871 | 0.881 | 1,756 | 13,546 | 7.71 | 2.48 | 0.92× | 1.25× |
| FSQ: output42 (best individual fit) | ±1 day, anchor | 0.790 | 0.647 | 978 | 6,014 | 6.15 | 2.62 | 1.05× | 0.92× |
| FSQ: output8, three-city boxes, old+new | ±1 day, anchor, ≥2 | 0.777 | 0.836 | 1,251 | 9,601 | 7.68 | 3.33 | 0.91× | 0.47× |
| FSQ: pipeline subset, old+new | ±1 day, anchor, ≥2 | 0.731 | 0.806 | 1,202 | 8,510 | 7.08 | 3.28 | 0.92× | 0.35× |

- **Foursquare:** whole-history output8 still wins on all eight numbers (0.878), ahead of the
  best individual fit (0.790) and the three-city-only build (0.777). The cost is stated: 25% more
  check-ins than theirs, some outside the three cities.
- **Gowalla changed with the fix.** The ±3 h / ≥3-member rule had been over-counting; its group
  check-ins drop from 9,243 to 4,297 (1.98 per group, so those groups barely recur) and it falls
  to 0.856. The best Gowalla reading is now **±30 s, exact member-set**: check-ins per group 3.54,
  exactly theirs, and match8 0.941. But its groups average 2.07 members. Choice pending with the user.
- **Weeplace:** unchanged (0.943 on groups, 0.902 on all eight, the 1.36× check-ins charged).

## 2.12 The group files (built 2026-11-19)

`python src/build_llmgpr_match_groups.py` → `data/groups_llmgpr_match/<DATASET>/` (2.6 MB in
all): `groups.csv`, `group_members.csv`, `group_checkins.csv` (time-ordered, who was present,
LLMGPR leave-one-out split), `manifest.json` (exact rule, source, counts, match). Integrity checks
pass on all three: membership rows, sequence lengths, present ⊆ members, anchor ∈ members, time
order, last check-in = test.

| dataset | rule | groups | group ck | ck/group | users/group | users in groups | groups with ≥3 ck (train history) | match (groups) / match8 |
|---|---|---|---|---|---|---|---|---|
| GOWALLA | ±30 s, exact member-set | 2,101 | 7,434 | 3.54 | 2.07 (1,964 pairs) | 3,165 | 543 | 0.906 / 0.941 |
| WEEPLACE | ±60 s, anchor, ≥3 ck | 920 | 12,775 | 13.89 | 3.96 (max 27) | 1,210 | 920 | 0.943 / 0.902 |
| LLMGPR (Foursquare) | ±1 h, anchor, old+new friends | 1,817 | 13,258 | 7.30 | 2.58 (max 17) | 1,817 | 927 | 0.895 / 0.878 |

Notes for whoever uses them:
- **Leave-one-out on short sequences.** LLMGPR's protocol (last = test, second-last = val) leaves
  a 1-check-in group with no group history (test only; its members' own histories are still
  available), and a 2-check-in group with val + test only. That's 1,558 of Gowalla's groups and
  890 of Foursquare's; every Weeplace group has ≥3. Report them or filter them, but say which.
- **Id spaces.** Gowalla = `data/gowalla` ids. Weeplace = `data/weeplace_from_branch` (the
  `llmgpr-Weeplace` branch's ids). Foursquare user_id = the pipeline's `data/llmgpr` user_id, but
  its poi_idx indexes the whole-history venues of `data/llmgpr_full/house8_whole`; **join to
  the pipeline's POIs on `venue_id`**. Some Foursquare group check-ins are outside the three
  cities (whole history).
- **Anchor groups overlap by design.** Every user who appears in a Foursquare/Weeplace group is
  also the anchor of their own group, so the same outing can sit in several groups' sequences.
- **Foursquare uses friendship_new**, which postdates the check-ins. That's right for
  reproducing LLMGPR's groups, but it's a leak if these groups feed our own model's training
  as-is.

## 2.13 Is this safe to use? Assessment + test leakage (2026-11-19)

**Leakage, measured.** A test target counts as leaked when the same outing (same POI, within the
window, ≥2 shared people) also sits in *another* group's training history:

| dataset | test targets | leaked | share |
|---|---|---|---|
| Gowalla | 2,101 | 26 | 1.2% |
| Weeplace | 920 | 165 | **17.9%** |
| Foursquare | 1,817 | 242 | **13.3%** |

It comes from the anchor identity: a and b's outing is in a's sequence and in b's. Per-group
leave-one-out can put it in test for one and in train for the other. **This must be fixed before
any evaluation:** mask every training check-in that is the same outing as another group's val/test
target, or split by a global time cutoff.

**Logical:** the co-presence rule (direct friends, same POI, short window) and the anchor
identity (GroupIM's published "an individual and her subset of friends") are both citable.

**Not safe as they stand, for four reasons:**
1. Leakage (above).
2. **The parameters were fitted to a baseline's statistics**, per dataset (30 s / 60 s / 1 h,
   two identities, different filters), out of hundreds of configurations. Defensible as "the
   closest reconstruction of LLMGPR's setup", not as our own definition of a group.
3. **Anchor groups are ego networks, not groups that went out together.** Members of a Weeplace
   or Foursquare group may never have been out at the same time; each check-in has a different
   subset present (`present` column). Fine for reproducing LLMGPR; weak as a model of joint
   decisions.
4. **Foursquare uses friendship_new** (future information) and check-ins outside the three
   cities. Fine for reproduction; a leak and an inconsistency with the paper's own text for our
   training.

**Recommendation:** two group sets.
- **Reproduction set** (these files, leakage fixed): used only for the comparison with LLMGPR,
  framed as a best-effort reconstruction with our Table 1 next to theirs.
- **Our own set**: one rule for all datasets, chosen on its own merits (old friendship only,
  three-city check-ins, groups that were actually out together, a global time split), for our
  main results and the analyses (group size, heterogeneity).

## 2.14 Can we fit their group *behaviour*, not just totals? (2026-11-19)

Everything LLMGPR publishes about its groups, read in full (v1 and v2):
1. Table 1: #groups, #group check-ins, check-ins per group (= mean outings per group), users
   per group. Means only; no distribution of sizes, recurrence or overlap.
2. Cold-start (§4.6): "200 randomly selected group-level check-in sequences from each dataset …
   with fewer than 10 check-ins each", so at least 200 groups per dataset have under 10 check-ins.
3. §1: group check-ins are "approximately 50–100 times sparser" than individual ones.
4. v1 only: check-ins per group exactly 7 (FSQ) and 12 (Weeplace), i.e. fixed-length group
   sequences in that version (§2.6).
5. Protocol: maximum sequence length 200; leave-one-out per sequence.

Our group files against 2–3:

| | groups with <10 ck (need ≥200) | sparsity ours / theirs | ck per group p25 / median / p75 / max | seen once / twice / ≥3 |
|---|---|---|---|---|
| Gowalla | 1,987 ✅ | 121× / 111× | 1 / 1 / 3 / 494 | 57% / 18% / 26% |
| Weeplace | 525 ✅ | 66× / 52× | 4 / 8 / 16 / 163 | 0% / 0% / 100% |
| Foursquare | 1,479 ✅ | 114× / 96× | 1 / 3 / 7 / 303 | 35% / 14% / 51% |

**We already fit every group statistic they published**, and the extra constraints are weak and
satisfied. The behaviour beyond the means (how recurrence and size are distributed) isn't in
the paper, so it can't be reverse-engineered from it. Only the authors can supply it.
Their 200-length cap would truncate a handful of our longest sequences (max 494 / 163 / 303);
applying it is a cheap step toward faithfulness.

## 2.15 Cap + leakage fix applied (2026-11-19)

`src/build_llmgpr_match_groups.py` now:
1. **Caps each group sequence at its most recent 200 check-ins** (LLMGPR §4.1's maximum
   sequence length), before the leave-one-out split.
2. **Masks cross-group leaks.** A train check-in that is the same outing as another group's
   val/test check-in (same POI, within the window, at least 60 s for Gowalla's 30 s rule, ≥2
   people in common) gets `split = masked`. Only train rows change, so every group keeps exactly
   one test and its val. **Drop `masked` rows from training targets and histories.**
   The build asserts zero leaking test targets afterwards.

| dataset | cap: groups truncated / check-ins dropped | leaking test targets before → after | train rows masked | groups / group ck / ck per group / size | match |
|---|---|---|---|---|---|
| Gowalla | 2 / 304 | 26 → **0** | 42 | 2,101 / 7,130 / 3.39 / 2.07 | 0.886 |
| Weeplace | 0 / 0 | 165 → **0** | 487 | 920 / 12,775 / 13.89 / 3.96 | 0.943 |
| Foursquare | 2 / 206 | 242 → **0** | 552 | 1,817 / 13,052 / 7.18 / 2.58 | 0.895 |

The "before" counts equal the independent measurement in §2.13, which cross-checks the masking.
The cap lowers Gowalla's check-ins per group from 3.54 to 3.39 (one pair alone had 494), so
its group match goes from 0.906 to 0.886. Integrity checks pass on all three (sequence lengths
≤ 200, one test per group, present ⊆ members, last check-in = test).

## 2.3 What this means for comparing with LLMGPR

- **We can still compare.** A comparison runs their method and ours on the **same** groups
  with their protocol (leave-one-out, 500 nearest candidates; `src/eval_protocol.py`). Matching
  their dataset statistics is not required for that, and their groups were never released, so
  nobody can reuse them.
- What we cannot claim is that our groups are *identical* to theirs. The paper should state our
  rule precisely and report our Table-1 statistics next to theirs.
- **Best-supported reading for Gowalla: same minute + direct friends** (three of four statistics
  within 10%). For Foursquare, the question stays open until the full dataset is in place.

## 2.4 Next checks (not run yet)

1. **Weeplace** (the `llmgpr-Weeplace` branch has the data): a third dataset, with the largest
   groups (4.37) and most check-ins per group (12.97). It tells us whether same-minute fits
   two datasets or one.
2. **Full Foursquare** (`llmgpr_cat_checkins.parquet`, 7,849 users / 1.19M check-ins, from the
   `llmgpr-catalogue-recovery` Kaggle run): the subset can't decide anything.
3. **Email the authors** (Jing Long, UQ) for the window and the group definition. Every free
   parameter here is one sentence for them to answer.

Sources: [LLMGPR v2](https://arxiv.org/html/2411.13415v2), [LLMGPR v1](https://arxiv.org/html/2411.13415v1),
[CubeRec](https://arxiv.org/abs/2204.02592), [GroupIM](https://arxiv.org/abs/2006.03736),
[GroupIM code + Weeplaces](https://github.com/CrowdDynamicsLab/GroupIM),
[Yin et al. CAGR/TKDE](https://arxiv.org/abs/2010.00813),
[Papers with Code entry](https://paperswithcode.com/paper/unleashing-the-power-of-large-language-models-1).

## Reproduce

```bash
python src/group_rule_sweep.py --self-check
python src/group_rule_sweep.py --dataset LLMGPR  --data-dir ./data/llmgpr    # ~10 s
python src/prepare_gowalla_csvs.py   # only if the gitignored Gowalla CSVs are missing
python src/group_rule_sweep.py --dataset GOWALLA --data-dir ./data/gowalla   # ~4-8 min, 5-hop BFS dominates
# Part 2
python src/group_rule_sweep.py --dataset GOWALLA --data-dir ./data/gowalla --copresence ego15 ego60 rolling15 \
       --rules clique hops1_cc --out-dir ./data/group_rule_sweep/groupim_rule
python src/group_rule_sweep.py --dataset GOWALLA --data-dir ./data/gowalla --copresence bucket1 \
       --rules hops5_cc hops1_cc clique --scopes all --out-dir ./data/group_rule_sweep/minute
python src/group_identity_probe.py --dataset GOWALLA --data-dir ./data/gowalla               # 15/60 min
python src/group_identity_probe.py --dataset GOWALLA --data-dir ./data/gowalla --windows 0 1 2 5 \
       --out-dir ./data/group_rule_sweep/identity_tight
python src/group_identity_probe.py --dataset GOWALLA --data-dir ./data/gowalla --seconds --windows 0 10 30 \
       --out-dir ./data/group_rule_sweep/identity_seconds
python src/group_fullgraph_probe.py           # needs data/gowalla_raw/gowalla_friendship.csv (gitignored)
# (the same LLMGPR commands work with --dataset LLMGPR --data-dir ./data/llmgpr)
```
