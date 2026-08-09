# The group pipeline, stage by stage

Companion to `README.md`. This is the *why* — what each stage does, what was measured, which
design choices were forced by the data, and what is still unproven.

Everything below was measured on Foursquare NYC (TSMC2014): **1,073 users, 5,120 POIs,
147,539 check-ins**, per-user chronological 70/10/20, all mining restricted to the train split.

---

## 1 · Group construction — `src/build_groups.py`, `src/affinity.py`

### The finding that forced the design

The natural task — mine real groups that move together, POI A → POI B, predict B — **does not
survive contact with the data**. Across window ∈ {30, 60, 120, 180} min and horizon ∈ {240, 720,
1440} min, with membership matched by exact set *and* by Jaccard ≥ 0.34:

```
real group -> group transitions:   17 .. 62      (GROUP_REC_PROPOSAL's "trainable" bar: 5,000)
relaxed to ">=2 members visit the same next POI within 24h":   70 .. 182
```

There is no parameter setting that makes real group transitions trainable. `phase0_diagnostics.py`
D3 and the older `legacy/build_group_relations.py` agree independently (~21 and ~22). So the
21 real transitions are reported as a curiosity set, and the trainable task is built the standard
way (AGREE / GroupIM / KCGRS "occasional groups"): **a real individual next-POI event becomes a
group event by adding companions.**

### Real co-presence: bounded anchor windows, not chaining

Each check-in opens a window `[t, t+60min]`; members are the distinct users inside it; nested
subsets are dropped. **Every group therefore spans ≤ 60 min by construction.**

The earlier approach (`legacy/group-kg.ipynb`) used single-linkage clustering in time, which
chains consecutive check-ins. Measured on the same data with the same 60-minute parameter:

```
single-linkage:   3,073 groups,  max size 87,  max span 713 min   <- a 60-min window
anchor windows:   2,608 groups,  max size  8,  max span  60 min
```

An 87-person "group" spanning 12 hours is a busy venue's evening, not a group.

### The three regimes = KCGRS's own taxonomy, actually constructed

KCGRS names Established / Occasional / Random and never says how any is built.

| regime | construction | train n |
|---|---|---|
| `established` | a **clique** in the affinity graph — every pair above threshold, not just anchor-to-member | 18,969 |
| `occasional` | a subset of a **real observed co-presence set**, rare venue, before *t* | 4,794 |
| `random` | uniform sampling | 10,881 |

### The affinity function is validated, not asserted

Real co-presence is too thin to be the group *source* (2,608 events, 80% pairs) but it is an
excellent **label**. Scoring every candidate signal against it over all 575,128 user pairs, with
each signal recomputed so it cannot contain the co-presence events that produced the labels
(encounters forced ≥3 h apart):

```
taste (category L2 cosine)        AUC 0.6477      <- strongest single signal
rhythm (hour-of-week cosine)      AUC 0.6459
territory (locality cosine)       AUC 0.6218
far co-visit (same POI, >=3h)     AUC 0.5522
------------------------------------------------
z-sum of the four                 AUC 0.7219

top 1.0% of pairs -> 5,752 pairs / 954 users, 8.68% precision vs real co-presence, 12.1x lift
```

> **Methodological trap, documented because it cost a wrong conclusion once.** Scoring
> `same POI same day` *without* the ≥3 h gap gives AUC 0.98, which looks like a triumph and is
> pure tautology: two users co-present within an hour are necessarily at the same POI on the same
> day, so the feature contains the label. Always keep the gap larger than the co-presence window.

### Filters, each chosen from a measurement

| # | filter | applies to | why |
|---|---|---|---|
| 1 | bounded 60-min window, nested subsets dropped | real groups | span ≤ window by construction |
| 2 | venue ≤ 50 distinct visitors | occasional | Crandall et al. — co-presence at a tourist magnet means nothing |
| 3 | ≥1 anchor–companion pair met ≥2× | occasional | 56.9% of events were one-off in *every* pair |
| 4 | co-presence event strictly before *t* | occasional | causality |
| 5 | top-1% affinity **+ full clique** | established | a star around the anchor is not a group |
| 6 | every member causally knew the target's category | all | removes selection-by-hindsight |
| 7 | every member ≥5 check-ins before *t* | all | otherwise members are placeholders |
| 8 | joint history not 100% anchor | all | 8.0% of groups had a one-person "group trajectory" |

Filters 3, 5, 6 and 7 are **asserted at build time**, not just applied.

**Why filter 3 is "one recurring pair" and not "all pairs recur".** The stricter rule sounds
better and destroys the data:

```
venue <=50 only              1,523 events   2:1343 3:103 4:30 5:24 6:12 7:4 8:7
+ >=1 pair met >=2x            657 events   2:520  3:67  4:25 5:22 6:12 7:4 8:7
+ ALL pairs met >=2x           535 events   2:520  3:15   <- everything above size 3 gone
+ exact member SET recurs      478 events   2:470  3:8    <- same
```

A 5-person outing needs all 10 of its pairs to recur, which never happens. A real group is a core
dyad plus occasional joiners.

### Output

```
data/groups/
  constructed_groups.csv          33,831   one row per GROUP (regime, size, members, n_examples)
  constructed_group_members.csv            long form, for joins
  ephemeral_groups.csv             2,608   real co-presence EVENTS
  group_members.csv                        long form -> MEMBER_OF in the KG
  co_attended.csv                  4,133   IDF-weighted user-user ties
  real_group_transitions.csv          21   genuine group moves; a curiosity set, not trainable
  group_examples_{split}.jsonl    34,644 / 6,486 / 13,420   the task data
  groups_manifest.json                     every count, every config value, the affinity validation
```

IDs reuse the existing spaces: `poi_idx` (0–5119, identical to `poi_metadata_NYC.csv` and to the
`<poi_i>` tokens), `user_id`, `venue_id`. The only new id is `group_id`. `example_id` in the JSONL
is a *separate namespace* from `ephemeral_groups.csv`'s integer `group_id`, deliberately, so the
two can never be joined by accident.

---

## 2 · Knowledge graph — `src/build_kg.py`

The POI structural layer is rebuilt from `poi_metadata` + train check-ins, and the group layer is
added **on top of it**, not instead. 9,445 entities / 175,727 triples / 11 relations.

Node counts reproduce the Stage-1 documentation exactly (CATEGORY 500, LOCALITY 142, REGION 2,
POI 5,120), which is asserted in the notebook — a cheap guard against a silently different graph.

`PREFERS_CATEGORY` is the highest-value new relation: it places *users* in the taxonomy, so they
acquire a meaningful hyperbolic depth instead of floating. Each user attaches to the deepest node
their behaviour justifies (dominant child must hold ≥ τ of their visits), so a specialist reaches
a leaf and a generalist stops at level 1. Attaching every user to every leaf they touched would
put all users at the same depth and destroy exactly the signal we want.

Also emitted: `kg_hierarchy.pt`, a `[m, 3]` tensor of `(parent, child, relation)` pairs — the
relation id is carried so the depth regulariser can weight each hierarchy type equally rather
than letting `HAS_CATEGORY` (5,120 edges) drown out `SUBCATEGORY_OF` (417).

---

## 3 · RotH with a depth regulariser — `src/train_roth.py`

### The problem

Diagnostic D1 asks: does taxonomy depth predict hyperbolic radius? On the shipped embeddings:

```
rho = +0.0301   ABSENT      radii by depth:  1.256  1.253  1.258  1.256  1.273
```

Flat. RotH is trained for link prediction, and nothing in that objective asks for a radial
ordering — so it is a hoped-for by-product, and it did not happen.

This matters because the group-consensus mechanism *is* radial: the weighted gyromidpoint of
disagreeing members lies closer to the origin, i.e. at a more general node of the taxonomy. On a
ball with no radial ordering, that statement is decorative.

### The fix

A margin ranking term over the hierarchy pairs from stage 2 — children pushed outward relative to
parents — plus a weak root-pull:

```
depth_weight 5.0, depth_margin 0.3, root_pull 0.01, 120 epochs
```

### The result

```
                      D1 Spearman   verdict    radius by depth (1..5)
OLD (shipped)              +0.0301   ABSENT    1.256 1.253 1.258 1.256 1.273
NEW (+depth reg)           +0.8485   STRONG    0.767 0.868 0.978 1.241 1.501
control (depth_weight=0)   -0.1055   ABSENT    2.609 2.461 2.471 2.552 2.426
```

Reproduced independently on Kaggle (+0.8485) and locally (+0.8478), so it is not a seed artifact.
Without the regulariser the ordering is not merely absent but slightly **inverted**.

> **A high ρ with non-monotonic per-depth radii is a false pass** — the self-check guards against
> exactly that. Always read the radii table, not just ρ.

### The trade-off, stated rather than buried

Overall link-prediction MRR **drops** (0.1939 with vs 0.2088 control). Per relation, the split is
clean — every hierarchical relation improves, every flat one degrades:

| hierarchy | Δ MRR | | flat / behavioural | Δ MRR |
|---|---|---|---|---|
| GROUP_PREFERS | +0.3940 | | MEMBER_OF | −0.2447 |
| LOCATED_IN | +0.2873 | | CO_ATTENDED | −0.1311 |
| PREFERS_CATEGORY | +0.2108 | | IS_NEAR_TO | −0.0508 |
| HAS_CATEGORY | +0.0796 | | VISITED | −0.0252 |
| SUBCATEGORY_OF | +0.0067 | | FOLLOWED_BY | −0.0031 |

The pooled drop is arithmetic: `FOLLOWED_BY` and `IS_NEAR_TO` are 2,651 of 4,000 eval triples, so
the overall number is dominated by flat relations. `MEMBER_OF` losing 0.24 is the one that
deserves a sentence in the paper — plausibly `GROUP_PREFERS` now pins groups radially, which
constrains where `MEMBER_OF` can place them.

> **The control run is currently missing** — it was lost with a scratch directory. Re-running
> `--depth-weight 0 --root-pull 0` is ~40 min and it is the row that turns stage 3 from an
> assertion into an argument.

---

## 4 · POI-POI triples for the adapter — `src/build_poi_poi_triples.py`

Feeds the KG-triple-preservation term in `stage6b_run2_server.ipynb` §6b. Reads the tensor-format
KG from stage 2 via `--kg-dir` — no crosswalk file, no gpickle, no dependency on any external
project.

`--derive taxonomy` matters: both *native* POI-POI relations (`IS_NEAR_TO`, `FOLLOWED_BY`) are
**flat**, so without derived taxonomy-sibling relations the TransE term has no hierarchical
structure to preserve — which is the whole point of the loss.

---

## 5 · What is still unproven

**Nothing has been tested at the decision layer.** D1, link prediction, group construction — all
upstream. A hierarchical ball does not by itself imply better recommendations.

**The scoring path may discard the geometry.** The existing `tied` path computes
`MLP(h) · logmap0(z_p)`, a Euclidean inner product, and an inner product cannot express a geodesic
distance. `objective.py::HyperbolicScorer` exists to fix this. If the hyperbolic-vs-Euclidean
ablation is run on the current path and comes back null, that is an *architectural* artifact and
would license a wrong conclusion.

**The group baselines have not been run.** AVG / least-misery / most-pleasure over the existing
individual checkpoint need no training. If a trivial fuse matches the group model, the paper's
premise changes. Run them before the fine-tune, not after.

**Comparability with the two group-rec papers is limited**, and the limits should be stated
explicitly rather than papered over:

| | KCGRS (DSS 2025) | MAC-GPR (WWW 2026) | this work |
|---|---|---|---|
| dataset | Yelp, 4 cities | Gowalla / Foursquare-US / Swarm | FSQ-NYC (TSMC2014) |
| sequential? | **no** | **no** | **yes** |
| group source | Yelp friendship graph | ephemeral, 1-h co-location | ephemeral + affinity cliques |
| group sizes | 5–10 | 20–46 | 2–5 |
| metrics | Hit@10, NDCG@10 | GS, Fairness, Agreement, P@k, nDCG@k | Acc@1/5/10, MRR |

The nearest comparator by scale is **KCGRS Tucson** (1,359 users / 154,101 visits vs our 1,073 /
147,539), where KCGRS reports Hit@10 = 29.95 at group size 5. Only k=5 overlaps with their range;
MAC-GPR's 20–46 is unreachable on a single-city dataset — at their own 1-hour window FSQ-NYC
yields **zero** groups of size ≥20.

Two cautions before citing them:

- Neither paper reports **how many groups** it formed. Ours is therefore not comparable to a
  published count, because no published count exists.
- KCGRS's Tampa column is internally inconsistent — the same baselines score ~2.7× their
  Philadelphia values, and `GeoSoCa-Add` reports Hit 20.63 / NDCG 50.89, i.e. **NDCG@10 > Hit@10**,
  which is impossible with a single relevant item. Cite Philadelphia and Tucson; leave Tampa alone.

Our candidate set is **all 5,120 POIs, no negative sampling**. Many group-rec papers rank against
a small sampled candidate set, which inflates Hit@k substantially. State this explicitly — it is a
strength, and it pre-empts the obvious objection.
