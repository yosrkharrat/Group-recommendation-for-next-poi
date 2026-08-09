# From individual to group next-POI recommendation

**A design proposal for the next phase of the diffusion-LLM + hyperbolic-embedding project**
Working name: **HyGro-POI** — *Hyperbolic Group Consensus for Next-POI Recommendation with a Diffusion LLM*
Drafted 31 July 2026. Inputs: `PROJECT_DOCUMENTATION.md`, the H-RLPOI COMPSAC 2026 submission, and Pervin et al., *Knowledge-based Context-aware Group Recommender System* (KCGRS), Decision Support Systems 196 (2025) 114485.

---

## 1. The recommendation in one page

**Do the group pivot. It is the right move, and for a reason that is not the obvious one.**

The obvious reason is that group recommendation is a natural next problem. The real reason is
strategic: your individual model is currently at **Acc@1 0.1699** against a leaderboard where
GNPR-SID sits at 0.3618 and your own prior H-RLPOI at 0.3421. Even after the full-data run
(the biggest untapped lever, ~20× more training signal), catching those numbers is uncertain.
Individual next-POI on FSQ-NYC is a crowded, well-optimised benchmark, and you are entering it
with a model whose distinguishing features — masked-diffusion decoding and hyperbolic geometry —
are not yet visibly paying for themselves in the metric.

Group **next**-POI recommendation is a different situation. KCGRS, the strongest recent
knowledge-based group POI system, is **not sequential** — it recommends POIs to a group from
aggregated static preferences, with no trajectory, no "next", and no temporal ordering of the
group's own history. Meanwhile every strong next-POI model (GETNext, STHGCN, LLM4POI, GNPR-SID,
H-RLPOI) is single-user. The intersection — *sequential, trajectory-conditioned next-POI
prediction for a group* — is where your existing pipeline already lives and where the
comparison set is thin.

And there is a real technical claim to make there, not just an empty niche:

> **Group consensus is a geometric operation in hyperbolic space.** The weighted hyperbolic
> midpoint of members who disagree lies closer to the origin — i.e. at a *more general* node
> of the category taxonomy — than any member. Consensus-by-generalisation, which every group
> recommender approximates with hand-designed aggregation heuristics (average / least-misery /
> most-pleasure), falls out of the curvature for free. Group heterogeneity stops being a
> descriptive statistic and becomes a coordinate: the radius of the consensus point.

That claim is testable on the artifacts you already have, on a CPU, in about a minute — and
**it should be tested before any GPU time is spent**, because if `poi_hyperbolic_embs.npy` has
no radial hierarchy, the mechanism has no substrate. §9 gives you the script.

One caution up front: **this pivot does not excuse you from the full-data run.** The group
model reuses the individual model's frozen scorer, so a weak individual model caps the group
model. The good news is the ordering is cheap — see §8.

---

## 2. Honest read of where the project stands

What is genuinely solid, and should be said plainly in the paper:

- The pipeline is complete and reproducible end-to-end, with unusually good engineering
  hygiene (causality assertions, contiguous-ID asserts, non-zero-head asserts, the
  symptom→cause checklist). That is worth a paragraph; most papers cannot claim it.
- **The embedding-level geometry evidence is your strongest existing result** and it is
  currently undersold. Against an equal-capacity node2vec control on the same graph and split,
  RotH gains ΔMRR **+0.077 on HAS_CATEGORY** and **+0.138 on SUBCATEGORY_OF**, and ≈0 on the
  flat relations. That is a clean, well-controlled result showing the hyperbolic advantage
  appears *exactly and only* where the theory says it should. It is a better argument for the
  geometry than any end-to-end Acc@1 number will be.
- A 5%-trained model already matching H-RLPOI's fully-trained non-RL ablations is a real
  signal, provided the caveat is stated as prominently as the number.

What is weak, stated without euphemism:

- **The headline Acc@1 is not competitive** and the paper cannot lead with it.
- **The geometry is not yet doing work at the decision layer.** This is the deepest technical
  problem and it is fixable. The `tied` scoring path computes `MLP(h) · logmap0(z_p)` — a
  Euclidean inner product against a randomly projected tangent vector. An inner product
  *cannot express a geodesic distance*. So after `logmap0` + a fixed random projection to 2048-d
  + a dot product, the hierarchy survives only as feature correlation, never as geometry. The
  hyperbolic-vs-euclidean ablation at the LLaDA level may well come back near-null for this
  reason alone — and you would wrongly conclude the geometry does not help, when in fact the
  architecture discarded it. §7/R2 fixes this, and the fix helps the individual model too.
- The `EMB_CONDITION ∈ {euclidean, random}` ablation — the paper's central claim at system
  level — has never been trained. Right now the geometric claim rests entirely on the
  link-prediction ablation.
- Loss and ranking quality diverged under the regularization guards (val loss 5.2548 → 5.2414,
  val Acc@1 0.1809 → 0.1594). Unresolved, and it will be asked about. §7/R3.

---

## 3. What to take from KCGRS — and its four exploitable weaknesses

Take the **skeleton**: KG → entity embeddings → context infusion → learned-weight aggregation
into a group vector → treat the group as a hypothetical user → score with the same head. That
architecture is sound and maps almost 1:1 onto what you already have (KG v3 → RotH → LLaDA
context → *new group layer* → existing scorer). You are not rebuilding; you are inserting one
module.

Then exploit the gaps. Each of these is a defensible contribution, not a cheap shot:

**W1 — KCGRS is not sequential.** It recommends POIs to a group; it does not predict a group's
*next* POI. No trajectory, no temporal ordering, no "where does this group go after here". Your
entire pipeline is built around exactly that. This is the single largest gap and it defines
the paper.

**W2 — Its knowledge-graph embeddings are Euclidean and translational.** KCGRS Eq. (1) is
`‖h + r − t‖` — plain TransE. Their own related-work section concedes translational scoring
struggles with hierarchical and complex relations. They then aggregate group members by a
Euclidean weighted sum (Eq. 9). So a system whose stated purpose is capturing hierarchical
domain knowledge uses the geometry least suited to it. Your RotH + gyromidpoint is a direct,
like-for-like upgrade of that exact component — and the Euclidean version is recoverable as a
`c → 0` ablation on the *same code path*, which makes the comparison airtight.

**W3 — The affinity loss cannot beat averaging, by construction.** This is the important one.
KCGRS Eq. (10) trains the group embedding to minimise

    L_affin(g,p) = ( R̂[Z_g : Z_p] − (1/K) Σ_i R̂[Z_ui : Z_p] )²

i.e. it regresses the group's predicted score onto the **arithmetic mean of member scores**.
The global optimum of that objective is a group embedding that *reproduces average
aggregation*. Any margin KCGRS shows over an AVG baseline therefore comes from the inductive
bias of the embedding space and from failing to fit its own objective — not from the objective
itself. There is no supervision from a real group outcome anywhere in it. Replacing this with
a ranking loss on observed group visits plus an explicit fairness term (§5, C4) is a
methodological correction with a clear, statable rationale.

**W4 — Group weights do not generalise to unseen groups.** KCGRS learns a free scalar `w_i` per
member per group by gradient descent on that group's own pre-visit history (Eq. 12). A group
with no history has no weights. Making the weights a *function* of member features (§5, C1)
fixes cold-start groups, which is the realistic deployment case.

*(Minor, worth noting once and not labouring: KCGRS's "Contextual Transformer" is a
three-layer feed-forward MLP trained with MSE on ratings — there is no attention and no
transformer in it, and an MSE-on-ratings objective is misaligned with a ranking task.)*

---

## 4. Problem formulation

Let `U` be users, `P` the 5,120 NYC POIs, and let a group `g = {u₁ … u_k}`, `2 ≤ k ≤ 5`, be a
set of users observed co-present at a POI within a time window. Given

- the group's **joint recent trajectory** `H_g = [(p₁,t₁) … (p_L,t_L)]` — POIs the group
  visited together, most recent last;
- each member's **causal individual profile** `π(u_i)` — the top-5 POIs / top-3 categories /
  top-3 hours from that member's strict prefix, exactly as built today by
  `_profile_from_prefix`;
- a query time `(dow, hour)`;

predict the next POI the group visits together. Evaluate with Acc@1/5/10, MRR and NDCG@k over
all 5,120 POIs, plus the fairness and heterogeneity measures of §6.

The key difference from KCGRS: the target is a *transition*, conditioned on where the group
already is. The key difference from your current model: the conditioning is a *set* of users
with a *shared* trajectory, and the answer must satisfy all of them.

---

## 5. Proposed method

Five components. C1–C2 are the scientific core; C3 is the capability that only a diffusion LLM
gives you; C4 fixes KCGRS's objective; C5 is optional and cheap.

### C1 — Hyperbolic consensus by weighted gyromidpoint

Members carry RotH embeddings `z_{u_i}` on the Poincaré ball `D^64_c`. Aggregate them with the
**weighted gyromidpoint**, computed as the Einstein midpoint in the Klein model:

    x_K = 2x / (1 + c‖x‖²)                          Poincaré → Klein
    γ_i = 1 / √(1 − c‖x_{K,i}‖²)                    Lorentz factor
    m_K = Σ_i w_i γ_i x_{K,i} / Σ_i w_i γ_i         weighted Einstein midpoint
    z_g = m_K / (1 + √(1 − c‖m_K‖²))                Klein → Poincaré

Closed-form, differentiable, permutation-invariant. Implemented and unit-tested in
[group/hyperbolic_group.py](group/hyperbolic_group.py); all 16 self-tests pass. Two properties
verified numerically there and worth stating in the paper:

- **`c → 0` recovers the Euclidean weighted mean exactly**, i.e. KCGRS Eq. (9). One curvature
  knob, one code path, a perfectly controlled ablation. (Verified to 1e-4.)
- **It approximates the true Fréchet mean to within 4% of the group's own spread** on random
  5-member groups in d=64, while being closed-form rather than iterative. The iterative Fréchet
  mean is provided as a check but should not be used in training.

Member weights come from a **`GeometricAttention`** module: `w_i = softmax(MLP([logmap₀(z_i),
d_c(z_i, z_g⁽⁰⁾), side_i]))`, where `z_g⁽⁰⁾` is a provisional uniform consensus, `d_c` is the
geodesic distance to it, and `side_i` carries scalar context (member activity count, recency).
Two refinement rounds let an outlier be down-weighted after the first consensus estimate.
Because the weights are a *function* of features rather than free per-group parameters, they
transfer to groups never seen in training — fixing W4.

### C2 — Heterogeneity as a coordinate, and the "consensus naming" figure

`group_heterogeneity()` returns four geometric descriptors: `dispersion` (weighted mean
geodesic distance from members to consensus — the Fréchet standard deviation), `spread` (mean
pairwise member distance), `consensus_r` (radius of the consensus), and

    depth_drop = Σ_i w_i · r(z_{u_i})  −  r(z_g)

the **specificity the group gives up to agree**. This is the mechanistic version of KCGRS's
descriptive `H_gr`. Two things follow:

1. **Heterogeneity-stratified evaluation.** Bucket test groups by `spread` and plot Acc@1 per
   bucket. The prediction the paper should commit to in advance: hyperbolic aggregation
   degrades more gracefully than Euclidean as heterogeneity rises, because the Euclidean mean
   of divergent members lands in a semantically empty region while the gyromidpoint lands on
   the common ancestor. If this holds, it is the paper's best figure. If it does not, you have
   learned something real and should report it.

2. **Consensus naming — an explainability result KCGRS structurally cannot produce.** Your KG
   contains 500 Category nodes as first-class entities with their own RotH embeddings. Decode
   the group consensus by finding its nearest category node: `argmin_cat d_c(z_g, z_cat)`. The
   system can then *say* "this group's consensus sits at `Dining and Drinking`, depth 1.2 —
   the members' individual preferences were `Cocktail Bar` and `Sports Bar`". A Euclidean
   average has no reason to land near any taxonomy node; a geodesic midpoint does. This is a
   qualitative table that costs nothing and is memorable.

> **Prerequisite.** Both of these need the RotH ball to actually be radially ordered — general
> categories near the origin, specific near the boundary. **This is not guaranteed.** RotH is
> trained for link prediction, not for Poincaré-style hierarchy embedding, so the radial
> ordering is a hoped-for by-product. Diagnostic **D1** in §9 measures it in seconds. Read §9
> before anything else.

### C3 — Group prompt and multi-mask itinerary decoding

**Prompt.** Reuse `prompt_text()` with a group block. Token budget is the binding constraint:
the current single-user prompt averages 483 tokens against `MAX_LEN=1024`, so five full member
profiles would blow the budget. Compress:

```
You are a POI recommendation expert. Predict the next POI this group of 3 visits together.
[group] <grp>   heterogeneity: moderate
[member 1] top: <poi_31> (Bar), <poi_884> (Coffee Shop) | hours: 18,19,22
[member 2] top: <poi_77> (Ramen), <poi_12> (Bar)        | hours: 12,19
[member 3] top: <poi_450> (Museum)                      | hours: 14,15
[joint check-ins]
<poi_884> (Coffee Shop, 17:00)
<poi_31>  (Bar, 19:00)
[current time] Thursday 21:00
[next POI] <MASK>
```

≈60–80 tokens per compact member block + a shared joint history ≈ 480 tokens at k=4. Fits.
Randomise member order per epoch as augmentation, so nothing latches onto position.

**The `<grp>` token is the elegant part, and it is nearly free.** You already wrap the
embedding lookup (`_mixed_embedding_forward`) to route POI ids to a frozen `W_POI`. Reserve one
more id, `<grp>`, and route it to a *per-example* vector supplied by the collator — the
gyromidpoint consensus, log-mapped and projected through the same fixed seeded projection as
`W_POI`, so it lands in the same statistical scale as every other POI token. The group
consensus then enters the LLM as a first-class token in a space where POI tokens already live.
This is a ~20-line change to the existing wrapper. It also justifies the compression: the
member text blocks can be short precisely because `<grp>` carries the aggregate.

**Multi-mask decoding — the diffusion-specific contribution.** Place `M` masks instead of one:

```
[next POIs] <MASK> <MASK> <MASK>
```

LLaDA unmasks them in parallel with confidence-based remasking, so the M POIs are decoded
*jointly and conditioned on each other* rather than independently. For a group this is exactly
the right output shape: groups do not want a ranked list of independent items, they want a
small mutually-coherent shortlist to agree on — or a short itinerary. An autoregressive LLM
cannot produce a jointly-consistent set in one pass without committing to an order. **This is
the clearest thing you can point to that requires the diffusion backbone**, and it turns "we
used a diffusion LLM" from an implementation choice into a capability argument. Evaluate at
M ∈ {1, 3, 5}, and report both ranking metrics (M=1) and set-level coherence for M>1.

### C4 — A group objective that can actually beat averaging

Replace KCGRS's mean-regression with:

    L  =  L_rank  +  α · L_fair  +  β · L_Fréchet

- **`L_rank`** — cross-entropy over the 5,120 restricted POI logits at the masked position,
  target = the group's observed next POI. Identical machinery to today; only the target
  changes. This is real supervision from a real group outcome, which KCGRS has nowhere.
- **`L_fair`** — a soft-min over members of their *individual* model's log-probability for the
  POI the group model recommends. Directly optimises the least-misery criterion that group
  recommenders are judged on, and gradients reach the aggregation weights. Implemented as
  `least_misery_loss()`.
- **`L_Fréchet`** — `Σ_i w_i d_c(z_g, z_{u_i})²`. Its minimiser over `z_g` *is* the weighted
  Fréchet mean, so this regulariser keeps whatever the attention learns geometrically faithful
  rather than letting it drift into an arbitrary point of the ball. `frechet_variance_loss()`.

Keep KCGRS's mean-regression as an ablation row — it is a fair reimplementation of their
method and it should visibly cap out at the AVG baseline, which is the empirical demonstration
of W3.

### C5 — Groups as first-class KG nodes *(optional, cheap, do it only if time allows)*

Add `Group` nodes with `MEMBER_OF` (user→group) and `GROUP_VISITED` (group→POI) edges to KG v3
and retrain RotH. Group embeddings then come out of the hierarchy-aware model directly, usable
as an initialiser or regulariser for the aggregation module. Mirrors KCGRS's UPKG but
hyperbolic and with genuine group entities. **Leakage warning:** only train-split group visits
may enter the graph, exactly as `FOLLOWED_BY` is already restricted to train.

---

## 6. Group construction — the make-or-break

This is the highest-risk part of the whole plan and deserves your attention before the
modelling. **TSMC2014-NYC has no friendship edges** (unlike KCGRS's Yelp, or Gowalla). Groups
must be constructed, and how you do it determines what the results mean.

**T1 — Implicit co-visit groups (real signal; the scientifically valuable option).**
Users co-checked-in at the same POI within Δt ∈ {15, 30, 60, 120} minutes form a candidate
group. A **group transition** — the group co-present at POI A at time t, then co-present at POI
B within a horizon — is a genuine group next-POI target. This is real behaviour, not
simulation, and it would be a modest data contribution in its own right.

**The risk is sparsity, and it is serious.** FSQ-NYC after filtering has 1,073 users and
147,699 check-ins across ~10 months. Co-presence requires two of those 1,073 users at the same
venue in the same half-hour; *sequential* co-presence requires it twice in a row. Groups of
size ≥3 may be very rare, and the transition count could be in the hundreds rather than the
thousands. **Diagnostic D3 in §9 gives you the exact counts in about a minute.** Decide from
the number, not from hope. The three regimes:

| D3 transitions | What it means |
|---|---|
| ≥ 5,000 | T1 is viable for training *and* testing. Best case; lead with it. |
| 300 – 5,000 | Train on T2 synthetic groups, hold out T1 real groups as the test set. This is still a strong story — "trained on simulated groups, evaluated on real co-visits". |
| < 300 | T1 is not viable. Go T2 + T3. |

**T2 — Synthetic groups (comparability; do this regardless).**
The standard protocol in the group-rec literature (AGREE, GroupIM, and KCGRS's own "occasional
groups"). Sample groups of size 2–5 under two regimes: **random** (heterogeneous) and
**similarity-based** (homogeneous, sampled by category-profile or embedding similarity). This
is not just a fallback — it is what gives you *controlled* heterogeneity, and therefore the
stratified analysis of C2. Ground truth: a POI in the intersection of members' held-out
check-ins at a shared timestamp. Report T2 alongside T1 always; they answer different
questions.

**T3 — A second dataset with real social ties (recommended, if the calendar allows).**
- **Yelp** — KCGRS's own dataset, with their four cities (Philadelphia, Tucson, Indianapolis,
  Tampa) and published numbers. Enables a **head-to-head comparison against KCGRS on their own
  benchmark**, which is worth more than any number on NYC. Weakness: Yelp reviews are a poor
  proxy for sequential visits, so the "next" framing weakens.
- **Gowalla** — friendship edges *and* timestamped check-ins. The only common dataset that
  supports real social groups *and* genuine sequential next-POI. If T1 comes back thin, this
  is the better scientific answer; if you want the direct KCGRS comparison, Yelp.

**Recommendation:** run D3 first. Then T2 as the workhorse (it always works and gives you the
heterogeneity axis), T1 as the real-behaviour evaluation if the counts allow, and Yelp as the
stretch goal for a direct KCGRS comparison.

---

## 7. Improvements you need regardless of the group pivot

**R1 — Verify, and if necessary repair, the radial hierarchy.** Run D1. If Spearman(taxonomy
depth, hyperbolic radius) is below ~0.1, RotH did not learn a radial hierarchy and C1/C2 have
no substrate. The repair is cheap: add a depth-ranking regulariser to the RotH loss, e.g.
`Σ_{(child,parent)} max(0, μ − (r(child) − r(parent)))` over `SUBCATEGORY_OF` edges, pushing
children outward relative to parents. This costs one RotH retrain (~hours on CPU/small GPU),
not a LLaDA run, and it would improve the individual model too.

**R2 — Add a genuine hyperbolic scoring path.** As argued in §2, the current `tied` path cannot
express geodesic distance, so the geometry never reaches the decision surface.
`HyperbolicScorer` in [group/hyperbolic_group.py](group/hyperbolic_group.py) maps the hidden
state into the 64-d tangent space, exponentiates onto the ball, and ranks by
`−d_c(q, z_p) + b_p`, with learnable curvature. Add it as a third term alongside `poi_head` and
the tied path (`SCORING_MODE='all'`). It is 64-dimensional, so it costs almost nothing. **Its
ablation is the cleanest possible test of the paper's central geometric claim** — and it may
well rescue a hyperbolic-vs-euclidean comparison that would otherwise come back null for
architectural reasons rather than geometric ones.

**R3 — Resolve the guards anomaly.** Val loss improved while ranking degraded (0.1809 → 0.1594).
The plausible mechanism: weight decay 0.1 on the head over-shrinks the logit scale, which
lowers cross-entropy while flattening the ranking. Diagnostic: compare logit norms and the
head/tied contribution ratio between the two checkpoints, and evaluate ranking metrics per
epoch rather than at the end. Cheap, and it converts an embarrassing table row into an
analysis paragraph.

**R4 — The full-data run is still the biggest single lever.** Nothing here replaces it. The
group layer consumes the individual scorer; a better scorer lifts everything downstream.

---

## 8. Sequencing, with honest costs

| Phase | Work | Cost | Gate |
|---|---|---|---|
| **0** | D1/D2/D3 diagnostics (§9) | **~1 min, CPU** | D1 verdict decides whether C1/C2 survive as designed; D3 decides the group protocol |
| **0b** | R1 RotH repair, *only if D1 fails* | hours | — |
| **1** | R4 full-data individual run + R2 hyperbolic scorer | ~30 h/epoch on T4 → **needs the L40S** | the number everything else inherits |
| **2** | T2 group construction + group layer training | **cheap** — backbone frozen, only attention + scorer train; minutes/epoch | — |
| **3** | Baselines + ablations + heterogeneity stratification | moderate, mostly inference | — |
| **4** | T1 real-group evaluation; Yelp or Gowalla replication | large | stretch |

**The critical scheduling insight:** the group layer is *not* another LLaDA fine-tune. With the
backbone and `W_POI` frozen, you are training a small attention module and optionally the
scorer — minutes per epoch against 93 minutes for the current 5% run. The expensive thing is
Phase 1, which you already need. That is what makes this pivot affordable rather than a restart.

On the supervisor question in `PROJECT_DOCUMENTATION.md` §9 about compute: **the L40S access
matters more now, not less.** Phase 1 at full data is ~30 h/epoch on a T4 and the group work
sits downstream of it.

---

## 9. Do this first — Phase-0 diagnostics

[group/phase0_diagnostics.py](group/phase0_diagnostics.py). CPU only, no GPU, ~1 minute. Run on
Kaggle where `kushflq` is mounted:

```bash
python phase0_diagnostics.py --data-dir /kaggle/input/kushflq
python phase0_diagnostics.py --self-check    # synthetic fixture, validates the script itself
```

**D1 — radial hierarchy.** Spearman(taxonomy depth, hyperbolic radius) over all 5,120 POIs,
plus mean radius per depth level. Verdict thresholds: >0.30 strong, >0.10 weak, else stop and
do R1. *This is the single highest-value experiment available to you right now* — it costs
seconds and it determines whether the paper's central mechanism exists.

**D2 — consensus semantics.** For sampled POI pairs, does the gyromidpoint's `depth_drop` grow
with taxonomic divergence?

> One methodological point that cost me some care and will cost you a wrong conclusion if
> missed: **the pooled correlation here is confounded and must not be reported.** Two POIs
> that are both deep in the taxonomy start at large radii, so their midpoint drops further in
> absolute terms regardless of how related they are. On a synthetic hierarchy where the effect
> is present *by construction*, the pooled statistic reads ρ = +0.09 — looks like a null result
> — while the equal-depth strata read up to **ρ = +0.66**, with sharing one taxonomy level
> cutting the drop from 0.353 to 0.221. The script therefore stratifies by member depth and
> reports the equal-depth weighted mean as the headline, printing the pooled value only for
> contrast. Use the same stratification in the paper.

**D3 — group mining feasibility.** Co-visit events, distinct pairs, groups, groups of size ≥3,
and **group transitions** at Δt ∈ {15, 30, 60, 120} min. Read the transitions column against
the table in §6 and pick the protocol.

---

## 10. Evaluation design

**Metrics.** Acc@1/5/10, MRR, NDCG@10 at group level (KCGRS reports Hit ratio and NDCG — match
them for comparability). Then the ones that make it a *group* paper:

- **Fairness / least misery:** minimum member satisfaction, Gini coefficient of member
  satisfaction, and member coverage@k (fraction of members with ≥1 relevant item in the top-k).
- **Diversity and coverage:** intra-list category diversity and catalogue coverage — KCGRS
  reports both, and it is where a consensus-generalising method should look good.
- **Heterogeneity-stratified Acc@1**, per C2. The headline figure.

**Baselines, cheapest first.**

1. **Score aggregation over your existing individual model** — AVG, least-misery, most-pleasure,
   Borda/Copeland rank fusion. **These require zero training**: k forward passes and a fuse.
   They are also the baselines most likely to be competitive, so run them early and know the
   number you have to beat. Do not skip these; a group paper without them is not credible.
2. **Euclidean embedding aggregation** — `c → 0` on the same code path. This *is* KCGRS's Eq. (9).
3. **KCGRS reimplementation** — TransE-style KG + MLP + weighted sum + mean-regression loss. A
   faithful version is roughly a day's work and it is the paper's key related-work comparison.
4. **AGREE / GroupIM** — attention-based and self-supervised group recommenders. Use published
   code; do not reimplement from scratch.

**Ablations.** Aggregation {gyromidpoint, Fréchet, Euclidean mean, attention-only, AVG};
geometry {hyperbolic, euclidean/node2vec, random}; scoring {head, tied, +hyperbolic (R2)};
loss {rank, rank+fair, rank+fair+Fréchet, KCGRS mean-regression}; `<grp>` token on/off;
multi-mask M ∈ {1,3,5}; group size k ∈ {2,3,4,5}.

---

## 11. Risks, and what would kill each claim

| Risk | Kills | Mitigation |
|---|---|---|
| RotH ball has no radial hierarchy | C1, C2 entirely | **D1, before anything else.** Repair via R1 |
| Too few real co-visit transitions in NYC | T1 as a training set | D3 decides; fall back to T2 train / T1 test, or Gowalla |
| Group baselines (AVG / least-misery) match the learned model | the paper's premise | Run them in Phase 2, not at the end. If they match, the heterogeneity-stratified result becomes the contribution rather than the aggregate number |
| Hyperbolic ≈ Euclidean end-to-end | the geometric claim | R2 first — the current architecture *cannot* show a difference. Fall back on the link-prediction evidence, which is already clean |
| Reviewers reject synthetic groups | external validity | Report T1 real groups even if small; KCGRS's own user study is the precedent for the concern |
| Scope creep across two novelties | the paper | See below |

**On scope** — this answers the open question in `PROJECT_DOCUMENTATION.md` §9. Lead with
**one** headline. My recommendation: **the group formulation is the headline, hyperbolic
consensus is the mechanism, and the diffusion backbone is the enabler** (justified concretely
by multi-mask decoding, C3, not by novelty-for-its-own-sake). Three co-equal contributions
will read as unfocused; one claim with two supporting pillars reads as a paper.

**Novelty due diligence before committing.** I have not searched the literature for this, and
you should, specifically: *group next-POI recommendation*, *sequential group recommendation*,
*hyperbolic group recommendation*, *gyromidpoint aggregation recommender*. Hyperbolic
recommender systems (HGCF, HRCF, HyperML) and hyperbolic attention with Einstein midpoints
(Gulcehre et al., 2019) are established — the gyromidpoint is not itself novel and **must be
cited as prior art**, not presented as new. What is plausibly new is the *combination*:
gyromidpoint aggregation as a group-consensus operator, with the taxonomy-generalisation
reading and the heterogeneity-as-radius diagnostic, in a sequential next-POI setting on a
diffusion LLM. Verify that framing holds before writing the intro.

---

## 12. Files delivered with this proposal

| File | Status |
|---|---|
| [group/hyperbolic_group.py](group/hyperbolic_group.py) | Gyromidpoint, Fréchet mean, geodesic ops, `GeometricAttention`, `HyperbolicScorer`, `least_misery_loss`, `frechet_variance_loss`. **16/16 self-tests pass** (`python hyperbolic_group.py`) |
| [group/phase0_diagnostics.py](group/phase0_diagnostics.py) | D1/D2/D3 with verdict thresholds; `--self-check` validates against a synthetic hierarchy with no project data |

Both are standalone: `hyperbolic_group.py` needs only torch, `phase0_diagnostics.py` needs only
numpy + pandas. Neither touches the existing notebooks.

Two implementation notes discovered while testing, worth carrying into the real code:

- **`randn(d) * s` is not a ball sample.** In d=64 its norm concentrates at `s√d`, so nearly
  every draw lands outside the ball and gets clamped onto the boundary where float32 `artanh`
  saturates. Sample the radius explicitly (`random_ball_points`). This is exactly the kind of
  bug that would silently produce a null geometric result.
- **If RotH embeddings sit near ‖z‖ = 1, aggregate in float64.** float32 stays within 1.2e-4 of
  float64 up to r = 0.999, which is fine — but check the actual norm distribution in D1's
  output before trusting float32 in training.

---

## 13. Suggested next three actions

1. **Run D1 and D3** on Kaggle against `kushflq` (~1 minute). These two numbers determine the
   shape of the entire project. Nothing else should be decided before them.
2. **Take the D1 verdict to your supervisor** along with the scope recommendation in §11 and
   the compute request in §8 — the L40S question is now on the critical path.
3. **In parallel, implement R2** (the hyperbolic scoring path). It is small, it helps the
   individual model, and it is a prerequisite for the geometry ablation being meaningful at all.
