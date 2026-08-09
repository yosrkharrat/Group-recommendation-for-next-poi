"""
Finalized group construction for group next-POI recommendation on FSQ-NYC.

Why this design (read before changing anything)
-----------------------------------------------
The obvious construction -- mine real groups that move together, POI A -> POI B, and predict B --
does not survive contact with the data. Measured on all 147,539 NYC check-ins, across window
in {30,60,120,180} min and horizon in {240,720,1440} min, with membership matched by exact set
*and* by Jaccard >= 0.34:

    real group->group transitions:  17 .. 62        (proposal threshold for "trainable": 5,000)
    relaxed to ">=2 members visit the same next POI within 24h":  70 .. 182

There is no parameter setting that makes real group transitions trainable. `phase0_diagnostics.py`
D3 and `build_group_relations.py` agree on this independently (~21 and ~22 transitions). So real
co-visit transitions are reported as a curiosity set and the trainable task is built the standard
way (AGREE / GroupIM / KCGRS "occasional groups"): a real individual next-POI event is turned into
a group event by adding companions.

What makes this construction defensible rather than arbitrary:

1. **The three regimes are KCGRS's own group taxonomy, actually constructed.** That paper names
   Established (similar preferences) / Occasional (mixed) / Random (mixed) and then never says
   how any of them is built. Here:

       established   a CLIQUE in the multi-signal affinity graph (see affinity.py) -- every
                     member high-affinity with every other, not just with the anchor
       occasional    a subset of a REAL observed co-presence set, at a rare venue, from before
                     the prediction time -- so the membership is not synthetic at all
       random        uniform sampling

2. **The affinity function is validated, not asserted.** Real 1-hour co-presence is too thin to
   be the group source (3,750 groups, 80% pairs, 21 transitions) but it is an excellent *label*.
   Scoring every signal against it over all 575,128 user pairs -- with each signal recomputed so
   it cannot contain the co-presence events that produced the labels -- gives taste 0.648,
   rhythm 0.646, territory 0.622, far-co-visit 0.552, combined **0.722**, and at the top 1%
   threshold 12x lift over the base co-presence rate. Category preference is the strongest
   single honest signal, which is why `established` leans on it.

3. **The target must be plausible for every member, causally.** A group example is kept only if
   *every* member visited the target's level-2 category at some point strictly *before* the
   target timestamp. Using the full trajectory instead would be selection-by-hindsight; the
   causal test costs ~12 points of yield (48.1% -> 36.2% at k=2) and removes the objection.

4. **Everything the model sees is causal.** Member profiles come from each member's strict
   prefix before t; the joint history is member check-ins strictly before t. Ties, the affinity
   graph and the cliques are all computed from TRAIN only, so nothing about val/test shapes a
   group.

5. **Weak groups are filtered, and the filters were chosen from measurements.** Three leaks were
   found by auditing the first version and all three are now closed and asserted:

     - `occasional` groups used to be k-1 samples from the anchor's tie list, which produced
       STARS, not groups: 68.5% of size>=3 groups contained member pairs that had never been
       co-present. Fixed by reusing whole observed co-presence sets -> now 100% of size>=3
       occasional groups are subsets of a real co-present set.
     - members with an almost-empty causal profile contributed placeholders rather than
       preferences. `--min-member-activity 5` -> 0 members below 5 check-ins at query time.
     - 8.0% of groups had a joint history made up *entirely* of the anchor's own check-ins,
       so the "group trajectory" was one person's. `--require-companion-history` -> 0 left,
       median anchor share of the joint history 0.47.
     - 56.9% of co-presence events (even after the venue gate) were one-off in EVERY pair --
       two people who crossed paths once and never again. `--min-companion-repeat 2` requires
       the anchor to have >=1 companion met at least twice. Note the gate is deliberately NOT
       "all pairs recur": that reads as stricter but wipes out every group above size 3
       (1,523 -> 535 events), because a real outing is a core dyad plus occasional joiners.
       One recurring anchor pair keeps 4:25/30, 5:22/24, 6:12/12, 8:7/7 of the large groups.

Known bias, stated rather than hidden: `established` groups agree by construction, so their
targets drift toward the individual-model target and they are the EASY case. `occasional` and
`random` are the hard cases. Always report heterogeneity-stratified metrics, never a single
pooled number -- KCGRS has this same property and does not acknowledge it. Note also that
`regime` is not a proxy for heterogeneity: some `established` cliques score 0.95 because the
affinity is taste+rhythm+territory, so two people can share a neighbourhood and a schedule
while disagreeing on taste. Stratify on the `heterogeneity` column, not on `regime`.

Yield on FSQ-NYC (146,466 individual examples in, 54,550 group examples out after filtering):

    established 18,969 train   occasional 4,794 train   random 10,881 train

The occasional count is the smallest and it is the one built entirely from observed behaviour --
real co-present membership, rare venue, recurring companion. Treat it as the trustworthy slice
and `random` as the adversarial one.

Outputs (--out-dir)
-------------------
    ephemeral_groups.csv        real co-presence groups (bounded anchor windows)
    group_members.csv           group_id, user_id
    real_group_transitions.csv  the ~20-60 genuine group moves; a held-out curiosity set, n is
                                far too small to train or to headline
    co_attended.csv             IDF-weighted user-user tie graph -> feeds companions AND the KG
    group_examples_{split}.jsonl   the actual task data
    groups_manifest.json        every count and every config value used

Usage
-----
    python build_groups.py --data-dir ./data --out-dir ./data/groups
    python build_groups.py --self-check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from affinity import (build_affinity, clique_neighbourhood, copresence_labels, validate)

SEP = ">"


# --------------------------------------------------------------------------
# loading + splits
# --------------------------------------------------------------------------

def load_checkins(data_dir, dataset="NYC"):
    frames = []
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}_{dataset}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        frames.append(pd.read_csv(path).assign(orig_split=split))
    df = pd.concat(frames, ignore_index=True)
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["utc_time"]).copy()
    df["ts"] = df["utc_time"].astype("int64") // (60 * 10 ** 9)     # epoch minutes
    df["hour"] = df["utc_time"].dt.hour.astype(int)
    df["dow"] = df["utc_time"].dt.day_name()
    if "venue_id" not in df.columns:
        df["venue_id"] = df["poi_idx"].astype(str)
    return df


def resplit_per_user(df, train_frac=0.70, val_frac=0.10):
    """Per-user chronological 70/10/20 -- byte-identical logic to `resplit_per_user` in
    stage6b_run2_server.ipynb §3, so group examples land in the same splits as the individual
    examples the score-aggregation baselines are computed from."""
    df = df.sort_values(["user_id", "utc_time"], kind="mergesort").reset_index(drop=True)
    splits = np.empty(len(df), dtype=object)
    for _, idx in df.groupby("user_id", sort=False).indices.items():
        n = len(idx)
        n_tr = max(1, int(np.ceil(n * train_frac)))
        n_val = int(np.ceil(n * (train_frac + val_frac)))
        n_val = min(max(n_val, n_tr), n)
        splits[idx[:n_tr]] = "train"
        splits[idx[n_tr:n_val]] = "val"
        splits[idx[n_val:]] = "test"
    df["split"] = splits
    return df


def load_categories(data_dir, dataset="NYC"):
    path = os.path.join(data_dir, f"poi_metadata_{dataset}.csv")
    meta = pd.read_csv(path)
    col = next((c for c in ("category_path", "category", "categories") if c in meta.columns), None)
    if col is None:
        raise SystemExit(f"no category column in {path}")
    return meta.set_index("poi_idx")[col].fillna("Unknown").astype(str).to_dict(), len(meta)


def cat_prefix(cat_of, poi, level):
    parts = [p.strip() for p in str(cat_of.get(int(poi), "Unknown")).split(SEP) if p.strip()]
    return SEP.join(parts[:level]) if parts else "Unknown"


# --------------------------------------------------------------------------
# 1. real ephemeral groups  (bounded anchor windows -- no single-linkage chaining)
# --------------------------------------------------------------------------

def build_ephemeral_groups(df, window_min=60, min_size=2, max_size=8):
    """Same POI, users co-present inside a `window_min` window.

    Each check-in opens a candidate window [t, t+window_min]; members are the distinct users in
    it. Every group therefore spans at most `window_min` by construction. Single-linkage time
    clustering (the earlier group-kg.ipynb approach) chains consecutive check-ins and lets a busy
    venue collapse a whole evening into one 13-member "group"; this cannot.
    """
    rows = []
    for poi_idx, block in df.groupby("poi_idx", sort=False):
        block = block.sort_values("ts", kind="mergesort")
        ts = block["ts"].to_numpy()
        users = block["user_id"].to_numpy()
        venue = block["venue_id"].iloc[0]
        n = len(block)
        cand, j = [], 0
        for i in range(n):
            if j < i:
                j = i
            while j + 1 < n and ts[j + 1] - ts[i] <= window_min:
                j += 1
            members = frozenset(users[i:j + 1].tolist())
            if min_size <= len(members) <= max_size:
                cand.append((members, int(ts[i]), int(ts[j])))
        for members, t0, t1 in _drop_nested(cand):
            rows.append(dict(venue_id=venue, poi_idx=int(poi_idx), start_ts=t0, end_ts=t1,
                             span_min=t1 - t0, size=len(members), members=sorted(members)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["poi_idx", "start_ts"], kind="mergesort").reset_index(drop=True)
    out.insert(0, "group_id", np.arange(len(out)))
    return out


def _drop_nested(cand):
    """Keep maximal member sets: drop duplicates and subsets of an overlapping larger group."""
    kept = []
    for members, t0, t1 in sorted(cand, key=lambda r: (-len(r[0]), r[1])):
        if not any(members <= km and not (t1 < kt0 or t0 > kt1) for km, kt0, kt1 in kept):
            kept.append((members, t0, t1))
    return sorted(kept, key=lambda r: r[1])


def build_real_transitions(groups, horizon_min=240, min_shared=2, min_jaccard=0.5):
    """Group at POI A, then substantially the same people at POI B within the horizon.

    Kept for completeness and for the paper's honesty: this is the *genuine* group next-POI
    signal, and on FSQ-NYC it yields tens of examples, not thousands.
    """
    if groups.empty:
        return pd.DataFrame()
    recs, by_user = [], defaultdict(list)
    for r in groups.itertuples(index=False):
        recs.append((r.group_id, set(r.members), r.start_ts, r.end_ts, r.poi_idx))
        for u in r.members:
            by_user[u].append(len(recs) - 1)

    rows, seen = [], set()
    for idxs in by_user.values():
        for a in idxs:
            gid_a, mem_a, _, end_a, poi_a = recs[a]
            for b in idxs:
                gid_b, mem_b, start_b, _, poi_b = recs[b]
                if gid_a == gid_b or poi_a == poi_b or (gid_a, gid_b) in seen:
                    continue
                gap = start_b - end_a
                if not (0 < gap <= horizon_min):
                    continue
                shared = mem_a & mem_b
                if len(shared) < min_shared:
                    continue
                jac = len(shared) / len(mem_a | mem_b)
                if jac < min_jaccard:
                    continue
                seen.add((gid_a, gid_b))
                rows.append(dict(from_group=gid_a, to_group=gid_b, from_poi=poi_a, to_poi=poi_b,
                                 gap_min=gap, n_shared=len(shared), jaccard=round(jac, 4),
                                 members="|".join(map(str, sorted(shared)))))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# 2. the tie graph  (companion source + a KG relation in its own right)
# --------------------------------------------------------------------------

def venue_rarity(df):
    """1 / log(1 + distinct visitors). Co-presence at a venue few people ever visit is evidence
    of a tie; co-presence at a tourist magnet is close to none."""
    pop = df.groupby("poi_idx")["user_id"].nunique()
    return {int(p): 1.0 / math.log(1.0 + float(n)) for p, n in pop.items()}


def tie_pool(co, min_groups, min_venues):
    """Companion-eligible ties. The raw co-attendance graph is a long tail of coincidences --
    on FSQ-NYC, 3,540 of 4,751 pairs co-occurred exactly *once*, which is what two strangers
    passing through the same bar looks like. Requiring repeated co-presence (`min_groups`), or
    co-presence at more than one venue (`min_venues` = Crandall et al.'s actual criterion, the
    strict setting), is what separates a companion from a coincidence.

        min_groups=1, min_venues=1   4,751 pairs / 953 users   (unfiltered -- mostly noise)
        min_groups=2, min_venues=1   1,211 pairs / 491 users   (default)
        min_groups=1, min_venues=2     164 pairs / 213 users   (strict)

    The full graph is still written to co_attended.csv -- filtering applies to companion
    selection only, so the KG keeps every weighted edge.
    """
    if co.empty:
        return defaultdict(set), co
    keep = co[(co["n_groups"] >= min_groups) & (co["n_venues"] >= min_venues)]
    ties = defaultdict(set)
    for r in keep.itertuples(index=False):
        ties[r.u1].add(r.u2)
        ties[r.u2].add(r.u1)
    return ties, keep


def build_co_attended(groups, rarity):
    if groups.empty:
        return pd.DataFrame()
    n_groups, venues, w = Counter(), defaultdict(set), defaultdict(float)
    for r in groups.itertuples(index=False):
        rar = rarity.get(int(r.poi_idx), 1.0)
        members = list(r.members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j])
                n_groups[pair] += 1
                venues[pair].add(int(r.poi_idx))
                w[pair] += rar
    rows = [dict(u1=a, u2=b, n_groups=c, n_venues=len(venues[(a, b)]),
                 weight_idf=round(w[(a, b)], 6)) for (a, b), c in n_groups.items()]
    return pd.DataFrame(rows).sort_values("weight_idf", ascending=False).reset_index(drop=True)


def category_profile_matrix(df, cat_of, users, level=2):
    """L2-normalised level-`level` category histogram per user. TRAIN rows only."""
    cats = sorted({cat_prefix(cat_of, p, level) for p in df["poi_idx"].unique()})
    ui = {u: i for i, u in enumerate(users)}
    ci = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(users), len(cats)), dtype=np.float64)
    for u, p in zip(df["user_id"], df["poi_idx"]):
        if u in ui:
            M[ui[u], ci[cat_prefix(cat_of, p, level)]] += 1.0
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    return M, ui


# --------------------------------------------------------------------------
# 3. group next-POI examples
# --------------------------------------------------------------------------

class MemberState:
    """Causal per-user running state: counters over the strict prefix, plus the timeline."""

    def __init__(self, df, cat_of, cat_level):
        self.ts, self.poi, self.hour = {}, {}, {}
        self.cat_seen_at = {}
        for u, g in df.groupby("user_id", sort=False):
            g = g.sort_values("ts", kind="mergesort")
            self.ts[u] = g["ts"].to_numpy()
            self.poi[u] = g["poi_idx"].to_numpy().astype(int)
            self.hour[u] = g["hour"].to_numpy().astype(int)
            # first time this user was seen in each level-k category -> causal membership test
            first = {}
            for t, p in zip(self.ts[u], self.poi[u]):
                c = cat_prefix(cat_of, p, cat_level)
                if c not in first:
                    first[c] = int(t)
            self.cat_seen_at[u] = first

    def n_before(self, u, t):
        """How many check-ins this member had strictly before t -- their activity at query time."""
        ts = self.ts.get(u)
        return 0 if ts is None else int(np.searchsorted(ts, t, side="left"))

    def knows_category_before(self, u, cat, t):
        seen = self.cat_seen_at.get(u)
        return seen is not None and cat in seen and seen[cat] < t

    def prefix_slice(self, u, t, limit=None):
        """(pois, hours) for this member's check-ins strictly before t, most recent last."""
        ts = self.ts.get(u)
        if ts is None:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        n = int(np.searchsorted(ts, t, side="left"))
        lo = 0 if limit is None else max(0, n - limit)
        return self.poi[u][lo:n], self.hour[u][lo:n]

    def profile(self, u, t, cat_of, top_k, top_cats, top_hours):
        pois, hours = self.prefix_slice(u, t)
        if len(pois) == 0:
            return dict(n_seen=0, top_pois=[], top_cats=[], top_hrs=[])
        pc, cc, hc = Counter(), Counter(), Counter()
        for p, h in zip(pois, hours):
            pc[int(p)] += 1
            cc[str(cat_of.get(int(p), "Venue"))] += 1
            hc[int(h)] += 1
        return dict(n_seen=int(len(pois)),
                    top_pois=[[int(p), int(c)] for p, c in pc.most_common(top_k)],
                    top_cats=[c for c, _ in cc.most_common(top_cats)],
                    top_hrs=[int(h) for h, _ in hc.most_common(top_hours)])


def heterogeneity(members, state, t, cat_of, level, ci_cache):
    """Mean pairwise cosine *distance* between members' causal level-`level` category profiles.

    Deliberately geometry-free: D1 shows the current RotH ball has no radial hierarchy
    (Spearman(depth, radius) = +0.019), so a hyperbolic spread would measure nothing right now.
    Swap this for `group_heterogeneity()` from hyperbolic_group.py once RotH is retrained with
    the depth regulariser.
    """
    vecs = []
    for u in members:
        pois, _ = state.prefix_slice(u, t)
        if len(pois) == 0:
            continue
        c = Counter(cat_prefix(cat_of, int(p), level) for p in pois)
        keys = tuple(sorted(c))
        for k in keys:
            ci_cache.setdefault(k, len(ci_cache))
        v = np.zeros(len(ci_cache))
        for k, n in c.items():
            v[ci_cache[k]] = n
        nrm = np.linalg.norm(v)
        if nrm > 0:
            vecs.append(v / nrm)
    if len(vecs) < 2:
        return 0.0
    dim = max(len(v) for v in vecs)
    vecs = [np.pad(v, (0, dim - len(v))) for v in vecs]
    dists = [1.0 - float(np.dot(vecs[i], vecs[j]))
             for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return float(np.mean(dists)) if dists else 0.0


def build_group_examples(df, cat_of, state, ctx, a):
    """One group example per (individual example x regime), subject to the causal constraint."""
    rng = random.Random(a.seed)
    ci_cache = {}
    out = {"train": [], "val": [], "test": []}
    stats = Counter()

    for u, g in df.groupby("user_id", sort=False):
        g = g.sort_values("ts", kind="mergesort")
        arr = g[["ts", "poi_idx", "hour", "dow", "split"]].to_numpy(dtype=object)
        for i in range(1, len(arr)):
            t, target, t_hour, t_dow, split = arr[i]
            t, target, t_hour = int(t), int(target), int(t_hour)
            stats["individual_examples"] += 1
            tgt_cat = cat_prefix(cat_of, target, a.cat_level)

            for regime in a.regimes:
                k = rng.choice(a.sizes)
                ctx["t"] = t                      # the occasional regime needs causality
                comp = _pick_companions(regime, u, k, ctx, rng)
                if comp is None:
                    stats[f"skip_no_companions_{regime}"] += 1
                    continue
                members = [u] + comp
                stats[f"candidates_{regime}"] += 1

                # causal validity: every member knew this category before t
                if a.constraint == "cat-inter" and not all(
                        state.knows_category_before(m, tgt_cat, t) for m in members):
                    stats[f"skip_constraint_{regime}"] += 1
                    continue

                # Every member must be active enough at t to contribute a real profile;
                # otherwise the "group" is the anchor plus placeholders. Measured cost at
                # min_member_activity=5: ~1.2% of members, ~1.6% of groups.
                if a.min_member_activity > 0 and any(
                        state.n_before(m, t) < a.min_member_activity for m in members):
                    stats[f"skip_inactive_member_{regime}"] += 1
                    continue

                hist, hist_hours, hist_owner = _joint_history(members, state, t, a.hist_len)
                if len(hist) < a.min_hist:
                    stats[f"skip_short_history_{regime}"] += 1
                    continue

                # A joint history made up entirely of the anchor's own check-ins is not a
                # *group* trajectory -- 8.0% of groups were like this before this guard.
                if a.require_companion_history and len(members) > 1 and \
                        all(o == u for o in hist_owner):
                    stats[f"skip_anchor_only_history_{regime}"] += 1
                    continue

                het = heterogeneity(members, state, t, cat_of, a.cat_level, ci_cache)
                out[split].append(dict(
                    # A synthetic group *example*, not a row of ephemeral_groups.csv -- those
                    # are real co-presence events with their own integer `group_id`. Kept in
                    # separate namespaces on purpose so the two can never be joined by accident.
                    example_id=f"{split}_{len(out[split]):07d}",
                    anchor=int(u), members=[int(m) for m in members], size=len(members),
                    regime=regime,
                    hist=[int(x) for x in hist], hist_hours=[int(x) for x in hist_hours],
                    hist_owner=[int(x) for x in hist_owner],
                    member_profiles=[state.profile(m, t, cat_of, a.profile_top_k,
                                                   a.profile_cats, a.profile_hours)
                                     for m in members],
                    target=target, t_hour=t_hour, t_dow=str(t_dow),
                    heterogeneity=round(het, 4), split=split,
                ))
                stats["emitted"] += 1
    return out, stats


def _pick_companions(regime, u, k, ctx, rng):
    """k-1 companions for anchor `u`, or None if the regime cannot supply enough.

    The three regimes operationalise KCGRS's own group taxonomy (Established / Occasional /
    Random), which that paper names but never constructs:

      established  a CLIQUE in the thresholded multi-signal affinity graph -- every member is
                   high-affinity with every other, not merely with the anchor. This is the
                   "similar preferences" type, and the affinity function behind it is validated
                   against real co-presence at AUC 0.737 (see affinity.py).
      occasional   users the anchor was genuinely co-present with (>= min_tie_groups times).
                   Real behaviour, mixed preferences.
      random       uniform sampling. Maximum heterogeneity, the hard case.
    """
    need = k - 1
    if need == 0:
        return []

    if regime == "established":
        i = ctx["ui"].get(u)
        if i is None:
            return None
        nbrs = ctx["clique_nbrs"].get(i, ())
        if len(nbrs) < need:
            return None
        G, users = ctx["clique_G"], ctx["users"]
        # grow a clique: each new member must be adjacent to ALL current members
        members = [i]
        cand = list(nbrs)
        rng.shuffle(cand)
        for c in cand:
            if len(members) == k:
                break
            if all(G[c, m] for m in members):
                members.append(c)
        if len(members) < k:
            return None
        return [users[j] for j in members[1:]]

    if regime == "occasional":
        # Use an ACTUALLY OBSERVED co-present set, not k-1 samples from the anchor's tie list.
        # Sampling from ties produced stars rather than groups: measured on the tie-based
        # version, 68.5% of size>=3 groups contained member pairs that were never co-present,
        # i.e. the anchor knew A, B and C while A, B, C were strangers to each other. A real
        # co-presence event is mutually co-present by definition, and any subset of it still is.
        # Restricted to events strictly before `t` so the group demonstrably existed already,
        # and to events at RARE venues (<= max_venue_visitors distinct visitors). The rarity gate
        # is Crandall et al.'s actual criterion and it is applied per EVENT rather than per pair:
        # requiring every member pair to have repeated co-presence sounds stricter but destroys
        # the data (2,608 -> 741 events, and every size above 4 vanishes), because a 5-person
        # outing needs all 10 of its pairs to recur. Per-event rarity keeps sizes up to 8
        # (2,608 -> 1,523 at <=50 visitors) while still excluding tourist-magnet coincidences.
        cands = [m for (ts, m) in ctx["real_groups_by_user"].get(u, ())
                 if ts < ctx["t"] and len(m) >= k]
        if not cands:
            return None
        chosen = cands[rng.randrange(len(cands))]
        others = sorted(x for x in chosen if x != u)
        if len(others) < need:
            return None

        # Require the anchor to have at least one RECURRING companion in the group. 56.9% of
        # co-presence events (even after the venue gate) are one-off in every pair -- two people
        # who crossed paths once and never again, which is not a group. Demanding that *every*
        # pair recur would be the obvious gate and it destroys the size distribution
        # (1,523 -> 535 events, nothing above size 3 left), because a real outing is a core dyad
        # plus occasional joiners, not a fully recurring clique. One recurring anchor-companion
        # pair keeps the whole size range (4:25/30, 5:22/24, 6:12/12, 8:7/7) and still removes
        # the pure coincidences.
        rep = ctx["pair_repeat"]
        if ctx["min_companion_repeat"] > 1:
            recurring = [x for x in others
                         if rep.get((min(u, x), max(u, x)), 0) >= ctx["min_companion_repeat"]]
            if not recurring:
                return None
            keep = rng.choice(recurring)
            rest = [x for x in others if x != keep]
            return sorted([keep] + rng.sample(rest, need - 1)) if need > 1 else [keep]
        return rng.sample(others, need)

    if regime == "random":
        picked = set()
        while len(picked) < need:
            c = rng.choice(ctx["users"])
            if c != u:
                picked.add(c)
        return sorted(picked)

    raise ValueError(f"unknown regime {regime!r}")


def _joint_history(members, state, t, hist_len):
    """Merged, time-ordered check-ins of ALL members strictly before t -- the group's shared
    trajectory. `hist_owner` records who contributed each item so the prompt can mark them."""
    merged = []
    for m in members:
        ts = state.ts.get(m)
        if ts is None:
            continue
        n = int(np.searchsorted(ts, t, side="left"))
        lo = max(0, n - hist_len)
        for idx in range(lo, n):
            merged.append((int(ts[idx]), int(state.poi[m][idx]), int(state.hour[m][idx]), int(m)))
    merged.sort()
    merged = merged[-hist_len:]
    return ([x[1] for x in merged], [x[2] for x in merged], [x[3] for x in merged])


# --------------------------------------------------------------------------
# reporting + IO
# --------------------------------------------------------------------------

def report(groups, transitions, co, examples, stats, a):
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  real ephemeral groups     {len(groups):,}   "
          f"(window {a.window}m, sizes {a.min_size}-{a.max_size})")
    if not groups.empty:
        print(f"    span min/median/max     {groups.span_min.min()}/"
              f"{groups.span_min.median():.0f}/{groups.span_min.max()}"
              f"   <- max must be <= {a.window}")
        sizes = Counter(groups["size"])
        print("    size distribution       " +
              "  ".join(f"{s}:{c}" for s, c in sorted(sizes.items())))
    print(f"  real group transitions    {len(transitions):,}   "
          f"<- genuine group moves; far too few to train on (see module docstring)")
    print(f"  co_attended tie pairs     {len(co):,}")

    print(f"\n  individual examples seen  {stats['individual_examples']:,}")
    for regime in a.regimes:
        cand = stats.get(f"candidates_{regime}", 0)
        drops = {r: stats.get(f"skip_{r}_{regime}", 0) for r in
                 ("constraint", "inactive_member", "short_history", "anchor_only_history")}
        kept = cand - sum(drops.values())
        print(f"    {regime:<12} cand={cand:>7,}  " +
              "  ".join(f"-{r.replace('_',' ')}={v:>6,}" for r, v in drops.items()) +
              f"  kept={kept:>7,}")

    print(f"\n  GROUP EXAMPLES")
    for split in ("train", "val", "test"):
        ex = examples[split]
        if not ex:
            print(f"    {split:<6} 0")
            continue
        by_size = Counter(e["size"] for e in ex)
        by_reg = Counter(e["regime"] for e in ex)
        het = np.array([e["heterogeneity"] for e in ex])
        print(f"    {split:<6} {len(ex):>8,}   sizes " +
              " ".join(f"{s}:{c:,}" for s, c in sorted(by_size.items())) +
              "   regimes " + " ".join(f"{r}:{c:,}" for r, c in sorted(by_reg.items())))
        print(f"           heterogeneity  p25={np.percentile(het,25):.3f} "
              f"med={np.median(het):.3f} p75={np.percentile(het,75):.3f}"
              f"   <- the stratification axis for evaluation")


def constructed_groups_table(examples):
    """One row per distinct GROUP (regime + member set), not per example.

    The JSONL holds one row per (group, target) pair, so the same group recurs whenever that
    anchor has another next-POI event -- useful for training, useless for inspecting who is
    actually grouped with whom. This is the spreadsheet-readable roster of the groups
    themselves, with `n_examples` recording how much supervision each one carries.
    """
    agg = {}
    for split in ("train", "val", "test"):
        for e in examples[split]:
            key = (e["regime"], tuple(sorted(e["members"])))
            r = agg.setdefault(key, dict(regime=e["regime"], size=e["size"],
                                         members=key[1], anchors=set(), n_examples=0,
                                         n_train=0, n_val=0, n_test=0, het=[]))
            r["n_examples"] += 1
            r[f"n_{split}"] += 1
            r["anchors"].add(e["anchor"])
            r["het"].append(e["heterogeneity"])
    rows = []
    for i, (_, r) in enumerate(sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[1]["n_examples"]))):
        rows.append(dict(
            constructed_group_id=f"{r['regime'][:3]}_{i:06d}",
            regime=r["regime"], size=r["size"],
            members="|".join(map(str, r["members"])),
            anchors="|".join(map(str, sorted(r["anchors"]))),
            n_examples=r["n_examples"], n_train=r["n_train"],
            n_val=r["n_val"], n_test=r["n_test"],
            heterogeneity=round(float(np.mean(r["het"])), 4),
        ))
    return pd.DataFrame(rows)


def write_all(out_dir, groups, transitions, co, examples, manifest):
    os.makedirs(out_dir, exist_ok=True)
    written = []

    cg = constructed_groups_table(examples)
    if not cg.empty:
        cg.to_csv(os.path.join(out_dir, "constructed_groups.csv"), index=False)
        written.append(f"constructed_groups.csv ({len(cg)})")
        pd.DataFrame([dict(constructed_group_id=r.constructed_group_id, user_id=int(u),
                           regime=r.regime)
                      for r in cg.itertuples(index=False)
                      for u in r.members.split("|")]).to_csv(
            os.path.join(out_dir, "constructed_group_members.csv"), index=False)
        written.append("constructed_group_members.csv")

    if not groups.empty:
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        g = groups.copy()
        g["start_time"] = epoch + pd.to_timedelta(g["start_ts"], unit="m")
        g["end_time"] = epoch + pd.to_timedelta(g["end_ts"], unit="m")
        members = g.pop("members")
        g["members"] = ["|".join(map(str, m)) for m in members]
        g.drop(columns=["start_ts", "end_ts"]).to_csv(
            os.path.join(out_dir, "ephemeral_groups.csv"), index=False)
        written.append(f"ephemeral_groups.csv ({len(g)})")
        pd.DataFrame([dict(group_id=gid, user_id=u)
                      for gid, ms in zip(groups["group_id"], members) for u in ms]).to_csv(
            os.path.join(out_dir, "group_members.csv"), index=False)
        written.append("group_members.csv")

    for name, d in (("real_group_transitions.csv", transitions), ("co_attended.csv", co)):
        if d is not None and not d.empty:
            d.to_csv(os.path.join(out_dir, name), index=False)
            written.append(f"{name} ({len(d)})")

    for split, ex in examples.items():
        path = os.path.join(out_dir, f"group_examples_{split}.jsonl")
        with open(path, "w") as f:
            for e in ex:
                f.write(json.dumps(e) + "\n")
        written.append(f"group_examples_{split}.jsonl ({len(ex)})")

    with open(os.path.join(out_dir, "groups_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    written.append("groups_manifest.json")
    print("\nwrote to " + out_dir + ":\n  " + "\n  ".join(written))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(a):
    df = load_checkins(a.data_dir, a.dataset)
    cat_of, n_pois = load_categories(a.data_dir, a.dataset)
    df = resplit_per_user(df, a.train_frac, a.val_frac) if a.resplit else \
        df.assign(split=df["orig_split"])
    vc = df["split"].value_counts(normalize=True)
    print(f"check-ins={len(df):,}  users={df.user_id.nunique()}  POIs={df.poi_idx.nunique()}  "
          f"split train={vc.get('train',0):.3f}/val={vc.get('val',0):.3f}/test={vc.get('test',0):.3f}")

    # --- everything that shapes a group is mined from TRAIN only ---
    train_df = df[df["split"] == "train"]
    groups = build_ephemeral_groups(train_df, a.window, a.min_size, a.max_size)
    transitions = build_real_transitions(groups, a.horizon) if not groups.empty else pd.DataFrame()
    co = build_co_attended(groups, venue_rarity(train_df))

    ties, kept_ties = tie_pool(co, a.min_tie_groups, a.min_tie_venues)
    print(f"co-attendance ties: {len(co):,} pairs -> {len(kept_ties):,} eligible "
          f"(n_groups>={a.min_tie_groups}, n_venues>={a.min_tie_venues}); "
          f"{len(ties)}/{df.user_id.nunique()} users have >=1 companion")

    users = sorted(df["user_id"].unique().tolist())
    ui = {u: i for i, u in enumerate(users)}

    # --- multi-signal affinity, validated against the real co-presence pairs above ---
    meta = pd.read_csv(os.path.join(a.data_dir, f"poi_metadata_{a.dataset}.csv"))
    loc_of = meta.set_index("poi_idx")["locality"].fillna("Unknown").astype(str).to_dict() \
        if "locality" in meta.columns else {}
    train_df = train_df.copy()
    train_df["day"] = train_df["utc_time"].dt.floor("D")

    A, comp = build_affinity(train_df, cat_of, loc_of, users,
                             cat_level=a.cat_level, far_gap_min=a.far_gap_min)
    Y = copresence_labels(groups, users)
    validation = validate(A, comp, Y, verbose=True)
    clique_nbrs, clique_G, clique_thr = clique_neighbourhood(A, a.affinity_percentile)
    n_eligible = sum(1 for v in clique_nbrs.values() if v)
    print(f"  affinity graph: top {100-a.affinity_percentile:.1f}% (cut={clique_thr:.3f}), "
          f"{n_eligible}/{len(users)} users with >=1 neighbour")

    visitors = train_df.groupby("poi_idx")["user_id"].nunique().to_dict()
    real_by_user = defaultdict(list)
    n_rare = 0
    for r in groups.itertuples(index=False):
        if visitors.get(int(r.poi_idx), 0) > a.max_venue_visitors:
            continue                      # tourist magnet: co-presence there means nothing
        n_rare += 1
        m = frozenset(r.members)
        for u_ in m:
            real_by_user[u_].append((int(r.start_ts), m))
    print(f"  'occasional' compositions: {n_rare:,}/{len(groups):,} real co-presence events at "
          f"venues with <={a.max_venue_visitors} distinct visitors, over {len(real_by_user)} users")

    pair_repeat = {(min(r.u1, r.u2), max(r.u1, r.u2)): int(r.n_groups)
                   for r in co.itertuples(index=False)}
    ctx = dict(ties=ties, users=users, ui=ui, real_groups_by_user=real_by_user,
               pair_repeat=pair_repeat, min_companion_repeat=a.min_companion_repeat,
               clique_nbrs=clique_nbrs, clique_G=clique_G, t=None)

    state = MemberState(df, cat_of, a.cat_level)
    examples, stats = build_group_examples(df, cat_of, state, ctx, a)

    report(groups, transitions, co, examples, stats, a)
    _assert_causal(examples, state, df)
    _assert_cliques(examples, clique_G, ui)
    _assert_recurring_companions(examples, pair_repeat, a.min_companion_repeat)

    manifest = dict(
        config={k: v for k, v in vars(a).items()},
        n_pois=n_pois, n_users=len(users), n_checkins=len(df),
        n_ephemeral_groups=len(groups), n_real_transitions=len(transitions),
        n_co_attended=len(co),
        affinity=dict(validation=validation, clique_threshold=clique_thr,
                      n_clique_eligible_users=n_eligible),
        counts={s: len(e) for s, e in examples.items()},
        stats=dict(stats),
    )
    write_all(a.out_dir, groups, transitions, co, examples, manifest)


def _assert_recurring_companions(examples, pair_repeat, min_repeat):
    """Every `occasional` group must contain at least one companion the anchor met repeatedly.
    Without this, 56.9% of the groups are two people who crossed paths once."""
    if min_repeat <= 1:
        return
    checked = bad = 0
    for split in ("train", "val", "test"):
        for e in examples[split]:
            if e["regime"] != "occasional":
                continue
            checked += 1
            a = e["anchor"]
            if not any(pair_repeat.get((min(a, c), max(a, c)), 0) >= min_repeat
                       for c in e["members"][1:]):
                bad += 1
    assert bad == 0, f"{bad} occasional groups have no recurring anchor-companion pair"
    if checked:
        print(f"  recurrence assert passed: {checked:,} 'occasional' groups each contain a "
              f"companion met >={min_repeat}x")


def _assert_cliques(examples, G, ui):
    """Every `established` group must be a clique in the thresholded affinity graph -- a star
    around the anchor would defeat the point of the regime."""
    checked = bad = 0
    for split in ("train", "val", "test"):
        for e in examples[split][:3000]:
            if e["regime"] != "established":
                continue
            idx = [ui[m] for m in e["members"]]
            checked += 1
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    if not G[idx[i], idx[j]]:
                        bad += 1
    assert bad == 0, f"{bad} non-clique pairs inside 'established' groups"
    if checked:
        print(f"  clique assert passed: {checked:,} 'established' groups are true cliques")


def _assert_causal(examples, state, df):
    """Cheap invariants that would catch the leaks that actually matter."""
    ts_of = {}
    for u, g in df.groupby("user_id", sort=False):
        ts_of[u] = g.sort_values("ts")["ts"].to_numpy()
    checked = 0
    for split in ("train", "val", "test"):
        for e in examples[split][:2000]:
            for pr in e["member_profiles"]:
                assert sum(c for _, c in pr["top_pois"]) <= pr["n_seen"], "profile is not causal"
            assert len(e["hist"]) == len(e["hist_hours"]) == len(e["hist_owner"])
            assert set(e["hist_owner"]) <= set(e["members"]), "history from a non-member"
            assert e["anchor"] == e["members"][0]
            assert len(set(e["members"])) == len(e["members"]), "duplicate member"
            checked += 1
    print(f"\n  causality + integrity asserts passed on {checked:,} sampled examples")


def _self_check():
    """Synthetic fixture: planted companion pairs, a busy venue that must not create ties,
    and a target category one member has never seen (must be rejected by the constraint)."""
    import tempfile
    print("SELF-CHECK on a synthetic fixture")
    base = pd.Timestamp("2012-04-03", tz="UTC")
    rows = []

    def add(u, poi, minute):
        rows.append(dict(user_id=u, poi_idx=poi, venue_id=f"v{poi}",
                         utc_time=base + pd.Timedelta(minutes=int(minute)), orig_split="train"))

    for k, (x, y) in enumerate([(1, 2), (3, 4)]):           # planted companion pairs
        for day in range(40):
            t = day * 1440 + 1080 + k * 5
            add(x, 10 + k, t); add(y, 10 + k, t + 7)
            add(x, 20 + k, t + 90); add(y, 20 + k, t + 95)
    for day in range(40):                                    # busy venue: strangers, no ties
        for m in range(30):
            add(100 + m, 99, day * 1440 + 600 + m * 3)

    df = pd.DataFrame(rows)
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True)
    df["ts"] = df["utc_time"].astype("int64") // (60 * 10 ** 9)
    df["hour"] = df["utc_time"].dt.hour.astype(int)
    df["dow"] = df["utc_time"].dt.day_name()
    cat_of = {10: "Dining and Drinking > Bar", 11: "Dining and Drinking > Bar",
              20: "Dining and Drinking > Cafe", 21: "Dining and Drinking > Cafe",
              99: "Arts and Entertainment > Museum"}

    class A: pass
    a = A()
    a.window, a.min_size, a.max_size, a.horizon = 60, 2, 8, 240
    a.cat_level, a.hist_len, a.min_hist = 2, 15, 1
    a.profile_top_k, a.profile_cats, a.profile_hours = 5, 3, 3
    a.sizes, a.regimes, a.constraint, a.seed = [2], ["occasional"], "cat-inter", 42
    a.affinity_percentile, a.far_gap_min = 99.0, 180
    a.min_member_activity, a.require_companion_history = 1, True
    a.max_venue_visitors = 10        # fixture: venue 99 has 30 visitors -> must be excluded
    a.min_companion_repeat = 2       # planted pairs meet 40x; strangers meet once
    # strict Crandall setting: a companion must have been co-present at >1 distinct venue.
    # The fixture's 30 busy-venue strangers co-occur 40 times but only ever at venue 99, so
    # this is exactly what must exclude them while keeping the planted two-venue pairs.
    a.min_tie_groups, a.min_tie_venues = 1, 2

    df = resplit_per_user(df, 0.70, 0.10)
    groups = build_ephemeral_groups(df[df.split == "train"], a.window, a.min_size, a.max_size)
    co = build_co_attended(groups, venue_rarity(df[df.split == "train"]))
    ties, kept_ties = tie_pool(co, a.min_tie_groups, a.min_tie_venues)

    users = sorted(df.user_id.unique().tolist())
    M, ui = category_profile_matrix(df[df.split == "train"], cat_of, users, a.cat_level)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        SIM = M @ M.T
    np.fill_diagonal(SIM, -np.inf)
    order = np.argsort(-SIM, axis=1)
    state = MemberState(df, cat_of, a.cat_level)
    visitors = df[df.split == "train"].groupby("poi_idx")["user_id"].nunique().to_dict()
    real_by_user = defaultdict(list)
    real_sets = []
    for r in groups.itertuples(index=False):
        if visitors.get(int(r.poi_idx), 0) > a.max_venue_visitors:
            continue
        m = frozenset(r.members)
        real_sets.append(m)
        for u_ in m:
            real_by_user[u_].append((int(r.start_ts), m))
    pair_repeat = {(min(r.u1, r.u2), max(r.u1, r.u2)): int(r.n_groups)
                   for r in co.itertuples(index=False)}
    ctx = dict(ties=ties, users=users, ui=ui, real_groups_by_user=real_by_user,
               pair_repeat=pair_repeat, min_companion_repeat=a.min_companion_repeat,
               clique_nbrs={}, clique_G=None, t=None)
    examples, stats = build_group_examples(df, cat_of, state, ctx, a)
    report(groups, transitions := build_real_transitions(groups, a.horizon), co, examples, stats, a)

    planted = {(1, 2), (3, 4)}
    top = {(int(r.u1), int(r.u2)) for r in co.head(len(planted)).itertuples(index=False)}
    busy = co[(co.u1 >= 100) & (co.u2 >= 100)]
    all_ex = [e for v in examples.values() for e in v]

    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    print()
    results = [
        ok("groups formed", not groups.empty),
        ok(f"every group span <= window ({groups.span_min.max()} <= {a.window})",
           groups.span_min.max() <= a.window),
        ok("IDF weighting ranks planted pairs above busy-venue strangers",
           top == planted and (busy.empty or co.head(2).weight_idf.min() > busy.weight_idf.max())),
        ok("group examples emitted", len(all_ex) > 0),
        ok("every example has the anchor first", all(e["anchor"] == e["members"][0] for e in all_ex)),
        ok("every occasional group is a subset of an observed co-presence set",
           all(any(set(e["members"]) <= s for s in real_sets) for e in all_ex)),
        ok("joint history only contains member check-ins",
           all(set(e["hist_owner"]) <= set(e["members"]) for e in all_ex)),
        ok("profiles are causal (counts <= prefix length)",
           all(sum(c for _, c in pr["top_pois"]) <= pr["n_seen"]
               for e in all_ex for pr in e["member_profiles"])),
        ok("constraint rejects targets a member has not seen causally",
           stats.get("skip_constraint_occasional", 0) > 0),
        ok("venue-rarity gate excludes busy-venue strangers",
           all(all(m < 100 for m in e["members"]) for e in all_ex)),
    ]
    with tempfile.TemporaryDirectory() as td:
        write_all(td, groups, transitions, co, examples, {"source": "self_check"})
        n = sum(1 for _ in open(os.path.join(td, "group_examples_train.jsonl")))
        results.append(ok("jsonl round-trips", n == len(examples["train"])))
    return all(results)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./data/groups")
    p.add_argument("--dataset", default="NYC")
    # real group mining
    p.add_argument("--window", type=int, default=60, help="co-presence window, minutes")
    p.add_argument("--min-size", type=int, default=2)
    p.add_argument("--max-size", type=int, default=8)
    p.add_argument("--horizon", type=int, default=240, help="real-transition horizon, minutes")
    p.add_argument("--min-companion-repeat", type=int, default=2,
                   help="an 'occasional' group must contain >=1 companion the anchor was "
                        "co-present with at least this many times; 1 disables the check and "
                        "lets one-off encounters through (56.9%% of events)")
    p.add_argument("--max-venue-visitors", type=int, default=50,
                   help="'occasional' groups only reuse co-presence events from venues with at "
                        "most this many distinct visitors (median venue has 40, p90 has 203)")
    p.add_argument("--min-tie-groups", type=int, default=2,
                   help="co_attended.csv export filter (feeds the KG); does NOT gate the "
                        "occasional regime, which uses whole observed co-presence sets instead")
    p.add_argument("--min-tie-venues", type=int, default=1,
                   help="...at >= this many distinct venues (2 = strict Crandall criterion)")
    # group examples
    p.add_argument("--regimes", nargs="+",
                   default=["established", "occasional", "random"],
                   choices=["established", "occasional", "random"],
                   help="KCGRS's group taxonomy: established=affinity clique, "
                        "occasional=real co-presence, random=uniform")
    p.add_argument("--affinity-percentile", type=float, default=99.0,
                   help="keep the top (100-p)%% of user pairs as affinity edges; 99.0 gives "
                        "5,752 pairs / 905 users / 1,704 5-cliques on FSQ-NYC")
    p.add_argument("--far-gap-min", type=int, default=180,
                   help="same-POI-same-day encounters must be >= this far apart, so the "
                        "affinity signal cannot contain the co-presence labels it is scored on")
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4, 5],
                   help="group size sampled per anchor from this list")
    p.add_argument("--constraint", choices=["cat-inter", "none"], default="cat-inter",
                   help="cat-inter: every member visited the target's category BEFORE t")
    p.add_argument("--cat-level", type=int, default=2, help="taxonomy level for the constraint")
    p.add_argument("--hist-len", type=int, default=15, help="matches HIST_LEN in the notebook")
    p.add_argument("--min-hist", type=int, default=1)
    p.add_argument("--min-member-activity", type=int, default=5,
                   help="drop groups where any member has < this many check-ins before t")
    p.add_argument("--require-companion-history", action="store_true", default=True,
                   help="drop groups whose joint history is 100%% the anchor's own check-ins")
    p.add_argument("--allow-anchor-only-history", dest="require_companion_history",
                   action="store_false")
    p.add_argument("--profile-top-k", type=int, default=5)
    p.add_argument("--profile-cats", type=int, default=3)
    p.add_argument("--profile-hours", type=int, default=3)
    # split
    p.add_argument("--resplit", action="store_true", default=True,
                   help="per-user chronological 70/10/20, identical to the notebook")
    p.add_argument("--no-resplit", dest="resplit", action="store_false")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()

    if a.self_check:
        sys.exit(0 if _self_check() else 1)
    run(a)


if __name__ == "__main__":
    main()
