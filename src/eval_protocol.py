"""
LLMGPR's evaluation protocol (Long et al., CIKM'25, Sec. 4.1), implemented so that HyGro-POI's
row in the comparison tables is measured the way every other row in them was.

Why this exists
---------------
The stage-5 notebooks rank the true POI against the ENTIRE catalogue (`_val_pass`: strict-greater
rank over all N_POI logits, then Acc@k / MRR). LLMGPR does not. Its published numbers -- and every
baseline number it reports -- come from a different protocol, quoted here so nothing is
paraphrased away:

    "we adopt the leave-one-out protocol [...] for each of the check-in sequences, the last
     check-in POI is for testing, the second last POI is for validation, and all others are for
     training. In addition, the maximum sequence length is set to 200. For each ground truth
     check-in POI, [...] we only pair it with 500 unvisited and nearest POIs within the same
     region as the candidates for ranking. [...] we leverage two ranking metrics, namely Hit
     Ratio at Rank k (HR@k) and Normalized Discounted Cumulative Gain at Rank k (NDCG@k)"

A random ranker scores HR@5 = 1.0% under that protocol and 0.10% against our 5,120 POIs, and
sampled metrics are not monotonically consistent with full-ranking ones (Krichene & Rendle,
KDD'20), so the two cannot be reconciled by a correction factor afterwards. The only way to put
our number in their table is to compute it their way. This module does exactly that, and nothing
else: the full-catalogue numbers stay where they are, reported as a separate protocol.

What "their way" means here, decision by decision
--------------------------------------------------
* **Candidates.** ground truth + `n_cand` (500) POIs that are (a) not in the example's own
  input history, (b) in the same region as the ground truth, (c) nearest to it by haversine
  distance. "Nearest to it" is read as nearest to the GROUND-TRUTH POI, the reading in Long et
  al.'s earlier work; `anchor="last"` gives nearest-to-the-last-check-in instead, and the
  manifest records which was used.
* **Region.** a metadata column: `locality` is the city on the 3-city exports (LLMGPR,
  GOWALLA), which is what "region" means in the paper. On TSMC FSQ-NYC `locality` is a
  neighbourhood (142 values), and the whole dataset is one metropolitan region, so the region
  filter is the whole catalogue there. `--region-col auto` makes that choice from the column's
  cardinality and prints it; never guess silently.
* **POIs without coordinates.** LLMGPR's 10 km catalogue covers ~43% of the venues actually
  visited on the 3-city Foursquare export, so 57% of those POIs have no lat/lon. A POI without
  coordinates cannot be "nearest" to anything. When the geo-sorted pool runs short, the set is
  filled with seeded-random same-region unvisited POIs and the fill is COUNTED (`n_fill`,
  `frac_examples_filled`) -- an anchor without coordinates yields an all-random set and is
  counted separately (`n_anchor_no_coord`). Both go into the manifest. NYC and GOWALLA have
  100% coverage, so there the fill path never fires.
* **Split.** leave-one-out is applied to the EXAMPLES, not by retraining: the last check-in of
  each user's sequence is the test example, the second-last the validation example. Under the
  pipeline's per-user chronological 70/10/20, the last check-in is always in the test tail, so
  a checkpoint trained on the 70% prefix has never seen it -- the model just saw LESS training
  data than LLMGPR's all-but-two. That is the conservative direction and it is stated in the
  paper, not hidden.
* **Groups under leave-one-out.** LLMGPR's group sequences have a "last check-in"; our
  constructed group examples do not -- build_groups.py emits one example per (anchor
  check-in, regime) and a member set recurs many times. The analogue of "the last check-in of
  each sequence" is ONE example per member set, the chronologically last. Examples carry no
  timestamp, but every member profile carries `n_seen` = that member's check-ins strictly
  before t, so within a fixed member set the SUM of `n_seen` is monotone in t: the example
  with the largest sum is the latest (ties -> later `example_id`, which is chronological
  within an anchor). `--group-select all` keeps every example instead; the manifest says which.
* **Sequence length 200.** `truncate_history` is provided for completeness; the prompt budget
  (MAX_LEN=1024 tokens) already caps history well below 200 POIs, so it is a no-op in practice.
* **Rank.** strict-greater over the candidate scores plus one -- the same tie rule as
  `_val_pass`, restricted to the candidate columns. The ground truth is in the set, and
  `s > s_target` is False for itself, so it is never counted as its own competitor.
* **Metrics.** HR@k = 1[rank <= k]; NDCG@k = 1[rank <= k] / log2(rank + 1). One relevant item
  per example, so the ideal DCG is 1. Reported at k = 5 and 10, matching the tables. Acc@k and
  MRR are deliberately NOT what this module reports.

Use from the notebook (after the checkpoint is loaded and `_forward_multimask` exists)
-------------------------------------------------------------------------------------
    from eval_protocol import (CandidateIndex, last_per_user, last_per_group, ranks_from_logits,
                               hr_ndcg, visited_before_target)
    index   = CandidateIndex(meta_df, region_col="auto")
    loo_ex  = last_per_user(test_ex)                        # leave-one-out test examples
    # group task: loo_grp = last_per_group(group_test_ex); visited = each example's own `hist`
    visited = visited_before_target(full_df)                # user -> set of POIs before the last
    cands   = np.stack([index.candidates(ex["target"], visited[ex["user"]])[0] for ex in loo_ex])
    ranks   = []
    for batch, cand_b in batches(loo_ex, cands):            # same collate as _val_pass
        logits, tgt = _forward_multimask(batch)
        ranks.extend(ranks_from_logits(logits[:, 0, :], tgt, cand_b))
    print(hr_ndcg(ranks))                                    # {'HR@5': .., 'NDCG@5': .., ...}

Or build the candidate sets once, offline, as a reproducible artifact (this file's CLI), and
load the .npz in the notebook -- then the protocol is a versioned input, not a notebook cell.

    python src/eval_protocol.py --data-dir data --dataset NYC --task individual --split test \
        --out data/eval_candidates_NYC_individual_test.npz
    python src/eval_protocol.py --data-dir data/llmgpr --dataset LLMGPR --task group \
        --groups-dir data/llmgpr/groups_social --split test \
        --out data/llmgpr/eval_candidates_LLMGPR_group_test.npz
    python src/eval_protocol.py --self-check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
DEFAULT_N_CAND = 500
DEFAULT_KS = (5, 10)


# ──────────────────────────────────────────────────────────────────────────────
# geometry
# ──────────────────────────────────────────────────────────────────────────────
def haversine_km(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from one point to an array of points."""
    p0, l0 = math.radians(lat0), math.radians(lon0)
    p1, l1 = np.radians(lats), np.radians(lons)
    dphi, dlmb = p1 - p0, l1 - l0
    a = np.sin(dphi / 2.0) ** 2 + math.cos(p0) * np.cos(p1) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ──────────────────────────────────────────────────────────────────────────────
# candidate sets
# ──────────────────────────────────────────────────────────────────────────────
class CandidateIndex:
    """Nearest-unvisited-same-region candidate sets over a POI metadata table.

    `meta_df` needs `poi_idx` (contiguous 0..N-1), `latitude`, `longitude` and, unless
    `region_col` is None, the region column. NaN coordinates are allowed and handled (see the
    module docstring).
    """

    AUTO_REGION_MAX_UNIQUE = 10   # `locality` is a city on the 3-city exports (3 values),
                                  # a neighbourhood on TSMC NYC (142) -> whole catalogue there

    def __init__(self, meta_df: pd.DataFrame, region_col: str | None = "auto",
                 lat_col: str = "latitude", lon_col: str = "longitude",
                 seed: int = 42, cache_slack: int = 1000):
        m = meta_df.sort_values("poi_idx").reset_index(drop=True)
        assert (m["poi_idx"].to_numpy() == np.arange(len(m))).all(), \
            "poi_idx must be contiguous 0..N-1 (it is in every poi_metadata_*.csv the pipeline writes)"
        self.n = len(m)
        self.lat = m[lat_col].to_numpy(dtype=float)
        self.lon = m[lon_col].to_numpy(dtype=float)
        self.has_coord = np.isfinite(self.lat) & np.isfinite(self.lon)

        if region_col == "auto":
            if "locality" in m.columns and m["locality"].nunique(dropna=True) <= self.AUTO_REGION_MAX_UNIQUE:
                region_col = "locality"
            else:
                region_col = None
        self.region_col = region_col
        if region_col is None:
            self.region = np.zeros(self.n, dtype=np.int64)
            self.region_names = {0: "<whole catalogue>"}
        else:
            codes, uniques = pd.factorize(m[region_col].fillna("<unknown>"), sort=True)
            self.region = codes.astype(np.int64)
            self.region_names = dict(enumerate(uniques.tolist()))
        self._pool_by_region = {r: np.flatnonzero(self.region == r) for r in np.unique(self.region)}
        self._geo_pool_by_region = {r: p[self.has_coord[p]] for r, p in self._pool_by_region.items()}

        self.seed = seed
        self.cache_slack = cache_slack
        self._nearest_cache: dict[tuple[int, int], np.ndarray] = {}

    # -- description ---------------------------------------------------------------------------
    def describe(self) -> dict:
        return dict(
            n_poi=int(self.n),
            region_col=self.region_col,
            n_regions=len(self._pool_by_region),
            regions={self.region_names[r]: int(len(p)) for r, p in self._pool_by_region.items()},
            coord_coverage=float(self.has_coord.mean()),
        )

    # -- nearest neighbours ----------------------------------------------------------------------
    def nearest_sorted(self, anchor: int, k: int) -> np.ndarray:
        """Same-region POIs WITH coordinates, sorted by distance to `anchor`, anchor excluded,
        at most `k` of them. Cached per (anchor, k-bucket) so repeated targets are free."""
        assert self.has_coord[anchor], "nearest_sorted needs an anchor with coordinates"
        pool = self._geo_pool_by_region[self.region[anchor]]
        k_eff = min(k, len(pool) - 1)
        key = (int(anchor), int(k_eff))
        hit = self._nearest_cache.get(key)
        if hit is not None:
            return hit
        d = haversine_km(self.lat[anchor], self.lon[anchor], self.lat[pool], self.lon[pool])
        d[pool == anchor] = np.inf
        if k_eff < len(pool):
            part = np.argpartition(d, k_eff)[:k_eff]
            order = part[np.argsort(d[part], kind="mergesort")]
        else:
            order = np.argsort(d, kind="mergesort")
        out = pool[order][:k_eff].astype(np.int32)
        self._nearest_cache[key] = out
        return out

    # -- the protocol ----------------------------------------------------------------------------
    def candidates(self, target: int, visited, n: int = DEFAULT_N_CAND,
                   anchor: int | None = None, rng: np.random.Generator | None = None):
        """-> (cands int32[n+1] with the target at position 0, info dict).

        `visited`: POIs in the example's own input history (unvisited = not in it).
        `anchor`:  POI to measure distance from; default the target itself.
        """
        target = int(target)
        anchor = target if anchor is None else int(anchor)
        excl = set(int(v) for v in visited)
        excl.add(target)
        rng = rng if rng is not None else np.random.default_rng(self.seed + target)

        picked: list[int] = []
        anchor_has_coord = bool(self.has_coord[anchor])
        if anchor_has_coord:
            # ask for progressively more neighbours until n unvisited ones are in hand or the
            # region's geo pool is exhausted -- a heavy user may have visited many of the nearest
            k = n + len(excl) + self.cache_slack
            pool_size = len(self._geo_pool_by_region[self.region[anchor]]) - 1
            while True:
                order = self.nearest_sorted(anchor, k)
                picked = [int(p) for p in order if int(p) not in excl][:n]
                if len(picked) >= n or k >= pool_size:
                    break
                k = min(k * 2, pool_size)
        n_geo = len(picked)

        n_fill = n - n_geo
        if n_fill > 0:
            # not enough geo-rankable unvisited POIs in the region: fill from the rest of the
            # region (mostly POIs without coordinates), seeded, so the set is still deterministic
            pool = self._pool_by_region[self.region[target]]
            taken = excl | set(picked)
            rest = np.array([int(p) for p in pool if int(p) not in taken], dtype=np.int64)
            take = min(n_fill, len(rest))
            if take > 0:
                picked.extend(int(p) for p in rng.choice(rest, size=take, replace=False))
            n_fill = n - n_geo  # what the protocol asked for beyond geo, even if region ran short

        cands = np.array([target] + picked, dtype=np.int32)
        assert len(set(cands.tolist())) == len(cands), "duplicate candidate"
        assert not (set(picked) & excl), "a visited POI or the target leaked into the negatives"
        assert (self.region[cands] == self.region[target]).all(), "candidate outside the target's region"
        info = dict(n_geo=n_geo, n_fill=max(0, n - n_geo), n_short=max(0, n - (len(cands) - 1)),
                    anchor_has_coord=anchor_has_coord)
        return cands, info


# ──────────────────────────────────────────────────────────────────────────────
# leave-one-out selection and history helpers
# ──────────────────────────────────────────────────────────────────────────────
def leave_one_out_split(df: pd.DataFrame, user_col: str = "user_id",
                        time_col: str = "utc_time") -> pd.DataFrame:
    """Overwrite `split`: per user, last check-in -> test, second-last -> val, rest -> train.
    Users with fewer than three check-ins go entirely to train (nothing to hold out safely)."""
    df = df.sort_values([user_col, time_col] if time_col in df.columns else [user_col],
                        kind="mergesort").reset_index(drop=True)
    split = np.full(len(df), "train", dtype=object)
    for _, idx in df.groupby(user_col, sort=False).indices.items():
        if len(idx) >= 3:
            split[idx[-1]] = "test"
            split[idx[-2]] = "val"
    df["split"] = split
    return df


def last_per_user(examples: list[dict], user_key: str = "user", position_key=None) -> list[dict]:
    """Leave-one-out selection over already-built examples: keep, per user, the example whose
    target is the LAST check-in of the sequence. Examples carry `profile.n_seen` (the prefix
    length) when built by the notebook's `build_split_examples`; that is the default ordering.
    Deterministic given the input order (ties keep the later example)."""
    if position_key is None:
        def position_key(ex):
            prof = ex.get("profile")
            return prof["n_seen"] if isinstance(prof, dict) and "n_seen" in prof else 0
    best: dict = {}
    for ex in examples:
        u = ex[user_key]
        if u not in best or position_key(ex) >= position_key(best[u]):
            best[u] = ex
    return list(best.values())


# Group examples come in two shapes and the helpers below accept both:
#   * the jsonl written by build_groups.py: members=[int], member_profiles=[{n_seen,..}],
#     hist=[int] (joint history), hist_owner=[int]
#   * the notebook's rebuilt dict (build_group_example): members=[{user, profile{n_seen,..},
#     hist, hist_hours}] and no top-level hist -- the joint history split per owner
def group_members(ex: dict) -> list[int]:
    return [int(m["user"]) if isinstance(m, dict) else int(m) for m in ex["members"]]


def group_n_seen(ex: dict) -> list[int]:
    if "member_profiles" in ex:
        return [int((p or {}).get("n_seen", 0)) for p in ex["member_profiles"]]
    return [int((m.get("profile") or {}).get("n_seen", 0)) for m in ex["members"] if isinstance(m, dict)]


def group_visited(ex: dict) -> set[int]:
    """POIs in the example's own input history -- the joint history, whichever shape carries it."""
    if "hist" in ex:
        return {int(p) for p in ex["hist"]}
    out: set[int] = set()
    for m in ex["members"]:
        if isinstance(m, dict):
            out.update(int(p) for p in m.get("hist", []))
    return out


def group_time_key(ex: dict) -> tuple:
    """Monotone-in-time proxy for a group example: sum of members' `n_seen` (each member's
    check-ins strictly before t), then example_id (chronological within an anchor)."""
    return (sum(group_n_seen(ex)), str(ex.get("example_id", "")))


def last_per_group(examples: list[dict]) -> list[dict]:
    """Leave-one-out over constructed group examples: keep, per member set, the chronologically
    last example (see the module docstring). Input order does not matter; output is sorted by
    example_id for reproducibility."""
    best: dict[frozenset, dict] = {}
    for ex in examples:
        g = frozenset(group_members(ex))
        if g not in best or group_time_key(ex) > group_time_key(best[g]):
            best[g] = ex
    return sorted(best.values(), key=lambda e: str(e.get("example_id", "")))


def visited_before_target(full_df: pd.DataFrame, user_col: str = "user_id", poi_col: str = "poi_idx",
                          time_col: str = "utc_time", drop_last: int = 1) -> dict[int, set]:
    """user -> set of POIs in the sequence BEFORE its last `drop_last` check-ins. With
    drop_last=1 this is the leave-one-out test example's 'visited' set; drop_last=2 the
    validation one's."""
    df = full_df.sort_values([user_col, time_col] if time_col in full_df.columns else [user_col],
                             kind="mergesort")
    out: dict[int, set] = {}
    for u, g in df.groupby(user_col, sort=False):
        seq = g[poi_col].tolist()
        out[int(u)] = set(int(p) for p in (seq[:-drop_last] if drop_last else seq))
    return out


def truncate_history(hist: list, max_len: int = 200) -> list:
    """LLMGPR caps sequences at 200; the prompt budget caps ours far lower, so a no-op in practice."""
    return hist[-max_len:] if max_len and len(hist) > max_len else hist


# ──────────────────────────────────────────────────────────────────────────────
# ranking and metrics
# ──────────────────────────────────────────────────────────────────────────────
def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


def ranks_from_logits(logits, targets, cands) -> list[int]:
    """logits [B, N_POI], targets [B], cands [B, n+1] -> rank of each target among its
    candidates (1 = best). Strict-greater tie rule, identical to the notebook's `_val_pass`."""
    L, T, C = _to_numpy(logits), _to_numpy(targets).astype(np.int64), _to_numpy(cands).astype(np.int64)
    assert L.ndim == 2 and C.ndim == 2 and L.shape[0] == C.shape[0] == T.shape[0]
    assert (C[:, 0] == T).all(), "candidates must carry the target at position 0"
    cand_scores = np.take_along_axis(L, C, axis=1)              # [B, n+1]
    tgt_scores = L[np.arange(L.shape[0]), T][:, None]           # [B, 1]
    return ((cand_scores > tgt_scores).sum(1) + 1).astype(int).tolist()


def hr_ndcg(ranks, ks=DEFAULT_KS) -> dict:
    """HR@k and NDCG@k for one relevant item per example (ideal DCG = 1)."""
    r = np.asarray(ranks, dtype=float)
    out = {"n": int(len(r))}
    for k in ks:
        hit = r <= k
        out[f"HR@{k}"] = float(hit.mean()) if len(r) else float("nan")
        out[f"NDCG@{k}"] = float((hit / np.log2(r + 1.0)).mean()) if len(r) else float("nan")
    return out


def format_metrics(m: dict, ks=DEFAULT_KS) -> str:
    return "  ".join(f"{name}@{k}={m[f'{name}@{k}']:.4f}" for k in ks for name in ("HR", "NDCG")) \
        + f"  (n={m['n']})"


# ──────────────────────────────────────────────────────────────────────────────
# offline artifact: build candidate sets for a split and save them
# ──────────────────────────────────────────────────────────────────────────────
def _find(data_dir: str, fname: str) -> str:
    for root in (data_dir, ".", os.path.join(data_dir, "..")):
        p = os.path.join(root, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{fname} not under {data_dir}")


def build_individual_candidates(data_dir: str, dataset: str, split: str, index: CandidateIndex,
                                n_cand: int, anchor_mode: str):
    """Leave-one-out over the pipeline's CSVs: one example per user (last check-in for test,
    second-last for val). Returns (keys, targets, cands [E, n+1], infos)."""
    frames = [pd.read_csv(_find(data_dir, f"{s}_{dataset}.csv")) for s in ("train", "val", "test")]
    full = pd.concat(frames, ignore_index=True)
    drop_last = 1 if split == "test" else 2
    full = full.sort_values(["user_id", "utc_time"] if "utc_time" in full.columns else ["user_id"],
                            kind="mergesort")
    keys, targets, cands, infos = [], [], [], []
    for u, g in full.groupby("user_id", sort=False):
        seq = [int(p) for p in g["poi_idx"].tolist()]
        if len(seq) < 3:
            continue
        target = seq[-drop_last]
        prefix = seq[:-drop_last]
        anchor = None if anchor_mode == "target" else prefix[-1]
        c, info = index.candidates(target, prefix, n=n_cand, anchor=anchor)
        keys.append(int(u)); targets.append(target); cands.append(c); infos.append(info)
    return keys, targets, cands, infos


def build_group_candidates(groups_dir: str, split: str, index: CandidateIndex, n_cand: int,
                           anchor_mode: str, group_select: str = "last"):
    """Over build_groups.py's group_examples_<split>.jsonl: `visited` is the joint history the
    example is predicted from; keyed by example_id. `group_select="last"` keeps one example per
    member set (leave-one-out analogue, see module docstring); "all" keeps every example."""
    path = os.path.join(groups_dir, f"group_examples_{split}.jsonl")
    with open(path) as f:
        examples = [json.loads(line) for line in f]
    n_all = len(examples)
    if group_select == "last":
        examples = last_per_group(examples)
    print(f"[eval-protocol] group examples: {n_all:,} in {split}; "
          f"{len(examples):,} kept ({group_select}: "
          f"{'one per member set, chronologically last' if group_select == 'last' else 'every example'})")
    keys, targets, cands, infos = [], [], [], []
    for ex in examples:
        target = int(ex["target"])
        hist = [int(p) for p in ex["hist"]]
        anchor = None if anchor_mode == "target" or not hist else hist[-1]
        c, info = index.candidates(target, group_visited(ex), n=n_cand, anchor=anchor)
        keys.append(ex["example_id"]); targets.append(target); cands.append(c); infos.append(info)
    return keys, targets, cands, infos


def summarize_infos(infos: list[dict], n_cand: int) -> dict:
    n = len(infos)
    return dict(
        n_examples=n,
        n_cand=n_cand,
        frac_examples_filled=float(np.mean([i["n_fill"] > 0 for i in infos])) if n else 0.0,
        mean_fill_per_example=float(np.mean([i["n_fill"] for i in infos])) if n else 0.0,
        n_anchor_no_coord=int(sum(not i["anchor_has_coord"] for i in infos)),
        n_examples_short=int(sum(i["n_short"] > 0 for i in infos)),
    )


def build_artifact(a) -> dict:
    meta = pd.read_csv(_find(a.data_dir, f"poi_metadata_{a.dataset}.csv"))
    index = CandidateIndex(meta, region_col=(None if a.region_col == "none" else a.region_col),
                           seed=a.seed)
    desc = index.describe()
    print(f"[eval-protocol] {a.dataset}: {desc['n_poi']:,} POIs | region column: {desc['region_col']} "
          f"({desc['n_regions']} region(s)) | coordinate coverage {desc['coord_coverage']:.1%}")
    if desc["coord_coverage"] < 1.0:
        print(f"[eval-protocol] WARNING: {1 - desc['coord_coverage']:.1%} of POIs have no coordinates -- "
              f"'nearest' is undefined for them; the fill path will fire and be counted below")

    if a.task == "individual":
        keys, targets, cands, infos = build_individual_candidates(
            a.data_dir, a.dataset, a.split, index, a.n_cand, a.anchor)
    else:
        groups_dir = a.groups_dir or os.path.join(a.data_dir, "groups_social")
        keys, targets, cands, infos = build_group_candidates(groups_dir, a.split, index, a.n_cand,
                                                             a.anchor, a.group_select)

    stats = summarize_infos(infos, a.n_cand)
    manifest = dict(protocol="LLMGPR CIKM'25 Sec. 4.1: leave-one-out, ground truth + n_cand unvisited "
                             "nearest same-region POIs, HR@k / NDCG@k",
                    dataset=a.dataset, task=a.task, split=a.split, anchor=a.anchor,
                    group_select=(a.group_select if a.task == "group" else None),
                    n_cand=a.n_cand, seed=a.seed, index=desc, stats=stats)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        np.savez_compressed(a.out, keys=np.array(keys), targets=np.array(targets, dtype=np.int64),
                            cands=np.stack(cands).astype(np.int32))
        with open(os.path.splitext(a.out)[0] + "_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[eval-protocol] wrote {a.out}  ({len(keys):,} examples x {a.n_cand + 1} candidates) "
              f"+ manifest")
    print(f"[eval-protocol] {a.task}/{a.split}: {stats['n_examples']:,} examples | "
          f"filled {stats['frac_examples_filled']:.1%} of them (mean {stats['mean_fill_per_example']:.1f} "
          f"random fills) | anchors without coords {stats['n_anchor_no_coord']} | "
          f"regions too small for {a.n_cand}: {stats['n_examples_short']}")
    return manifest


# ──────────────────────────────────────────────────────────────────────────────
# self-check
# ──────────────────────────────────────────────────────────────────────────────
def _self_check() -> bool:
    res = []
    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]

    # a 20x20 grid of POIs ~1 km apart, two regions (left half / right half), some without coords
    n_side = 20
    lat0, lon0 = 40.70, -74.00
    rows = []
    for i in range(n_side):
        for j in range(n_side):
            idx = i * n_side + j
            has = not (idx % 7 == 3)               # every 7th POI lacks coordinates
            rows.append(dict(poi_idx=idx,
                             latitude=lat0 + 0.009 * i if has else np.nan,
                             longitude=lon0 + 0.012 * j if has else np.nan,
                             locality="West" if j < n_side // 2 else "East"))
    meta = pd.DataFrame(rows)
    n = len(meta)
    index = CandidateIndex(meta, region_col="auto", seed=0)
    res.append(ok("auto region picks `locality` when it has few values", index.region_col == "locality"))

    # 1. candidates: target first, unique, unvisited, same region, geo-nearest come first in order
    target = 5 * n_side + 3                              # West, has coords
    visited = {target - 1, target + 1, target + n_side}  # its three nearest neighbours
    cands, info = index.candidates(target, visited, n=50)
    res.append(ok("target sits at position 0", cands[0] == target))
    res.append(ok("n+1 unique candidates", len(cands) == 51 and len(set(cands.tolist())) == 51))
    res.append(ok("no visited POI among the negatives", not (set(cands[1:].tolist()) & visited)))
    res.append(ok("all candidates in the target's region", (index.region[cands] == index.region[target]).all()))
    geo = cands[1:1 + info["n_geo"]]
    d = haversine_km(index.lat[target], index.lon[target], index.lat[geo], index.lon[geo])
    res.append(ok("geo candidates are sorted by distance", bool((np.diff(d) >= -1e-9).all())))
    # the nearest unvisited POI with coordinates must be present
    pool = index._geo_pool_by_region[index.region[target]]
    dd = haversine_km(index.lat[target], index.lon[target], index.lat[pool], index.lon[pool])
    order = pool[np.argsort(dd, kind="mergesort")]
    nearest_unvisited = next(int(p) for p in order if int(p) != target and int(p) not in visited)
    res.append(ok("the nearest unvisited POI is in the set", nearest_unvisited in set(cands.tolist())))
    res.append(ok("no fill needed when the geo pool suffices", info["n_fill"] == 0))

    # 2. fill path: ask for more than the region's geo pool can give
    n_geo_pool = len(index._geo_pool_by_region[index.region[target]]) - 1
    cands2, info2 = index.candidates(target, set(), n=n_geo_pool + 5)
    res.append(ok("fill fires only for the shortfall", info2["n_geo"] == n_geo_pool and info2["n_fill"] == 5))
    res.append(ok("filled POIs are the region's coordinate-less ones",
                  all(not index.has_coord[p] for p in cands2[1 + info2["n_geo"]:])))
    res.append(ok("fill is deterministic under the seed",
                  np.array_equal(cands2, index.candidates(target, set(), n=n_geo_pool + 5)[0])))

    # 3. anchor without coordinates -> counted, all-random same-region set
    no_coord = next(i for i in range(n) if not index.has_coord[i])
    cands3, info3 = index.candidates(no_coord, set(), n=20)
    res.append(ok("coordinate-less anchor is flagged and still yields n candidates",
                  not info3["anchor_has_coord"] and info3["n_geo"] == 0 and len(cands3) == 21))

    # 4. ranks and metrics, by hand
    N = 10
    logits = np.zeros((3, N))
    logits[0] = np.arange(N)                       # target 9 is the best  -> rank 1
    logits[1] = np.arange(N)                       # target 2 among cands {2,5,7,9}: 3 above -> rank 4
    logits[2] = np.arange(N); logits[2, 6] = 8.0   # tie with 8: strict-greater -> target 8 rank 1
    cands_m = np.array([[9, 0, 1, 2], [2, 5, 7, 9], [8, 6, 0, 1]])
    ranks = ranks_from_logits(logits, np.array([9, 2, 8]), cands_m)
    res.append(ok("ranks by hand [1, 4, 1]", ranks == [1, 4, 1]))
    m = hr_ndcg(ranks, ks=(1, 3, 5))
    res.append(ok("HR@1 = 2/3, HR@3 = 2/3, HR@5 = 1", abs(m["HR@1"] - 2 / 3) < 1e-12 and
                  abs(m["HR@3"] - 2 / 3) < 1e-12 and m["HR@5"] == 1.0))
    exp_ndcg5 = (1 + 1 + 1 / math.log2(5)) / 3
    res.append(ok("NDCG@5 by hand", abs(m["NDCG@5"] - exp_ndcg5) < 1e-12))
    res.append(ok("NDCG@3 excludes the rank-4 hit", abs(m["NDCG@3"] - 2 / 3) < 1e-12))

    # 5. a random scorer lands at the protocol's floor: HR@5 ~ 5/(n+1)
    rng = np.random.default_rng(0)
    n_c = 100
    B = 4000
    rl = rng.standard_normal((B, n_c + 1))
    fake_c = np.tile(np.arange(n_c + 1), (B, 1))
    r = ranks_from_logits(rl, np.zeros(B, dtype=int), fake_c)
    hr5 = hr_ndcg(r)["HR@5"]
    res.append(ok(f"random scorer HR@5 ~ 5/{n_c + 1} (got {hr5:.4f})", abs(hr5 - 5 / (n_c + 1)) < 0.01))

    # 6. leave-one-out split and selection
    df = pd.DataFrame(dict(user_id=[1] * 5 + [2] * 2 + [3] * 3, poi_idx=list(range(10)),
                           utc_time=pd.date_range("2012-01-01", periods=10, freq="h")))
    s = leave_one_out_split(df)
    by_u = {u: g.sort_values("utc_time")["split"].tolist() for u, g in s.groupby("user_id")}
    res.append(ok("LOO: last=test, second-last=val, rest=train",
                  by_u[1] == ["train", "train", "train", "val", "test"] and by_u[3] == ["train", "val", "test"]))
    res.append(ok("LOO: users with <3 check-ins stay in train", by_u[2] == ["train", "train"]))
    vis = visited_before_target(df)
    res.append(ok("visited-before-target drops exactly the last check-in", vis[1] == {0, 1, 2, 3} and vis[3] == {7, 8}))
    ex = [dict(user=1, target=3, profile=dict(n_seen=3)), dict(user=1, target=4, profile=dict(n_seen=4)),
          dict(user=2, target=6, profile=dict(n_seen=1))]
    lp = {e["user"]: e["target"] for e in last_per_user(ex)}
    res.append(ok("last_per_user keeps the example with the longest prefix", lp == {1: 4, 2: 6}))
    res.append(ok("truncate_history caps at 200", len(truncate_history(list(range(300)))) == 200))

    # 7. one example per member set, the chronologically last (sum of n_seen, then example_id)
    def gex(eid, members, n_seen):
        return dict(example_id=eid, members=members, target=0,
                    member_profiles=[dict(n_seen=n) for n in n_seen])
    gexs = [gex("test_0000000", [0, 430], [138, 237]),     # group A, earlier
            gex("test_0000001", [430, 0], [140, 237]),     # group A, later (same set, other anchor)
            gex("test_0000002", [7, 9], [5, 5]),           # group B
            gex("test_0000003", [0, 430], [140, 237])]     # group A, same time -> later id wins
    kept = last_per_group(gexs)
    res.append(ok("last_per_group keeps one example per member set", len(kept) == 2))
    res.append(ok("...the one with the largest summed n_seen, ties -> later example_id",
                  {e["example_id"] for e in kept} == {"test_0000003", "test_0000002"}))
    res.append(ok("member-set key is order-insensitive",
                  frozenset(gexs[0]["members"]) == frozenset(gexs[1]["members"])))
    # the notebook's rebuilt shape: members are dicts carrying profile + per-member hist
    nb_ex = dict(example_id="test_0000009", target=3,
                 members=[dict(user=0, profile=dict(n_seen=140), hist=[10, 11]),
                          dict(user=430, profile=dict(n_seen=237), hist=[12])])
    res.append(ok("notebook-shaped example: members / n_seen / visited read correctly",
                  group_members(nb_ex) == [0, 430] and group_n_seen(nb_ex) == [140, 237]
                  and group_visited(nb_ex) == {10, 11, 12}))
    kept2 = {e["example_id"] for e in last_per_group(gexs + [nb_ex])}
    res.append(ok("both shapes of one member set compete: same n_seen sum, later id wins",
                  kept2 == {"test_0000009", "test_0000002"}))
    res.append(ok("jsonl-shaped visited = joint hist",
                  group_visited(dict(members=[1, 2], hist=[5, 6, 5], member_profiles=[])) == {5, 6}))

    good = all(res)
    print(f"\n{'ALL PASS' if good else 'SOME FAILED'}  ({sum(res)}/{len(res)})")
    return good


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--dataset", default="NYC")
    p.add_argument("--task", choices=["individual", "group"], default="individual")
    p.add_argument("--groups-dir", default=None, help="group_examples_<split>.jsonl location "
                                                       "(default <data-dir>/groups_social)")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--n-cand", type=int, default=DEFAULT_N_CAND)
    p.add_argument("--anchor", choices=["target", "last"], default="target",
                   help="measure 'nearest' from the ground truth (paper reading) or the last check-in")
    p.add_argument("--group-select", choices=["last", "all"], default="last",
                   help="group task: one example per member set (the chronologically last -- the "
                        "leave-one-out analogue) or every constructed example")
    p.add_argument("--region-col", default="auto",
                   help="metadata column defining 'same region'; 'auto' = locality if it is a city "
                        "(<= 10 values) else the whole catalogue; 'none' = whole catalogue")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help=".npz to write (keys, targets, cands) + _manifest.json")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args(argv)
    if a.self_check:
        return 0 if _self_check() else 1
    build_artifact(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
