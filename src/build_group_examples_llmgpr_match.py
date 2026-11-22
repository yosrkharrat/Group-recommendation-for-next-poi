"""
Group next-POI examples from the LLMGPR-matched group files, in the schema stage 5 reads.

Why
---
`build_llmgpr_match_groups.py` writes the matched groups as tables (one row per group outing).
Stage 5 reads group EXAMPLES (`group_examples_{train,val,test}.jsonl`, the schema of
`build_groups.py`). This script is the step between the two, so stage 5 can train and evaluate on
the matched groups instead of the 180-min / 5-hop `groups_social` set.

One example per group outing
----------------------------
    target      the outing's venue, mapped to the pipeline `poi_idx` by `venue_id` (the group
                files carry another id space in their own `poi_idx` column: never used here)
    members     the users PRESENT at that outing, in the file's order. Every example is therefore
                a real joint outing of friends, even where the group's identity (an anchor and
                every friend ever co-present) contains people who never met each other.
    hist*       each member's own check-ins strictly before the outing, the most recent
                `--hist-per-member` each, merged in time order with an owner tag
    member_profiles
                `build_groups.MemberState`'s causal profile dict (n_seen, top_pois, top_cats,
                top_hrs) over the same check-ins
    group_hist* the group's OWN earlier outings (LLMGPR's group sequence), the most recent
                `--group-hist-len`; rows with split == "masked" are never used
    split       the group file's leave-one-out split: last outing test, second last val, rest train
    regime      "llmgpr_match"

Two overlap settings (`--overlap`)
----------------------------------
A group outing is built from its members' own check-ins, so the outing to predict also exists as
individual check-ins of the members.

    remove   (default) the clean setting, the same rules as stage 5's STAGE6B_GROUP_OVERLAP=remove:
             * individual TRAIN rows that are a held-out (val/test/masked) outing of a guarded
               group set are dropped (`--guard-sets`, each set's manifest window);
             * individual VAL/TEST rows that are a TRAIN outing of a trained group set are dropped
               (`--train-sets`), so the group training data is kept and the individual
               evaluation gives those rows up;
             * on top of that, a member's own check-in at the target venue within its set's window
               before the outing never enters that example's history or profile
               (it would put the answer in the prompt).
    keep     nothing is removed. Histories are still strictly before the outing time. This is
             LLMGPR's protocol as described (user and group sequences are not separated).

Histories are causal in both settings (strictly before the outing time).

Outputs (--out-dir, default <group dir>/examples_<overlap>)
----------------------------------------------------------
    group_examples_{train,val,test}.jsonl
    group_examples_manifest.json      inputs (+ sha256), config, counts, self-check results

Usage
-----
    python src/build_group_examples_llmgpr_match.py                    # LLMGPR_3CITY, remove
    python src/build_group_examples_llmgpr_match.py --overlap keep
    python src/build_group_examples_llmgpr_match.py --self-check       # asserts on the built files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import MemberState, heterogeneity, load_categories, load_checkins  # noqa: E402
from build_social_graph import drop_outing_checkins, holdout_outings  # noqa: E402
from group_outing_guard import group_guard_policy, group_window_minutes  # noqa: E402

SPLITS = ("train", "val", "test")
DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def load_individual(data_dir, dataset, overlap, guard_dirs, train_dirs, guard_min=None):
    """All individual check-ins (train + val + test) with `ts` in epoch minutes, after the
    overlap rules. Returns (frame, counts of rows removed per split)."""
    df = load_checkins(data_dir, dataset)
    removed = dict(train=0, val=0, test=0)
    if overlap == "remove":
        held = holdout_outings(guard_dirs)                       # val / test / masked outings
        trained = holdout_outings(train_dirs, splits=("train",))
        parts = []
        for split, outings in (("train", held), ("val", trained), ("test", trained)):
            part = df[df["orig_split"] == split]
            part, removed[split] = drop_outing_checkins(part, outings, guard_min)
            assert drop_outing_checkins(part, outings, guard_min)[1] == 0
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
    return df, removed


def load_outings(group_dir, venue_to_poi):
    """Non-masked group outings with the pipeline poi id, the outing time in epoch minutes and
    the present members. Sorted by (group, seq)."""
    g = pd.read_csv(os.path.join(group_dir, "group_checkins.csv"))
    n_masked = int((g["split"] == "masked").sum())
    g = g[g["split"] != "masked"].copy()
    t = pd.to_datetime(g["utc_time"], utc=True)
    g["ts"] = t.dt.as_unit("ns").astype("int64") // (60 * 10 ** 9)
    g["t_hour"] = t.dt.hour.astype(int)
    g["t_dow"] = t.dt.dayofweek.map(lambda d: DOW[int(d)])
    g["target"] = g["venue_id"].map(venue_to_poi)
    missing = int(g["target"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} group outings are at venues outside the POI catalogue")
    g["target"] = g["target"].astype(int)
    g["members"] = [[int(x) for x in str(p).split("|")] for p in g["present"]]
    g = g.sort_values(["group_id", "seq"], kind="mergesort").reset_index(drop=True)
    return g, n_masked


# --------------------------------------------------------------------------
# one member at one outing
# --------------------------------------------------------------------------

def member_prefix(state, u, t, target, guard_min, exclude_target):
    """(ts, pois, hours) of the member's check-ins strictly before t. With `exclude_target`,
    their check-ins at the target venue within `guard_min` before t are left out."""
    ts = state.ts.get(u)
    if ts is None:
        e = np.empty(0, dtype=np.int64)
        return e, e, e
    n = int(np.searchsorted(ts, t, side="left"))
    ts, pois, hours = ts[:n], state.poi[u][:n], state.hour[u][:n]
    if exclude_target and n:
        keep = ~((pois == target) & (ts >= t - guard_min))
        if not keep.all():
            ts, pois, hours = ts[keep], pois[keep], hours[keep]
    return ts, pois, hours


def profile_of(pois, hours, cat_of, top_k, top_cats, top_hours):
    """Same dict as `MemberState.profile`, over an explicit prefix."""
    if len(pois) == 0:
        return dict(n_seen=0, top_pois=[], top_cats=[], top_hrs=[])
    pc, cc, hc = Counter(), Counter(), Counter()
    for p, h in zip(pois.tolist(), hours.tolist()):
        pc[p] += 1
        cc[str(cat_of.get(p, "Venue"))] += 1
        hc[h] += 1
    return dict(n_seen=int(len(pois)),
                top_pois=[[int(p), int(c)] for p, c in pc.most_common(top_k)],
                top_cats=[c for c, _ in cc.most_common(top_cats)],
                top_hrs=[int(h) for h, _ in hc.most_common(top_hours)])


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(outings, state, cat_of, a, regime):
    exclude = a.overlap == "remove"
    out = {s: [] for s in SPLITS}
    stats = Counter()
    checks = Counter()
    ci_cache = {}
    for gid, grp in outings.groupby("group_id", sort=False):
        g_ts = grp["ts"].to_numpy()
        g_poi = grp["target"].to_numpy()
        g_hour = grp["t_hour"].to_numpy()
        for k, r in enumerate(grp.itertuples(index=False)):
            t, target = int(r.ts), int(r.target)
            merged, profiles = [], []
            for u in r.members:
                ts, pois, hours = member_prefix(state, u, t, target, a.profile_guard_min, exclude)
                profiles.append(profile_of(pois, hours, cat_of, a.profile_top_k,
                                           a.profile_cats, a.profile_hours))
                lo = max(0, len(ts) - a.hist_per_member)
                merged.extend(zip(ts[lo:].tolist(), pois[lo:].tolist(), hours[lo:].tolist(),
                                  [u] * (len(ts) - lo)))
                # checked while the timestamps are still at hand
                checks["hist_not_before_target"] += int((ts >= t).sum())
                if exclude:
                    checks["target_copy_in_history"] += int(
                        ((pois == target) & (ts >= t - a.profile_guard_min)).sum())
            if not any(p["n_seen"] for p in profiles):
                stats[f"skip_no_member_history_{r.split}"] += 1
                continue
            merged.sort()
            # the group's own earlier outings: strictly earlier in the group's sequence AND in time
            earlier = np.flatnonzero(g_ts[:k] < t)
            earlier = earlier[-a.group_hist_len:]
            checks["group_hist_not_before_target"] += int((g_ts[earlier] >= t).sum())
            rec = dict(
                example_id=f"{r.split}_g{int(gid)}_s{int(r.seq)}",
                group_id=int(gid), anchor=int(r.members[0]), members=[int(u) for u in r.members],
                size=len(r.members), regime=regime,
                hist=[int(x[1]) for x in merged], hist_hours=[int(x[2]) for x in merged],
                hist_owner=[int(x[3]) for x in merged],
                member_profiles=profiles,
                group_hist=[int(p) for p in g_poi[earlier]],
                group_hist_hours=[int(h) for h in g_hour[earlier]],
                target=target, t_hour=int(r.t_hour), t_dow=str(r.t_dow),
                heterogeneity=round(heterogeneity(r.members, state, t, cat_of, a.cat_level,
                                                  ci_cache), 6),
                split=str(r.split))
            out[r.split].append(rec)
            stats[f"emitted_{r.split}"] += 1
            stats["members_with_no_history"] += sum(p["n_seen"] == 0 for p in profiles)
            stats["examples_without_group_hist"] += int(len(earlier) == 0)
    return out, stats, checks


# --------------------------------------------------------------------------
# self-check on the written files (independent of the builder's own bookkeeping)
# --------------------------------------------------------------------------

def self_check(out_dir, group_dir, data_dir, dataset, n_poi):
    """Re-reads the written examples and the raw inputs. Raises on any violation."""
    man = json.load(open(os.path.join(out_dir, "group_examples_manifest.json")))
    cfg = man["config"]
    res = {}
    recs = {s: [json.loads(line) for line in open(os.path.join(out_dir, f"group_examples_{s}.jsonl"))]
            for s in SPLITS}
    g = pd.read_csv(os.path.join(group_dir, "group_checkins.csv"))
    meta = pd.read_csv(os.path.join(data_dir, f"poi_metadata_{dataset}.csv"))
    v2p = dict(zip(meta["venue_id"], meta["poi_idx"]))
    g["target"] = g["venue_id"].map(v2p)
    g["ts"] = pd.to_datetime(g["utc_time"], utc=True).dt.as_unit("ns").astype("int64") // (60 * 10 ** 9)
    key = {(int(r.group_id), int(r.seq)): r for r in g.itertuples(index=False)}

    seen = set()
    for split in SPLITS:
        for rec in recs[split]:
            gid, seq = rec["group_id"], int(rec["example_id"].rsplit("_s", 1)[1])
            row = key[(gid, seq)]
            assert row.split == split, f"{rec['example_id']}: split {row.split} != {split}"
            assert row.split != "masked"
            assert rec["target"] == int(row.target) and 0 <= rec["target"] < n_poi
            assert rec["members"] == [int(x) for x in str(row.present).split("|")]
            assert len(rec["members"]) == len(rec["member_profiles"]) == rec["size"]
            assert len(rec["hist"]) == len(rec["hist_hours"]) == len(rec["hist_owner"])
            assert set(rec["hist_owner"]) <= set(rec["members"])
            assert all(0 <= p < n_poi for p in rec["hist"] + rec["group_hist"])
            for pr in rec["member_profiles"]:
                assert sum(c for _, c in pr["top_pois"]) <= pr["n_seen"], "profile is not causal"
            assert (gid, seq) not in seen
            seen.add((gid, seq))
    res["examples"] = {s: len(recs[s]) for s in SPLITS}
    for s in ("val", "test"):
        per_group = Counter(r["group_id"] for r in recs[s])
        assert max(per_group.values()) == 1, f"a group has several {s} examples"
    res["one_val_and_test_example_per_group"] = True

    # the group's own history must be exactly its earlier non-masked outings (never a later one,
    # never a masked one), so no group example can read a held-out outing of its own group
    by_group = {gid: grp.sort_values("seq") for gid, grp in g[g["split"] != "masked"].groupby("group_id")}
    bad = 0
    for split in SPLITS:
        for rec in recs[split]:
            seq = int(rec["example_id"].rsplit("_s", 1)[1])
            grp = by_group[rec["group_id"]]
            t = int(grp.loc[grp["seq"] == seq, "ts"].iloc[0])
            exp = grp[(grp["seq"] < seq) & (grp["ts"] < t)]["target"].astype(int).tolist()
            bad += rec["group_hist"] != exp[-cfg["group_hist_len"]:]
    assert bad == 0, f"{bad} examples have a group history that is not their earlier outings"
    res["group_hist_is_earlier_outings_only"] = True

    # member histories against the raw individual check-ins, by timestamp
    df = load_checkins(data_dir, dataset)
    times = {}
    for (u, p), part in df.groupby(["user_id", "poi_idx"], sort=False):
        times[(int(u), int(p))] = np.sort(part["ts"].to_numpy())
    first_ts = df.groupby("user_id")["ts"].min().to_dict()
    n_future = n_copy = n_impossible = 0
    for split in SPLITS:
        for rec in recs[split]:
            seq = int(rec["example_id"].rsplit("_s", 1)[1])
            row = key[(rec["group_id"], seq)]
            t = int(row.ts)
            for u in rec["members"]:
                if first_ts.get(u, t) >= t and any(o == u for o in rec["hist_owner"]):
                    n_impossible += 1
            for p, u in zip(rec["hist"], rec["hist_owner"]):
                ts = times.get((u, p))
                assert ts is not None, "history item is not a check-in of its owner"
                n_future += int(not (ts < t).any())        # no visit of that POI before t at all
            if cfg["overlap"] == "remove":
                # the target may be in a member's history only through a visit older than the guard
                for u in set(rec["members"]):
                    ts = times.get((u, rec["target"]))
                    shown = sum(1 for p, o in zip(rec["hist"], rec["hist_owner"])
                                if o == u and p == rec["target"])
                    if ts is None or not shown:
                        continue
                    older = int((ts < t - cfg.get("profile_guard_min", cfg["guard"])).sum())
                    n_copy += int(shown > older)
    assert n_future == 0, f"{n_future} history items have no visit before the outing"
    assert n_impossible == 0
    assert n_copy == 0, f"{n_copy} examples show the target through a check-in inside the guard"
    res["histories_are_before_the_outing"] = True
    res["no_target_copy_inside_guard"] = cfg["overlap"] == "remove" or "not applied (keep)"

    if cfg["overlap"] == "remove":
        # nothing an example shows may be a removed individual row: recount removed rows
        held = holdout_outings(cfg["guard_sets"])
        trained = holdout_outings(cfg["train_sets"], splits=("train",))
        left = 0
        clean, _ = load_individual(data_dir, dataset, "remove", cfg["guard_sets"], cfg["train_sets"],
                                   cfg["guard"])
        for split, outings in (("train", held), ("val", trained), ("test", trained)):
            left += drop_outing_checkins(clean[clean["orig_split"] == split], outings, cfg["guard"])[1]
        assert left == 0
        res["overlap_rows_left_in_individual_data"] = 0
    return res


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(a):
    t0 = time.time()
    group_dir = a.group_dir.rstrip("/")
    out_dir = a.out_dir or os.path.join(group_dir, f"examples_{a.overlap}")
    guard_dirs = a.guard_sets or [group_dir]
    train_dirs = a.train_sets or [group_dir]
    a.profile_guard_min = group_window_minutes(group_dir, a.guard)
    guard_policy = group_guard_policy(guard_dirs, a.guard)
    train_policy = group_guard_policy(train_dirs, a.guard)
    os.makedirs(out_dir, exist_ok=True)

    cat_of, n_poi = load_categories(a.data_dir, a.dataset)
    meta = pd.read_csv(os.path.join(a.data_dir, f"poi_metadata_{a.dataset}.csv"))
    assert meta["venue_id"].is_unique and meta["poi_idx"].is_unique
    venue_to_poi = dict(zip(meta["venue_id"], meta["poi_idx"]))

    df, removed = load_individual(a.data_dir, a.dataset, a.overlap, guard_dirs, train_dirs, a.guard)
    # the individual files' own poi_idx must be the catalogue's (the group files' is not)
    agree = (df["venue_id"].map(venue_to_poi) == df["poi_idx"]).mean()
    assert agree == 1.0, f"individual poi_idx disagrees with the catalogue on {1 - agree:.2%} of rows"
    state = MemberState(df, cat_of, a.cat_level)
    outings, n_masked = load_outings(group_dir, venue_to_poi)
    regime = a.regime

    out, stats, checks = build(outings, state, cat_of, a, regime)
    assert checks["hist_not_before_target"] == 0, "a history item is not before its outing"
    assert checks["group_hist_not_before_target"] == 0
    assert checks["target_copy_in_history"] == 0, "a member's copy of the outing is in the history"

    for split in SPLITS:
        with open(os.path.join(out_dir, f"group_examples_{split}.jsonl"), "w", encoding="utf-8") as f:
            for rec in out[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sizes = Counter(r["size"] for s in SPLITS for r in out[s])
    hist_per_member = [sum(1 for o in r["hist_owner"] if o == u)
                       for s in SPLITS for r in out[s] for u in r["members"]]
    inputs = [os.path.join(a.data_dir, f"{s}_{a.dataset}.csv") for s in SPLITS]
    inputs += [os.path.join(a.data_dir, f"poi_metadata_{a.dataset}.csv"),
               os.path.join(group_dir, "group_checkins.csv")]
    inputs += [os.path.join(d, "group_checkins.csv") for d in guard_dirs + train_dirs
               if d.rstrip("/") != group_dir]
    manifest = dict(
        script="src/build_group_examples_llmgpr_match.py",
        built=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        config=dict(group_dir=group_dir, data_dir=a.data_dir, dataset=a.dataset, overlap=a.overlap,
                    guard=a.guard, guard_sets=guard_dirs, train_sets=train_dirs,
                    profile_guard_min=a.profile_guard_min,
                    guard_policy=guard_policy, train_guard_policy=train_policy,
                    hist_per_member=a.hist_per_member, group_hist_len=a.group_hist_len,
                    profile_top_k=a.profile_top_k, profile_cats=a.profile_cats,
                    profile_hours=a.profile_hours, cat_level=a.cat_level, regime=regime,
                    members="users present at the outing"),
        inputs={p: sha256(p) for p in dict.fromkeys(inputs)},
        counts=dict(
            n_pois=int(n_poi), individual_checkins=int(len(df)),
            individual_rows_removed=removed,
            group_outings_non_masked=int(len(outings)), group_outings_masked=n_masked,
            examples={s: len(out[s]) for s in SPLITS},
            groups={s: len({r["group_id"] for r in out[s]}) for s in SPLITS},
            skipped={k: int(v) for k, v in stats.items() if k.startswith("skip")},
            members_with_no_history=int(stats["members_with_no_history"]),
            examples_without_group_hist=int(stats["examples_without_group_hist"]),
            size_hist={str(k): int(v) for k, v in sorted(sizes.items())},
            history_items_per_member=dict(
                mean=round(float(np.mean(hist_per_member)), 2),
                p50=float(np.percentile(hist_per_member, 50)),
                share_empty=round(float(np.mean(np.array(hist_per_member) == 0)), 4))),
        seconds=round(time.time() - t0, 1))
    with open(os.path.join(out_dir, "group_examples_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    manifest["self_check"] = self_check(out_dir, group_dir, a.data_dir, a.dataset, n_poi)
    with open(os.path.join(out_dir, "group_examples_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    c = manifest["counts"]
    print(f"[{a.overlap}] {out_dir}")
    print(f"  individual rows removed: {removed}")
    print(f"  outings: {c['group_outings_non_masked']:,} used, {n_masked:,} masked (never used)")
    print(f"  examples: {c['examples']}   groups: {c['groups']}   skipped: {c['skipped']}")
    print(f"  members with no history: {c['members_with_no_history']:,}   "
          f"examples without group history: {c['examples_without_group_hist']:,}")
    print(f"  history items per member: {c['history_items_per_member']}")
    print(f"  self-check: {manifest['self_check']}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--group-dir", default="./data/groups_llmgpr_match/LLMGPR_3CITY")
    p.add_argument("--data-dir", default="./data/llmgpr")
    p.add_argument("--dataset", default="LLMGPR")
    p.add_argument("--out-dir", default=None, help="default: <group-dir>/examples_<overlap>")
    p.add_argument("--overlap", choices=("remove", "keep"), default="remove")
    p.add_argument("--guard", type=float, default=None,
                   help="default: each set's manifest window; override may only widen it")
    p.add_argument("--guard-sets", nargs="*", default=None,
                   help="group dirs whose held-out outings are removed from individual train "
                        "(default: --group-dir only; stage 5 passes its GROUP_OVERLAP_SETS)")
    p.add_argument("--train-sets", nargs="*", default=None,
                   help="group dirs whose TRAIN outings are removed from individual val/test "
                        "(default: --group-dir only)")
    p.add_argument("--hist-per-member", type=int, default=30,
                   help="recent check-ins kept per member (stage 5's GROUP_HIST_PER_MEMBER)")
    p.add_argument("--group-hist-len", type=int, default=30, help="recent own outings kept")
    p.add_argument("--profile-top-k", type=int, default=10)
    p.add_argument("--profile-cats", type=int, default=3)
    p.add_argument("--profile-hours", type=int, default=3)
    p.add_argument("--cat-level", type=int, default=2, help="taxonomy level for heterogeneity")
    p.add_argument("--regime", default="llmgpr_match")
    p.add_argument("--self-check", action="store_true",
                   help="only re-check files already written to --out-dir")
    a = p.parse_args()
    if a.self_check:
        group_dir = a.group_dir.rstrip("/")
        out_dir = a.out_dir or os.path.join(group_dir, f"examples_{a.overlap}")
        _, n_poi = load_categories(a.data_dir, a.dataset)
        print(json.dumps(self_check(out_dir, group_dir, a.data_dir, a.dataset, n_poi), indent=1))
        return
    run(a)


if __name__ == "__main__":
    main()
