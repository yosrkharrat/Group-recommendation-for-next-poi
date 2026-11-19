"""
Build the group files for each dataset under its closest reading of LLMGPR's group rule.

LLMGPR (CIKM'25) states its rule in one sentence and never fixes the window, the group identity or
any filter. GROUP_RULE_SWEEP.md §2.7-2.9 searched 264 readings per dataset against LLMGPR's own
Table 1 (src/group_match_search.py); this script materialises the chosen reading per dataset.

Co-presence (all datasets): an ANCHOR check-in (u, p, t) plus u's DIRECT friends with a check-in at
the same POI p within +-W seconds (GroupIM's "an individual and her subset of friends").
Group identity:
    anchor      one group per anchor user; members = the anchor + every friend ever co-present
                with them; the group's check-ins = the anchor's co-present check-ins
    member_set  one group per exact set of co-present users; its check-ins = outings of that set
                (views of one outing from each member's side are merged)

Outputs, data/groups_llmgpr_match/<DATASET>/:
    groups.csv          group_id, anchor (anchor identity only), size, members ("|"), n_checkins
    group_members.csv   group_id, user_id
    group_checkins.csv  group_id, seq, poi_idx, venue_id, utc_time, present (users co-present,
                        "|"), split -- LLMGPR's leave-one-out: last = test, second-last = val,
                        rest = train (a 1-check-in group is test only; 2 -> val + test);
                        'masked' = a train row that is the same outing as another group's
                        val/test row: DROP these from training targets and histories.
                        Sequences keep their most recent 200 check-ins (LLMGPR's max length).
    manifest.json       rule, data source, counts, LLMGPR Table-1 targets and the match

Id spaces: user_id and poi_idx are those of the input data dir. GOWALLA = data/gowalla (the
pipeline's ids). WEEPLACE = the llmgpr-Weeplace branch CSVs. LLMGPR (Foursquare) = house8_whole,
whose user_id equals the pipeline's data/llmgpr user_id (rank of the raw id) but whose poi_idx
indexes ALL venues of the users' whole histories; join to the pipeline's POIs on venue_id.

    python src/build_llmgpr_match_groups.py                 # all three
    python src/build_llmgpr_match_groups.py --datasets GOWALLA
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import load_checkins, load_friend_graph  # noqa: E402
from group_identity_probe import copresent_events  # noqa: E402
from group_match_search import TARGETS, match, member_set_outings, same_minute_events  # noqa: E402

FSQ = "./data/llmgpr_full/house8_whole"
CONFIGS = {
    # window_s=None -> same calendar minute. Values chosen in GROUP_RULE_SWEEP.md §2.11 (match8).
    "WEEPLACE": dict(data_dir="./data/weeplace_from_branch",
                     friends=["friendship_old_WEEPLACE.csv"],
                     window_s=60, identity="anchor", min_checkins=3, min_size=2),
    "LLMGPR":   dict(data_dir=FSQ,
                     friends=["friendship_old_LLMGPR.csv", "friendship_new_only_LLMGPR.csv"],
                     window_s=3600, identity="anchor", min_checkins=1, min_size=2),
    # Chosen 2026-11-19 over "+-3 h, >=3 members" (match8 0.856, groups barely recur after the
    # counting fix): best match8 0.941, check-ins per group exactly theirs (3.54), size 2.07.
    "GOWALLA":  dict(data_dir="./data/gowalla",
                     friends=["friendship_old_GOWALLA.csv"],
                     window_s=30, identity="member_set", min_checkins=1, min_size=2),
}
OUT_ROOT = "./data/groups_llmgpr_match"
MAX_SEQ = 200   # LLMGPR §4.1: "the maximum sequence length is set to 200" (most recent kept)


def loo_split(n):
    """LLMGPR §4.1: last check-in = test, second-last = validation, the rest = training."""
    s = ["train"] * n
    if n >= 1:
        s[-1] = "test"
    if n >= 2:
        s[-2] = "val"
    return s


def mask_cross_group_leaks(cks, window_s):
    """A group's TRAIN check-in that is the same outing as ANOTHER group's val/test check-in
    (same POI, anchor times within the window, >= 2 people in common) becomes split='masked'.
    Anchor groups share outings by construction (a's outing with b sits in a's and in b's
    sequence), so per-sequence leave-one-out would otherwise train on other groups' test targets
    (measured before this fix: Weeplace 17.9%, Foursquare 13.3%, Gowalla 1.2% of test targets).
    Masked rows stay in the file for transparency; exclude them from training targets AND
    histories. Only train rows change, so each group's val/test positions are untouched."""
    w = 0 if window_s is None else window_s
    pres = cks["present"].str.split("|").map(lambda x: frozenset(map(int, x)))
    held = cks[cks["split"].isin(["val", "test"])]
    held_by_poi = defaultdict(list)
    for i in held.index:
        held_by_poi[cks.at[i, "poi_idx"]].append((cks.at[i, "group_id"], cks.at[i, "_t"], pres[i]))
    masked = []
    for i in cks.index[cks["split"] == "train"]:
        for gid, t, pr in held_by_poi.get(cks.at[i, "poi_idx"], ()):
            if gid != cks.at[i, "group_id"] and abs(t - cks.at[i, "_t"]) <= max(w, 60) \
                    and len(pr & pres[i]) >= 2:
                masked.append(i)
                break
    cks.loc[masked, "split"] = "masked"
    return len(masked)


def leaked_targets(cks, window_s):
    """Test targets that also appear (same outing) in another group's TRAIN rows. Must be 0."""
    w = 0 if window_s is None else window_s
    pres = cks["present"].str.split("|").map(lambda x: frozenset(map(int, x)))
    train_by_poi = defaultdict(list)
    for i in cks.index[cks["split"] == "train"]:
        train_by_poi[cks.at[i, "poi_idx"]].append((cks.at[i, "group_id"], cks.at[i, "_t"], pres[i]))
    n = 0
    for i in cks.index[cks["split"] == "test"]:
        if any(gid != cks.at[i, "group_id"] and abs(t - cks.at[i, "_t"]) <= max(w, 60)
               and len(pr & pres[i]) >= 2 for gid, t, pr in train_by_poi.get(cks.at[i, "poi_idx"], ())):
            n += 1
    return n


def build(ds, cfg):
    t0 = time.time()
    df = load_checkins(cfg["data_dir"], ds)
    df["ts"] = df["utc_time"].dt.as_unit("ns").astype("int64") // 10 ** 9
    venue_of = df.drop_duplicates("poi_idx").set_index("poi_idx")["venue_id"].to_dict()
    G = load_friend_graph([os.path.join(cfg["data_dir"], f) for f in cfg["friends"]])
    w = cfg["window_s"]
    ev = same_minute_events(df, G) if w is None else copresent_events(df, G, w)

    # group -> list of (t, poi, present-set)
    seqs, members, anchor_of = defaultdict(list), {}, {}
    if cfg["identity"] == "anchor":
        for a, g, p, t in ev:
            seqs[a].append((t, p, g))
            members.setdefault(a, set()).update(g)
        anchor_of = {a: a for a in seqs}
    else:
        for g, p, t in member_set_outings(ev, 0 if w is None else w):
            seqs[g].append((t, p, g))
            members[g] = set(g)
    keep = [k for k in seqs if len(seqs[k]) >= cfg["min_checkins"] and len(members[k]) >= cfg["min_size"]]
    # deterministic group ids: by first check-in time, then sorted members
    keep.sort(key=lambda k: (min(x[0] for x in seqs[k]), sorted(members[k])))

    g_rows, m_rows, c_rows = [], [], []
    n_capped, ck_dropped_by_cap = 0, 0
    for gid, k in enumerate(keep):
        seq = sorted(seqs[k], key=lambda x: (x[0], x[1]))
        if len(seq) > MAX_SEQ:
            n_capped += 1
            ck_dropped_by_cap += len(seq) - MAX_SEQ
            seq = seq[-MAX_SEQ:]
        mem = sorted(members[k])
        g_rows.append(dict(group_id=gid, anchor=anchor_of.get(k, ""), size=len(mem),
                           members="|".join(map(str, mem)), n_checkins=len(seq)))
        m_rows += [dict(group_id=gid, user_id=u) for u in mem]
        for i, ((t, p, pres), split) in enumerate(zip(seq, loo_split(len(seq)))):
            c_rows.append(dict(group_id=gid, seq=i, poi_idx=int(p), venue_id=venue_of.get(int(p), ""),
                               utc_time=pd.Timestamp(int(t), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S+00:00"),
                               present="|".join(map(str, sorted(pres))), split=split, _t=int(t)))
    groups, mems, cks = pd.DataFrame(g_rows), pd.DataFrame(m_rows), pd.DataFrame(c_rows)
    leaks_before = leaked_targets(cks, w)
    n_masked = mask_cross_group_leaks(cks, w)
    leaks_after = leaked_targets(cks, w)
    assert leaks_after == 0, f"{leaks_after} test targets still leak after masking"
    cks = cks.drop(columns="_t")

    stats = dict(groups=len(groups), group_checkins=len(cks),
                 ck_per_group=len(cks) / len(groups), users_per_group=float(groups["size"].mean()))
    score, parts = match(stats, TARGETS[ds])
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = ""
    manifest = dict(
        dataset=ds, built=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        git_head=commit, script="src/build_llmgpr_match_groups.py",
        rule=dict(copresence="anchor check-in + direct friends at the same POI within +-window",
                  window_seconds=w if w is not None else "same calendar minute",
                  identity=cfg["identity"], min_checkins=cfg["min_checkins"], min_size=cfg["min_size"]),
        source=dict(data_dir=cfg["data_dir"], friendship_files=cfg["friends"],
                    friendship_edges=G.number_of_edges(), n_checkins=int(len(df)),
                    n_users=int(df.user_id.nunique())),
        counts=dict(groups=stats["groups"], group_checkins=stats["group_checkins"],
                    ck_per_group=round(stats["ck_per_group"], 3),
                    users_per_group=round(stats["users_per_group"], 3),
                    users_in_groups=int(mems.user_id.nunique()),
                    size_hist={int(k): int(v) for k, v in groups["size"].value_counts().sort_index().items()},
                    split=cks["split"].value_counts().to_dict(),
                    max_seq_cap=dict(cap=MAX_SEQ, groups_truncated=n_capped,
                                     checkins_dropped=ck_dropped_by_cap),
                    leakage=dict(test_targets_leaking_before_mask=leaks_before,
                                 train_rows_masked=n_masked, test_targets_leaking_after=leaks_after),
                    groups_with_train_history=int((groups["n_checkins"] >= 3).sum())),
        llmgpr_table1=TARGETS[ds], match=round(score, 4),
        match_parts=dict(zip(["groups", "group_checkins", "ck_per_group", "users_per_group"],
                             [round(x, 3) for x in parts])),
        reference="GROUP_RULE_SWEEP.md §2.7-2.10")

    out = os.path.join(OUT_ROOT, ds)
    os.makedirs(out, exist_ok=True)
    groups.to_csv(os.path.join(out, "groups.csv"), index=False)
    mems.to_csv(os.path.join(out, "group_members.csv"), index=False)
    cks.to_csv(os.path.join(out, "group_checkins.csv"), index=False)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{ds:<8} groups {stats['groups']:>5,}  group ck {stats['group_checkins']:>6,}  "
          f"ck/group {stats['ck_per_group']:6.2f}  users/group {stats['users_per_group']:.2f}  "
          f"match {score:.3f}  | cap: {n_capped} groups, -{ck_dropped_by_cap} ck | leaking test "
          f"targets {leaks_before} -> {leaks_after} ({n_masked} train rows masked)  -> {out}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    a = ap.parse_args()
    for ds in a.datasets:
        build(ds, CONFIGS[ds])


if __name__ == "__main__":
    main()
