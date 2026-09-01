#!/usr/bin/env python3
"""Convert our check-ins + friendship edges into KDD24-GBSR's input format.

GBSR (yimutianyang/KDD24-GBSR) denoises the USER-USER SOCIAL GRAPH; it has no notion
of a group. Verified against its own yelp files:

    traindata.npy    defaultdict{int user -> list[int item]}    19,539 users / 367,645 pairs
    testdata.npy     defaultdict{int user -> list[int item]}    19,539 users /  83,239 pairs
    user_users_d.npy defaultdict{int user -> set[int user]}     18,862 users / 727,384 directed
                                                                (100% reciprocal = stored undirected)

Constraints the loader imposes (rec_dataset.py / run_GBSR.py):
  * ids must be contiguous 0..num_user-1 and 0..num_item-1 -- negative sampling draws
    random.randint(0, num_item-1) and indexes nn.Embedding directly
  * implicit feedback only: no timestamps, no sequence, one row per (user, item) pair
  * the social graph must be symmetric
  * every user in testdata should also appear in traindata (negative sampling reads
    traindata[u]); users with no social edges are fine

Usage:
  python src/to_gbsr.py --checkins data/llmgpr_final_checkins.parquet \
                        --edges    data/llmgpr_final_friendship_old.parquet \
                        --out      datasets/fsq_nyc --split ratio --test-frac 0.2
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd


def read_any(path, **kw):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep, **kw)


def pick(df, wanted, fallback_idx):
    """Locate a column by name, else fall back to position."""
    for w in wanted:
        if w in df.columns:
            return w
    return df.columns[fallback_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkins", required=True)
    p.add_argument("--edges", default=None, help="two-column user-user edge list")
    p.add_argument("--out", required=True, help="output directory (GBSR data_path)")
    p.add_argument("--split", choices=["ratio", "loo"], default="ratio",
                   help="ratio: hold out a fraction of each user's items (matches GBSR's "
                        "own data, ~4.3 test items/user). loo: one item per user.")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--time-col", default=None,
                   help="if given, the split takes each user's LATEST items as test; "
                        "otherwise the split is random under --seed")
    p.add_argument("--min-train", type=int, default=1,
                   help="drop users left with fewer than this many training items")
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--emit-val", action="store_true",
                   help="also write valdata.npy (see the note this prints)")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    ck = read_any(a.checkins)
    ucol = pick(ck, ["user_id", "user", "uid"], 0)
    icol = pick(ck, ["venue_id", "item_id", "poi_id", "item", "iid"], 1)
    ck = ck[[ucol, icol] + ([a.time_col] if a.time_col else [])].dropna(subset=[ucol, icol])
    ck[ucol] = ck[ucol].astype(str); ck[icol] = ck[icol].astype(str)
    print(f"check-ins: {len(ck):,} rows | {ck[ucol].nunique():,} users | {ck[icol].nunique():,} items")

    # implicit feedback: one row per (user, item). Keep the latest occurrence when timed.
    if a.time_col:
        ck = ck.sort_values(a.time_col).drop_duplicates([ucol, icol], keep="last")
    else:
        ck = ck.drop_duplicates([ucol, icol])
    print(f"deduped to {len(ck):,} unique (user, item) pairs")

    # ---- split before reindexing, so users that vanish don't leave id holes ----
    groups = {u: g for u, g in ck.groupby(ucol, sort=False)}
    train_pairs, test_pairs = [], []
    for u, g in groups.items():
        items = g[icol].tolist()
        if a.time_col:
            order = list(range(len(items)))            # already sorted ascending
        else:
            order = list(rng.permutation(len(items)))
        n_test = 1 if a.split == "loo" else int(round(len(items) * a.test_frac))
        n_test = min(n_test, max(len(items) - a.min_train, 0))
        test_idx = set(order[len(order) - n_test:]) if n_test else set()
        for k, it in enumerate(items):
            (test_pairs if k in test_idx else train_pairs).append((u, it))

    tr = pd.DataFrame(train_pairs, columns=["u", "i"])
    te = pd.DataFrame(test_pairs, columns=["u", "i"])
    keep_u = tr["u"].value_counts()
    keep_u = set(keep_u[keep_u >= a.min_train].index)
    tr = tr[tr["u"].isin(keep_u)]
    te = te[te["u"].isin(keep_u)]        # test users must be in train (evaluate reads traindata[u])
    print(f"split({a.split}): train {len(tr):,} | test {len(te):,} | "
          f"users kept {len(keep_u):,}")

    # ---- contiguous reindex ----
    # The item vocabulary spans train UNION test. GBSR's own yelp files keep test items
    # that never appear in training (num_item comes from argparse, and every id indexes a
    # real embedding), so dropping them would make our test set easier than their protocol.
    users = sorted(keep_u); items = sorted(set(tr["i"]) | set(te["i"]))
    u2i = {u: k for k, u in enumerate(users)}
    i2i = {v: k for k, v in enumerate(items)}

    def to_dict(df, as_set=False):
        d = defaultdict(set if as_set else list)
        for u, i in zip(df["u"].map(u2i), df["i"].map(i2i)):
            d[int(u)].add(int(i)) if as_set else d[int(u)].append(int(i))
        return d

    traindata, testdata = to_dict(tr), to_dict(te)
    np.save(os.path.join(a.out, "traindata.npy"), traindata)
    np.save(os.path.join(a.out, "testdata.npy"), testdata)

    # ---- social graph: symmetrise, restrict to retained users ----
    n_edges = 0
    social = defaultdict(set)
    if a.edges:
        e = read_any(a.edges, header=None if a.edges.endswith((".tsv", ".txt")) else "infer")
        c1, c2 = e.columns[0], e.columns[1]
        e = e[[c1, c2]].dropna().astype(str)
        e = e[e[c1].isin(u2i) & e[c2].isin(u2i)]
        for x, y in zip(e[c1].map(u2i), e[c2].map(u2i)):
            if x != y:
                social[int(x)].add(int(y)); social[int(y)].add(int(x))
        n_edges = sum(len(v) for v in social.values()) // 2
    np.save(os.path.join(a.out, "user_users_d.npy"), social)

    if a.emit_val:
        np.save(os.path.join(a.out, "valdata.npy"), testdata)

    cov = len(social) / max(len(users), 1)
    print(f"\nwrote -> {a.out}/")
    print(f"  traindata.npy    {len(traindata):,} users, {sum(map(len,traindata.values())):,} pairs")
    print(f"  testdata.npy     {len(testdata):,} users, {sum(map(len,testdata.values())):,} pairs")
    print(f"  user_users_d.npy {len(social):,} users, {n_edges:,} undirected edges")
    print(f"\nnum_user = {len(users)}   num_item = {len(items)}")
    print(f"run it with:\n  python run_GBSR.py --dataset {os.path.basename(a.out)} "
          f"--num_user {len(users)} --num_item {len(items)} --beta 2.0 --sigma 0.25")

    print(f"\nsocial coverage: {len(social):,}/{len(users):,} users have >=1 edge ({cov:.1%})")
    if cov < 0.5:
        print("  WARNING: GBSR's own datasets sit at ~97% (yelp: 18,862/19,539). Below ~50% the")
        print("  social branch touches a minority of users and the denoiser has little to do.")
    print("\nNOTE: rec_dataset.py loads valdata AND testdata from testdata.npy, and run_GBSR.py")
    print("  tracks the best NDCG@20 on testdata every epoch -- model selection happens on the")
    print("  test set. For a clean protocol, emit a real validation split and repoint that line.")


if __name__ == "__main__":
    main()
