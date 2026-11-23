"""Mean +- std over training seeds: the supervisor's protocol (slides + seed.ipynb, 2026-11-23).

Train the same model with SEEDS = [42, 43, 44] and report the mean over the runs with the std
below it. One run of notebooks/stage5_lbsn_finetune.ipynb is one seed (STAGE6B_SEED=<seed>); it
writes its results into <out_root>/seed_<seed>/. This module reads those files back and builds
the two tables of seed.ipynb:

  per_seed  one row per (seed, variant) with the metrics of that run
  summary   per variant, mean and std over the seeds (pandas std: sample std, ddof=1 -- the
            definition the slide's 0.4004 +- 0.0037 example uses)

Variants are the test results the notebook already writes:
  individual  section 11 test (full catalogue)       acc1, acc5, acc10, mrr
  group       section 11b group test                 acc1, acc5, acc10, mrr
and, in their own tables (different metrics), section 11c's LLMGPR comparison protocol:
  individual / group                                  HR@5, NDCG@5, HR@10, NDCG@10

The notebook fixes DATA_SEED, so every seed is scored on the same test examples and the same 11c
candidates; the spread is the training randomness only (initial weights, data order).

    python src/seed_summary.py --out-root outputs --dataset LLMGPR
    python src/seed_summary.py --self-check
"""
import argparse
import json
import os

import pandas as pd

SEEDS = [42, 43, 44]
METRICS = ["acc1", "acc5", "acc10", "mrr"]
CP_METRICS = ["HR@5", "NDCG@5", "HR@10", "NDCG@10"]


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _check_seed(res, seed, path):
    """A file that records its seed must sit in the folder of that seed."""
    if res is not None and res.get("seed") is not None and int(res["seed"]) != int(seed):
        raise ValueError(f"{path} records seed {res['seed']} but sits in seed_{seed}/")


def load_runs(out_root, dataset, condition="hyperbolic", seeds=SEEDS):
    """Per-seed rows from <out_root>/seed_<seed>/. Returns (per_seed, cp_per_seed, missing)."""
    rows, cp_rows, missing = [], [], []
    for seed in seeds:
        d = os.path.join(out_root, f"seed_{seed}")
        p_ind = os.path.join(d, f"stage6b_results_{dataset}_{condition}.json")
        p_grp = os.path.join(d, f"stage5_group_results_{dataset}_{condition}.json")
        p_cp = os.path.join(d, f"comparison_protocol_results_{dataset}_{condition}.json")
        ind, grp, cp = _read(p_ind), _read(p_grp), _read(p_cp)
        for res, path in ((ind, p_ind), (grp, p_grp), (cp, p_cp)):
            _check_seed(res, seed, path)
        if ind is None:
            missing.append(seed)
            continue
        rows.append({"seed": seed, "variant": "individual", "n": ind["test"]["n"],
                     **{m: ind["test"][m] for m in METRICS}})
        if grp is not None:
            rows.append({"seed": seed, "variant": "group", "n": grp["group_test"]["n"],
                         **{m: grp["group_test"][m] for m in METRICS}})
        if cp is not None:
            for task in ("individual", "group"):
                if task in cp:
                    m = cp[task]["comparison_protocol"]
                    cp_rows.append({"seed": seed, "variant": task, "n": m["n"],
                                    **{k: m[k] for k in CP_METRICS if k in m}})
    return pd.DataFrame(rows), pd.DataFrame(cp_rows), missing


def mean_std(per_seed, metrics):
    """seed.ipynb: per_seed.groupby("variant")[METRICS].agg(["mean", "std"])."""
    if per_seed.empty:
        return per_seed
    cols = [m for m in metrics if m in per_seed.columns]
    out = per_seed.groupby("variant")[cols].agg(["mean", "std"])
    out.insert(0, ("n_seeds", ""), per_seed.groupby("variant")["seed"].nunique())
    return out


def summarize(out_root, dataset, condition="hyperbolic", seeds=SEEDS, save=True):
    per_seed, cp_per_seed, missing = load_runs(out_root, dataset, condition, seeds)
    summary = mean_std(per_seed, METRICS)
    cp_summary = mean_std(cp_per_seed, CP_METRICS)
    if missing:
        print(f"[seeds] WARNING: no results for seed(s) {missing} under {out_root} -- "
              f"the mean +- std below covers {len(seeds) - len(missing)} of {len(seeds)} seeds")
    if save and not per_seed.empty:
        per_seed.to_csv(os.path.join(out_root, f"main_results_per_seed_{dataset}.csv"), index=False)
        summary.to_csv(os.path.join(out_root, f"main_results_{dataset}.csv"))
        if not cp_per_seed.empty:
            cp_per_seed.to_csv(os.path.join(out_root, f"comparison_protocol_per_seed_{dataset}.csv"),
                               index=False)
            cp_summary.to_csv(os.path.join(out_root, f"comparison_protocol_{dataset}.csv"))
        print(f"[seeds] saved: {out_root}")
    return dict(per_seed=per_seed, summary=summary, cp_per_seed=cp_per_seed,
                cp_summary=cp_summary, missing=missing)


def _self_check():
    import tempfile
    # the slide's example: 0.4046 / 0.3987 / 0.3979 -> 0.4004 +- 0.0037
    acc1 = {42: 0.4046, 43: 0.3987, 44: 0.3979}
    with tempfile.TemporaryDirectory() as root:
        for seed, a in acc1.items():
            d = os.path.join(root, f"seed_{seed}")
            os.makedirs(d)
            m = dict(n=5, loss=0.0, acc1=a, acc5=a, acc10=a, mrr=a)
            with open(os.path.join(d, "stage6b_results_X_hyperbolic.json"), "w") as f:
                json.dump(dict(seed=seed, test=m), f)
        out = summarize(root, "X", save=False)
        s = out["summary"].loc["individual"]
        assert round(s[("acc1", "mean")], 4) == 0.4004, s
        assert round(s[("acc1", "std")], 4) == 0.0037, s
        assert int(s[("n_seeds", "")]) == 3 and out["missing"] == []
        # a file in the wrong seed folder must be refused
        with open(os.path.join(root, "seed_43", "stage6b_results_X_hyperbolic.json"), "w") as f:
            json.dump(dict(seed=42, test=dict(n=5, acc1=0, acc5=0, acc10=0, mrr=0)), f)
        try:
            summarize(root, "X", save=False)
            raise AssertionError("seed mismatch was not caught")
        except ValueError:
            pass
    print("seed_summary self-check: OK (slide example 0.4004 +- 0.0037 reproduced)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="outputs", help="folder holding seed_<seed>/ subfolders")
    p.add_argument("--dataset", default="LLMGPR")
    p.add_argument("--condition", default="hyperbolic")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        _self_check()
    else:
        r = summarize(a.out_root, a.dataset, a.condition, a.seeds)
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            for k in ("per_seed", "summary", "cp_per_seed", "cp_summary"):
                if not r[k].empty:
                    print(f"\n== {k}\n{r[k].round(4)}")
