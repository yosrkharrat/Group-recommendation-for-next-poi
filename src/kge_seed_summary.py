"""Report KGE metrics as mean and sample std over matched training seeds.

No GPU or PyTorch required. Input files are kge_bench.py result JSONs.
Example: python src/kge_seed_summary.py --runs 'outputs/foursquare_kge/*_s4?.json' \
    --expected-seeds 42 43 44 --out outputs/foursquare_kge/training_summary
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd


METRICS = ["best_val_mrr", "test_mrr", "test_hits1", "test_hits3", "test_hits10",
           "test_mr", "d1_best", "purity10_leaf", "purity10_top", "train_seconds",
           "sec_per_epoch", "peak_mem_mb", "n_params"]


def seed_statistics(frame, metrics, expected_seeds=None):
    """Keep original metric names for means, append _std (ddof=1).

    Refuse duplicate/missing seeds and unequal seed sets across models. A single
    run is allowed for smoke checks, with undefined std rather than a false zero.
    """
    if frame.empty:
        raise ValueError("No KGE runs to summarize")
    if frame[["model", "seed"]].isna().any().any():
        raise ValueError("Every KGE run must record model and training seed")
    if frame.duplicated(["model", "seed"]).any():
        raise ValueError("Duplicate (model, seed) runs; provide each run once")
    if expected_seeds is not None:
        if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
            raise ValueError("Expected seeds must be nonempty and unique")
        expected = set(expected_seeds)
    else:
        expected = set(frame["seed"])
    for model, group in frame.groupby("model"):
        actual = set(group["seed"])
        if actual != expected:
            raise ValueError(f"{model}: seeds {sorted(actual)}, expected {sorted(expected)}")
    grouped = frame.groupby("model")
    result = grouped[metrics].mean()
    std = grouped[metrics].std(ddof=1).add_suffix("_std")
    result = result.join(std)
    result.insert(0, "n_seeds", grouped["seed"].nunique())
    return result


def load_runs(paths):
    rows, protocol = [], None
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            run = json.load(stream)
        cfg = run["config"]
        # These must match across the comparison, independently of the model.
        current = {key: cfg.get(key) for key in
                   ("dataset", "split_seed", "dim", "val_frac", "test_frac", "max_val", "max_test")}
        current["split_sizes"] = run.get("split_sizes")
        current["n_test"] = run["test"]["n"]
        if protocol is None:
            protocol = current
        elif current != protocol:
            raise ValueError(f"Incompatible split/dimension/evaluation protocol: {path}")
        row = dict(run=Path(path).stem, model=cfg["model"], seed=cfg["seed"],
                   split_seed=cfg["split_seed"], n_test=run["test"]["n"])
        row.update({metric: run.get(metric) for metric in METRICS if not metric.startswith("test_")})
        row.update({f"test_{metric}": run["test"][metric]
                    for metric in ("mrr", "hits1", "hits3", "hits10", "mr")})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out", required=True, help="output prefix")
    args = parser.parse_args()
    paths = sorted({path for pattern in args.runs for path in glob.glob(pattern)})
    frame = load_runs(paths)
    summary = seed_statistics(frame, METRICS, args.expected_seeds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out + "_per_seed.csv", index=False)
    summary.to_csv(args.out + ".csv")
    # pandas serializes undefined std (one seed) as null, not nonstandard JSON NaN.
    payload = dict(std_ddof=1, expected_seeds=args.expected_seeds,
                   per_model=json.loads(summary.reset_index().to_json(orient="records", double_precision=15)),
                   per_run=json.loads(frame.to_json(orient="records", double_precision=15)))
    Path(args.out + ".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string())
    print(f"Saved {args.out}.csv / .json and {args.out}_per_seed.csv")


if __name__ == "__main__":
    main()
