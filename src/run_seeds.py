"""Run the stage-5 pipeline once per seed, then report mean +- std (supervisor's protocol, 2026-11-23).

The loop of seed.ipynb:

    for seed in SEEDS:
        runs[seed] = run_sft(cfg(seed=seed))

on our pipeline. Each seed executes notebooks/stage5_lbsn_finetune.ipynb top to bottom in a FRESH
kernel with STAGE6B_SEED=<seed> (the GPU is freed between seeds and nothing from one seed's session
leaks into the next), writing into <out-root>/seed_<seed>/. The executed notebook is saved there
too, as the run's log. When every seed is done, src/seed_summary.py builds the per-seed table and
the mean +- std per variant.

Every other STAGE6B_* setting is inherited from the environment, so all seeds share one config.
A seed whose results file already exists is skipped (rerun the command after an interruption;
the notebook's own RESUME continues a seed that stopped mid-training). --force reruns it.
The loop stops at the first failing seed rather than averaging fewer runs.

    export STAGE6B_RUN_PROFILE=full STAGE6B_GROUP_SET=llmgpr_match ...   # as for a single run
    python src/run_seeds.py --out-root outputs                           # seeds 42 43 44
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seed_summary as SS  # noqa: E402

NOTEBOOK = os.path.join(HERE, "..", "notebooks", "stage5_lbsn_finetune.ipynb")


def results_file(out_root, seed, dataset, condition):
    return os.path.join(out_root, f"seed_{seed}", f"stage6b_results_{dataset}_{condition}.json")


def run_seed(seed, out_root, notebook=NOTEBOOK):
    """Execute the notebook for one seed in a fresh kernel."""
    seed_dir = os.path.join(out_root, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    env = dict(os.environ, STAGE6B_SEED=str(seed), STAGE6B_OUT_DIR=out_root)
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=-1", "--output-dir", seed_dir,
           "--output", f"stage5_seed{seed}.ipynb", notebook]
    print(f"[seeds] === seed {seed} -> {seed_dir}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    print(f"[seeds] seed {seed} done in {(time.time() - t0) / 3600:.2f} h", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default=os.environ.get("STAGE6B_OUT_DIR", "outputs"))
    p.add_argument("--seeds", type=int, nargs="+", default=SS.SEEDS)
    p.add_argument("--notebook", default=NOTEBOOK)
    p.add_argument("--force", action="store_true", help="rerun seeds that already have results")
    a = p.parse_args()

    out_root = os.path.abspath(a.out_root)
    dataset = os.environ.get("STAGE6B_DATASET", "LLMGPR")
    condition = "hyperbolic"                       # EMB_CONDITION is fixed in the notebook
    for seed in a.seeds:
        if not a.force and os.path.exists(results_file(out_root, seed, dataset, condition)):
            print(f"[seeds] seed {seed}: results already in {out_root}/seed_{seed} -- skipped "
                  f"(--force to rerun)", flush=True)
            continue
        run_seed(seed, out_root, os.path.abspath(a.notebook))

    r = SS.summarize(out_root, dataset, condition, a.seeds)
    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        for k in ("per_seed", "summary", "cp_per_seed", "cp_summary"):
            if not r[k].empty:
                print(f"\n== {k}\n{r[k].round(4)}")


if __name__ == "__main__":
    main()
