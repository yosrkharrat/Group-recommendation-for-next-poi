"""
Does the depth regulariser buy D1 by spending hierarchical link prediction?

The Gowalla raw-arm RotH run (`LLMGPR_GOWALLA.md` §5, stage 3) came out lopsided:

    D1 rho = +0.8683 STRONG           (FSQ managed +0.3245, barely over its 0.30 gate)
    HAS_CATEGORY MRR = 0.0105         (FSQ's best run reached 0.191)
    LOCATED_IN   MRR = 0.0027         (26x random -- the weakest relation in the KG)

and at epoch 50 the objective decomposes as kge 0.011972 + 5.0 x depth 0.003035, so the depth
term is **55.9% of the loss**. At `--depth-weight 5.0` it is not a regulariser, it is the larger
half of the objective, and the two relations it acts on most directly are the two worst in the
table. Hypothesis: the depth weight trades hierarchical link prediction for radial ordering.

This probe runs the matched comparison the claim needs -- identical seed, data, budget and
evaluation, varying ONLY `depth_weight` -- and reports D1 against per-relation MRR for each.
Both directions are informative:

  * If D1 falls and HAS_CATEGORY/LOCATED_IN rise as the weight drops, the tradeoff is real and
    the operating point is a choice to make deliberately before stage 5 consumes the embeddings.
  * If HAS_CATEGORY stays flat at every weight, the depth term is exonerated and the cause is
    elsewhere (epoch budget, the 324-category vocabulary, or relation weighting) -- which is a
    different and equally useful answer.

Budget note: a full 50-epoch run is ~4.5 h on this machine, so the probe uses a short shared
budget. It compares configurations against each other at equal epochs, never against the
50-epoch headline number.

Usage
-----
    python src/roth_depth_weight_probe.py --weights 0 1 5 --epochs 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

RELS = ("HAS_CATEGORY", "LOCATED_IN", "PREFERS_CATEGORY", "GROUP_PREFERS",
        "IS_NEAR_TO", "SUBCATEGORY_OF")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg-dir", default="./data/gowalla/kg_raw")
    p.add_argument("--data-dir", default="./data/gowalla")
    p.add_argument("--dataset", default="GOWALLA")
    p.add_argument("--work-dir", default="./data/gowalla/_depth_probe")
    p.add_argument("--out", default="./data/gowalla/roth_depth_weight_probe.json")
    p.add_argument("--weights", type=float, nargs="+", default=[0.0, 1.0, 5.0])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-neg", type=int, default=32)
    p.add_argument("--max-eval", type=int, default=4000)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    os.makedirs(a.work_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for w in a.weights:
        out_dir = os.path.join(a.work_dir, f"dw{w:g}")
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, "-u", os.path.join(here, "train_roth.py"),
               "--kg-dir", a.kg_dir, "--data-dir", a.data_dir, "--dataset", a.dataset,
               "--out-dir", out_dir, "--epochs", str(a.epochs),
               "--batch-size", str(a.batch_size), "--n-neg", str(a.n_neg),
               "--max-eval", str(a.max_eval), "--log-every", str(max(1, a.epochs // 2)),
               "--depth-weight", str(w), "--depth-margin", "0.3", "--root-pull", "0.01",
               "--seed", str(a.seed), "--device", a.device]
        print(f"\n{'='*70}\ndepth_weight={w:g}  ({a.epochs} epochs)\n{'='*70}", flush=True)
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode})")
            continue
        res = json.load(open(os.path.join(out_dir, "roth_results.json")))
        pr = res["link_prediction"]["per_relation"]
        rows.append(dict(depth_weight=w, d1=res["d1"]["spearman"], verdict=res["d1"]["verdict"],
                         mrr=res["link_prediction"]["mrr"],
                         hits10=res["link_prediction"]["hits10"],
                         per_relation={k: pr.get(k, {}).get("mrr") for k in RELS},
                         final=res["history"][-1]))

    print("\n" + "=" * 78)
    print(f"MATCHED COMPARISON at {a.epochs} epochs (seed {a.seed}) -- D1 vs link prediction")
    print("=" * 78)
    hdr = f"{'depth_w':>8}{'D1 rho':>9}{'verdict':>9}{'MRR':>8}{'H@10':>8}"
    hdr += "".join(f"{r[:11]:>12}" for r in ("HAS_CATEGORY", "LOCATED_IN", "GROUP_PREFERS"))
    print(hdr); print("-" * len(hdr))
    for r in rows:
        line = (f"{r['depth_weight']:>8.4g}{r['d1']:>9.4f}{r['verdict']:>9}"
                f"{r['mrr']:>8.4f}{r['hits10']:>8.4f}")
        for k in ("HAS_CATEGORY", "LOCATED_IN", "GROUP_PREFERS"):
            v = r["per_relation"].get(k)
            line += f"{v:>12.4f}" if v is not None else f"{'-':>12}"
        print(line)

    if len(rows) >= 2:
        lo = min(rows, key=lambda r: r["depth_weight"])
        hi = max(rows, key=lambda r: r["depth_weight"])
        hc_lo, hc_hi = lo["per_relation"].get("HAS_CATEGORY"), hi["per_relation"].get("HAS_CATEGORY")
        print(f"\ndepth_weight {lo['depth_weight']:g} -> {hi['depth_weight']:g}: "
              f"D1 {lo['d1']:+.4f} -> {hi['d1']:+.4f}", end="")
        if hc_lo is not None and hc_hi is not None:
            print(f"   HAS_CATEGORY MRR {hc_lo:.4f} -> {hc_hi:.4f}")
            print("VERDICT: " + ("tradeoff CONFIRMED -- the depth term buys D1 and costs "
                                "hierarchical link prediction"
                                if hc_hi < hc_lo and hi["d1"] > lo["d1"] else
                                "tradeoff NOT reproduced at this budget -- look elsewhere "
                                "(epochs, category vocabulary, relation weighting)"))
    with open(a.out, "w") as f:
        json.dump(dict(config=vars(a), rows=rows), f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
