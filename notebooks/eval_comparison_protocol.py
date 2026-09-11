# eval_comparison_protocol.py -- HyGro-POI under LLMGPR's evaluation protocol.
#
# This file is section 11c of notebooks/stage5_lbsn_finetune.ipynb, kept as a script so it can
# also be run against a notebook version that predates that section:
#
#     %run -i notebooks/eval_comparison_protocol.py        # after section 11 has run
#
# `-i` runs it in the notebook's namespace, so it scores with the very same model, prompts,
# frozen W_POI and trained heads that section 11 just used. A stand-alone loader would have to
# re-implement ~15 notebook cells, and any drift there would silently change the numbers --
# which is exactly what a comparison table cannot afford.
#
# Normal use needs neither this file nor %run: set EVAL_CKPT in the notebook's config cell to
# the trained checkpoint (local folder or HF repo id) and run the notebook top to bottom --
# sections 11, 11b and 11c run on it with no training. See notebooks/COMPARISON_PROTOCOL_RUN.md.
# Keep this file and section 11c identical.
# ── 11c · Comparison protocol (LLMGPR CIKM'25 §4.1): leave-one-out, 500 nearest unvisited
#          same-region candidates, HR@k / NDCG@k -- the numbers for the paper's comparison tables.
#
# Run AFTER §11 (and §11b if the session has group_ex), with the trained checkpoint loaded.
# Same checkpoint, same `collate`, same logits as §11/§11b. Only two things change:
#   (1) WHICH examples: the last check-in per user (individual) / one example per member set,
#       the chronologically last (group) -- LLMGPR's leave-one-out, applied to the examples;
#   (2) WHAT the target is ranked against: 500 unvisited POIs nearest to it in the same region,
#       instead of the whole catalogue.
# Both ranks are read off the SAME forward pass, so the full-catalogue number on exactly these
# examples is reported alongside as a control. Every protocol decision (and why) is in
# src/eval_protocol.py's docstring; nothing about the protocol is decided in this cell.
#
# Names this cell relies on (all defined earlier in stage5_lbsn_finetune.ipynb):
#   meta_df, full_df, test_ex, build_split_examples, collate, model, BATCH_SIZE, SEED,
#   TEST_MAX, SMOKE_TEST, OUT_DIR, DATASET, EMB_CONDITION, MODEL_NAME, SCORING_MODE, N_MASKS,
#   USE_ACC_AT_T_OBJECTIVE, _forward_multimask / _forward_single,
#   and, for the group task, group_ex, build_group_example, _load_group_examples (§9b).
import json, os, subprocess, sys as _sys
import numpy as np, torch
from torch.utils.data import DataLoader

_REPO_RAW = globals().get("REPO_RAW", "https://raw.githubusercontent.com/yosrkharrat/"
                                       "Group-recommendation-for-next-poi/llmgpr-gowalla/src")
_CODE_STAGE = globals().get("CODE_STAGE", "./_code")
try:
    import eval_protocol as EP
except ImportError:                       # §0b predates the module: fetch it from the repo branch
    os.makedirs(_CODE_STAGE, exist_ok=True)
    _dst = os.path.join(_CODE_STAGE, "eval_protocol.py")
    subprocess.run(["wget", "-q", f"{_REPO_RAW}/eval_protocol.py", "-O", _dst], check=True)
    if _CODE_STAGE not in _sys.path:
        _sys.path.insert(0, _CODE_STAGE)
    import eval_protocol as EP
_sc = subprocess.run([_sys.executable, EP.__file__, "--self-check"], capture_output=True, text=True)
assert _sc.returncode == 0, "eval_protocol.py self-check FAILED -- do not trust anything below:\n" + _sc.stdout[-2000:]
print(f"[11c] eval_protocol self-check: {_sc.stdout.strip().splitlines()[-1]}")

CP_N_CAND, CP_ANCHOR, CP_KS = 500, "target", (5, 10)   # the protocol as published; anchor="target" =
                                                        # nearest to the ground-truth POI (confirmed reading)
cp_index = EP.CandidateIndex(meta_df, region_col="auto", seed=SEED)
cp_desc = cp_index.describe()
print(f"[11c] {DATASET}: {cp_desc['n_poi']:,} POIs | region = {cp_desc['region_col']} "
      f"({cp_desc['n_regions']} region(s)) | coordinate coverage {cp_desc['coord_coverage']:.1%}")
if cp_desc["coord_coverage"] < 0.999:
    print("[11c] WARNING: POIs without coordinates -> part of the candidate sets is random, not "
          "'nearest'. The fill fraction is recorded in the results json; a column built like this "
          "must not be reported as LLMGPR's protocol.")


@torch.no_grad()
def comparison_protocol_pass(examples, cands, name):
    """One forward pass -> rank among the 501 candidates AND rank against the full catalogue."""
    model.eval()
    loader = DataLoader(list(range(len(examples))), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda idx: (collate([examples[i] for i in idx]),
                                                cands[np.asarray(idx)]))
    r_cand, r_full = [], []
    for batch, cand_b in loader:
        if globals().get("USE_ACC_AT_T_OBJECTIVE", False):
            logits, tgt = _forward_multimask(batch)
            logits = logits[:, 0, :]                        # slot 0 = the ranked position, as in _val_pass
        else:
            logits, tgt = _forward_single(batch)
        r_cand.extend(EP.ranks_from_logits(logits, tgt, cand_b))
        r_full.extend(((logits > logits.gather(1, tgt[:, None])).sum(1) + 1).tolist())
    res = dict(comparison_protocol=EP.hr_ndcg(r_cand, ks=CP_KS),
               full_catalogue_same_examples=EP.hr_ndcg(r_full, ks=CP_KS))
    print(f"[11c] {name:10s} comparison protocol           : {EP.format_metrics(res['comparison_protocol'], CP_KS)}")
    print(f"[11c] {name:10s} full catalogue, same examples : {EP.format_metrics(res['full_catalogue_same_examples'], CP_KS)}")
    return res


cp_results = dict(model=MODEL_NAME, dataset=DATASET, condition=EMB_CONDITION, scoring_mode=SCORING_MODE,
                  n_masks=N_MASKS, n_cand=CP_N_CAND, anchor=CP_ANCHOR, index=cp_desc,
                  protocol="LLMGPR CIKM'25 Sec. 4.1 -- leave-one-out; ground truth + 500 unvisited "
                           "POIs nearest to it within the same region; HR@k / NDCG@k")

# ── individual: the last check-in of each user ────────────────────────────────────────────────
# TEST_MAX (probe profile) subsamples test_ex, which can drop a user's last check-in; rebuild the
# full test split in that case so the selection really is leave-one-out.
_test_all = test_ex if (TEST_MAX is None and not SMOKE_TEST) else build_split_examples(full_df, "test")
loo_ind = EP.last_per_user(_test_all)
_visited = EP.visited_before_target(full_df, drop_last=1)
_sort_keys = ["user_id", "utc_time"] if "utc_time" in full_df.columns else ["user_id"]
_last_poi = full_df.sort_values(_sort_keys, kind="mergesort").groupby("user_id")["poi_idx"].last()
assert all(int(_last_poi[ex["user"]]) == int(ex["target"]) for ex in loo_ind[:500]), \
    "leave-one-out selection is not the user's last check-in -- check the split / TEST_MAX"
_ind = [cp_index.candidates(ex["target"], _visited[ex["user"]], n=CP_N_CAND) for ex in loo_ind]
ind_cands = np.stack([c for c, _ in _ind])
ind_stats = EP.summarize_infos([i for _, i in _ind], CP_N_CAND)
print(f"[11c] individual: {len(loo_ind):,} leave-one-out examples (one per user) | "
      f"filled {ind_stats['frac_examples_filled']:.1%} | anchors w/o coords {ind_stats['n_anchor_no_coord']}")
cp_results["individual"] = dict(n=len(loo_ind), candidate_stats=ind_stats,
                                **comparison_protocol_pass(loo_ind, ind_cands, "individual"))

# ── group: one example per member set, the chronologically last ───────────────────────────────
_gex = globals().get("group_ex")
loo_grp, grp_cands = None, None
if _gex and _gex.get("test"):
    _grp_all = _gex["test"]
    if TEST_MAX is not None and len(_grp_all) >= TEST_MAX:        # capped -> reload the full split
        _grp_all = [build_group_example(r) for r in _load_group_examples("test")]
    loo_grp = EP.last_per_group(_grp_all)
    _grp = [cp_index.candidates(ex["target"], EP.group_visited(ex), n=CP_N_CAND) for ex in loo_grp]
    grp_cands = np.stack([c for c, _ in _grp])
    grp_stats = EP.summarize_infos([i for _, i in _grp], CP_N_CAND)
    print(f"[11c] group: {len(_grp_all):,} test examples -> {len(loo_grp):,} kept (one per member set, "
          f"chronologically last) | filled {grp_stats['frac_examples_filled']:.1%} | "
          f"anchors w/o coords {grp_stats['n_anchor_no_coord']}")
    cp_results["group"] = dict(n=len(loo_grp), n_before_selection=len(_grp_all), group_select="last",
                               candidate_stats=grp_stats,
                               **comparison_protocol_pass(loo_grp, grp_cands, "group"))
else:
    print("[11c] no group_ex in this session -- individual task only")

# ── persist: results + the exact candidate sets, so the numbers can be audited ────────────────
_out = f"{OUT_DIR}/comparison_protocol_results_{DATASET}_{EMB_CONDITION}.json"
with open(_out, "w") as f:
    json.dump(cp_results, f, indent=2)
_npz = dict(ind_keys=np.array([ex["user"] for ex in loo_ind]), ind_cands=ind_cands)
if loo_grp is not None:
    _npz.update(grp_keys=np.array([ex["example_id"] for ex in loo_grp]), grp_cands=grp_cands)
np.savez_compressed(f"{OUT_DIR}/comparison_protocol_candidates_{DATASET}.npz", **_npz)
print(f"[11c] saved {_out}\n[11c] saved {OUT_DIR}/comparison_protocol_candidates_{DATASET}.npz"
      f"\n[11c] -> send both files back together with this cell's printed output.")
