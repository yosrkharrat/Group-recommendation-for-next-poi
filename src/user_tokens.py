"""
User tokens: one `<user_i>` token per user, living in the same embedding space as the POI tokens.

Why
---
Design direction (2026-11-19): the social graph is a graph of USERS, and users carry more
information than a group, so the model should see users as tokens. A group
prompt then names each member by their token (`[member 1] <user_524>`), and the LLM combines
them. A single `<grp>` consensus token, if kept, becomes an ablation built from these same vectors.

How a user's vector is built (no new training, nothing new to align)
-------------------------------------------------------------------
1. **Signature on the ball.** The weighted gyromidpoint of the user's top-`top_k` most visited POIs
   in the TRAIN split, weighted by visit count. It is the same recipe as stage 5's
   `_member_signature` (cell 34), which already places members in the POI space.
2. **Optional one-hop social smoothing** over `data/social_graph/<DS>/social_edges_<DS>.csv`
   (`src/build_social_graph.py`): the user's point is pulled towards their ties,
       s'_u = gyromidpoint([s_u, s_v1, ..., s_vk], weights [1, beta * a_uv / Σ_v a_uv])
   with a_uv = Σ over the selected relations of FRIEND_OF → 1 and CO_PRESENT → weight / median.
   The neighbours' mass is `beta` whatever their number. Smoothing reads only the UNSMOOTHED
   neighbour points (one hop, not iterated). A user with no train history but with ties gets the
   ties' midpoint (cold start). Relation sets: none | friends | copresence | both (the ablation).
3. **Into the LLM space through the SAME trained adapter as the POIs** (`alignment.ManifoldAwareAdapter`,
   the supervisor's), then the SAME per-dimension affine as `W_POI`. `match_llm_embedding_distribution`
   standardises with the statistics of whatever batch it is given, so calling it on the user rows
   would place users in a different space from the POIs. `affine_like_pois` reuses the POI
   batch's statistics instead. Check: a user whose only POI is p gets exactly W_POI[p]
   (`_self_test`, and asserted in the notebook).

Does smoothing carry a social signal? (offline, `--diagnose`, LLMGPR)
--------------------------------------------------------------------
Proxy task with no LLM: rank every POI by geodesic distance to the user's point, target = the
user's first VALIDATION POI not visited in train (5,386 users). MRR for the users smoothing moves,
unsmoothed -> smoothed; `both_random` = the same edges with partners shuffled (control).

foursquare-v2 embeddings (RefH on the POI-only KG of the extended catalogue, data/llmgpr/kg_poi,
15,315 POIs, every POI with coordinates), 2026-11-19:

    beta   friends          copresence       both             both_random      low-history, both
    0.1    .0131 -> .0126   .0107 -> .0117   .0127 -> .0123   .0127 -> .0119   .0158 -> .0142
    0.5    .0131 -> .0123   .0107 -> .0124   .0127 -> .0121   .0127 -> .0108   .0158 -> .0116
    1.0    .0131 -> .0118   .0107 -> .0126   .0127 -> .0120   .0127 -> .0089   .0158 -> .0138
    2.0    .0131 -> .0109   .0107 -> .0105   .0127 -> .0118   .0127 -> .0059   .0158 -> .0125

Co-presence helps at every beta (+18% at beta 1, all-user MRR .0138 -> .0143). Declared friendship
now HURTS at every beta, though always less than random ties. Ties carry signal, but declared
friends live elsewhere on this more geographic map. So the notebook recommends
SOCIAL_SMOOTHING=copresence with SOCIAL_BETA=1.0.

On the OLD embeddings (kg_denoised RefH, 14,402 POIs, 43% with coordinates), friendship helped too
(both: .0104 -> .0140 at beta 2, random .0104 -> .0092). The finding depends on the embedding
geometry, which is why the stage 5 ablation (none / friends / copresence / both) decides, not this
proxy.

Leakage
-------
The table is built from each user's TRAIN-split check-ins: a transductive user representation, the
same footing as the POI KG (built on train). The splits are per-user chronological, so for every
val/test example the user's token only uses that user's earlier check-ins. For a TRAIN example the
token can include later train check-ins of the same user. This is the standard trade-off for user
embeddings (MF, KGE user nodes), and it is weaker than the KG itself, which already encodes the
train transitions. The per-example `[user profile]` text stays strictly causal. Social edges are
train-only by construction.

Usage
-----
    python src/user_tokens.py --self-test
    python src/user_tokens.py --diagnose      # offline: does smoothing move signatures towards
                                              # users' VALIDATION POIs? (no LLM needed)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyperbolic_group import gyromidpoint, logmap0 as _hg_logmap0  # noqa: E402

RELATIONS = {"none": (), "friends": ("FRIEND_OF",), "copresence": ("CO_PRESENT",),
             "both": ("FRIEND_OF", "CO_PRESENT")}


# --------------------------------------------------------------------------
# 1. signatures
# --------------------------------------------------------------------------

def user_top_pois(df, n_users, top_k):
    """(idx, w): [n_users, top_k] POI ids and visit counts, most visited first. Ties go to the POI
    visited first (df row order), exactly like stage 5's profile (`Counter.most_common`), so the
    user token and the [user profile] text agree on which POIs are the top ones."""
    d = df[["user_id", "poi_idx"]].assign(_pos=np.arange(len(df)))
    cnt = d.groupby(["user_id", "poi_idx"]).agg(n=("_pos", "size"), first=("_pos", "min")).reset_index()
    cnt = cnt.sort_values(["user_id", "n", "first"], ascending=[True, False, True], kind="mergesort")
    cnt["rank"] = cnt.groupby("user_id").cumcount()
    cnt = cnt[cnt["rank"] < top_k]
    idx = np.zeros((n_users, top_k), dtype=np.int64)
    w = np.zeros((n_users, top_k), dtype=np.float64)
    idx[cnt["user_id"].to_numpy(), cnt["rank"].to_numpy()] = cnt["poi_idx"].to_numpy()
    w[cnt["user_id"].to_numpy(), cnt["rank"].to_numpy()] = cnt["n"].to_numpy()
    return idx, w


def user_signatures(df, poi_embs, n_users, top_k=10, c=1.0):
    """[n_users, d] float64 ball points and a bool `has` mask (False = no check-ins in df)."""
    idx, w = user_top_pois(df, n_users, top_k)
    X = torch.as_tensor(np.asarray(poi_embs), dtype=torch.float64)[torch.as_tensor(idx)]
    W = torch.as_tensor(w)
    mask = W > 0
    sig = gyromidpoint(X, W, c=c, mask=mask)
    has = mask.any(-1)
    sig[~has] = 0.0
    return sig, has


# --------------------------------------------------------------------------
# 2. social smoothing
# --------------------------------------------------------------------------

def neighbour_table(edges, relations, n_users):
    """{u: (neighbour ids, raw affinities a_uv)} over the selected relations, symmetric."""
    if not relations:
        return {}
    e = edges[edges["relation"].isin(relations)][["u1", "u2", "relation", "weight"]].copy()
    if e.empty:
        return {}
    e["s"] = 1.0
    cp = e["relation"] == "CO_PRESENT"
    if cp.any():
        e.loc[cp, "s"] = e.loc[cp, "weight"] / float(e.loc[cp, "weight"].median())
    a = e.groupby(["u1", "u2"])["s"].sum().reset_index()
    both = pd.concat([a, a.rename(columns={"u1": "u2", "u2": "u1"})], ignore_index=True)
    both = both[(both["u1"] < n_users) & (both["u2"] < n_users)]
    return {int(u): (g["u2"].to_numpy(), g["s"].to_numpy(dtype=np.float64))
            for u, g in both.groupby("u1")}


def smooth_signatures(sig, has, edges, relations, beta=0.5, c=1.0):
    """One-hop smoothing towards ties. Returns (new sig, new has, number of users moved)."""
    out, has_out = sig.clone(), has.clone()
    if not relations or beta <= 0:
        return out, has_out, 0
    moved = 0
    for u, (nb, a) in neighbour_table(edges, relations, sig.shape[0]).items():
        ok = has[torch.as_tensor(nb)].numpy()
        if not ok.any():
            continue
        nb, a = nb[ok], a[ok]
        a = torch.as_tensor(a / a.sum())
        if has[u]:
            pts = torch.cat([sig[u:u + 1], sig[torch.as_tensor(nb)]], 0)
            wts = torch.cat([torch.ones(1, dtype=torch.float64), beta * a])
        else:                                           # cold start: the ties' midpoint
            pts, wts = sig[torch.as_tensor(nb)], a
            has_out[u] = True
        out[u] = gyromidpoint(pts.unsqueeze(0), wts.unsqueeze(0), c=c).squeeze(0)
        moved += 1
    return out, has_out, moved


# --------------------------------------------------------------------------
# 3. into the LLM's embedding space
# --------------------------------------------------------------------------

def affine_like_pois(raw, raw_poi, llm_weight):
    """`alignment.match_llm_embedding_distribution`'s formula, with the POI batch's statistics
    (so users and POIs share one affine map). affine_like_pois(raw_poi, raw_poi, W) is exactly
    match_llm_embedding_distribution(raw_poi, W)."""
    tm = llm_weight.mean(dim=0, keepdim=True)
    ts = llm_weight.std(dim=0, keepdim=True).clamp_min(1e-5)
    pm = raw_poi.mean(dim=0, keepdim=True)
    ps = raw_poi.std(dim=0, keepdim=True).clamp_min(1e-5)
    return (raw - pm) / ps * ts + tm


@torch.no_grad()
def user_token_table(adapter, user_ball, poi_embs, llm_weight, c=1.0, logmap=None):
    """[n_users, d_model] fp32. `logmap` must be the SAME log map the notebook used for W_POI."""
    logmap = logmap or _hg_logmap0
    adapter.eval()
    dev = next(adapter.parameters()).device
    poi = torch.as_tensor(np.asarray(poi_embs), dtype=torch.float32, device=dev)
    raw_poi = adapter(logmap(poi, c=c)).float()
    raw = adapter(logmap(user_ball.to(device=dev, dtype=torch.float32), c=c)).float()
    return affine_like_pois(raw, raw_poi, llm_weight.float().to(dev))


def build_user_tokens(train_df, poi_embs, n_users, adapter, llm_weight, *, top_k=10, c=1.0,
                      edges=None, smoothing="none", beta=0.5, logmap=None):
    """Everything the notebook needs in one call. Returns (table, info dict)."""
    sig, has = user_signatures(train_df, poi_embs, n_users, top_k=top_k, c=c)
    rel = RELATIONS[smoothing]
    if rel and edges is None:
        raise ValueError(f"smoothing={smoothing!r} needs the social edges file")
    sig_s, has_s, moved = smooth_signatures(sig, has, edges, rel, beta=beta, c=c) if rel else (sig, has, 0)
    table = user_token_table(adapter, sig_s, poi_embs, llm_weight, c=c, logmap=logmap)
    info = dict(n_users=n_users, with_history=int(has.sum()), after_smoothing=int(has_s.sum()),
                moved_by_smoothing=moved, smoothing=smoothing, beta=beta, top_k=top_k)
    return table, sig_s, info


# --------------------------------------------------------------------------
# offline diagnostic: does smoothing bring a user's point closer to where they go next?
# --------------------------------------------------------------------------

def _ball_dist_matrix(U, P, c=1.0, chunk=1024):
    """Poincare distances [n_u, n_p] via the arccosh form (no [n, m, d] tensor)."""
    p2 = (P * P).sum(-1)
    out = []
    for s in range(0, U.shape[0], chunk):
        q = U[s:s + chunk]
        q2 = (q * q).sum(-1, keepdim=True)
        sq = (q2 + p2[None] - 2.0 * q @ P.T).clamp_min(0)
        den = (1 - c * q2).clamp_min(1e-15) * (1 - c * p2[None]).clamp_min(1e-15)
        out.append(torch.arccosh((1 + 2 * c * sq / den).clamp_min(1 + 1e-12)) / c ** 0.5)
    return torch.cat(out)


def diagnose(data_dir, dataset, emb_path, edges_path, top_k=10, beta=0.5, c=1.0, seed=0):
    t0 = time.time()
    tr = pd.read_csv(os.path.join(data_dir, f"train_{dataset}.csv"), usecols=["user_id", "poi_idx"])
    va = pd.read_csv(os.path.join(data_dir, f"val_{dataset}.csv"), usecols=["user_id", "poi_idx"])
    emb = np.load(emb_path)
    edges = pd.read_csv(edges_path)
    n_users = int(max(tr["user_id"].max(), va["user_id"].max())) + 1
    sig, has = user_signatures(tr, emb, n_users, top_k=top_k, c=c)
    n_tr = tr.groupby("user_id").size().reindex(range(n_users), fill_value=0).to_numpy()
    seen = tr.groupby("user_id")["poi_idx"].apply(set).to_dict()
    # target: each user's FIRST val POI not already seen in train (a revisit is trivially near)
    tgt = {}
    for u, g in va.groupby("user_id", sort=False):
        for p in g["poi_idx"]:
            if p not in seen.get(u, ()):
                tgt[int(u)] = int(p)
                break
    users = np.array(sorted(tgt))
    P = torch.as_tensor(emb, dtype=torch.float64)
    print(f"{len(users):,} users with a new (unvisited) validation POI; {len(emb):,} POIs; "
          f"beta={beta}; target = first validation POI not visited in train")

    ut = torch.as_tensor(users)
    tcol = torch.as_tensor([tgt[u] for u in users])

    def rr_of(points):
        D = _ball_dist_matrix(points[ut], P, c=c)
        d_t = D[torch.arange(len(users)), tcol]
        return 1.0 / ((D < d_t[:, None]).sum(1).numpy() + 1)

    rr0 = rr_of(sig)
    low = n_tr[users] <= np.quantile(n_tr[users], 0.25)
    res = {}
    print(f"{'smoothing':<11}{'MRR all':>9}{'moved':>8}{'MRR moved: none -> smoothed':>32}"
          f"{'low-history moved: none -> smoothed':>40}")
    # Control: the same edges with each relation's partners shuffled (every user keeps their number
    # of ties and the weights, but the ties go to random users). Pulling a point towards ANY users
    # moves it towards general taste, which can help a nearest-POI ranking on its own; the social
    # signal is the gap between "both" and "both_random".
    rnd = edges.copy()
    g = np.random.default_rng(seed)
    for r in rnd["relation"].unique():
        m = (rnd["relation"] == r).to_numpy()
        rnd.loc[m, "u2"] = g.permutation(rnd.loc[m, "u2"].to_numpy())
    rnd = rnd[rnd["u1"] != rnd["u2"]]
    settings = [(n, rel, edges) for n, rel in RELATIONS.items()] + [("both_random", RELATIONS["both"], rnd)]
    for name, rel, E in settings:
        s, _, _ = smooth_signatures(sig, has, E, rel, beta=beta, c=c)
        rr = rr_of(s) if rel else rr0
        mv = (s[ut] != sig[ut]).any(-1).numpy()
        lm = mv & low
        res[name] = dict(mrr=float(rr.mean()), n_moved=int(mv.sum()),
                         moved_before=float(rr0[mv].mean()) if mv.any() else None,
                         moved_after=float(rr[mv].mean()) if mv.any() else None,
                         n_low_moved=int(lm.sum()),
                         low_before=float(rr0[lm].mean()) if lm.any() else None,
                         low_after=float(rr[lm].mean()) if lm.any() else None)
        r = res[name]
        mv_s = f"{r['moved_before']:.4f} -> {r['moved_after']:.4f}" if mv.any() else "-"
        lo_s = (f"n={r['n_low_moved']:>4}  {r['low_before']:.4f} -> {r['low_after']:.4f}"
                if lm.any() else "-")
        print(f"{name:<11}{r['mrr']:>9.4f}{r['n_moved']:>8,}{mv_s:>32}{lo_s:>40}")
    print(f"({time.time() - t0:.0f}s)")
    return res


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test():
    from alignment import ManifoldAwareAdapter, match_llm_embedding_distribution
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_poi, d, d_model = 40, 8, 16
    poi = rng.normal(size=(n_poi, d))
    poi = poi / np.linalg.norm(poi, axis=1, keepdims=True) * rng.uniform(0.1, 0.8, (n_poi, 1))

    # user 0: only POI 7 (x3); user 1: POIs 1,2,3; user 2: no history; user 3: POI 5 + POI 9
    df = pd.DataFrame(dict(user_id=[0, 0, 0, 1, 1, 1, 1, 3, 3, 3],
                           poi_idx=[7, 7, 7, 1, 2, 3, 3, 5, 9, 9]))
    sig, has = user_signatures(df, poi, 4, top_k=10)
    assert has.tolist() == [True, True, False, True]
    assert torch.allclose(sig[0], torch.as_tensor(poi[7]), atol=1e-10), "single-POI signature != POI"
    idx, w = user_top_pois(df, 4, 2)
    assert idx[1].tolist() == [3, 1] and w[1].tolist() == [2, 1], (idx[1], w[1])   # top-k by count
    tie = pd.DataFrame(dict(user_id=[0, 0, 0], poi_idx=[9, 4, 6]))                    # all count 1
    assert user_top_pois(tie, 1, 2)[0][0].tolist() == [9, 4], "ties must go to the first-visited POI"
    from collections import Counter
    assert [p for p, _ in Counter([9, 4, 6]).most_common(2)] == [9, 4]

    adapter = ManifoldAwareAdapter(d, d_model, hidden_dim=32, num_layers=2, dropout=0.1)
    llm_w = torch.randn(100, d_model) * 0.02 + 0.01
    tbl = user_token_table(adapter, sig, poi, llm_w)
    with torch.no_grad():
        adapter.eval()
        raw_poi = adapter(_hg_logmap0(torch.as_tensor(poi, dtype=torch.float32)))
        w_poi = match_llm_embedding_distribution(raw_poi, llm_w)
    assert torch.allclose(affine_like_pois(raw_poi, raw_poi, llm_w), w_poi, atol=1e-6)
    assert torch.allclose(tbl[0], w_poi[7], atol=1e-5), "user with one POI must get exactly that POI's token"
    print("signature + token table: single-POI user == its W_POI row; affine == match_llm_embedding_distribution")

    edges = pd.DataFrame(dict(u1=[0, 0, 1], u2=[1, 2, 3], relation=["FRIEND_OF", "CO_PRESENT", "CO_PRESENT"],
                              weight=[1.0, 0.8, 1.6]))
    s0, h0, m0 = smooth_signatures(sig, has, edges, (), beta=0.5)
    assert m0 == 0 and torch.equal(s0, sig)
    s1, h1, m1 = smooth_signatures(sig, has, edges, ("FRIEND_OF",), beta=0.5)
    assert m1 == 2 and torch.equal(s1[3], sig[3]) and not h1[2]
    from hyperbolic_group import geodesic_distance
    assert geodesic_distance(s1[0], sig[1]) < geodesic_distance(sig[0], sig[1]), "smoothing must pull towards the tie"
    s2, h2, m2 = smooth_signatures(sig, has, edges, ("FRIEND_OF", "CO_PRESENT"), beta=0.5)
    assert h2[2] and torch.allclose(s2[2], sig[0], atol=1e-10), "cold-start user = its only tie's point"
    nt = neighbour_table(edges, ("CO_PRESENT",), 4)
    assert np.allclose(sorted(nt[1][1]), [1.6 / 1.2]) and np.allclose(nt[0][1], [0.8 / 1.2])   # / median 1.2
    s3, _, _ = smooth_signatures(sig, has, edges, ("FRIEND_OF",), beta=0.0)
    assert torch.equal(s3, sig)
    print("smoothing: none/beta=0 are identity, pulls towards ties, cold start, CO_PRESENT median scaling")
    print("self-test passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--data-dir", default="./data/llmgpr")
    p.add_argument("--dataset", default="LLMGPR")
    p.add_argument("--emb", default="./data/llmgpr/kg_poi/poi_hyperbolic_embs_LLMGPR_refh.npy")
    p.add_argument("--edges", default="./data/social_graph/LLMGPR/social_edges_LLMGPR.csv")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=10)
    a = p.parse_args()
    if a.self_test:
        _self_test()
    if a.diagnose:
        diagnose(a.data_dir, a.dataset, a.emb, a.edges, top_k=a.top_k, beta=a.beta)
    if not (a.self_test or a.diagnose):
        p.print_help()


if __name__ == "__main__":
    main()
