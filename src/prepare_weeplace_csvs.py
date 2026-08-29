from __future__ import annotations

import argparse
import json
import math
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_groups import resplit_per_user

DS_DEFAULT = "WEEPLACE"
MIN_INTERACTIONS = 10
WINDOW_MINUTES = 180
MIN_GROUP_SIZE = 2
MIN_RECURRENCE = 1

CITY_MAP = {"new york": "New York", "chicago": "Chicago", "los angeles": "Los Angeles"}
STATE_OF = {"New York": "NY", "Chicago": "IL", "Los Angeles": "CA"}

REFERENCE = dict(users=4_560, groups=923, pois=44_194, cats=625, user_ck=623_654,
                  group_ck=11_974, ck_per_user=136.77, ck_per_group=12.97, users_per_group=4.37)

# LLMGPR Table 1, Weeplace column only.
TABLE1 = {
    "users":           dict(label="#users",             weeplace=4_560),
    "groups":          dict(label="#groups",             weeplace=923),
    "pois":            dict(label="#POIs",                weeplace=44_194),
    "cats":            dict(label="#categories",          weeplace=625),
    "user_ck":         dict(label="#user check-ins",      weeplace=623_654),
    "group_ck":        dict(label="#group check-ins",     weeplace=11_974),
    "ck_per_user":     dict(label="#check-ins per user",  weeplace=136.77),
    "ck_per_group":    dict(label="#check-ins per group", weeplace=12.97),
    "users_per_group": dict(label="#users per group",     weeplace=4.37),
}
CATALOGUE_SCOPED = {"pois", "cats"}   # read from raw (pre-filter) check-ins -- see `run`


# --------------------------------------------------------------------------
# loading + the literal filter
# --------------------------------------------------------------------------

def load_checkins(path):
    df = pd.read_csv(path, dtype=str)
    city_norm = df["city"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()
    df = df[city_norm.isin(CITY_MAP)].copy()
    df["city"] = city_norm[city_norm.isin(CITY_MAP)].map(CITY_MAP)
    df = df.rename(columns={"userid": "user_id", "placeid": "venue_id"})
    df["utc_time"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["category"] = df["category"].fillna("Unknown").str.replace(":", ">", regex=False)
    df = df.dropna(subset=["user_id", "venue_id", "utc_time", "lat", "lon"]).reset_index(drop=True)
    return df[["user_id", "venue_id", "utc_time", "lat", "lon", "city", "category"]]


def load_friends(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = ["user_id", "friend_id"]
    return df.dropna()


def literal_filter(df, min_inter=MIN_INTERACTIONS):
    """Users with <T interactions removed, then POIs with <T visits removed, single pass,
    user-first."""
    uc = df["user_id"].value_counts()
    df = df[df["user_id"].isin(uc[uc >= min_inter].index)]
    pc = df["venue_id"].value_counts()
    df = df[df["venue_id"].isin(pc[pc >= min_inter].index)].copy()
    return df


# --------------------------------------------------------------------------
# compact ids, split, house-format assembly
# --------------------------------------------------------------------------

def assemble(ck, friends):
    raw_users = sorted(ck["user_id"].unique())
    uid_of = {u: i for i, u in enumerate(raw_users)}
    raw_venues = sorted(ck["venue_id"].unique())
    pid_of = {v: i for i, v in enumerate(raw_venues)}

    venue_first = ck.drop_duplicates("venue_id").set_index("venue_id")

    # Prefer a real (non-"Unknown") category for a venue over whichever row happened to come
    # first in the file; every check-in at that venue then shares the resolved value.
    cat_of = venue_first["category"].copy()
    real_cat = (ck[ck["category"] != "Unknown"]
                .drop_duplicates("venue_id").set_index("venue_id")["category"])
    cat_of.update(real_cat)

    meta = pd.DataFrame({"venue_id": raw_venues})
    meta["poi_idx"] = np.arange(len(meta))
    meta["venue_category_name"] = cat_of.loc[raw_venues].to_numpy()
    meta["category"] = meta["venue_category_name"]
    meta["locality"] = venue_first.loc[raw_venues, "city"].to_numpy()
    meta["region"] = [STATE_OF[c] for c in meta["locality"]]
    meta["latitude"] = venue_first.loc[raw_venues, "lat"].to_numpy()
    meta["longitude"] = venue_first.loc[raw_venues, "lon"].to_numpy()
    meta["name"] = ""
    meta["description"] = ""

    df = ck.copy()
    df["user_id"] = df["user_id"].map(uid_of)
    df["poi_idx"] = df["venue_id"].map(pid_of)
    df["venue_category_name"] = df["venue_id"].map(cat_of)
    df["venue_category_id"] = ""
    df["timezone_offset"] = 0
    df = df.merge(meta[["poi_idx", "latitude", "longitude"]], on="poi_idx", how="left")
    df = resplit_per_user(df, 0.70, 0.10)

    users = pd.DataFrame({"user_id": range(len(raw_users)), "raw_id": raw_users})

    u1 = friends["user_id"].map(uid_of)
    u2 = friends["friend_id"].map(uid_of)
    keep = u1.notna() & u2.notna() & (u1 != u2)
    e = pd.DataFrame({"u1": u1[keep].astype(int), "u2": u2[keep].astype(int)})
    edges = pd.DataFrame(np.sort(e.to_numpy(), axis=1), columns=["u1", "u2"])
    edges = edges.drop_duplicates().reset_index(drop=True)

    return df, meta, users, edges


# --------------------------------------------------------------------------
# group construction: literal rule, rolling window, cliques
# --------------------------------------------------------------------------

def build_groups(df, edges, window_minutes=WINDOW_MINUTES, min_group_size=MIN_GROUP_SIZE,
                  min_recurrence=MIN_RECURRENCE):
    G = nx.Graph()
    G.add_edges_from(edges[["u1", "u2"]].itertuples(index=False, name=None))

    window = np.timedelta64(int(window_minutes * 60), "s")
    member_events = {}          # frozenset(members) -> occurrence count
    rows_by_key = {}            # frozenset(members) -> list of (event_id, user_id, poi_idx, utc_time)
    event_id = 0

    ordered = df.sort_values(["poi_idx", "utc_time"], kind="mergesort")
    for _, sub in ordered.groupby("poi_idx", sort=False):
        times = sub["utc_time"].to_numpy()
        if len(times) > 1:
            gaps = np.diff(times) > window
            cluster = np.concatenate([[0], np.cumsum(gaps)])
        else:
            cluster = np.zeros(len(times), dtype=int)
        sub = sub.assign(_cluster=cluster)
        for _, clu in sub.groupby("_cluster", sort=False):
            present = set(clu["user_id"])
            if len(present) < min_group_size:
                continue
            sg = G.subgraph(present)
            for clique in nx.find_cliques(sg):
                if len(clique) < min_group_size:
                    continue
                key = frozenset(clique)
                member_events[key] = member_events.get(key, 0) + 1
                rows = clu[clu["user_id"].isin(clique)]
                rows_by_key.setdefault(key, []).extend(
                    (event_id, r.user_id, r.poi_idx, r.utc_time) for r in rows.itertuples())
                event_id += 1

    keys = sorted((k for k, n in member_events.items() if n >= min_recurrence), key=sorted)
    group_rows, ck_rows = [], []
    for gid, key in enumerate(keys):
        group_rows.append(dict(group_id=gid, members="|".join(map(str, sorted(key))),
                                size=len(key), n_events=member_events[key]))
        for ev, uid, pid, ts in rows_by_key[key]:
            ck_rows.append(dict(group_id=gid, event_id=ev, user_id=uid, poi_idx=pid, utc_time=ts))

    groups_df = pd.DataFrame(group_rows, columns=["group_id", "members", "size", "n_events"])
    group_ck_df = pd.DataFrame(ck_rows, columns=["group_id", "event_id", "user_id", "poi_idx", "utc_time"])
    return groups_df, group_ck_df


def build_groups_social(df, edges, window_minutes=WINDOW_MINUTES, min_clique_size=2,
                        attendance_frac=0.7, min_recurrence=4):
    G = nx.Graph()
    G.add_edges_from(edges[["u1", "u2"]].itertuples(index=False, name=None))
    cliques = [frozenset(c) for c in nx.find_cliques(G) if len(c) >= min_clique_size]

    window = np.timedelta64(int(window_minutes * 60), "s")
    by_user = {u: sub[["poi_idx", "utc_time"]].to_numpy() for u, sub in df.groupby("user_id")}

    group_rows, ck_rows = [], []
    gid = 0
    for clique in cliques:
        need = max(2, math.ceil(len(clique) * attendance_frac))
        rows = sorted(((pid, ts, u) for u in clique for pid, ts in by_user.get(u, [])),
                     key=lambda r: (r[0], r[1]))
        if not rows:
            continue
        events, i, n = [], 0, len(rows)
        while i < n:
            j = i
            present = {rows[i][2]: [(rows[i][0], rows[i][1])]}
            while j + 1 < n and rows[j + 1][0] == rows[i][0] and \
                    (rows[j + 1][1] - rows[j][1]) <= window:
                j += 1
                present.setdefault(rows[j][2], []).append((rows[j][0], rows[j][1]))
            if len(present) >= need:
                events.append(present)
            i = j + 1
        if len(events) < min_recurrence:
            continue
        group_rows.append(dict(group_id=gid, members="|".join(map(str, sorted(clique))),
                                size=len(clique), n_events=len(events)))
        for ev_id, present in enumerate(events):
            for uid, visits in present.items():
                for pid, ts in visits:
                    ck_rows.append(dict(group_id=gid, event_id=ev_id, user_id=uid,
                                        poi_idx=pid, utc_time=ts))
        gid += 1

    groups_df = pd.DataFrame(group_rows, columns=["group_id", "members", "size", "n_events"])
    group_ck_df = pd.DataFrame(ck_rows, columns=["group_id", "event_id", "user_id", "poi_idx", "utc_time"])
    return groups_df, group_ck_df


def group_stats(groups_df):
    if len(groups_df) == 0:
        return dict(groups=0, group_ck=0, users_per_group=0.0, ck_per_group=0.0)
    n_groups = len(groups_df)
    n_events = int(groups_df["n_events"].sum())
    return dict(groups=n_groups, group_ck=n_events,
                users_per_group=float(groups_df["size"].mean()),
                ck_per_group=n_events / n_groups)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def write_outputs(out_dir, ds, df, meta, users, edges, groups_df, group_ck_df, manifest):
    os.makedirs(out_dir, exist_ok=True)
    split_cols = ["user_id", "venue_id", "venue_category_id", "venue_category_name",
                  "latitude", "longitude", "timezone_offset", "utc_time", "poi_idx"]
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        part[split_cols].to_csv(os.path.join(out_dir, f"{split}_{ds}.csv"), index=False)
        print(f"  {split}_{ds}.csv  {len(part):,} rows")

    meta_cols = ["venue_id", "name", "category", "locality", "region", "description",
                 "poi_idx", "latitude", "longitude"]
    meta[meta_cols].to_csv(os.path.join(out_dir, f"poi_metadata_{ds}.csv"), index=False)
    print(f"  poi_metadata_{ds}.csv  {len(meta):,} POIs")

    edges.to_csv(os.path.join(out_dir, f"friendship_{ds}.csv"), index=False)
    print(f"  friendship_{ds}.csv  {len(edges):,} edges")

    users.to_csv(os.path.join(out_dir, f"users_{ds}.csv"), index=False)

    groups_df.to_csv(os.path.join(out_dir, f"groups_{ds}.csv"), index=False)
    group_ck_df.to_csv(os.path.join(out_dir, f"group_checkins_{ds}.csv"), index=False)
    print(f"  groups_{ds}.csv  {len(groups_df):,} groups   "
          f"group_checkins_{ds}.csv  {len(group_ck_df):,} rows")

    with open(os.path.join(out_dir, "weeplace_prep_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def print_table1_comparison(stats):
    """Print LLMGPR's Table 1, Weeplace column only, next to what this run produced."""
    print(f"\n{'':<22}{'ours':>13}{'Weeplace':>13}{'ratio':>9}")
    print("-" * 60)
    for k, row in TABLE1.items():
        ours = stats.get(k)
        theirs = row["weeplace"]
        flag = " *" if k in CATALOGUE_SCOPED else ""
        ratio = f"{ours / theirs:.3f}x" if theirs else "n/a"
        print(f"{row['label']:<22}{ours:>13,.2f}{row['weeplace']:>13,.2f}{ratio:>9}{flag}")
    print("\n* read as the region CATALOGUE (every venue/category in the 3-city raw check-ins, "
          "before the >=10 filter) -- LLMGPR_TRACK.md Sec 1.4 recovers this reading for "
          "Foursquare from their own tables; it lands at 1.03x/1.07x here too.")


def run(a):
    print("loading + city-filtering checkins ...")
    raw = load_checkins(a.checkins)
    print(f"  {len(raw):,} check-ins in New York / Chicago / Los Angeles, "
          f"{raw.user_id.nunique():,} users, {raw.venue_id.nunique():,} venues")

    print(f"applying literal >={a.min_interactions}/>={a.min_interactions} filter "
          f"(in-region, single pass, user-first) ...")
    ck = literal_filter(raw, a.min_interactions)
    print(f"  {len(ck):,} check-ins, {ck.user_id.nunique():,} users, {ck.venue_id.nunique():,} POIs")

    print("loading friendships ...")
    friends = load_friends(a.friends)
    both_in = friends["user_id"].isin(set(ck.user_id)) & friends["friend_id"].isin(set(ck.user_id))
    print(f"  {len(friends):,} directed edges, {both_in.mean():.1%} with both endpoints "
          f"in the filtered population")

    df, meta, users, edges = assemble(ck, friends)

    if a.group_mode == "social-clique":
        print(f"building groups (mode=social-clique, window={a.window_minutes}min, "
              f"min_clique_size={a.min_clique_size}, attendance_frac={a.attendance_frac}, "
              f"min_recurrence={a.min_recurrence}) ...")
        groups_df, group_ck_df = build_groups_social(df, edges, a.window_minutes,
                                                      a.min_clique_size, a.attendance_frac,
                                                      a.min_recurrence)
    else:
        print(f"building groups (mode=simultaneous, window={a.window_minutes}min, "
              f"min_size={a.min_group_size}, min_recurrence={a.min_recurrence}) ...")
        groups_df, group_ck_df = build_groups(df, edges, a.window_minutes, a.min_group_size,
                                              a.min_recurrence)
    gstats = group_stats(groups_df)
    print(f"  {gstats['groups']:,} groups, {gstats['group_ck']:,} co-presence events, "
          f"{gstats['users_per_group']:.2f} users/group, {gstats['ck_per_group']:.2f} events/group")
    if len(groups_df):
        size_dist = groups_df["size"].value_counts().sort_index()
        print("  group size distribution:", ", ".join(f"{s}={n:,}" for s, n in size_dist.items()))
        recur = groups_df["n_events"].value_counts().sort_index()
        print("  recurrence distribution (co-presences per group):",
              ", ".join(f"{r}x={n:,}" for r, n in recur.items()))

    # #POIs/#categories: region catalogue (raw, pre-filter check-ins), not the post-filter count.
    stats = dict(users=ck.user_id.nunique(), pois=raw.venue_id.nunique(),
                cats=raw["category"].nunique(), user_ck=len(ck),
                ck_per_user=len(ck) / ck.user_id.nunique(), **gstats)
    print_table1_comparison(stats)

    manifest = dict(config=vars(a), n_checkins=len(df), n_users=int(df.user_id.nunique()),
                    n_pois=len(meta), split_sizes=df["split"].value_counts().to_dict(),
                    city_counts=ck["city"].value_counts().to_dict(),
                    n_friendship_edges=len(edges), stats=stats, reference=REFERENCE)
    write_outputs(a.out_dir, a.dataset, df, meta, users, edges, groups_df, group_ck_df, manifest)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def _self_check():
    import tempfile
    print("SELF-CHECK on a synthetic fixture")
    t0 = pd.Timestamp("2012-04-03T18:00:00")

    def fmt(ts):
        return ts.isoformat()

    rows = []
    # a,b,c (mutual friend triangle) + friendless d all visit v1 together, 12 times, spaced
    # 3h apart (>> the 60min window) -> 12 SEPARATE co-presence events for clique {a,b,c};
    # d is present at every one of them but must never appear in a group (no friend edges).
    for i in range(12):
        for u in ("a", "b", "c", "d"):
            rows.append((u, "v1", fmt(t0 + pd.Timedelta(hours=3 * i)), 40.7, -73.9,
                         "New York", "Food:Pizza"))
    # user "e" checks in only twice -> below the interaction floor, dropped entirely
    for i in range(2):
        rows.append(("e", "v1", fmt(t0 + pd.Timedelta(hours=i)), 40.7, -73.9,
                     "New York", "Food:Pizza"))
    # a Boston check-in -> dropped by the city filter regardless of volume, city-name-only noise
    for i in range(20):
        rows.append(("a", "v_bos", fmt(t0 + pd.Timedelta(hours=i)), 42.35, -71.08,
                     "Boston", "Food:Pizza"))
    # a and b co-occur at v2 twice, a day apart (>> window) -> two separate co-presence events
    rows.append(("a", "v2", fmt(t0), 40.71, -73.95, "New York", "Nightlife:Bar"))
    rows.append(("b", "v2", fmt(t0 + pd.Timedelta(minutes=5)), 40.71, -73.95, "New York",
                 "Nightlife:Bar"))
    rows.append(("a", "v2", fmt(t0 + pd.Timedelta(days=1)), 40.71, -73.95, "New York",
                 "Nightlife:Bar"))
    rows.append(("b", "v2", fmt(t0 + pd.Timedelta(days=1, minutes=5)), 40.71, -73.95, "New York",
                 "Nightlife:Bar"))
    # pad v2 to clear the POI floor (>=10) with solo visits by c, far in time from a/b's visits
    # so they land in their own clusters and never join (or inflate) the a/b group
    for i in range(6):
        rows.append(("c", "v2", fmt(t0 + pd.Timedelta(days=10, hours=3 * i)), 40.71, -73.95,
                     "New York", "Nightlife:Bar"))

    ck = pd.DataFrame(rows, columns=["userid", "placeid", "datetime", "lat", "lon", "city",
                                     "category"])
    friend = pd.DataFrame([("a", "b"), ("b", "c"), ("a", "c"),   # a,b,c mutual triangle
                           ("a", "e")],                          # e is filtered out anyway
                          columns=["userid1", "userid2"])

    with tempfile.TemporaryDirectory() as td:
        ckp, frp = os.path.join(td, "ck.csv"), os.path.join(td, "fr.csv")
        ck.to_csv(ckp, index=False)
        friend.to_csv(frp, index=False)

        class A:
            pass
        a = A()
        a.checkins, a.friends, a.out_dir, a.dataset = ckp, frp, os.path.join(td, "out"), "TEST"
        a.min_interactions = 10
        a.window_minutes = 60
        a.group_mode = "simultaneous"
        a.min_group_size = 2
        a.min_recurrence = 1

        run(a)
        tr = pd.concat([pd.read_csv(os.path.join(a.out_dir, f"{s}_TEST.csv"))
                        for s in ("train", "val", "test")])
        me = pd.read_csv(os.path.join(a.out_dir, "poi_metadata_TEST.csv"))
        us = pd.read_csv(os.path.join(a.out_dir, "users_TEST.csv"))
        gr = pd.read_csv(os.path.join(a.out_dir, "groups_TEST.csv"))
        gc = pd.read_csv(os.path.join(a.out_dir, "group_checkins_TEST.csv"))
        n_friend_edges = len(pd.read_csv(os.path.join(a.out_dir, "friendship_TEST.csv")))

    raw_to_uid = dict(zip(us["raw_id"], us["user_id"]))
    key_abc = {str(raw_to_uid["a"]), str(raw_to_uid["b"]), str(raw_to_uid["c"])}
    key_ab = {str(raw_to_uid["a"]), str(raw_to_uid["b"])}
    member_sets = gr["members"].apply(lambda m: set(m.split("|")))

    ok = lambda n, c: (print(f"  {'PASS' if c else 'FAIL'}  {n}"), c)[1]
    print()
    res = [
        ok("city filter drops Boston, keeps New York variants",
           "v_bos" not in set(me.venue_id) and set(me.venue_id) == {"v1", "v2"}),
        ok("user with <10 interactions ('e') is dropped entirely", "e" not in raw_to_uid),
        ok("friend edge to a filtered-out user ('e') never reaches friendship_TEST.csv",
           n_friend_edges == 3),
        ok("user/poi ids are 0-based contiguous ranks",
           sorted(tr.user_id.unique()) == list(range(len(raw_to_uid)))
           and sorted(me.poi_idx) == list(range(len(me)))),
        ok("category ':' becomes '>'", me.category.eq("Food>Pizza").any()
           and not me.category.str.contains(":").any()),
        ok("locality/region set from the city field",
           me.loc[me.venue_id == "v1", "locality"].iloc[0] == "New York"
           and me.loc[me.venue_id == "v1", "region"].iloc[0] == "NY"),
        ok("a/b/c form a clique group at v1; friendless 'd' never appears in any group",
           (member_sets == key_abc).any()
           and not gr["members"].str.contains(str(raw_to_uid["d"])).any()),
        ok("the {a,b,c} member-set recurs as ONE persistent group across all 12 co-presences",
           gr.loc[member_sets == key_abc, "n_events"].iloc[0] == 12),
        ok("a widely time-separated pair at v2 forms TWO events, not one merged cluster",
           gr.loc[member_sets == key_ab, "n_events"].iloc[0] == 2),
        ok("split files carry the house columns",
           list(tr.columns) == ["user_id", "venue_id", "venue_category_id",
                                "venue_category_name", "latitude", "longitude",
                                "timezone_offset", "utc_time", "poi_idx"]),
        ok("group_checkins rows only reference kept groups",
           set(gc.group_id).issubset(set(gr.group_id))),
    ]
    return all(res)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkins", default="./data/Weeplace/weeplace_checkins.csv")
    p.add_argument("--friends", default="./data/Weeplace/weeplace_friends.csv")
    p.add_argument("--out-dir", default="./data/weeplace")
    p.add_argument("--dataset", default=DS_DEFAULT)
    p.add_argument("--min-interactions", type=int, default=MIN_INTERACTIONS)
    p.add_argument("--window-minutes", type=float, default=WINDOW_MINUTES)
    p.add_argument("--group-mode", choices=["simultaneous", "social-clique"],
                   default="social-clique",
                   help="'simultaneous' = a friendship clique, all of whom must be co-present; "
                        "'social-clique' (default) = anchor to a maximal clique of the static "
                        "friendship graph and count an occasion whenever >=--attendance-frac of "
                        "it co-attends")
    p.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE,
                   help="mode=simultaneous only: minimum co-present clique size")
    p.add_argument("--min-clique-size", type=int, default=2,
                   help="mode=social-clique only: minimum size of the static friendship clique "
                        "a group is anchored to")
    p.add_argument("--attendance-frac", type=float, default=0.7,
                   help="mode=social-clique only: fraction of a clique that must be "
                        "simultaneously co-present for one occasion to count (rounded up, "
                        "minimum 2)")
    p.add_argument("--min-recurrence", type=int, default=None,
                   help="keep only groups that recur at least this many times; defaults to 4 "
                        f"for mode=social-clique, {MIN_RECURRENCE} for mode=simultaneous")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.min_recurrence is None:
        a.min_recurrence = 4 if a.group_mode == "social-clique" else MIN_RECURRENCE
    if a.self_check:
        sys.exit(0 if _self_check() else 1)
    run(a)


if __name__ == "__main__":
    main()
