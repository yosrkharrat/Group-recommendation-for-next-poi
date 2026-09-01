#!/usr/bin/env python3
"""Check whether Yang's Section-3 dump reproduces LLMGPR's Foursquare statistics,
and whether Section-5 friendships can be attached to it.

LLMGPR (CIKM'25) cites its Foursquare data as [4] = Chen et al., IMWUT'20, whose
dataset section reads: "33,278,683 check-in records of 266,909 users at 3,680,126
unique POIs between April 2012 and September 2013 in the most checked 415 cities".
That is verbatim Section 3 of Dingqi Yang's page -- NOT the Section-5 WWW2019 dump
we have been extracting from. Section 3 has no friendship file, hence this probe.

  --stats    rebuild LLMGPR Table 1 from Section 3 (NY / LA / Chicago, >=10 core)
  --idprobe  test whether Section-5 user ids are the same id space as Section 3
             (and, if not, whether a check-in fingerprint recovers the mapping)
"""
import argparse, collections, os, random, sys
import pandas as pd

CITY_BBOX = {  # identical to the current notebook so the numbers are comparable
    "New York":    dict(lon_min=-74.3,  lon_max=-73.6,  lat_min=40.4, lat_max=41.0),
    "Chicago":     dict(lon_min=-88.0,  lon_max=-87.5,  lat_min=41.6, lat_max=42.1),
    "Los Angeles": dict(lon_min=-118.7, lon_max=-117.6, lat_min=33.6, lat_max=34.4),
}
CHUNK = 2_000_000
CK_COLS  = ["user_id", "venue_id", "utc_time", "tz_offset"]
POI_COLS = ["venue_id", "lat", "lon", "category", "country"]


def read_checkins(path, **kw):
    return pd.read_csv(path, sep="\t", header=None, names=CK_COLS,
                       dtype={"user_id": str, "venue_id": str, "utc_time": str},
                       usecols=[0, 1, 2, 3], on_bad_lines="skip", **kw)


def city_venue_sets(poi_path):
    pois = pd.read_csv(poi_path, sep="\t", header=None, names=POI_COLS,
                       dtype={"venue_id": str}, low_memory=False)
    pois["lat"] = pd.to_numeric(pois["lat"], errors="coerce")
    pois["lon"] = pd.to_numeric(pois["lon"], errors="coerce")
    print(f"POIs in file: {len(pois):,}")
    out, cats = {}, {}
    for city, b in CITY_BBOX.items():
        m = (pois["lon"].between(b["lon_min"], b["lon_max"]) &
             pois["lat"].between(b["lat_min"], b["lat_max"]))
        sub = pois.loc[m]
        out[city] = set(sub["venue_id"])
        cats[city] = dict(zip(sub["venue_id"], sub["category"]))
        print(f"  {city}: {len(sub):,} venues inside bbox")
    return out, cats


def k_core(df, k=10):
    """LLMGPR 4.1: 'users and POIs with less than 10 interactions are removed'
    -> keep >= k.  Iterated, because each removal can push the other side under."""
    n0 = len(df)
    while True:
        before = len(df)
        vc = df["user_id"].value_counts();  df = df[df["user_id"].isin(vc[vc >= k].index)]
        vc = df["venue_id"].value_counts(); df = df[df["venue_id"].isin(vc[vc >= k].index)]
        if len(df) == before or df.empty:
            break
    print(f"  {k}-core: {n0:,} -> {len(df):,} check-ins")
    return df


def cmd_stats(a):
    venues, cats = city_venue_sets(a.pois)
    keep = set().union(*venues.values())
    parts, seen = [], 0
    for ch in read_checkins(a.checkins, chunksize=CHUNK):
        seen += len(ch)
        parts.append(ch[ch["venue_id"].isin(keep)])
        print(f"\rscanned {seen:,}", end="", file=sys.stderr)
    df = pd.concat(parts, ignore_index=True)
    print(f"\nin-bbox check-ins (3 cities, unfiltered): {len(df):,}")

    print(f"\n{'city':<12}{'users':>9}{'POIs':>9}{'cats':>7}{'check-ins':>12}{'ck/user':>9}")
    tot = []
    for city, vset in venues.items():
        d = k_core(df[df["venue_id"].isin(vset)].copy(), a.min_interactions)
        tot.append(d)
        nc = pd.Series([cats[city].get(v) for v in d["venue_id"].unique()]).nunique()
        print(f"{city:<12}{d['user_id'].nunique():>9,}{d['venue_id'].nunique():>9,}"
              f"{nc:>7,}{len(d):>12,}{len(d)/max(d['user_id'].nunique(),1):>9.1f}")
    allc = pd.concat(tot, ignore_index=True)
    cat_all = {}
    for c in cats.values():
        cat_all.update(c)
    nc = pd.Series([cat_all.get(v) for v in allc["venue_id"].unique()]).nunique()
    print(f"{'ALL 3':<12}{allc['user_id'].nunique():>9,}{allc['venue_id'].nunique():>9,}"
          f"{nc:>7,}{len(allc):>12,}{len(allc)/max(allc['user_id'].nunique(),1):>9.1f}")
    print("\nLLMGPR Table 1 target (Foursquare, NY+LA+Chicago):")
    print(f"{'TARGET':<12}{7507:>9,}{80962:>9,}{436:>7,}{1214631:>12,}{162.8:>9.1f}")
    print("arXiv v1 of the same paper, NYC only: 6,078 users / 63,445 POIs / 923,856 check-ins")
    if a.out:
        allc.to_csv(a.out, index=False)
        print(f"wrote {a.out}")


def cmd_idprobe(a):
    """Are Section-5 user ids the same id space as Section-3's?

    Fingerprint = exact (venue_id, utc_time) pair. Sample S5 users, collect their
    pairs, then stream S3 and see which S3 user those pairs land on."""
    random.seed(0)
    s5_users = set()
    for f in (a.friend_old, a.friend_new):
        if f and os.path.exists(f):
            e = pd.read_csv(f, sep="\t", header=None, names=["u", "v"], dtype=str)
            s5_users |= set(e["u"]) | set(e["v"])
    print(f"S5 users carrying >=1 friendship edge: {len(s5_users):,}")

    sample = set(random.sample(sorted(s5_users), min(a.sample, len(s5_users))))
    fp = {}                                    # (venue, utc) -> s5 user
    for ch in read_checkins(a.s5_checkins, chunksize=CHUNK):
        ch = ch[ch["user_id"].isin(sample)]
        for u, v, t in zip(ch["user_id"], ch["venue_id"], ch["utc_time"]):
            fp[(v, t)] = u
    print(f"fingerprints from {len(sample):,} sampled S5 users: {len(fp):,} check-ins")

    votes = collections.defaultdict(collections.Counter)
    hits = 0
    for ch in read_checkins(a.s3_checkins, chunksize=CHUNK):
        for u3, v, t in zip(ch["user_id"], ch["venue_id"], ch["utc_time"]):
            u5 = fp.get((v, t))
            if u5 is not None:
                votes[u5][u3] += 1
                hits += 1
    print(f"fingerprint hits in S3: {hits:,}")

    matched = [u5 for u5 in votes if votes[u5]]
    identical = sum(1 for u5 in matched if votes[u5].most_common(1)[0][0] == u5)
    purity = sorted(votes[u5].most_common(1)[0][1] / sum(votes[u5].values())
                    for u5 in matched)
    med = purity[len(purity) // 2] if purity else 0.0
    print(f"\nS5 users found in S3      : {len(matched):,} / {len(sample):,}"
          f" ({len(matched)/len(sample):.1%})")
    print(f"  ...with the SAME id     : {identical:,}"
          f"  <- if ~= matched, the two dumps share one id space: just join on user_id")
    print(f"  median vote purity      : {med:.2f}"
          f"  <- if ~= 1.0, each S5 user maps to exactly one S3 user:")
    print(f"  purity >= 0.9           : {sum(1 for x in purity if x >= 0.9):,}"
          f"     build the S5->S3 id map from these votes and carry the friendships over")
    print("  (a (venue_id, utc_time) pair is near-unique across 3.7M venues at second\n"
          "   resolution, so low purity on the real dumps means a genuine id mismatch)")
    if matched:
        print("\nexamples (s5_id -> top s3_id, votes):")
        for u5 in matched[:10]:
            top, n = votes[u5].most_common(1)[0]
            print(f"  {u5} -> {top}  ({n}/{sum(votes[u5].values())})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    s.add_argument("--checkins", required=True, help="dataset_TIST2015_Checkins.txt")
    s.add_argument("--pois", required=True, help="dataset_TIST2015_POIs.txt")
    s.add_argument("--min-interactions", type=int, default=10)
    s.add_argument("--out", default=None)
    i = sub.add_parser("idprobe"); i.set_defaults(fn=cmd_idprobe)
    i.add_argument("--s3-checkins", required=True)
    i.add_argument("--s5-checkins", required=True)
    i.add_argument("--friend-old", required=True)
    i.add_argument("--friend-new", default=None)
    i.add_argument("--sample", type=int, default=2000)
    a = p.parse_args(); a.fn(a)
