"""Shared exclusion windows for group outings (epoch minutes throughout).

Each set's manifest is authoritative. A numeric override may widen its exclusion
window, but cannot silently make it narrower than the construction window.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

POLICY_VERSION = "group-window-v1"
OUTING_COLUMNS = ["outing", "group_id", "split", "venue_id", "ev_ts", "user_id", "repro", "guard_min"]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def group_window_minutes(group_dir, override=None):
    path = Path(group_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    seconds = manifest.get("rule", {}).get("window_seconds")
    if seconds == "same calendar minute":
        window = 1.0
    elif isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
        window = float(seconds) / 60.0
    else:
        raise ValueError(f"{path}: missing or unsupported rule.window_seconds")
    if not math.isfinite(window) or window < 0:
        raise ValueError(f"{path}: invalid construction window")
    if override is not None:
        override = float(override)
        if not math.isfinite(override) or override < window:
            raise ValueError(f"guard {override} min is narrower than {path}'s {window:g} min window")
        return override
    return window


def group_guard_policy(group_dirs, override=None):
    if isinstance(group_dirs, (str, Path)):
        group_dirs = [group_dirs]
    sets = []
    for d in sorted({str(Path(d)) for d in (group_dirs or [])}):
        p = Path(d)
        sets.append(dict(name=p.name, guard_min=group_window_minutes(p, override),
                         manifest_sha256=file_sha256(p / "manifest.json"),
                         checkins_sha256=file_sha256(p / "group_checkins.csv")))
    if len({s["name"] for s in sets}) != len(sets):
        raise ValueError("group sets must have distinct directory names")
    return dict(version=POLICY_VERSION, mode="manifest" if override is None else "override",
                sets=sorted(sets, key=lambda s: s["name"]))


def holdout_outings(repro_dirs, splits=("val", "test", "masked")):
    if isinstance(repro_dirs, (str, Path)):
        repro_dirs = [repro_dirs]
    frames = []
    for d in repro_dirs or []:
        frames.append(pd.read_csv(Path(d) / "group_checkins.csv").assign(
            repro=Path(d).name, guard_min=group_window_minutes(d)))
    if not frames:
        return pd.DataFrame(columns=OUTING_COLUMNS)
    g = pd.concat(frames, ignore_index=True)
    g = g[g["split"].isin(splits)].copy()
    g["ev_ts"] = pd.to_datetime(g["utc_time"], utc=True).dt.as_unit("ns").astype("int64") // (60 * 10**9)
    g["outing"] = np.arange(len(g))
    g["user_id"] = g["present"].astype(str).str.split("|")
    g = g.explode("user_id")
    g["user_id"] = g["user_id"].astype(int)
    return g[OUTING_COLUMNS].reset_index(drop=True)


def drop_outing_checkins(train, outings, guard_min=None):
    """Exclude same-member, same-venue visits inside each outing's own +/- window."""
    if outings is None or outings.empty:
        return train, 0
    if guard_min is None:
        if "guard_min" not in outings:
            raise ValueError("outings need per-set guard_min; load them with holdout_outings")
        windows = pd.to_numeric(outings["guard_min"], errors="raise")
    else:
        windows = pd.Series(float(guard_min), index=outings.index)
        if "guard_min" in outings and (windows < outings["guard_min"]).any():
            raise ValueError("explicit guard is narrower than a group set's construction window")
    if not np.isfinite(windows).all() or (windows < 0).any():
        raise ValueError("invalid outing exclusion window")
    tr = train[["venue_id", "user_id", "ts"]].reset_index(drop=True)
    tr["_position"] = np.arange(len(train))
    events = outings[["venue_id", "user_id", "ev_ts"]].assign(_guard=windows.to_numpy())
    matched = tr.merge(events, on=["venue_id", "user_id"])
    hit = matched.loc[(matched["ts"] - matched["ev_ts"]).abs() <= matched["_guard"], "_position"].unique()
    keep = np.ones(len(train), dtype=bool)
    keep[hit] = False
    return train.iloc[keep], len(hit)


def validate_poi_artifacts(embedding_path, expected_policy, metadata_path=None):
    """Fail before fine-tuning if graph/embedding files predate the active exclusions."""
    p = Path(embedding_path)
    graph_path = p.parent / "kg_manifest.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if graph.get("outing_guard_policy") != expected_policy:
        raise ValueError("POI KG uses another outing guard. Rebuild it with the current group manifests.")
    sidecar = Path(str(p) + ".manifest.json")
    if not sidecar.exists():
        raise ValueError(f"Missing {sidecar.name}. Retrain KGE on the rebuilt graph with --save-embs, "
                         "then copy the selected .npy AND its .npy.manifest.json sidecar.")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = dict(version="kge-embedding-v1", embedding_sha256=file_sha256(p),
                    kg_manifest_sha256=file_sha256(graph_path),
                    kg_triples_sha256=file_sha256(p.parent / "kg_triples.pt"),
                    outing_guard_policy=expected_policy)
    if metadata_path is not None:
        expected["metadata_sha256"] = file_sha256(metadata_path)
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError("POI embeddings do not match the current graph/metadata/outing policy. "
                         "Retrain or select the matching GPU-run artifacts.")
    return provenance
