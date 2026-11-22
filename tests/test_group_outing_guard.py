"""Regression checks for per-set exclusion windows; no model training required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group_outing_guard import (drop_outing_checkins, file_sha256, group_guard_policy,
                                group_window_minutes, holdout_outings, validate_poi_artifacts)
from build_group_examples_llmgpr_match import member_prefix
from build_groups import MemberState


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.long = self.make_set("long", 86400, "A")
        self.short = self.make_set("short", 3600, "B")

    def tearDown(self):
        self.temp.cleanup()

    def make_set(self, name, window, venue):
        p = self.root / name
        p.mkdir()
        (p / "manifest.json").write_text(json.dumps(dict(rule=dict(window_seconds=window))))
        pd.DataFrame([dict(group_id=0, split="test", venue_id=venue,
                           utc_time="1970-01-01T00:00:00Z", present="0|1")]).to_csv(
            p / "group_checkins.csv", index=False)
        return p

    def test_mixed_windows_units_boundaries_and_participants(self):
        events = holdout_outings([self.long, self.short])
        self.assertEqual(set(events.guard_min), {1440, 60})
        rows = pd.DataFrame(dict(user_id=[0, 0, 0, 0, 0, 0, 9, 0],
                                 venue_id=["A", "A", "A", "A", "B", "B", "A", "C"],
                                 ts=[480, -480, 1440, 1441, 60, 90, 480, 0]), index=[0]*8)
        kept, count = drop_outing_checkins(rows, events)
        self.assertEqual(count, 4)
        self.assertEqual(kept.ts.tolist(), [1441, 90, 480, 0])
        self.assertEqual(drop_outing_checkins(kept, events)[1], 0)

    def test_manifest_and_override_validation(self):
        self.assertEqual(group_window_minutes(self.long), 1440)
        self.assertEqual(group_window_minutes(self.short, 120), 120)
        with self.assertRaises(ValueError):
            group_window_minutes(self.long, 120)
        with self.assertRaises(ValueError):
            drop_outing_checkins(pd.DataFrame(dict(user_id=[0],venue_id=["A"],ts=[0])),
                                 holdout_outings([self.long]), 120)
        with self.assertRaises(FileNotFoundError):
            group_window_minutes(self.root / "missing")
        minute = self.make_set("minute", "same calendar minute", "C")
        self.assertEqual(group_window_minutes(minute), 1)

    def test_split_filter_and_empty_sets(self):
        self.assertTrue(holdout_outings([self.long], splits=("train",)).empty)
        self.assertTrue(holdout_outings([]).empty)

    def test_policy_changes_when_inputs_change(self):
        policy = group_guard_policy([self.short, self.long])
        self.assertEqual(policy, group_guard_policy([self.long, self.short]))
        (self.long / "manifest.json").write_text(json.dumps(dict(rule=dict(window_seconds=172800))))
        self.assertNotEqual(policy, group_guard_policy([self.short, self.long]))

    def test_member_profile_uses_full_own_window(self):
        df = pd.DataFrame(dict(user_id=[0,0,0], poi_idx=[7,7,8], hour=[0,1,2], ts=[0,1000,1900]))
        state = MemberState(df, {7:"Food",8:"Art"}, 1)
        ts, pois, _ = member_prefix(state, 0, 2000, 7, 1440, True)
        self.assertEqual(ts.tolist(), [0,1900])
        self.assertEqual(pois.tolist(), [7,8])

    def test_stale_embeddings_rejected(self):
        policy = group_guard_policy([self.long, self.short])
        kg = self.root / "kg"
        kg.mkdir()
        emb = kg / "selected.npy"
        emb.write_bytes(b"fixture-embedding")
        (kg / "kg_triples.pt").write_bytes(b"fixture-triples")
        manifest = kg / "kg_manifest.json"
        manifest.write_text(json.dumps(dict(outing_guard_policy=policy)))
        with self.assertRaises(ValueError):
            validate_poi_artifacts(emb, policy)
        sidecar = dict(version="kge-embedding-v1", embedding_sha256=file_sha256(emb),
                       kg_triples_sha256=file_sha256(kg / "kg_triples.pt"),
                       kg_manifest_sha256=file_sha256(manifest), outing_guard_policy=policy)
        Path(str(emb)+".manifest.json").write_text(json.dumps(sidecar))
        validate_poi_artifacts(emb, policy)
        (kg / "kg_triples.pt").write_bytes(b"rebuilt-triples")
        with self.assertRaises(ValueError):
            validate_poi_artifacts(emb, policy)


if __name__ == "__main__":
    unittest.main()
