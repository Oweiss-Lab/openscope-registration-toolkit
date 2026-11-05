"""Tests for the manifest freeze (pipeline stage 5): acceptance rule, integrity gates, and consumer round-trip."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from openscope_registration.freeze import freeze_plane, plane_slug
from openscope_registration.diagnostics import (
    accepted_tracks_wide,
    derive_retained_and_reference,
)
from openscope_registration.pipeline import run_freeze_manifest

SESSIONS = ["s_a", "s_b", "s_c"]


def _pairs(rows):
    return pd.DataFrame(rows, columns=["session_id", "reference_roi_id", "moving_roi_id", "p_same", "p_same_raw", "mask_iou",
                                       "centroid_distance_px", "footprint_correlation", "mechanism", "candidate"])


def _good_pairs():
    rows = []
    for rid in (0, 1, 2):
        for s in SESSIONS:
            rows.append([s, rid, rid + 100, 0.97, 0.9, 0.7, 1.0, 0.8, "matched", True])
    rows.append(["s_a", 3, 103, 0.5, 0.4, 0.3, 3.0, 0.2, "matched", True])  # below threshold
    return _pairs(rows)


def _decisions(d: dict[int, str]):
    return pd.DataFrame({"track_id": list(d), "decision": list(d.values())})


class FreezePlaneTests(unittest.TestCase):
    def test_accepts_only_reviewed_tracks_above_threshold(self) -> None:
        frozen = freeze_plane("VISp_0", _decisions({0: "accept", 1: "accept", 2: "reject", 3: "reject"}), _good_pairs(), None)
        self.assertEqual(sorted(frozen.accepted_tracks.reference_roi_id.unique()), [0, 1])
        self.assertEqual(len(frozen.accepted_tracks), 6)
        self.assertEqual(frozen.summary["n_accepted"], 2)
        self.assertEqual(frozen.summary["n_rejected"], 2)
        self.assertEqual(frozen.summary["n_sessions"], 3)
        self.assertTrue((frozen.accepted_tracks["plane"] == "VISp_0").all())

    def test_rejects_undecided_review(self) -> None:
        with self.assertRaises(ValueError):
            freeze_plane("P", _decisions({0: "accept", 1: "undecided"}), _good_pairs(), None)

    def test_rejects_accepted_track_with_missing_session(self) -> None:
        pairs = _good_pairs()
        pairs = pairs.loc[~((pairs.reference_roi_id == 1) & (pairs.session_id == "s_b"))]
        with self.assertRaises(ValueError):
            freeze_plane("P", _decisions({0: "accept", 1: "accept"}), pairs, None)

    def test_rejects_accepted_track_below_threshold_everywhere(self) -> None:
        with self.assertRaises(ValueError):
            freeze_plane("P", _decisions({3: "accept"}), _good_pairs(), None)


class FreezeWorkflowTests(unittest.TestCase):
    def test_workflow_writes_manifest_consumable_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gui = tmp / "gui"; gui.mkdir()
            runs = {}
            for plane in ("VISp_0", "VISl_4"):
                run = tmp / f"refine_{plane}"; run.mkdir()
                _good_pairs().to_csv(run / "refined_pairs.csv", index=False)
                pd.DataFrame({"reference_roi_id": [0, 0, 0], "session_id": SESSIONS, "provenance": ["matched"] * 3}).to_csv(run / "track_day_provenance.csv", index=False)
                (run / "manifest.json").write_text(json.dumps({"excluded_sessions": ["s_excluded"]}))
                _decisions({0: "accept", 1: "accept", 2: "reject", 3: "reject"}).to_csv(gui / f"manual_review_decisions_999_{plane_slug(plane)}.csv", index=False)
                runs[plane] = run
            out = tmp / "frozen"
            manifest = run_freeze_manifest(gui, runs, out, subject_id="999")
            self.assertEqual(manifest["artifact_type"], "frozen_cell_manifest")
            self.assertEqual(manifest["status"], "frozen")
            self.assertEqual(manifest["excluded_sessions"], ["s_excluded"])
            self.assertEqual(manifest["summary"]["VISp_0"]["n_accepted"], 2)
            self.assertTrue((out / "accepted_tracks_visp0.csv").exists() and (out / "accepted_tracks_visl4.csv").exists())
            # provenance column merged in
            acc = pd.read_csv(out / "accepted_tracks_visp0.csv")
            self.assertIn("provenance", acc.columns)
            # downstream consumers read it with no study knowledge
            wide = accepted_tracks_wide(out, "visp0", "s_ref")
            self.assertEqual(list(wide.index), [0, 1])
            self.assertEqual(int(wide.loc[0, "s_b"]), 100)
            self.assertEqual(int(wide.loc[0, "s_ref"]), 0)
            retained, ref = derive_retained_and_reference([out], set(SESSIONS) | {"s_ref", "s_excluded"})
            self.assertEqual(ref, "s_ref")
            self.assertEqual(retained, sorted(SESSIONS + ["s_ref"]))


if __name__ == "__main__":
    unittest.main()
