"""Synthetic ground-truth tests for the structural registration workflow (pipeline stages 1-2)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openscope_registration.matching import build_tracks, session_tag
from .test_registration_diagnostics import make_session
from openscope_registration.pipeline import (
    fov_correspondence,
    register_plane,
    run_structural_registration,
)


class TracksTests(unittest.TestCase):
    def test_tracks_count_coverage_and_adjacent_consistency(self) -> None:
        ref_ids = np.array([0, 1, 2])
        sessions = ["ref", "a", "b"]
        tab = lambda pairs: pd.DataFrame(  # noqa: E731
            {"reference_roi_id": [p[0] for p in pairs], "moving_roi_id": [p[1] for p in pairs],
             "mask_iou": [0.8] * len(pairs), "candidate": [True] * len(pairs)})
        to_reference = {"a": tab([(0, 10), (1, 11)]), "b": tab([(0, 20), (1, 21), (2, 22)])}
        adjacent = {("ref", "a"): tab([(0, 10), (1, 11)]), ("a", "b"): tab([(10, 20), (11, 99)])}
        tracks = build_tracks(ref_ids, sessions, "ref", to_reference, adjacent).set_index("reference_roi_id")
        self.assertEqual(tracks.loc[0, "n_sessions_covered"], 3)
        self.assertTrue(tracks.loc[0, "covers_all_sessions"])
        self.assertEqual(tracks.loc[0, "adjacent_consistency_fraction"], 1.0)   # 10->20 confirmed
        self.assertEqual(tracks.loc[1, "adjacent_consistency_fraction"], 0.5)   # 11->21 contradicted by direct 11->99
        self.assertEqual(tracks.loc[2, "n_sessions_covered"], 2)                # missing in a
        self.assertEqual(tracks.loc[2, "n_adjacent_checks"], 0)
        self.assertEqual(tracks.loc[2, f"roi_id__{session_tag('a')}"], -1)
        self.assertEqual(tracks.loc[2, f"roi_id__{session_tag('b')}"], 22)


class StructuralRegistrationTests(unittest.TestCase):
    def test_register_plane_recovers_transform_and_identity(self) -> None:
        reference = make_session("multiplane_000_2026-01-01", seed=1)
        moving = [make_session("multiplane_000_2026-01-02", angle_deg=1.5, shift=(3.0, -2.0), seed=2),
                  make_session("multiplane_000_2026-01-03", angle_deg=-1.0, shift=(-2.5, 4.0), seed=3, drop=2)]
        result = register_plane(reference, moving, plane="P0", angle_step_deg=0.25)
        tf = result["transforms_table"].set_index("session_id")
        self.assertAlmostEqual(tf.loc["multiplane_000_2026-01-02", "angle_deg"], -1.5, delta=0.26)
        self.assertAlmostEqual(tf.loc["multiplane_000_2026-01-02", "shift_y_px"], -3.0, delta=0.6)
        self.assertAlmostEqual(tf.loc["multiplane_000_2026-01-02", "shift_x_px"], 2.0, delta=0.6)
        self.assertTrue((tf["rigid_ncc"] > 0.8).all())
        self.assertTrue((tf["inverse_shift_residual_px"] < 1.0).all())
        to_ref = result["matches"][result["matches"].comparison.eq("to_reference") & result["matches"].candidate]
        self.assertTrue((to_ref["reference_roi_id"] == to_ref["moving_roi_id"]).all(), "identity matching on same-layout sessions")
        tracks = result["tracks"]
        self.assertGreaterEqual(int(tracks["covers_all_sessions"].sum()), 25)
        self.assertEqual(int(tracks["adjacent_consistency_fraction"].dropna().lt(1.0).sum()), 0)
        self.assertEqual(len(result["adjacent"]), 2)

    def test_fov_correspondence_prefers_diagonal(self) -> None:
        ref0, ref1 = make_session("r", seed=1), make_session("r", seed=11, drop=5)
        ref1.projection = np.rot90(ref1.projection)  # make plane 1 structurally distinct
        ref1.footprints = np.rot90(ref1.footprints, axes=(1, 2))
        mov0 = make_session("m", angle_deg=0.5, shift=(1.0, 1.0), seed=2)
        mov1 = make_session("m", angle_deg=0.5, shift=(1.0, 1.0), seed=12, drop=5)
        mov1.projection = np.rot90(mov1.projection)
        mov1.footprints = np.rot90(mov1.footprints, axes=(1, 2))
        fov = fov_correspondence({"P0": {"r": ref0, "m": mov0}, "P1": {"r": ref1, "m": mov1}}, "r", angle_step_deg=0.5)
        self.assertEqual(len(fov), 4)
        self.assertTrue(fov.drop_duplicates(["session_id", "moving_plane"])["diagonal_is_best"].all())

    def test_workflow_writes_pilot_compatible_files(self) -> None:
        ref = make_session("multiplane_000_2026-01-01", seed=1)
        mov = make_session("multiplane_000_2026-01-02", angle_deg=1.0, shift=(2.0, -1.0), seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            res = run_structural_registration({"P0": {ref.session_id: ref, mov.session_id: mov}}, ref.session_id, out, angle_step_deg=0.5)
            for name in ("session_plane_inventory", "registration_transforms", "residual_tile_shifts", "roi_match_candidates", "candidate_track_coverage"):
                self.assertTrue((out / f"{name}.csv").exists(), name)
            self.assertFalse((out / "fov_correspondence_ncc.csv").exists(), "single plane -> no FOV table")
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["artifact_type"], "structural_registration")
            self.assertIn("P0", manifest["summary"])
            self.assertIn("median_rigid_ncc", manifest["summary"]["P0"])
            cands = pd.read_csv(out / "roi_match_candidates.csv")
            self.assertEqual(set(cands.comparison), {"to_reference", "adjacent_day"})
            self.assertEqual(int(cands.comparison.eq("adjacent_day").sum()), int(cands.comparison.eq("to_reference").sum()))
            with self.assertRaises(FileExistsError):
                run_structural_registration({"P0": {ref.session_id: ref, mov.session_id: mov}}, ref.session_id, out, angle_step_deg=0.5)
            self.assertEqual(res["summary"]["P0"]["n_reference_rois"], ref.n_rois)


if __name__ == "__main__":
    unittest.main()
