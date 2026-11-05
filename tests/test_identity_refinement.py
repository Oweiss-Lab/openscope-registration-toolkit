"""Synthetic tests for the identity-refinement workflow (rescue + P_same)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from openscope_registration.psame import (
    expected_false_matches,
    fit_p_same,
)
from openscope_registration.rescue import (
    backprojection_evidence,
)
from openscope_registration.data import RigidTransform, SessionStructuralData
from .test_registration_diagnostics import make_session
from openscope_registration.pipeline import run_identity_refinement

import pandas as pd


def with_is_soma(session: SessionStructuralData, non_soma_ids: set[int] = frozenset()) -> SessionStructuralData:
    session.metadata["is_soma"] = np.asarray([int(r) not in non_soma_ids for r in session.roi_ids], dtype=bool)
    return session


class PSameTests(unittest.TestCase):
    def test_posteriors_separate_planted_same_and_different(self) -> None:
        rng = np.random.default_rng(0)
        n = 150
        same = pd.DataFrame({
            "footprint_correlation": rng.normal(0.85, 0.06, n).clip(0, 1),
            "centroid_distance_px": rng.gamma(2.0, 0.7, n),
        })
        diff = pd.DataFrame({
            "footprint_correlation": rng.normal(0.05, 0.08, n),
            "centroid_distance_px": rng.uniform(2.0, 10.0, n),
        })
        pairs = pd.concat([same, diff], ignore_index=True)
        scored, model = fit_p_same(pairs, pixel_size_um=0.78)
        self.assertGreater(scored["p_same"].iloc[:n].median(), 0.95)
        self.assertLess(scored["p_same"].iloc[n:].median(), 0.05)
        n_accept, efm = expected_false_matches(scored["p_same"].to_numpy(), 0.95)
        self.assertGreaterEqual(n_accept, int(0.8 * n))
        self.assertLess(efm, 0.1 * n)
        self.assertEqual(model.n_pairs, 2 * n)

    def test_feature_scatter_grid_overlays_every_fit(self) -> None:
        from openscope_registration.plotting import (
            plot_p_same_feature_scatter_grid,
            plot_p_same_scatter,
        )

        rng = np.random.default_rng(1)

        def planted(n: int) -> pd.DataFrame:
            same = pd.DataFrame({"footprint_correlation": rng.normal(0.85, 0.06, n).clip(0, 1),
                                 "centroid_distance_px": rng.gamma(2.0, 0.7, n)})
            diff = pd.DataFrame({"footprint_correlation": rng.normal(0.05, 0.08, n),
                                 "centroid_distance_px": rng.uniform(2.0, 10.0, n)})
            return pd.concat([same, diff], ignore_index=True)

        fits = []
        for _ in range(2):
            scored, model = fit_p_same(planted(80), pixel_size_um=0.78)
            fits.append((scored, 0.78, model))
        units = {"unit A": fits, "unit B": fits[:1], "unit C": [(fits[0][0], 0.78, fits[0][2].as_dict())]}
        with tempfile.TemporaryDirectory() as tmp:
            grid = Path(tmp) / "grid.png"
            totals = plot_p_same_feature_scatter_grid(units, grid, title="synthetic", ncols=2, dpi=60)
            self.assertTrue(grid.exists() and grid.stat().st_size > 0)
            self.assertEqual(totals, {"unit A": 320, "unit B": 160, "unit C": 160})
            single = Path(tmp) / "p1.png"
            plot_p_same_scatter(fits[0][0], 0.78, single, title_context="synthetic", dpi=60, model=fits[0][2])
            self.assertTrue(single.exists() and single.stat().st_size > 0)


class IsotonicCalibrationTests(unittest.TestCase):
    def test_recalibration_corrects_underconfident_posteriors(self) -> None:
        from openscope_registration.psame import (
            apply_isotonic,
            fit_isotonic_calibration,
        )

        rng = np.random.default_rng(3)
        p_raw = rng.uniform(0.3, 0.97, 2000)
        true_prob = np.clip(1.6 * p_raw - 0.1, 0, 1)  # under-confident raw posterior
        outcomes = (rng.uniform(size=2000) < true_prob).astype(float)
        x_knots, y_knots = fit_isotonic_calibration(p_raw, outcomes)
        self.assertTrue((np.diff(y_knots) >= -1e-12).all())  # monotone
        p_cal = apply_isotonic(p_raw, x_knots, y_knots)
        raw_error = float(np.mean(np.abs(p_raw - true_prob)))
        cal_error = float(np.mean(np.abs(p_cal - true_prob)))
        self.assertLess(cal_error, 0.5 * raw_error)
        self.assertTrue(np.isnan(apply_isotonic(np.array([np.nan]), x_knots, y_knots)[0]))


class ManualReviewHtmlTests(unittest.TestCase):
    def test_builder_writes_selfcontained_html(self) -> None:
        from openscope_registration.review_html import (
            build_manual_review_html,
            encode_crop_png,
        )

        rng = np.random.default_rng(0)
        gray = rng.uniform(0, 1, (40, 40))
        outline = np.zeros((40, 40), dtype=bool)
        outline[15:25, 15:25] = True
        b64 = encode_crop_png(gray, outline)
        self.assertGreater(len(b64), 100)
        tracks = [
            {"track_id": i, "metrics": {"min P_same": 0.9 + 0.01 * i},
             "review_required": i != 2,
             "cells": {"d1": {"image_b64": b64, "category": "matched", "tooltip": "t"},
                       "d2": {"image_b64": b64, "category": "rescued_classifier_override", "tooltip": "t"}}}
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.html"
            build_manual_review_html(tracks, ["d1", "d2"], path, title="test review", context_note="note",
                                     export_slug="visp0")
            text = path.read_text()
        for needle in ["track-0", "track-2", "exportCsv", "localStorage", b64[:40], "rescued_classifier_override",
                       "manual_review_decisions_visp0.csv", 'data-review="0"', 'data-review="1"', "track_id,decision,source"]:
            self.assertIn(needle, text)


    def test_index_page_lists_guis_with_progress_keys(self) -> None:
        from openscope_registration.review_html import build_review_index_html

        entries = [
            {"label": "mouse X - VISl_4", "href": "a.html", "gui_title": "review A", "n_tracks": 12},
            {"label": "mouse X - VISl_5", "href": "b.html", "gui_title": "review B", "n_tracks": 7},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.html"
            build_review_index_html(entries, path, title="batch index")
            text = path.read_text()
        for needle in ["a.html", "b.html", '"review::review A"', '"review::review B"', 'data-total="12"', "localStorage"]:
            self.assertIn(needle, text)


class RescueTests(unittest.TestCase):
    def test_classifier_override_recovers_relabeled_cell(self) -> None:
        reference = with_is_soma(make_session("ref"))
        day = with_is_soma(make_session("day", angle_deg=1.0, shift=(-4.0, 2.0), seed=2), non_soma_ids={3, 7})
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_identity_refinement(
                reference, [day], Path(tmp) / "n", Path(tmp) / "f",
                transforms={"day": RigidTransform(-1.0, 4.0, -2.0, 0.95)},
                n_evidence_null=30, dpi=72,
            )
        self.assertGreaterEqual(summary["full_coverage_plus_override"], summary["full_coverage_base"] + 2)
        self.assertEqual(summary["full_coverage_plus_override"], 30)

    def test_backprojection_distinguishes_present_from_absent(self) -> None:
        reference = make_session("ref")
        projection = np.clip((reference.projection - reference.projection.mean()) / reference.projection.std(), -3, 6)
        present = backprojection_evidence(projection, reference.footprints, reference.roi_ids,
                                          reference.roi_ids[:5], n_null=100, seed=1)
        self.assertTrue((present["label"] == "present_but_undetected").all(), present["evidence_z"].tolist())
        blank = np.zeros_like(projection)
        absent = backprojection_evidence(blank, reference.footprints, reference.roi_ids,
                                         reference.roi_ids[:5], n_null=100, seed=1)
        self.assertTrue((absent["label"] == "absent_ambiguous").all())


class WorkflowTests(unittest.TestCase):
    def test_full_refinement_outputs_and_provenance(self) -> None:
        reference = with_is_soma(make_session("ref"))
        # day_a: 2 cells relabeled non-soma (rescuable); day_b: 3 cells dropped entirely (back-projection)
        day_a = with_is_soma(make_session("day_a", angle_deg=1.0, shift=(-4.0, 2.0), seed=2), non_soma_ids={3, 7})
        day_b = with_is_soma(make_session("day_b", angle_deg=-0.5, shift=(3.0, 3.0), seed=3, drop=3))
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_identity_refinement(
                reference, [day_a, day_b], Path(tmp) / "n", Path(tmp) / "f",
                transforms={"day_a": RigidTransform(-1.0, 4.0, -2.0, 0.95), "day_b": RigidTransform(0.5, -3.0, -3.0, 0.95)},
                n_evidence_null=30, dpi=72,
            )
            for path in summary["output_files"].values():
                self.assertTrue(Path(path).exists(), path)
            provenance = pd.read_csv(Path(tmp) / "n" / "track_day_provenance.csv")
            consistency = pd.read_csv(Path(tmp) / "n" / "transitive_consistency_pairs.csv")
            self.assertEqual(set(consistency.columns) >= {"session_id", "reference_roi_id", "moving_roi_id", "p_same_raw", "p_same", "parity", "consistent"}, True)
            self.assertEqual(set(consistency.parity) <= {"fit", "held_out"}, True)
            self.assertGreater(len(consistency), 0)
        self.assertEqual(summary["expected_false_matches"] < 2.0, True)
        by_cat = provenance.groupby(["session_id", "category"]).size()
        self.assertGreaterEqual(by_cat.get(("day_a", "rescued_classifier_override"), 0), 2)
        # dropped cells 0..2 on day_b are present in the image but unsegmented
        dropped = provenance.loc[provenance["session_id"].eq("day_b") & provenance["reference_roi_id"].isin([0, 1, 2])]
        self.assertTrue((dropped["category"] == "present_but_undetected").all(), dropped.to_string())
        self.assertEqual(summary["full_coverage_plus_backprojection"], 30)


if __name__ == "__main__":
    unittest.main()
