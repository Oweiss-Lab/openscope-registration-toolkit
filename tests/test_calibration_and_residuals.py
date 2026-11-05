"""Tests for calibration (transitive consistency, reliability bins) and residual-summary actions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openscope_registration.psame import (
    consistency_pairs_table,
    isotonic_curve_from_pairs,
    reliability_bins,
    transitive_consistency_flags,
)
from openscope_registration.diagnostics import (
    position_r2,
    residual_vector_summary_table,
    summarise_residual_vectors,
)
from openscope_registration.plotting import draw_residual_displacement_panel
from openscope_registration.plotting import plot_reliability_diagram


def _pairs():
    # ref 0: a->10, b->20; ref 1: a->11, b->21; ref 2 only in a
    return pd.DataFrame({
        "session_id": ["a", "b", "a", "b", "a"],
        "reference_roi_id": [0, 0, 1, 1, 2],
        "moving_roi_id": [10, 20, 11, 21, 12],
        "candidate": [True] * 5,
        "p_same_raw": [0.9, 0.8, 0.7, 0.6, 0.5],
        "p_same": [0.95, 0.9, 0.85, 0.8, 0.7],
    })


class ConsistencyTests(unittest.TestCase):
    def test_flags_pass_fail_and_na(self) -> None:
        pairs = _pairs()
        adjacent = {("a", "b"): {(10, 20), (11, 99)}}  # ref 0 confirmed, ref 1 contradicted
        flags = transitive_consistency_flags(pairs, adjacent)
        self.assertEqual(flags.loc[0], True)
        self.assertEqual(flags.loc[1], True)
        self.assertEqual(flags.loc[2], False)
        self.assertEqual(flags.loc[3], False)
        self.assertTrue(pd.isna(flags.loc[4]))  # ref 2 has no partner on b -> no check
        table = consistency_pairs_table(pairs, flags, fit_sessions={"a"})
        self.assertEqual(len(table), 4)
        self.assertEqual(set(table.parity), {"fit", "held_out"})
        self.assertEqual(table.consistent.sum(), 2)

    def test_reliability_bins_equal_count_and_fixed(self) -> None:
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 1000)
        y = (rng.uniform(0, 1, 1000) < s).astype(float)  # perfectly calibrated
        eq = reliability_bins(s, y, n_bins=10)
        self.assertEqual(len(eq), 10)
        self.assertTrue((eq["n"].between(90, 110)).all())
        self.assertLess(np.abs(eq["fraction"] - eq["score_mean"]).mean(), 0.08)
        fx = reliability_bins(s, y, n_bins=5, equal_count=False)
        self.assertEqual(len(fx), 5)

    def test_isotonic_curve_reconstructs_monotone_map(self) -> None:
        pairs = pd.DataFrame({"p_same_raw": [0.1, 0.5, 0.9], "p_same": [0.3, 0.6, 0.95]})
        curve = isotonic_curve_from_pairs(pairs, np.array([0.1, 0.3, 0.9]))
        np.testing.assert_allclose(curve, [0.3, 0.45, 0.95])


class ResidualSummaryTests(unittest.TestCase):
    def test_pure_translation_has_systematic_fraction_one(self) -> None:
        v = np.tile([0.5, -0.3], (50, 1))
        out = summarise_residual_vectors(v)
        self.assertAlmostEqual(out["systematic_fraction"], 1.0)
        self.assertAlmostEqual(out["mean_vector_magnitude_um"], np.hypot(0.5, 0.3))

    def test_pure_scatter_has_small_systematic_fraction_and_position_r2(self) -> None:
        rng = np.random.default_rng(1)
        v = rng.normal(0, 1, (400, 2)); pos = rng.uniform(0, 512, (400, 2))
        out = summarise_residual_vectors(v, positions=pos)
        self.assertLess(out["systematic_fraction"], 0.05)
        self.assertLess(out["position_r2"], 0.05)
        self.assertGreater(out["hotelling_p"], 0.01)

    def test_affine_field_has_high_position_r2(self) -> None:
        rng = np.random.default_rng(2)
        pos = rng.uniform(0, 512, (300, 2))
        v = 0.004 * (pos - 256) + rng.normal(0, 0.05, (300, 2))  # 0.4% scale gradient
        self.assertGreater(position_r2(v, pos), 0.9)

    def test_summary_table_groups_by_session(self) -> None:
        df = pd.DataFrame({"session_id": ["a"] * 10 + ["b"] * 10, "ref_y_px": np.arange(20.0), "ref_x_px": np.arange(20.0),
                           "residual_y_um": np.r_[np.ones(10), -np.ones(10)], "residual_x_um": np.zeros(20),
                           "residual_um": np.ones(20)})
        t = residual_vector_summary_table(df).set_index("session_id")
        self.assertAlmostEqual(t.loc["a", "mean_y_um"], 1.0)
        self.assertAlmostEqual(t.loc["b", "mean_y_um"], -1.0)
        self.assertIn("position_r2", t.columns)


class VisualTests(unittest.TestCase):
    def test_residual_panel_arrows_point_in_true_image_direction(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        res = pd.DataFrame({"ref_y_px": [100.0], "ref_x_px": [100.0], "residual_y_um": [1.0], "residual_x_um": [0.0], "residual_um": [1.0]})
        fig, ax = plt.subplots()
        q = draw_residual_displacement_panel(ax, res, (256, 256), mode="arrows", pixel_size_um=1.0, arrow_gain_px_per_um=10.0)
        self.assertAlmostEqual(float(q.V[0]), 1.0)  # +y (downward in image axes), not negated
        self.assertEqual(ax.get_ylim(), (256.0, 0.0))
        plt.close(fig)

    def test_reliability_diagram_writes_file(self) -> None:
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 500); y = (rng.uniform(0, 1, 500) < s).astype(float)
        series = {"fit": reliability_bins(s[:250], y[:250], n_bins=5), "held_out": reliability_bins(s[250:], y[250:], n_bins=5)}
        grid = np.linspace(0, 1, 50)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rel.png"
            plot_reliability_diagram(series, out, title="synthetic", curves=np.vstack([grid, grid ** 0.8]), grid=grid, dpi=60)
            self.assertTrue(out.exists() and out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
