"""Data-gated regression: the packaged `register` stage reproduces the original OpenScope pilot outputs.

Skipped unless both of these exist (set via environment variables):

  OSREG_PILOT_ROOT   directory of the original pilot run (registration_transforms.csv,
                     roi_match_candidates.csv, candidate_track_coverage.csv)
  OSREG_NEW_ROOT     directory written by ``osreg register`` on the same sessions/planes/params

The check is exact for the rigid transforms (same functions, same inputs) and for
the candidate pair set; summary columns are compared to float tolerance.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = os.environ.get("OSREG_PILOT_ROOT")
NEW = os.environ.get("OSREG_NEW_ROOT")


@unittest.skipUnless(PILOT and NEW and Path(PILOT).exists() and Path(NEW).exists(), "set OSREG_PILOT_ROOT and OSREG_NEW_ROOT")
class PilotReproductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pilot, self.new = Path(PILOT), Path(NEW)
        self.planes = sorted(pd.read_csv(self.new / "registration_transforms.csv")["plane"].unique())

    def _both(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        a = pd.read_csv(self.pilot / name)
        b = pd.read_csv(self.new / name)
        a = a.loc[a["plane"].isin(self.planes)]
        return a, b

    def test_transforms_identical(self) -> None:
        a, b = self._both("registration_transforms.csv")
        key = ["plane", "session_id"]
        a, b = a.set_index(key).sort_index(), b.set_index(key).sort_index()
        self.assertEqual(list(a.index), list(b.index))
        for col in ["angle_deg", "shift_y_px", "shift_x_px", "rigid_ncc", "identity_ncc",
                    "reverse_angle_deg", "reverse_shift_y_px", "reverse_shift_x_px", "n_full_tiles"]:
            np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=1e-6, err_msg=col)
        # The original pilot's per-tile residual used an integer-only phase correlation; the package uses the
        # same subpixel estimator as the global fit. Agreement is therefore expected only to +-1 px per tile.
        np.testing.assert_allclose(a["residual_median_px"].to_numpy(float), b["residual_median_px"].to_numpy(float), atol=1.0)

    def test_candidate_pairs_identical(self) -> None:
        a, b = self._both("roi_match_candidates.csv")
        flag_a = "pilot_candidate" if "pilot_candidate" in a else "candidate"
        key = ["plane", "comparison", "moving_session_id", "reference_session_id", "reference_roi_id", "moving_roi_id"]
        sa = set(map(tuple, a.loc[a[flag_a], key].itertuples(index=False)))
        sb = set(map(tuple, b.loc[b["candidate"], key].itertuples(index=False)))
        self.assertEqual(sa, sb, f"{len(sa ^ sb)} candidate pairs differ")
        m = a.merge(b, on=key, suffixes=("_a", "_b"))
        self.assertEqual(len(m), len(a))
        np.testing.assert_allclose(m["mask_iou_a"], m["mask_iou_b"], atol=1e-9)
        np.testing.assert_allclose(m["centroid_distance_px_a"], m["centroid_distance_px_b"], atol=1e-6)

    def test_track_coverage_identical(self) -> None:
        a, b = self._both("candidate_track_coverage.csv")
        key = ["plane", "reference_roi_id"]
        a, b = a.set_index(key).sort_index(), b.set_index(key).sort_index()
        self.assertEqual(list(a.index), list(b.index))
        for col in ["n_sessions_covered", "covers_all_sessions", "n_adjacent_checks", "n_adjacent_consistent"]:
            self.assertTrue((a[col].to_numpy() == b[col].to_numpy()).all(), col)
        np.testing.assert_allclose(a["median_iou_to_reference"], b["median_iou_to_reference"], atol=1e-9)
        roi_cols = [c for c in a.columns if c.startswith("roi_id__")]
        self.assertTrue(roi_cols and set(roi_cols) <= set(b.columns))
        self.assertTrue((a[roi_cols].to_numpy() == b[roi_cols].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
