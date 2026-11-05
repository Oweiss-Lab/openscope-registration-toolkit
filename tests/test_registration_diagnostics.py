"""Synthetic end-to-end tests for the registration diagnostics workflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

from openscope_registration.matching import (
    assign_matches,
    footprints_to_sparse,
    iou_matrix,
    sparse_centroids,
    weighted_footprint_correlation,
)
from openscope_registration.rigid import (
    preprocess_projection,
    register_rigid,
    warp_image,
)
from openscope_registration.data import RigidTransform, SessionStructuralData
from openscope_registration.pipeline import run_registration_diagnostics


def make_session(session_id: str, *, angle_deg: float = 0.0, shift: tuple[float, float] = (0.0, 0.0), seed: int = 0, drop: int = 0, size: int = 192) -> SessionStructuralData:
    rng = np.random.default_rng(7)  # same cell layout for every session
    grid = np.array([(y, x) for y in range(30, size - 20, 26) for x in range(30, size - 20, 26)])
    centers = grid[rng.permutation(len(grid))[:30]] + rng.integers(-4, 5, size=(30, 2))  # jittered, non-overlapping
    projection = np.zeros((size, size))
    footprints = []
    yy, xx = np.mgrid[:size, :size]
    for cy, cx in centers:
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 16).astype(float)
        footprints.append(mask)
        projection += mask * rng.uniform(0.5, 1.5)
    projection = ndimage.gaussian_filter(projection, 1.0)
    noise_rng = np.random.default_rng(seed)
    projection = projection + 0.02 * noise_rng.standard_normal(projection.shape)
    footprints_arr = np.asarray(footprints)
    if angle_deg or shift != (0.0, 0.0):
        projection = warp_image(projection, angle_deg, shift, order=1)
        footprints_arr = np.asarray([warp_image(m, angle_deg, shift, order=0) for m in footprints_arr])
    keep = slice(drop, None)
    return SessionStructuralData(
        session_id=session_id,
        projection=projection,
        footprints=np.clip(footprints_arr[keep], 0, None),
        roi_ids=np.arange(len(footprints))[keep],
        pixel_size_um=0.78,
    ).validate()


class ContractTests(unittest.TestCase):
    def test_validation_rejects_duplicate_ids_and_empty_masks(self) -> None:
        session = make_session("s")
        bad = SessionStructuralData("b", session.projection, session.footprints, np.zeros(len(session.roi_ids), dtype=int), 0.78)
        with self.assertRaises(ValueError):
            bad.validate()
        empty = session.footprints.copy()
        empty[0] = 0.0
        with self.assertRaises(ValueError):
            SessionStructuralData("e", session.projection, empty, session.roi_ids, 0.78).validate()


class RigidTests(unittest.TestCase):
    def test_recovers_known_transform(self) -> None:
        reference = make_session("ref")
        moving = make_session("mov", angle_deg=-1.5, shift=(5.0, -3.0), seed=1)
        transform = register_rigid(preprocess_projection(reference.projection), preprocess_projection(moving.projection))
        self.assertAlmostEqual(transform.angle_deg, 1.5, delta=0.3)
        self.assertGreater(transform.score, 0.8)


class MatchingTests(unittest.TestCase):
    def test_identity_matching_is_perfect_and_correlation_separates(self) -> None:
        session = make_session("s")
        sp = footprints_to_sparse(session.footprints)
        centroids = sparse_centroids(sp, session.projection.shape)
        matches = assign_matches(
            iou_matrix(sp, sp), session.roi_ids, session.roi_ids, centroids, centroids,
            candidate_iou=0.5, max_centroid_px=4.0, min_iou_margin=0.2, overlap_iou=0.2,
        )
        self.assertTrue(matches["candidate"].all())
        self.assertTrue((matches["reference_roi_id"] == matches["moving_roi_id"]).all())
        same = weighted_footprint_correlation(session.footprints[0], session.footprints[0])
        different = weighted_footprint_correlation(session.footprints[0], session.footprints[1])
        self.assertAlmostEqual(same, 1.0, places=6)
        self.assertLess(different, 0.3)


class WorkflowTests(unittest.TestCase):
    def test_full_battery_on_synthetic_sessions(self) -> None:
        reference = make_session("ref")
        moving = [
            make_session("day_a", angle_deg=1.0, shift=(-4.0, 2.0), seed=2),
            make_session("day_b", angle_deg=-0.5, shift=(3.0, 3.0), seed=3, drop=5),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_registration_diagnostics(
                reference, moving, Path(tmp) / "num", Path(tmp) / "fig",
                n_null=50, n_permutations=100, title_context="synthetic test", dpi=72,
            )
            for path in summary["output_files"].values():
                self.assertTrue(Path(path).exists(), path)
        self.assertGreaterEqual(summary["n_full_coverage_cells"], 20)
        self.assertLess(summary["median_match_distance_um"], 2.0)
        self.assertGreater(summary["median_footprint_correlation_match"], 0.8)
        self.assertLess(summary["median_footprint_correlation_shuffled"], 0.3)
        self.assertLess(summary["local_geometry_p_value"], 0.05)

    def test_supplied_transforms_are_used_verbatim(self) -> None:
        reference = make_session("ref")
        moving = [make_session("day_a", angle_deg=1.0, shift=(-4.0, 2.0), seed=2)]
        supplied = {"day_a": RigidTransform(angle_deg=-1.0, shift_y_px=4.0, shift_x_px=-2.0, score=0.9)}
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_registration_diagnostics(
                reference, moving, Path(tmp) / "num", Path(tmp) / "fig",
                transforms=supplied, n_null=20, n_permutations=50, dpi=72,
            )
        self.assertEqual(summary["transforms"]["day_a"]["angle_deg"], -1.0)


if __name__ == "__main__":
    unittest.main()
