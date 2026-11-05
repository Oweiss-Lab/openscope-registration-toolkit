"""Canonical contracts for the calcium-imaging toolbox.

The core contract is :class:`SessionStructuralData`: one imaging session's
structural evidence for cross-session registration — a 2-D structural
projection plus per-ROI weighted spatial footprints with stable ROI IDs.

Contract rules
--------------
- One sample = one ROI footprint (2-D weighted mask, same shape as projection).
- ``roi_ids`` are stable within the source session and unique.
- ``pixel_size_um`` is required; all reported distances are in micrometers.
- No NaN/inf anywhere; footprints must be non-negative with nonzero area.
- Metadata is free-form but ``metadata["depth_um"]`` is recommended for
  Z-drift diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SessionStructuralData:
    session_id: str
    projection: np.ndarray  # (H, W) float structural projection
    footprints: np.ndarray  # (n_rois, H, W) weighted masks, >= 0
    roi_ids: np.ndarray  # (n_rois,) int, unique, stable in-source IDs
    pixel_size_um: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "SessionStructuralData":
        projection = np.asarray(self.projection, dtype=float)
        footprints = np.asarray(self.footprints, dtype=float)
        roi_ids = np.asarray(self.roi_ids, dtype=int)
        if projection.ndim != 2:
            raise ValueError(f"{self.session_id}: projection must be 2-D, got {projection.shape}")
        if footprints.ndim != 3 or footprints.shape[1:] != projection.shape:
            raise ValueError(
                f"{self.session_id}: footprints must be (n, {projection.shape[0]}, {projection.shape[1]}), got {footprints.shape}"
            )
        if len(roi_ids) != len(footprints):
            raise ValueError(f"{self.session_id}: {len(roi_ids)} roi_ids vs {len(footprints)} footprints")
        if len(np.unique(roi_ids)) != len(roi_ids):
            raise ValueError(f"{self.session_id}: duplicate roi_ids")
        if not np.isfinite(projection).all() or not np.isfinite(footprints).all():
            raise ValueError(f"{self.session_id}: non-finite values")
        if (footprints < 0).any():
            raise ValueError(f"{self.session_id}: negative footprint weights")
        areas = (footprints > 0).sum(axis=(1, 2))
        if (areas == 0).any():
            raise ValueError(f"{self.session_id}: {int((areas == 0).sum())} empty footprints")
        if not (float(self.pixel_size_um) > 0):
            raise ValueError(f"{self.session_id}: pixel_size_um must be positive")
        self.projection = projection
        self.footprints = footprints
        self.roi_ids = roi_ids
        self.pixel_size_um = float(self.pixel_size_um)
        return self

    @property
    def n_rois(self) -> int:
        return len(self.roi_ids)

    def footprint_equivalent_diameters_um(self) -> np.ndarray:
        """Equivalent-circle diameter per footprint, in micrometers."""
        areas_px = (self.footprints > 0).sum(axis=(1, 2)).astype(float)
        return 2.0 * np.sqrt(areas_px / np.pi) * self.pixel_size_um


@dataclass
class RigidTransform:
    """Rigid alignment of a moving session onto a reference session."""

    angle_deg: float
    shift_y_px: float
    shift_x_px: float
    score: float = float("nan")  # NCC on overlap support after alignment

    def as_dict(self) -> dict[str, float]:
        return {
            "angle_deg": float(self.angle_deg),
            "shift_y_px": float(self.shift_y_px),
            "shift_x_px": float(self.shift_x_px),
            "score": float(self.score),
        }
