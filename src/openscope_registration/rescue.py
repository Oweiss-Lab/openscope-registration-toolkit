"""Rescue actions for detection-driven track loss (source-agnostic).

Two mechanisms for a reference cell unmatched on a day:
1. Classifier-override matching: re-run one-to-one assignment against the day's
   remaining non-primary-class ROIs (e.g., ROIs the soma classifier rejected).
2. Footprint back-projection: test image evidence for the cell at its expected
   location on the aligned day projection, against a shuffled-location null.

Absence of detection is not absence of the cell; every rescue carries an
explicit provenance flag and all thresholds are provisional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .matching import assign_matches, footprints_to_sparse, iou_matrix, sparse_centroids


def classifier_override_rescue(
    reference_footprints: np.ndarray,
    reference_ids: np.ndarray,
    unmatched_reference_ids: np.ndarray,
    warped_moving_footprints: np.ndarray,
    moving_ids: np.ndarray,
    eligible_moving_ids: np.ndarray,
    *,
    candidate_iou: float,
    max_centroid_px: float,
    min_iou_margin: float,
    overlap_iou: float,
) -> pd.DataFrame:
    """One-to-one assignment restricted to unmatched reference cells x eligible moving ROIs.

    ``eligible_moving_ids`` are typically the day's non-primary-class ROIs that were
    not already claimed by the base matching. Returns the same schema as
    :func:`assign_matches` plus ``rescue_mechanism = 'classifier_override'``.
    """
    ref_index = {int(r): i for i, r in enumerate(reference_ids)}
    mov_index = {int(m): i for i, m in enumerate(moving_ids)}
    ref_rows = [ref_index[int(r)] for r in unmatched_reference_ids if int(r) in ref_index]
    mov_rows = [mov_index[int(m)] for m in eligible_moving_ids if int(m) in mov_index]
    if not ref_rows or not mov_rows:
        empty = assign_matches(np.zeros((0, 0)), np.array([]), np.array([]), np.zeros((0, 2)), np.zeros((0, 2)),
                               candidate_iou=candidate_iou, max_centroid_px=max_centroid_px,
                               min_iou_margin=min_iou_margin, overlap_iou=overlap_iou)
        empty["rescue_mechanism"] = pd.Series(dtype=str)
        return empty
    ref_sub = reference_footprints[ref_rows]
    mov_sub = warped_moving_footprints[mov_rows]
    ref_sparse, mov_sparse = footprints_to_sparse(ref_sub), footprints_to_sparse(mov_sub)
    shape = reference_footprints.shape[1:]
    matches = assign_matches(
        iou_matrix(ref_sparse, mov_sparse),
        reference_ids[ref_rows], moving_ids[mov_rows],
        sparse_centroids(ref_sparse, shape), sparse_centroids(mov_sparse, shape),
        candidate_iou=candidate_iou, max_centroid_px=max_centroid_px,
        min_iou_margin=min_iou_margin, overlap_iou=overlap_iou,
    )
    matches["rescue_mechanism"] = "classifier_override"
    return matches


def footprint_annulus_contrast(image: np.ndarray, footprint: np.ndarray, annulus_px: tuple[int, int] = (2, 6)) -> float:
    """Mean image value inside the footprint minus mean in a surrounding annulus."""
    from scipy import ndimage

    core = footprint > 0
    if not core.any():
        return float("nan")
    inner = ndimage.binary_dilation(core, iterations=annulus_px[0])
    outer = ndimage.binary_dilation(core, iterations=annulus_px[1])
    ring = outer & ~inner
    if not ring.any():
        return float("nan")
    return float(image[core].mean() - image[ring].mean())


def backprojection_evidence(
    aligned_projection: np.ndarray,
    reference_footprints: np.ndarray,
    reference_ids: np.ndarray,
    unmatched_reference_ids: np.ndarray,
    *,
    n_null: int = 200,
    annulus_px: tuple[int, int] = (2, 6),
    evidence_z: float = 3.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Image evidence for each unmatched reference cell at its expected location.

    The day's projection is already warped into reference coordinates, so the
    reference footprint is evaluated in place. The null re-evaluates the same
    footprint at random in-bounds translations. ``evidence_z`` is a provisional
    threshold, recorded with every row.
    """
    rng = np.random.default_rng(seed)
    ref_index = {int(r): i for i, r in enumerate(reference_ids)}
    height, width = aligned_projection.shape
    rows = []
    for ref_id in np.asarray(unmatched_reference_ids, dtype=int):
        footprint = reference_footprints[ref_index[ref_id]]
        ys, xs = np.nonzero(footprint > 0)
        real = footprint_annulus_contrast(aligned_projection, footprint, annulus_px)
        margin_y = (int(ys.min()), height - 1 - int(ys.max()))
        margin_x = (int(xs.min()), width - 1 - int(xs.max()))
        null_values = []
        for _ in range(n_null):
            dy = int(rng.integers(-margin_y[0], margin_y[1] + 1))
            dx = int(rng.integers(-margin_x[0], margin_x[1] + 1))
            shifted = np.zeros_like(footprint)
            shifted[ys + dy, xs + dx] = footprint[ys, xs]
            null_values.append(footprint_annulus_contrast(aligned_projection, shifted, annulus_px))
        null_arr = np.asarray(null_values, dtype=float)
        null_std = float(np.nanstd(null_arr, ddof=1))
        z = float((real - np.nanmean(null_arr)) / null_std) if null_std > 0 else float("nan")
        rows.append(
            {
                "reference_roi_id": ref_id,
                "contrast": real,
                "null_mean": float(np.nanmean(null_arr)),
                "null_std": null_std,
                "evidence_z": z,
                "evidence_z_threshold": evidence_z,
                "label": "present_but_undetected" if np.isfinite(z) and z >= evidence_z else "absent_ambiguous",
            }
        )
    return pd.DataFrame(rows, columns=["reference_roi_id", "contrast", "null_mean", "null_std", "evidence_z", "evidence_z_threshold", "label"])


PROVENANCE_CATEGORIES = ["matched", "rescued_classifier_override", "present_but_undetected", "absent_ambiguous"]


def provenance_table(
    reference_ids: np.ndarray,
    session_order: list[str],
    accepted_base: dict[str, set[int]],
    accepted_rescue: dict[str, set[int]],
    evidence_by_session: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Per (reference cell, session) provenance category."""
    rows = []
    for ref_id in np.asarray(reference_ids, dtype=int):
        for session_id in session_order:
            if ref_id in accepted_base.get(session_id, set()):
                category = "matched"
            elif ref_id in accepted_rescue.get(session_id, set()):
                category = "rescued_classifier_override"
            else:
                evidence = evidence_by_session.get(session_id)
                label = "absent_ambiguous"
                if evidence is not None and len(evidence):
                    hit = evidence.loc[evidence["reference_roi_id"].eq(ref_id), "label"]
                    if len(hit):
                        label = str(hit.iloc[0])
                category = label
            rows.append({"reference_roi_id": int(ref_id), "session_id": session_id, "category": category})
    return pd.DataFrame(rows)
