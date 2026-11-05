"""Pixel-mask matching: IoU tables, Hungarian one-to-one assignment with provisional gates, weighted mask correlation, and template-anchored candidate tracks.

One-to-one Hungarian assignment on footprint IoU with centroid, uniqueness,
and split/merge gates. Thresholds are caller-supplied and must be treated as
preparation-specific (never universal pass/fail values).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment


def footprints_to_sparse(footprints: np.ndarray) -> sparse.csr_matrix:
    return sparse.csr_matrix((np.asarray(footprints) > 0).reshape(len(footprints), -1), dtype=np.float32)


def sparse_centroids(matrix: sparse.csr_matrix, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = matrix.nonzero()
    ys, xs = np.divmod(cols, shape[1])
    counts = np.bincount(rows, minlength=matrix.shape[0]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cy = np.bincount(rows, weights=ys, minlength=matrix.shape[0]) / counts
        cx = np.bincount(rows, weights=xs, minlength=matrix.shape[0]) / counts
    return np.column_stack([cy, cx])


def iou_matrix(reference: sparse.csr_matrix, moving: sparse.csr_matrix) -> np.ndarray:
    intersection = (reference @ moving.T).toarray()
    ref_area = np.asarray(reference.sum(axis=1)).ravel()
    mov_area = np.asarray(moving.sum(axis=1)).ravel()
    union = ref_area[:, None] + mov_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def weighted_footprint_correlation(reference: np.ndarray, moving: np.ndarray, pad_px: int = 3) -> float:
    """Pearson correlation of two weighted footprints over their padded union bounding box."""
    support = (reference > 0) | (moving > 0)
    if not support.any():
        return float("nan")
    ys, xs = np.nonzero(support)
    y0, y1 = max(ys.min() - pad_px, 0), min(ys.max() + pad_px + 1, reference.shape[0])
    x0, x1 = max(xs.min() - pad_px, 0), min(xs.max() + pad_px + 1, reference.shape[1])
    a = reference[y0:y1, x0:x1].astype(float).ravel()
    b = moving[y0:y1, x0:x1].astype(float).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def _second_largest(values: np.ndarray, axis: int) -> np.ndarray:
    if values.shape[axis] < 2:
        return np.zeros(values.shape[1 - axis])
    return -np.partition(-values, 1, axis=axis).take(1, axis=axis)


def assign_matches(
    iou: np.ndarray,
    reference_ids: np.ndarray,
    moving_ids: np.ndarray,
    reference_centroids: np.ndarray,
    moving_centroids: np.ndarray,
    *,
    candidate_iou: float,
    max_centroid_px: float,
    min_iou_margin: float,
    overlap_iou: float,
) -> pd.DataFrame:
    """One-to-one Hungarian assignment on IoU with ambiguity and split/merge flags.

    ``candidate`` marks pairs passing the caller's provisional thresholds; these
    are candidate identities only, never accepted matches.
    """
    columns = [
        "reference_roi_id", "moving_roi_id", "mask_iou", "centroid_distance_px",
        "reference_second_best_iou", "moving_second_best_iou", "iou_margin",
        "n_moving_overlapping_reference", "n_reference_overlapping_moving",
        "possible_split", "possible_merge", "candidate",
    ]
    if iou.size == 0:
        return pd.DataFrame(columns=columns)
    rows_idx, cols_idx = linear_sum_assignment(-iou)
    ref_second = _second_largest(iou, axis=1)
    mov_second = _second_largest(iou, axis=0)
    overlaps = iou >= overlap_iou
    out = []
    for r, c in zip(rows_idx, cols_idx):
        value = float(iou[r, c])
        if value <= 0:
            continue
        distance = float(np.hypot(*(reference_centroids[r] - moving_centroids[c])))
        margin = value - max(float(ref_second[r]), float(mov_second[c]))
        n_mov = int(overlaps[r].sum())
        n_ref = int(overlaps[:, c].sum())
        out.append(
            {
                "reference_roi_id": int(reference_ids[r]),
                "moving_roi_id": int(moving_ids[c]),
                "mask_iou": value,
                "centroid_distance_px": distance,
                "reference_second_best_iou": float(ref_second[r]),
                "moving_second_best_iou": float(mov_second[c]),
                "iou_margin": float(margin),
                "n_moving_overlapping_reference": n_mov,
                "n_reference_overlapping_moving": n_ref,
                "possible_split": n_mov > 1,
                "possible_merge": n_ref > 1,
                "candidate": bool(
                    value >= candidate_iou and distance <= max_centroid_px and margin >= min_iou_margin and n_mov <= 1 and n_ref <= 1
                ),
            }
        )
    return pd.DataFrame(out, columns=columns)



# ============================================================================
# merged from tracks.py



import numpy as np
import pandas as pd


def session_tag(session_id: str) -> str:
    """Stable short tag used for per-session ROI-id columns (text after the second underscore, else the id)."""
    parts = session_id.split("_", 2)
    return parts[-1] if len(parts) == 3 else session_id


def build_tracks(
    reference_ids: np.ndarray,
    sessions: list[str],
    reference_session: str,
    to_reference: dict[str, pd.DataFrame],
    adjacent: dict[tuple[str, str], pd.DataFrame],
    *,
    candidate_column: str = "candidate",
) -> pd.DataFrame:
    """One row per reference ROI: coverage, adjacent-day consistency, IoU summary, per-session member ids.

    ``to_reference[session]`` and ``adjacent[(a, b)]`` are ``assign_matches`` tables; only rows
    where ``candidate_column`` is True contribute members. Per-session member columns are
    ``roi_id__<session_tag>`` and hold ``-1`` where the track has no member.
    """
    reference_ids = np.asarray(reference_ids, dtype=int)
    members: dict[str, dict[int, int]] = {reference_session: {int(r): int(r) for r in reference_ids}}
    for session, matches in to_reference.items():
        passed = matches.loc[matches[candidate_column].astype(bool)]
        members[session] = dict(zip(passed["reference_roi_id"].astype(int), passed["moving_roi_id"].astype(int)))
    adjacent_pairs = {
        key: set(zip(df.loc[df[candidate_column].astype(bool), "reference_roi_id"].astype(int),
                     df.loc[df[candidate_column].astype(bool), "moving_roi_id"].astype(int)))
        for key, df in adjacent.items()
    }
    iou_lookup = {s: df.set_index("reference_roi_id")["mask_iou"] for s, df in to_reference.items()}
    rows = []
    for ref_id in reference_ids:
        ref_id = int(ref_id)
        covered = [s for s in sessions if ref_id in members.get(s, {})]
        checks = consistent = 0
        for a, b in zip(sessions[:-1], sessions[1:]):
            if ref_id in members.get(a, {}) and ref_id in members.get(b, {}):
                checks += 1
                consistent += int((members[a][ref_id], members[b][ref_id]) in adjacent_pairs.get((a, b), set()))
        ious = [float(iou_lookup[s].loc[ref_id]) for s in covered if s != reference_session and ref_id in iou_lookup[s].index]
        rows.append(
            {
                "reference_roi_id": ref_id,
                "n_sessions_covered": len(covered),
                "covers_all_sessions": len(covered) == len(sessions),
                "n_adjacent_checks": checks,
                "n_adjacent_consistent": consistent,
                "adjacent_consistency_fraction": consistent / checks if checks else np.nan,
                "min_iou_to_reference": min(ious) if ious else np.nan,
                "median_iou_to_reference": float(np.median(ious)) if ious else np.nan,
                **{f"roi_id__{session_tag(s)}": members.get(s, {}).get(ref_id, -1) for s in sessions},
            }
        )
    return pd.DataFrame(rows)
