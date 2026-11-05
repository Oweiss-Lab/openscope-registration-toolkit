"""Registration diagnostics: centroid distances, correlation vs shuffled null, per-cell residual displacement and its per-session summaries, local-geometry permutation test, coverage, size change, and frozen-manifest consumption.

Implements the plot-battery statistics from the calcium-imaging literature:
centroid-distance distributions, weighted footprint-correlation distributions
with shuffled nulls, per-cell residual displacement, local-geometry change
with permutation nulls, day-coverage summaries, and footprint-size change
(Z-drift probe). All distances are reported in micrometers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .matching import weighted_footprint_correlation


def centroid_distance_table(matches: pd.DataFrame, pixel_size_um: float, session_id: str) -> pd.DataFrame:
    out = matches.loc[matches["candidate"], ["reference_roi_id", "moving_roi_id", "centroid_distance_px", "mask_iou"]].copy()
    out["centroid_distance_um"] = out["centroid_distance_px"] * pixel_size_um
    out.insert(0, "session_id", session_id)
    return out


def footprint_correlation_table(
    reference_footprints: np.ndarray,
    warped_moving_footprints: np.ndarray,
    matches: pd.DataFrame,
    reference_index: dict[int, int],
    moving_index: dict[int, int],
    session_id: str,
    *,
    n_null: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Weighted footprint correlations for candidate pairs plus a shuffled-pair null."""
    rng = np.random.default_rng(seed)
    rows = []
    candidates = matches.loc[matches["candidate"]]
    for _, m in candidates.iterrows():
        ref = reference_footprints[reference_index[int(m["reference_roi_id"])]]
        mov = warped_moving_footprints[moving_index[int(m["moving_roi_id"])]]
        rows.append({"session_id": session_id, "kind": "match", "footprint_correlation": weighted_footprint_correlation(ref, mov)})
    n_ref, n_mov = len(reference_footprints), len(warped_moving_footprints)
    for _ in range(min(n_null, n_ref * n_mov)):
        r, c = int(rng.integers(n_ref)), int(rng.integers(n_mov))
        rows.append(
            {"session_id": session_id, "kind": "shuffled", "footprint_correlation": weighted_footprint_correlation(reference_footprints[r], warped_moving_footprints[c])}
        )
    return pd.DataFrame(rows)


def residual_displacement_table(
    matches: pd.DataFrame,
    reference_centroids: dict[int, np.ndarray],
    warped_moving_centroids: dict[int, np.ndarray],
    pixel_size_um: float,
    session_id: str,
) -> pd.DataFrame:
    """Per-matched-cell residual displacement (warped moving centroid minus reference centroid)."""
    rows = []
    for _, m in matches.loc[matches["candidate"]].iterrows():
        ref = reference_centroids[int(m["reference_roi_id"])]
        mov = warped_moving_centroids[int(m["moving_roi_id"])]
        rows.append(
            {
                "session_id": session_id,
                "reference_roi_id": int(m["reference_roi_id"]),
                "ref_y_px": float(ref[0]), "ref_x_px": float(ref[1]),
                "residual_y_um": float((mov[0] - ref[0]) * pixel_size_um),
                "residual_x_um": float((mov[1] - ref[1]) * pixel_size_um),
                "residual_um": float(np.hypot(mov[0] - ref[0], mov[1] - ref[1]) * pixel_size_um),
            }
        )
    return pd.DataFrame(rows)


def local_geometry_change(
    reference_centroids_by_session: list[np.ndarray],
    warped_moving_centroids_by_session: list[np.ndarray],
    pixel_size_um: float,
    *,
    n_neighbors: int = 4,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """Neighbor-pair distance/angle changes for matched cells vs a permutation null.

    Inputs are per-session aligned centroid arrays (row i of both arrays in one
    session is the same claimed cell). Neighborhoods and permutations are built
    within each session, then pooled. The null permutes moving identities within
    each session; if real matches are no better than the null, matching is
    indistinguishable from chance.
    """
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(seed)
    neighbor_sets = []
    for reference_centroids in reference_centroids_by_session:
        n = len(reference_centroids)
        if n < n_neighbors + 1:
            neighbor_sets.append(None)
            continue
        ref_dist = cdist(reference_centroids, reference_centroids)
        np.fill_diagonal(ref_dist, np.inf)
        neighbor_sets.append(np.argsort(ref_dist, axis=1)[:, :n_neighbors])
    if all(nb is None for nb in neighbor_sets):
        raise ValueError(f"No session has more than {n_neighbors} matches")

    def pair_changes(reference_centroids: np.ndarray, moving: np.ndarray, neighbor_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_d, delta_a = [], []
        for i in range(len(reference_centroids)):
            for j in neighbor_idx[i]:
                v_ref = reference_centroids[j] - reference_centroids[i]
                v_mov = moving[j] - moving[i]
                delta_d.append((np.linalg.norm(v_mov) - np.linalg.norm(v_ref)) * pixel_size_um)
                a = np.degrees(np.arctan2(v_mov[0], v_mov[1]) - np.arctan2(v_ref[0], v_ref[1]))
                delta_a.append((a + 180.0) % 360.0 - 180.0)
        return np.asarray(delta_d), np.asarray(delta_a)

    real_dd_parts, real_da_parts = [], []
    for ref, mov, nb in zip(reference_centroids_by_session, warped_moving_centroids_by_session, neighbor_sets):
        if nb is None:
            continue
        dd, da = pair_changes(ref, mov, nb)
        real_dd_parts.append(dd)
        real_da_parts.append(da)
    real_dd = np.concatenate(real_dd_parts)
    real_da = np.concatenate(real_da_parts)
    null_medians = np.empty(n_permutations)
    for k in range(n_permutations):
        parts = []
        for ref, mov, nb in zip(reference_centroids_by_session, warped_moving_centroids_by_session, neighbor_sets):
            if nb is None:
                continue
            dd, _ = pair_changes(ref, mov[rng.permutation(len(mov))], nb)
            parts.append(dd)
        null_medians[k] = np.median(np.abs(np.concatenate(parts)))
    real_median = float(np.median(np.abs(real_dd)))
    return {
        "delta_distance_um": real_dd,
        "delta_angle_deg": real_da,
        "real_median_abs_delta_distance_um": real_median,
        "null_median_abs_delta_distance_um": null_medians,
        "p_value": float((np.sum(null_medians <= real_median) + 1) / (n_permutations + 1)),
    }


def coverage_summary(matches_by_session: dict[str, pd.DataFrame], reference_ids: np.ndarray) -> pd.DataFrame:
    """Per-reference-cell coverage: which sessions contain its candidate match."""
    rows = []
    for ref_id in np.asarray(reference_ids, dtype=int):
        row: dict[str, object] = {"reference_roi_id": int(ref_id)}
        for session_id, matches in matches_by_session.items():
            hit = matches.loc[matches["candidate"] & matches["reference_roi_id"].eq(ref_id)]
            row[f"matched__{session_id}"] = bool(len(hit))
        row["n_sessions_matched"] = int(sum(bool(v) for k, v in row.items() if str(k).startswith("matched__")))
        rows.append(row)
    return pd.DataFrame(rows)


def footprint_size_change_table(
    reference_footprints: np.ndarray,
    warped_moving_footprints: np.ndarray,
    matches: pd.DataFrame,
    reference_index: dict[int, int],
    moving_index: dict[int, int],
    pixel_size_um: float,
    session_id: str,
) -> pd.DataFrame:
    """Equivalent-diameter ratio per matched pair (Z-drift / focal-plane probe)."""
    rows = []
    for _, m in matches.loc[matches["candidate"]].iterrows():
        ref_area = float((reference_footprints[reference_index[int(m["reference_roi_id"])]] > 0).sum())
        mov_area = float((warped_moving_footprints[moving_index[int(m["moving_roi_id"])]] > 0).sum())
        rows.append(
            {
                "session_id": session_id,
                "reference_roi_id": int(m["reference_roi_id"]),
                "reference_diameter_um": 2.0 * np.sqrt(ref_area / np.pi) * pixel_size_um,
                "diameter_ratio_moving_over_reference": float(np.sqrt(mov_area / ref_area)) if ref_area > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)



# ============================================================================
# merged from residual_summaries.py



import numpy as np
import pandas as pd


def hotelling_p(vectors: np.ndarray) -> float:
    """Two-sided Hotelling T² test of mean vector == 0 for 2-D samples (F approximation); NaN if n < 4."""
    from scipy import stats

    v = np.asarray(vectors, dtype=float)
    n = v.shape[0]
    if n < 4:
        return float("nan")
    mean = v.mean(axis=0)
    cov = np.cov(v, rowvar=False)
    try:
        t2 = n * mean @ np.linalg.solve(cov, mean)
    except np.linalg.LinAlgError:
        return float("nan")
    f = t2 * (n - 2) / (2 * (n - 1))
    return float(stats.f.sf(f, 2, n - 2))


def position_r2(vectors: np.ndarray, positions: np.ndarray) -> float:
    """Pooled R² (both components) of an affine fit ``r ~ 1 + y + x``; NaN if n < 6 or zero variance."""
    v = np.asarray(vectors, dtype=float); pos = np.asarray(positions, dtype=float)
    n = v.shape[0]
    if n < 6:
        return float("nan")
    X = np.column_stack([np.ones(n), pos - pos.mean(axis=0)])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    ss_res = float(((v - X @ beta) ** 2).sum())
    ss_tot = float(((v - v.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def summarise_residual_vectors(vectors: np.ndarray, *, unit: str = "um", positions: np.ndarray | None = None) -> dict[str, float]:
    """Scalar summary of one session's residual vectors ``(n, 2)`` ordered (y, x)."""
    v = np.asarray(vectors, dtype=float)
    n = v.shape[0]
    mean = v.mean(axis=0) if n else np.full(2, np.nan)
    se = v.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(2, np.nan)
    ms_total = float(np.mean(np.sum(v ** 2, axis=1))) if n else float("nan")
    ms_mean = float(np.sum(mean ** 2))
    out = {
        "n": int(n),
        f"mean_y_{unit}": float(mean[0]), f"mean_x_{unit}": float(mean[1]),
        f"mean_vector_magnitude_{unit}": float(np.hypot(*mean)),
        "mean_vector_angle_deg": float(np.degrees(np.arctan2(mean[0], mean[1]))),
        f"se_y_{unit}": float(se[0]), f"se_x_{unit}": float(se[1]),
        f"rms_residual_{unit}": float(np.sqrt(ms_total)),
        f"median_residual_magnitude_{unit}": float(np.median(np.hypot(v[:, 0], v[:, 1]))) if n else float("nan"),
        "systematic_fraction": ms_mean / ms_total if ms_total and ms_total > 0 else float("nan"),
        "hotelling_p": hotelling_p(v),
    }
    if positions is not None:
        out["position_r2"] = position_r2(v, positions)
    return out


def residual_vector_summary_table(residuals: pd.DataFrame, *, group_columns: tuple[str, ...] = ("session_id",)) -> pd.DataFrame:
    """One row per group of a ``residual_displacement_table`` frame (default: per session)."""
    rows = []
    for key, g in residuals.groupby(list(group_columns), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        v = g[["residual_y_um", "residual_x_um"]].to_numpy(float)
        pos = g[["ref_y_px", "ref_x_px"]].to_numpy(float) if {"ref_y_px", "ref_x_px"} <= set(g.columns) else None
        rows.append({**dict(zip(group_columns, key)), **summarise_residual_vectors(v, positions=pos)})
    return pd.DataFrame(rows)



# ============================================================================
# merged from manifest_consumption.py



import json
from pathlib import Path

import pandas as pd


def accepted_tracks_wide(manifest_root: Path, plane_slug: str, reference_session: str) -> pd.DataFrame:
    """Pivot a plane's accepted-tracks CSV to one row per track.

    Returns a DataFrame indexed by reference_roi_id with one column per
    session_id holding that session's matched ROI id; the reference session
    maps each track to itself. Raises if the plane has no accepted tracks.
    """
    csv = Path(manifest_root) / f"accepted_tracks_{plane_slug}.csv"
    long = pd.read_csv(csv)
    wide = long.pivot(index="reference_roi_id", columns="session_id", values="moving_roi_id")
    wide[reference_session] = wide.index
    if wide.empty:
        raise ValueError(f"No accepted tracks in {csv}")
    return wide.astype(int).sort_index()


def derive_retained_and_reference(
    manifest_roots: list[Path],
    eligible_sessions: set[str],
) -> tuple[list[str], str]:
    """Retained session ids (sorted) and the registration reference session.

    Retained sessions are those appearing in any accepted-tracks CSV across the
    given manifest roots. The reference session never appears there (tracks map
    to it), so it is recovered as the unique eligible session that is neither
    retained nor listed in any manifest's excluded_sessions. Raises if that
    session is not unique, or if a manifest references sessions outside the
    eligible set.
    """
    retained: set[str] = set()
    excluded: set[str] = set()
    for root in manifest_roots:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        excluded.update(manifest.get("excluded_sessions", []))
        for csv in root.glob("accepted_tracks_*.csv"):
            if csv.stat().st_size > 10:
                retained.update(pd.read_csv(csv)["session_id"].unique())
    missing = retained - eligible_sessions
    if missing:
        raise ValueError(f"Manifest sessions missing from eligible set: {sorted(missing)}")
    candidates = eligible_sessions - retained - excluded
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly 1 reference session, got {sorted(candidates)}")
    return sorted(retained | candidates), candidates.pop()
