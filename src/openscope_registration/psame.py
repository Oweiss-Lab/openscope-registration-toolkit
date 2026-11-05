"""Same-cell probability: two-component mixture on (mask correlation, log1p centroid distance), adjacent-day transitive consistency, isotonic recalibration, and reliability bins.

Fits a 2-D Gaussian mixture over (footprint correlation, log1p centroid distance)
for one-to-one assigned candidate pairs, with components anchored by
high-confidence same-cell pairs and clear different-cell pairs, refined by EM.
Posteriors replace fixed accept thresholds; the expected false-match count of an
accepted set is sum(1 - P_same).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PSameModel:
    mean_same: np.ndarray
    cov_same: np.ndarray
    mean_diff: np.ndarray
    cov_diff: np.ndarray
    prior_same: float
    n_pairs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "mean_same": self.mean_same.tolist(), "cov_same": self.cov_same.tolist(),
            "mean_diff": self.mean_diff.tolist(), "cov_diff": self.cov_diff.tolist(),
            "prior_same": self.prior_same, "n_pairs": self.n_pairs,
            "features": ["footprint_correlation", "log1p_centroid_distance_um"],
        }


def _features(pairs: pd.DataFrame, pixel_size_um: float) -> np.ndarray:
    return np.column_stack([
        pairs["footprint_correlation"].to_numpy(dtype=float),
        np.log1p(pairs["centroid_distance_px"].to_numpy(dtype=float) * pixel_size_um),
    ])


def _gauss_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = cov + 1e-6 * np.eye(2)
    delta = x - mean
    inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (np.einsum("ij,jk,ik->i", delta, inv, delta) + logdet + 2 * np.log(2 * np.pi))


def fit_p_same(
    pairs: pd.DataFrame,
    pixel_size_um: float,
    *,
    anchor_same: tuple[float, float] = (0.6, 3.0),  # corr >=, distance_um <=
    anchor_diff_corr: float = 0.2,
    n_em: int = 30,
) -> tuple[pd.DataFrame, PSameModel]:
    """Fit the mixture on all assigned pairs and return pairs with a P_same column.

    ``pairs`` must contain ``footprint_correlation`` and ``centroid_distance_px``.
    Anchors only initialize the EM; the final components are data-fit.
    """
    pairs = pairs.reset_index(drop=True)
    x = _features(pairs, pixel_size_um)
    finite = np.isfinite(x).all(axis=1)
    distance_um = pairs["centroid_distance_px"].to_numpy(dtype=float) * pixel_size_um
    same_init = finite & (pairs["footprint_correlation"] >= anchor_same[0]) & (distance_um <= anchor_same[1])
    diff_init = finite & (pairs["footprint_correlation"] < anchor_diff_corr)
    if same_init.sum() < 5:
        raise ValueError(f"Insufficient same-cell anchor pairs: {int(same_init.sum())}")
    fallback_diff = diff_init.sum() < 5
    if fallback_diff:
        # No clear different-cell population among assigned pairs (clean data):
        # fix the different component at a documented reference (corr ~ 0, far centroid)
        # and only fit the same component.
        mean_diff = np.array([0.0, np.log1p(10.0)])
        cov_diff = np.diag([0.15**2, 0.5**2])
    responsibility = np.where(same_init[finite], 0.99, np.where(diff_init[finite], 0.01, 0.5))
    xf = x[finite]
    prior = float(responsibility.mean())
    for _ in range(n_em):
        w_same = responsibility / responsibility.sum()
        mean_same = w_same @ xf
        cov_same = ((xf - mean_same).T * w_same) @ (xf - mean_same)
        if not fallback_diff:
            w_diff = (1 - responsibility) / max((1 - responsibility).sum(), 1e-9)
            mean_diff = w_diff @ xf
            cov_diff = ((xf - mean_diff).T * w_diff) @ (xf - mean_diff)
        log_same = _gauss_logpdf(xf, mean_same, cov_same) + np.log(prior)
        log_diff = _gauss_logpdf(xf, mean_diff, cov_diff) + np.log(1 - prior)
        responsibility = 1.0 / (1.0 + np.exp(np.clip(log_diff - log_same, -500, 500)))
        prior = float(np.clip(responsibility.mean(), 1e-3, 1 - 1e-3))
    p_same = np.full(len(pairs), np.nan)
    p_same[finite] = responsibility
    out = pairs.copy()
    out["p_same"] = p_same
    model = PSameModel(mean_same, cov_same, mean_diff, cov_diff, prior, int(finite.sum()))
    return out, model


def expected_false_matches(p_same: np.ndarray, threshold: float) -> tuple[int, float]:
    """(number accepted, expected false matches among them) at a posterior threshold."""
    accepted = p_same[np.isfinite(p_same) & (p_same >= threshold)]
    return int(len(accepted)), float(np.sum(1.0 - accepted))


def fit_isotonic_calibration(p_raw: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a monotone map raw-posterior -> empirical outcome rate via PAVA.

    ``outcomes`` are binary consistency indicators for the same pairs. Returns
    (x_knots, y_knots) for use with :func:`apply_isotonic` (np.interp).
    """
    finite = np.isfinite(p_raw) & np.isfinite(outcomes)
    if finite.sum() < 10:
        raise ValueError(f"Too few calibration points: {int(finite.sum())}")
    order = np.argsort(p_raw[finite], kind="stable")
    x = p_raw[finite][order]
    y = outcomes[finite][order].astype(float)
    # pool adjacent violators
    block_value: list[float] = []
    block_weight: list[float] = []
    block_x: list[float] = []
    for xi, yi in zip(x, y):
        block_value.append(yi)
        block_weight.append(1.0)
        block_x.append(xi)
        while len(block_value) > 1 and block_value[-2] > block_value[-1]:
            w = block_weight[-2] + block_weight[-1]
            v = (block_value[-2] * block_weight[-2] + block_value[-1] * block_weight[-1]) / w
            xm = (block_x[-2] * block_weight[-2] + block_x[-1] * block_weight[-1]) / w
            block_value[-2:] = [v]
            block_weight[-2:] = [w]
            block_x[-2:] = [xm]
    return np.asarray(block_x), np.clip(np.asarray(block_value), 0.0, 1.0)


def apply_isotonic(p_raw: np.ndarray, x_knots: np.ndarray, y_knots: np.ndarray) -> np.ndarray:
    """Apply the fitted monotone calibration map (constant extrapolation at the ends)."""
    out = np.interp(p_raw, x_knots, y_knots)
    return np.where(np.isfinite(p_raw), out, np.nan)



# ============================================================================
# merged from calibration.py



from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .matching import assign_matches, footprints_to_sparse, iou_matrix, sparse_centroids


def adjacent_direct_assignments(
    warped_footprints: Mapping[str, np.ndarray],
    roi_ids: Mapping[str, np.ndarray],
    session_order: Sequence[str],
    shape: tuple[int, int],
    **gates: float,
) -> dict[tuple[str, str], set[tuple[int, int]]]:
    """Direct one-to-one candidate assignments between consecutive sessions, all in the reference frame."""
    sparse = {s: footprints_to_sparse(warped_footprints[s]) for s in session_order}
    cents = {s: sparse_centroids(sparse[s], shape) for s in session_order}
    out: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for a, b in zip(session_order[:-1], session_order[1:]):
        direct = assign_matches(iou_matrix(sparse[a], sparse[b]), roi_ids[a], roi_ids[b], cents[a], cents[b], **gates)
        ok = direct.loc[direct["candidate"].astype(bool)]
        out[(a, b)] = set(zip(ok["reference_roi_id"].astype(int), ok["moving_roi_id"].astype(int)))
    return out


def transitive_consistency_flags(
    pairs: pd.DataFrame,
    adjacent: Mapping[tuple[str, str], set[tuple[int, int]]],
    *,
    candidate_column: str = "candidate",
) -> pd.Series:
    """Per-pair consistency flag (bool) indexed like ``pairs``; NaN where no adjacent check applies.

    ``pairs`` holds reference-mediated assignments (``session_id``, ``reference_roi_id``,
    ``moving_roi_id``, candidate flag). A pair passes if, for every adjacent session that also
    carries a candidate for the same reference ROI, the two moving ROIs are directly paired.
    """
    cand = pairs.loc[pairs[candidate_column].astype(bool)]
    lookup = {(r.session_id, int(r.reference_roi_id)): int(r.moving_roi_id) for r in cand.itertuples()}
    out = pd.Series(np.nan, index=pairs.index, dtype=object)
    for idx, r in cand.iterrows():
        s, rr, m = r["session_id"], int(r["reference_roi_id"]), int(r["moving_roi_id"])
        flags = []
        for (a, b), direct in adjacent.items():
            if s == a and (b, rr) in lookup:
                flags.append((m, lookup[(b, rr)]) in direct)
            elif s == b and (a, rr) in lookup:
                flags.append((lookup[(a, rr)], m) in direct)
        if flags:
            out.loc[idx] = bool(np.all(flags))
    return out


def consistency_pairs_table(
    pairs: pd.DataFrame,
    flags: pd.Series,
    fit_sessions: set[str],
    *,
    score_columns: Sequence[str] = ("p_same_raw", "p_same"),
) -> pd.DataFrame:
    """Long table of every checkable pair with parity (fit vs held-out) and its consistency flag."""
    ok = flags.notna()
    cols = ["session_id", "reference_roi_id", "moving_roi_id", *[c for c in score_columns if c in pairs.columns]]
    table = pairs.loc[ok, cols].copy()
    table["parity"] = np.where(table["session_id"].isin(fit_sessions), "fit", "held_out")
    table["consistent"] = flags.loc[ok].astype(bool).to_numpy()
    return table.reset_index(drop=True)


def reliability_bins(scores: np.ndarray, outcomes: np.ndarray, *, n_bins: int = 10, equal_count: bool = True) -> pd.DataFrame:
    """Observed outcome fraction per score bin with binomial SE; equal-count or fixed-width bins."""
    s = np.asarray(scores, dtype=float); y = np.asarray(outcomes, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]
    if equal_count:
        edges = np.quantile(s, np.linspace(0, 1, n_bins + 1)) if len(s) else np.linspace(0, 1, n_bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        edges[-1] = np.inf
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        frac = float(y[m].mean()); n = int(m.sum())
        rows.append({"bin": b, "score_mean": float(s[m].mean()), "score_lo": float(s[m].min()), "score_hi": float(s[m].max()),
                     "fraction": frac, "n": n, "se": float(np.sqrt(frac * (1 - frac) / n))})
    return pd.DataFrame(rows)


def isotonic_curve_from_pairs(pairs: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """The fitted calibration map evaluated on ``grid``, reconstructed from the stored (p_same_raw -> p_same) pairs."""
    ok = pairs["p_same_raw"].notna() & pairs["p_same"].notna()
    mapping = pairs.loc[ok, ["p_same_raw", "p_same"]].sort_values("p_same_raw").drop_duplicates("p_same_raw")
    return np.interp(np.asarray(grid, dtype=float), mapping["p_same_raw"].to_numpy(float), mapping["p_same"].to_numpy(float))
