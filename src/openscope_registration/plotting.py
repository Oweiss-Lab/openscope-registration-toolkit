"""All figures: the diagnostic battery (B1-B7), rescue/P_same figures (R1-R5, P1-P4), mixture-fit overlays, reliability diagrams, and per-cell residual panels.

Each function renders one source-specified diagnostic display from validated
numerical products. No figure computes statistics beyond simple summaries of
its inputs. Thresholds shown are caller-supplied, preparation-specific values.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _wrap(text: str, width: int = 95) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _display_labels(session_ids: list[str]) -> list[str]:
    """Strip the longest common prefix so long source IDs stay readable."""
    if len(session_ids) < 2:
        return list(session_ids)
    prefix = len(session_ids[0])
    for s in session_ids[1:]:
        while prefix and not s.startswith(session_ids[0][:prefix]):
            prefix -= 1
    return [s[prefix:] or s for s in session_ids]


def plot_centroid_distance_histogram(
    distances: pd.DataFrame, median_footprint_diameter_um: float, path: Path, *, title_context: str, dpi: int = 300
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    values = distances["centroid_distance_um"].to_numpy()
    ax.hist(values, bins=40, color="0.35")
    ax.axvline(median_footprint_diameter_um, color="tab:red", linestyle="--",
               label=f"median footprint diameter ({median_footprint_diameter_um:.1f} um)")
    ax.axvline(float(np.median(values)), color="tab:blue",
               label=f"median match distance ({np.median(values):.2f} um)")
    ax.set_xlabel("Centroid distance of matched pair (um)")
    ax.set_ylabel(f"Matched pairs (n={len(values)})")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(_wrap(f"Centroid-distance histogram, all candidate matches pooled across sessions — {title_context}. "
                       "Mass far below the footprint diameter indicates sub-cell-size residuals."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_footprint_correlation_distribution(correlations: pd.DataFrame, path: Path, *, title_context: str, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.2, 1.0, 49)
    for kind, color in (("match", "tab:blue"), ("shuffled", "0.6")):
        values = correlations.loc[correlations["kind"].eq(kind), "footprint_correlation"].dropna().to_numpy()
        ax.hist(values, bins=bins, color=color, alpha=0.7, density=True,
                label=f"{kind} pairs (n={len(values)}, median {np.median(values):.2f})")
    ax.set_xlabel("Weighted footprint Pearson correlation (union support)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(_wrap(f"Footprint-correlation distribution, matches vs shuffled pairs — {title_context}. "
                       "Matches must separate clearly from the shuffled null."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_roi_overlays_by_session(
    reference_projection_gray: np.ndarray,
    outlines_by_session: dict[str, list[np.ndarray]],
    path: Path,
    *,
    title_context: str,
    dpi: int = 300,
) -> None:
    """Matched-cell footprint outlines from every session drawn over the reference image."""
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(reference_projection_gray, cmap="gray", vmin=0, vmax=1)
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(outlines_by_session)))
    for (session_id, masks), color in zip(outlines_by_session.items(), colors):
        for mask in masks:
            ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=0.5, alpha=0.8)
        ax.plot([], [], color=color, label=session_id)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    ax.set_title(_wrap(f"Matched-cell footprint outlines colored by session, on the reference projection — {title_context}. "
                       "Consistent nested outlines = stable identities; systematic shifts or doubled outlines = misregistration."),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


RESIDUAL_ARROW_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "plasma_mid", plt.cm.plasma(np.linspace(0.12, 0.88, 256)))  # no black / near-white arrows on grey


def draw_residual_displacement_panel(
    ax,
    residuals: pd.DataFrame,
    image_shape: tuple[int, int],
    *,
    mode: str = "arrows",
    background: np.ndarray | None = None,
    background_alpha: float = 0.18,
    vmax_um: float = 2.5,
    arrow_gain_px_per_um: float = 12.0,
    pixel_size_um: float | None = None,
):
    """Draw one per-cell residual-displacement panel in image coordinates (y down).

    ``residuals`` is a ``residual_displacement_table`` frame (``ref_y_px``, ``ref_x_px``,
    ``residual_y_um``, ``residual_x_um``, ``residual_um``). ``mode="arrows"`` draws each
    residual vector from the reference centroid toward the warped moving centroid,
    exaggerated by ``arrow_gain_px_per_um`` (converted to image px via ``pixel_size_um``
    when given); ``mode="color"`` draws dots coloured by |r|. Returns the mappable.
    Arrows use ``angles="xy"`` with the raw (y down) components, so direction is true
    image direction; the earlier version negated y and drew mirrored arrows.
    """
    if background is not None:
        ax.imshow(background, cmap="gray", vmin=0, vmax=1, alpha=background_alpha)
    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)
    if mode == "color":
        return ax.scatter(residuals["ref_x_px"], residuals["ref_y_px"], c=residuals["residual_um"], cmap="magma",
                          vmin=0, vmax=vmax_um, s=14, linewidths=0.3, edgecolors="white", zorder=3)
    gain = arrow_gain_px_per_um / (pixel_size_um or 1.0)
    return ax.quiver(residuals["ref_x_px"], residuals["ref_y_px"], residuals["residual_x_um"], residuals["residual_y_um"],
                     residuals["residual_um"], cmap=RESIDUAL_ARROW_CMAP, clim=(0, vmax_um), angles="xy",
                     scale_units="xy", scale=1.0 / gain, width=0.0065, headwidth=3.2, headlength=3.6, headaxislength=3.2,
                     minlength=0.6, edgecolors="white", linewidths=0.5, zorder=3)


def plot_residual_displacement_map(residuals: pd.DataFrame, image_shape: tuple[int, int], path: Path, *, title_context: str,
                                   dpi: int = 300, pixel_size_um: float | None = None) -> None:
    """Per-cell residual displacement after rigid alignment, pooled over sessions (arrows, true image direction)."""
    fig, ax = plt.subplots(figsize=(9, 8))
    q = draw_residual_displacement_panel(ax, residuals, image_shape, mode="arrows", pixel_size_um=pixel_size_um)
    ax.set_xlabel("Reference x (px)")
    ax.set_ylabel("Reference y (px)")
    fig.colorbar(q, ax=ax, label="Residual displacement |r| (um)")
    ax.set_title(_wrap(f"Per-cell residual displacement after global rigid alignment (pooled sessions) — {title_context}. "
                       f"Median {residuals['residual_um'].median():.2f} um (n={len(residuals)} matched pairs). "
                       "Spatial clusters of aligned arrows indicate local deformation a rigid model cannot fix."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_local_geometry_null(result: dict[str, object], path: Path, *, title_context: str, dpi: int = 300) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    h = ax.hist2d(np.asarray(result["delta_distance_um"]), np.asarray(result["delta_angle_deg"]), bins=41, cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="Neighbor pairs")
    ax.set_xlabel("Change in neighbor distance (um)")
    ax.set_ylabel("Change in neighbor angle (deg)")
    ax.set_title("Local-geometry changes of matched cells\n(concentration at (0,0) expected)", fontsize=10)
    ax = axes[1]
    null = np.asarray(result["null_median_abs_delta_distance_um"])
    ax.hist(null, bins=40, color="0.6", label=f"permutation null (n={len(null)})")
    ax.axvline(result["real_median_abs_delta_distance_um"], color="tab:red",
               label=f"real matches ({result['real_median_abs_delta_distance_um']:.2f} um)")
    ax.set_xlabel("Median |neighbor-distance change| (um)")
    ax.set_ylabel("Permutations")
    ax.legend(fontsize=8)
    ax.set_title(f"Permutation test: p = {result['p_value']:.4g}", fontsize=10)
    fig.suptitle(_wrap(f"Local-geometry diagnostic vs shuffled null — {title_context}. Real matches must preserve "
                       "neighborhood geometry far better than permuted identities."), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_coverage_summary(coverage: pd.DataFrame, session_order: list[str], path: Path, *, title_context: str, dpi: int = 300) -> None:
    matched_cols = [f"matched__{s}" for s in session_order]
    matrix = coverage[matched_cols].to_numpy(dtype=bool)
    order = np.argsort(-matrix.sum(axis=1), kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1.0]})
    ax = axes[0]
    ax.imshow(matrix[order].T, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_yticks(range(len(session_order)), _display_labels(session_order), fontsize=7)
    ax.set_xlabel(f"Reference cells, sorted by coverage (n={len(coverage)})")
    ax.set_title("Candidate-match coverage (dark = matched)", fontsize=10)
    ax = axes[1]
    counts = coverage["n_sessions_matched"].value_counts().sort_index()
    ax.bar(counts.index, counts.to_numpy(), color="0.35")
    ax.set_xlabel("Sessions with candidate match")
    ax.set_ylabel("Reference cells")
    full = int((coverage["n_sessions_matched"] == len(session_order)).sum())
    ax.set_title(f"{full} cells matched in all {len(session_order)} sessions", fontsize=10)
    fig.suptitle(_wrap(f"Cross-session coverage of reference cells — {title_context}. The full-coverage intersection is the "
                       "candidate pool for longitudinal analyses."), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_footprint_size_change(size_changes: pd.DataFrame, session_order: list[str], path: Path, *, title_context: str, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    values = [size_changes.loc[size_changes["session_id"].eq(s), "diameter_ratio_moving_over_reference"].dropna().to_numpy() for s in session_order]
    labels = _display_labels(session_order)
    # `labels=` was removed from Axes.boxplot in matplotlib 3.11; set tick labels explicitly.
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(range(1, len(values) + 1))
    ax.set_xticklabels([f"{lab}\n(n={len(v)})" for lab, v in zip(labels, values)])
    rng = np.random.default_rng(0)
    for k, v in enumerate(values):
        ax.scatter(np.full(len(v), k + 1) + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color="0.3", alpha=0.4, zorder=3)
    ax.axhline(1.0, color="tab:red", linewidth=0.8)
    ax.set_ylabel("Footprint diameter ratio (day / reference)")
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.set_title(_wrap(f"Footprint-size change of matched cells per session (dots = cells) — {title_context}. "
                       "Systematic deviation from 1.0 on a day indicates focal-plane (Z) change, which X/Y registration cannot fix."),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)



# ============================================================================
# merged from registration_refinement.py



import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .rescue import PROVENANCE_CATEGORIES

CATEGORY_COLORS = {
    "matched": "#2066a8",
    "rescued_classifier_override": "#f6a600",
    "present_but_undetected": "#5ab06c",
    "absent_ambiguous": "#c8c8c8",
}


def _wrap(text: str, width: int = 95) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _short(session_ids: list[str]) -> list[str]:
    if len(session_ids) < 2:
        return list(session_ids)
    prefix = len(session_ids[0])
    for s in session_ids[1:]:
        while prefix and not s.startswith(session_ids[0][:prefix]):
            prefix -= 1
    return [s[prefix:] or s for s in session_ids]


def plot_provenance_matrix(provenance: pd.DataFrame, session_order: list[str], path: Path, *, title_context: str, dpi: int = 300) -> None:
    """R1: cells x sessions, colored by provenance category."""
    code = {c: i for i, c in enumerate(PROVENANCE_CATEGORIES)}
    pivot = provenance.pivot(index="reference_roi_id", columns="session_id", values="category").reindex(columns=session_order)
    matrix = pivot.replace(code).to_numpy(dtype=float)
    order = np.argsort(-(matrix <= 2).sum(axis=1), kind="stable")  # sort by non-absent coverage
    cmap = matplotlib.colors.ListedColormap([CATEGORY_COLORS[c] for c in PROVENANCE_CATEGORIES])
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.imshow(matrix[order].T, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_yticks(range(len(session_order)), _short(session_order), fontsize=7)
    ax.set_xlabel(f"Reference cells, sorted by non-absent coverage (n={pivot.shape[0]})")
    ax.legend(handles=[Patch(color=CATEGORY_COLORS[c], label=c) for c in PROVENANCE_CATEGORIES],
              fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title(_wrap(f"R1 Track-day provenance matrix — {title_context}. Why each track-day is present or missing; "
                       "rescue categories are candidate evidence, not accepted identities."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_provenance_stacked_bars(provenance: pd.DataFrame, session_order: list[str], path: Path, *, title_context: str, dpi: int = 300) -> None:
    """R2: per-session counts stacked by category."""
    counts = provenance.groupby(["session_id", "category"]).size().unstack(fill_value=0).reindex(
        index=session_order, columns=PROVENANCE_CATEGORIES, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bottom = np.zeros(len(session_order))
    for category in PROVENANCE_CATEGORIES:
        values = counts[category].to_numpy(dtype=float)
        ax.bar(range(len(session_order)), values, bottom=bottom, color=CATEGORY_COLORS[category], label=category)
        bottom += values
    ax.set_xticks(range(len(session_order)), _short(session_order), rotation=30, fontsize=7)
    ax.set_ylabel("Reference cells")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.set_title(_wrap(f"R2 Provenance per session — {title_context}. Days dominated by rescue/undetected categories lose "
                       "cells to detection, not geometry; days dominated by absent-ambiguous need investigation."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_backprojection_evidence(evidence: pd.DataFrame, path: Path, *, title_context: str, dpi: int = 300) -> None:
    """R3: contrast z-scores for unmatched cell-days vs the shuffled-location null (z=0,1 by construction)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    z = evidence["evidence_z"].dropna().to_numpy()
    threshold = float(evidence["evidence_z_threshold"].iloc[0]) if len(evidence) else 3.0
    ax.hist(z, bins=50, color="0.4", label=f"unmatched cell-days (n={len(z)})")
    ax.axvline(0.0, color="0.6", linewidth=0.8, label="shuffled-location null mean (z=0)")
    ax.axvline(threshold, color="tab:red", linestyle="--", label=f"provisional evidence threshold (z={threshold:g})")
    ax.set_xlabel("Footprint-vs-annulus contrast (z vs shuffled-location null)")
    ax.set_ylabel("Unmatched cell-days")
    n_present = int((evidence["label"] == "present_but_undetected").sum())
    ax.legend(fontsize=8)
    ax.set_title(_wrap(f"R3 Back-projection evidence — {title_context}. {n_present} of {len(evidence)} unmatched cell-days "
                       "show above-threshold image evidence (present but undetected by segmentation)."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_rescue_gain(gain: pd.DataFrame, path: Path, *, title_context: str, dpi: int = 300) -> None:
    """R4: full-coverage cells before vs after rescue, split by mechanism.

    ``gain`` columns: label, base, plus_override, plus_backprojection (cumulative counts).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(gain))
    ax.bar(x - 0.2, gain["base"], width=0.36, color=CATEGORY_COLORS["matched"], label="base matching")
    ax.bar(x + 0.2, gain["plus_override"], width=0.36, color=CATEGORY_COLORS["rescued_classifier_override"],
           label="+ classifier override")
    ax.bar(x + 0.2, gain["plus_backprojection"] - gain["plus_override"], width=0.36,
           bottom=gain["plus_override"], color=CATEGORY_COLORS["present_but_undetected"], label="+ back-projection evidence")
    ax.set_xticks(x, gain["label"], fontsize=9)
    ax.set_ylabel("Full-coverage reference cells")
    ax.legend(fontsize=8)
    ax.set_title(_wrap(f"R4 Rescue gain — {title_context}. Full-coverage counts: base matching alone, after "
                       "classifier-override rescue, and after counting present-but-undetected evidence."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_rescued_gallery(
    crops: dict[int, dict[str, tuple[np.ndarray, np.ndarray | None, str]]],
    session_order: list[str],
    path: Path,
    *,
    title_context: str,
    dpi: int = 300,
) -> None:
    """R5: rows = rescued cells, columns = sessions; crop, optional outline, border = category."""
    if not crops:
        return
    cells = list(crops.keys())
    fig, axes = plt.subplots(len(cells), len(session_order), figsize=(1.3 * len(session_order), 1.4 * len(cells)), squeeze=False)
    labels = _short(session_order)
    for r, ref_id in enumerate(cells):
        for c, session_id in enumerate(session_order):
            ax = axes[r, c]
            image, outline, category = crops[ref_id][session_id]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            if outline is not None:
                ax.contour(outline.astype(float), levels=[0.5], colors="yellow", linewidths=0.6)
            for spine in ax.spines.values():
                spine.set_edgecolor(CATEGORY_COLORS[category])
                spine.set_linewidth(2.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(labels[c], fontsize=6)
            if c == 0:
                ax.set_ylabel(f"ROI {ref_id}", fontsize=7)
    fig.suptitle(_wrap(f"R5 Rescued-cell review gallery — {title_context}. Border color = provenance category "
                       "(blue matched, orange classifier-override, green present-but-undetected, gray absent); "
                       "yellow outline = matched footprint. Manual review required.", 110), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


MIXTURE_COMPONENT_COLORS = {"same": "#08306b", "diff": "#67000d"}
_MIXTURE_LEGEND = [
    Line2D([], [], color=MIXTURE_COMPONENT_COLORS["same"], lw=1.2, label="same-cell component (2σ, × = mean)"),
    Line2D([], [], color=MIXTURE_COMPONENT_COLORS["diff"], lw=1.2, label="different-cell component (2σ, × = mean)"),
    Line2D([], [], color="0.15", lw=1.0, ls=":", label="p_raw = 0.5 decision contour"),
]


def _model_dict(model) -> dict:
    return model.as_dict() if hasattr(model, "as_dict") else dict(model)


def overlay_mixture_fit(
    ax,
    model,
    *,
    n_sigma: float = 2.0,
    distance_range_um: tuple[float, float] = (0.01, 16.0),
    correlation_range: tuple[float, float] = (-0.4, 1.02),
    alpha: float = 0.45,
) -> None:
    """Overlay a fitted ``PSameModel`` (or its ``as_dict()``) on a feature scatter.

    Draws the n-sigma covariance ellipse and mean marker of each component and
    the ``p_raw = 0.5`` decision contour. The model lives in
    (correlation, log1p(distance_um)); every overlay is mapped point-by-point
    into the unlogged display space (distance_um on x, correlation on y), so
    ellipses appear teardrop-shaped. Required by the figure-maker model-fit
    overlay protocol for any scatter showing data a mixture was fit to.
    """
    from .psame import _gauss_logpdf

    m = _model_dict(model)
    theta = np.linspace(0, 2 * np.pi, 200)
    for key, color in MIXTURE_COMPONENT_COLORS.items():
        mean = np.asarray(m[f"mean_{key}"], dtype=float)
        cov = np.asarray(m[f"cov_{key}"], dtype=float)
        vals, vecs = np.linalg.eigh(cov)
        circle = (vecs * (n_sigma * np.sqrt(np.clip(vals, 0, None)))) @ np.vstack([np.cos(theta), np.sin(theta)])
        ax.plot(np.expm1(mean[1] + circle[1]), mean[0] + circle[0], color=color, lw=0.7, alpha=alpha, zorder=6)
        ax.plot(np.expm1(mean[1]), mean[0], marker="x", color=color, ms=4, mew=1.1, alpha=min(1.0, alpha + 0.25), zorder=7)
    x = np.linspace(*distance_range_um, 220)
    y = np.linspace(*correlation_range, 180)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([yy.ravel(), np.log1p(xx.ravel())])
    prior = float(m["prior_same"])
    log_same = _gauss_logpdf(pts, np.asarray(m["mean_same"], float), np.asarray(m["cov_same"], float)) + np.log(prior)
    log_diff = _gauss_logpdf(pts, np.asarray(m["mean_diff"], float), np.asarray(m["cov_diff"], float)) + np.log(1 - prior)
    post = np.exp(log_same - np.logaddexp(log_same, log_diff)).reshape(xx.shape)
    ax.contour(xx, yy, post, levels=[0.5], colors="0.15", linewidths=0.7, linestyles=":", alpha=0.7, zorder=5)


def plot_p_same_scatter(
    pairs: pd.DataFrame,
    pixel_size_um: float,
    path: Path,
    *,
    title_context: str,
    dpi: int = 300,
    model=None,
    color_by: str = "p_same",
) -> None:
    """P1: feature scatter colored by posterior, with the fitted mixture overlaid when ``model`` is given."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    distance_um = pairs["centroid_distance_px"] * pixel_size_um
    sc = ax.scatter(distance_um, pairs["footprint_correlation"], c=pairs[color_by if color_by in pairs.columns else "p_same"],
                    s=7, cmap="coolwarm_r", vmin=0, vmax=1, linewidths=0)
    label = "p_raw (mixture posterior)" if color_by == "p_same_raw" else "P_same (posterior)"
    fig.colorbar(sc, ax=ax, label=label)
    if model is not None:
        overlay_mixture_fit(ax, model)
        ax.legend(handles=_MIXTURE_LEGEND, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=3, frameon=False)
    ax.set_xlabel("Centroid distance (um)")
    ax.set_ylabel("Weighted footprint correlation")
    ax.set_title(_wrap(f"P1 P_same feature scatter, all one-to-one assigned pairs (n={len(pairs)}) — {title_context}."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_p_same_feature_scatter_grid(
    units: dict[str, list[tuple[pd.DataFrame, float, object]]],
    path: Path,
    *,
    title: str,
    dpi: int = 300,
    ncols: int = 5,
    color_by: str = "p_same_raw",
    unit_noun: str = "fit",
    distance_range_um: tuple[float, float] = (-0.5, 16.0),
    correlation_range: tuple[float, float] = (-0.4, 1.02),
) -> dict[str, int]:
    """One panel per unit (animal, field of view, cohort, ...), pooling that unit's fits.

    ``units`` maps a panel label to a list of ``(pairs, pixel_size_um, model)``
    triples, one per independently fitted mixture (e.g. one per plane). Pairs
    from all fits in a unit are pooled into one scatter; each fit's mixture is
    overlaid separately so the number of ellipse pairs equals the number of
    fits. Returns the pooled pair count per unit.
    """
    labels = list(units)
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows + 0.6), sharex=True, sharey=True, squeeze=False)
    sc = None
    totals: dict[str, int] = {}
    for ax, label in zip(axes.ravel(), labels):
        fits = units[label]
        frames = []
        for pairs, px_um, _ in fits:
            col = color_by if color_by in pairs.columns else "p_same"
            frames.append(pd.DataFrame({
                "d_um": pairs["centroid_distance_px"].to_numpy(float) * float(px_um),
                "corr": pairs["footprint_correlation"].to_numpy(float),
                "c": pairs[col].to_numpy(float),
            }))
        d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["d_um", "corr", "c"])
        totals[label] = int(len(d))
        sc = ax.scatter(d["d_um"], d["corr"], c=d["c"], cmap="coolwarm_r", vmin=0, vmax=1, s=4, alpha=0.5, linewidths=0, rasterized=True)
        for _, _, model in fits:
            if model is not None:
                overlay_mixture_fit(ax, model, distance_range_um=(max(distance_range_um[0], 0.01), distance_range_um[1]),
                                    correlation_range=correlation_range)
        ax.set_title(f"{label}\nn={len(d):,} pairs, {len(fits)} {unit_noun}{'s' if len(fits) != 1 else ''}", fontsize=9)
        ax.grid(color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(*distance_range_um)
        ax.set_ylim(*correlation_range)
    for ax in axes.ravel()[len(labels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("centroid distance (µm)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("weighted pixel mask\ncorrelation ρ_mask", fontsize=9)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.012)
        cbar.set_label("p_raw (mixture posterior)" if color_by == "p_same_raw" else "P_same (posterior)")
    fig.legend(handles=_MIXTURE_LEGEND, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.45, -0.035))
    fig.suptitle(_wrap(title, 110), fontsize=12)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return totals


def plot_p_same_histogram(pairs: pd.DataFrame, accept_threshold: float, review_band: tuple[float, float], path: Path, *, title_context: str, dpi: int = 300) -> None:
    """P2: posterior histogram with accept threshold, review band, expected false matches."""
    from .psame import expected_false_matches

    p = pairs["p_same"].dropna().to_numpy()
    n_accept, efm = expected_false_matches(p, accept_threshold)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(p, bins=50, color="0.4", log=True)
    ax.axvspan(review_band[0], review_band[1], color="tab:orange", alpha=0.2,
               label=f"manual-review band ({review_band[0]:g}-{review_band[1]:g})")
    ax.axvline(accept_threshold, color="tab:red", linestyle="--", label=f"accept threshold ({accept_threshold:g})")
    ax.set_xlabel("P_same (calibrated posterior where calibration applied)")
    ax.set_ylabel("Pairs (log scale)")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_title(_wrap(f"P2 P_same posteriors (n={len(p)} pairs) — {title_context}. At threshold {accept_threshold:g}: "
                       f"{n_accept} accepted pairs, expected false matches = {efm:.1f} (sum of 1-P_same)."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_threshold_curve(pairs: pd.DataFrame, provenance_full_at: dict[float, int], path: Path, *, title_context: str, dpi: int = 300) -> None:
    """P3: accepted full-coverage tracks and expected false matches vs threshold."""
    from .psame import expected_false_matches

    thresholds = np.round(np.arange(0.5, 0.996, 0.005), 3)
    p = pairs["p_same"].dropna().to_numpy()
    efm = [expected_false_matches(p, t)[1] for t in thresholds]
    fig, ax = plt.subplots(figsize=(9, 5))
    track_thresholds = sorted(provenance_full_at)
    ax.plot(track_thresholds, [provenance_full_at[t] for t in track_thresholds], color="tab:blue", marker="o", markersize=3,
            label="full-coverage tracks (left axis)")
    ax.set_xlabel("P_same accept threshold")
    ax.set_ylabel("Full-coverage tracks", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(thresholds, efm, color="tab:red", label="expected false matches (right axis)")
    ax2.set_ylabel("Expected false matches, all accepted pairs", color="tab:red")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center left")
    ax.set_title(_wrap(f"P3 Threshold selection — {title_context}. Choose the operating point with the error cost visible."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_calibration(calibration: pd.DataFrame, path: Path, *, title_context: str, dpi: int = 300) -> None:
    """P4: adjacent-day transitive consistency per P_same bin vs the diagonal."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", label="perfect calibration")
    ax.errorbar(calibration["bin_center"], calibration["fraction_consistent"],
                yerr=calibration["se"], color="tab:blue", marker="o", linestyle="-", capsize=3,
                label="observed (mean +/- SE)")
    for _, row in calibration.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["bin_center"], row["fraction_consistent"]), fontsize=6,
                    textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel("P_same bin")
    ax.set_ylabel("Fraction passing adjacent-day transitive consistency")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(_wrap(f"P4 Posterior calibration on held-out sessions — {title_context}. Points near the "
                       "diagonal support the posterior's interpretation; consistency is necessary, not sufficient."), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_reliability_diagram(
    series: dict[str, pd.DataFrame],
    path: Path,
    *,
    title: str,
    curves: np.ndarray | None = None,
    grid: np.ndarray | None = None,
    dpi: int = 300,
    score_label: str = "raw posterior p_raw",
    outcome_label: str = "fraction passing adjacent-day transitive consistency",
) -> None:
    """Reliability diagram per the figure-maker model-fit overlay protocol.

    ``series`` maps a label (e.g. ``"fit"``, ``"held_out"``) to a ``reliability_bins`` frame.
    Top: observed fraction vs mean score per bin with binomial SE and bin extents, the identity
    line, and optional fitted calibration ``curves`` (rows evaluated on ``grid``; drawn thin with
    their median bold). Bottom: pairs per bin (log scale) by series. Legends sit outside the axes.
    """
    from matplotlib.lines import Line2D

    styles = {"fit": ("tab:blue", "o"), "held_out": ("tab:orange", "s")}
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(7.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1.0, label="identity (perfect calibration)")
    if curves is not None and grid is not None and len(curves):
        for c in curves:
            ax.plot(grid, c, color="tab:green", lw=0.6, alpha=0.25)
        ax.plot(grid, np.median(curves, axis=0), color="tab:green", lw=2.2, label=f"fitted isotonic map (median of {len(curves)})")
    handles = []
    for i, (label, bins) in enumerate(series.items()):
        color, marker = styles.get(label, (f"C{i}", "o"))
        ax.errorbar(bins["score_mean"], bins["fraction"], yerr=bins["se"],
                    xerr=[bins["score_mean"] - bins["score_lo"], bins["score_hi"] - bins["score_mean"]],
                    fmt=marker, color=color, ms=6, capsize=2, lw=1.0, zorder=4)
        axn.bar(bins["score_mean"], bins["n"], width=0.8 * (bins["score_hi"] - bins["score_lo"]).clip(lower=0.01),
                color=color, alpha=0.45, edgecolor=color)
        handles.append(Line2D([], [], color=color, marker=marker, ls="", ms=6, label=f"{label} ({int(bins['n'].sum()):,} pairs)"))
    ax.set_ylabel(outcome_label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(color="0.9", lw=0.6); ax.set_axisbelow(True)
    axn.set_yscale("log"); axn.set_ylabel("pairs per bin"); axn.set_xlabel(score_label)
    axn.grid(color="0.9", lw=0.6, axis="y"); axn.set_axisbelow(True)
    line_h, line_l = ax.get_legend_handles_labels()
    fig.legend(line_h + handles, line_l + [h.get_label() for h in handles], loc="upper left",
               bbox_to_anchor=(1.01, 0.95), fontsize=8.5, frameon=False)
    fig.suptitle(_wrap(title, 80), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
