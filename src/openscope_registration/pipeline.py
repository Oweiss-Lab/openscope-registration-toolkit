"""The five pipeline stages as plain functions: register (1-2), diagnose (3a), refine (3b), review (4), freeze (5). Each writes CSV tables, figures, and a manifest.

This is pipeline stage 1-2 ("align images" and "match cells one-to-one"). For
every moving session it estimates a rigid transform onto the reference by
rotation grid search plus bounded phase correlation, scores it with masked NCC,
records the reverse transform and inverse-consistency residuals, measures
leftover per-tile shifts, warps the moving pixel masks into the reference
frame, builds the IoU table, assigns one-to-one Hungarian matches with
provisional candidate gates, repeats the matching between consecutive sessions
(adjacent-day assignments used later for transitive-consistency calibration),
and summarises template-anchored candidate tracks.

Inputs are validated :class:`SessionStructuralData` objects, so the workflow
is independent of where the sessions came from. Outputs mirror the original
OpenScope pilot script (``registration_transforms.csv``,
``residual_tile_shifts.csv``, ``roi_match_candidates.csv``,
``candidate_track_coverage.csv``, ``manifest.json``) so downstream consumers
(diagnostics battery, identity refinement, review GUI, freeze) keep working.
Field-of-view correspondence across planes (``fov_correspondence_ncc.csv``) is
produced when more than one plane is supplied.

Candidate thresholds are provisional pilot values, never acceptance criteria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import time

import numpy as np
import pandas as pd

from .data import RigidTransform, SessionStructuralData
from .diagnostics import centroid_distance_table, coverage_summary, footprint_correlation_table, footprint_size_change_table, local_geometry_change, residual_displacement_table
from .freeze import freeze_plane, load_review_decisions, plane_slug
from .manifests import sha256, write_manifest
from .matching import assign_matches, build_tracks, footprints_to_sparse, iou_matrix, sparse_centroids, weighted_footprint_correlation
from .plotting import plot_backprojection_evidence, plot_calibration, plot_centroid_distance_histogram, plot_coverage_summary, plot_footprint_correlation_distribution, plot_footprint_size_change, plot_local_geometry_null, plot_p_same_histogram, plot_p_same_scatter, plot_provenance_matrix, plot_provenance_stacked_bars, plot_rescue_gain, plot_rescued_gallery, plot_residual_displacement_map, plot_roi_overlays_by_session, plot_threshold_curve
from .psame import adjacent_direct_assignments, apply_isotonic, consistency_pairs_table, expected_false_matches, fit_isotonic_calibration, fit_p_same, transitive_consistency_flags
from .rescue import backprojection_evidence, classifier_override_rescue, provenance_table
from .review_html import build_manual_review_html, encode_crop_png
from .rigid import normalized_correlation, preprocess_projection, register_rigid, residual_tile_shifts, valid_support, warp_footprints, warp_image

SOURCE = "openscope_registration.pipeline"



def _transform_row(forward: RigidTransform, reverse: RigidTransform, identity_ncc: float) -> dict[str, float]:
    return {
        "angle_deg": forward.angle_deg, "shift_y_px": forward.shift_y_px, "shift_x_px": forward.shift_x_px,
        "rigid_ncc": forward.score, "identity_ncc": identity_ncc,
        "reverse_angle_deg": reverse.angle_deg, "reverse_shift_y_px": reverse.shift_y_px, "reverse_shift_x_px": reverse.shift_x_px,
        "inverse_angle_residual_deg": forward.angle_deg + reverse.angle_deg,
        "inverse_shift_residual_px": float(np.hypot(forward.shift_y_px + reverse.shift_y_px, forward.shift_x_px + reverse.shift_x_px)),
    }


def register_plane(
    reference: SessionStructuralData,
    moving_sessions: list[SessionStructuralData],
    *,
    plane: str,
    session_order: list[str] | None = None,
    maximum_angle_deg: float = 3.0,
    angle_step_deg: float = 0.25,
    max_shift_px: float = 64.0,
    tile_size_px: int = 64,
    candidate_iou: float = 0.5,
    max_centroid_px: float = 4.0,
    min_iou_margin: float = 0.2,
    overlap_iou: float = 0.2,
    highpass_sigma_px: float = 20.0,
    smooth_sigma_px: float = 1.0,
) -> dict[str, Any]:
    """Register one plane: returns transforms, tiles, matches (to-reference + adjacent), tracks, and warped masks."""
    reference.validate()
    for s in moving_sessions:
        s.validate()
    by_id = {reference.session_id: reference, **{s.session_id: s for s in moving_sessions}}
    sessions = list(session_order) if session_order else sorted(by_id)
    missing = set(sessions) - set(by_id)
    if missing or reference.session_id not in sessions:
        raise ValueError(f"session_order must list the reference and every moving session; missing {sorted(missing)}")
    prepared = {sid: preprocess_projection(by_id[sid].projection, highpass_sigma_px, smooth_sigma_px) for sid in sessions}
    ref_pre = prepared[reference.session_id]
    shape = ref_pre.shape
    px_um = reference.pixel_size_um
    gates = dict(candidate_iou=candidate_iou, max_centroid_px=max_centroid_px, min_iou_margin=min_iou_margin, overlap_iou=overlap_iou)
    rigid_kwargs = dict(maximum_angle_deg=maximum_angle_deg, angle_step_deg=angle_step_deg, max_shift_px=max_shift_px)

    transform_rows, tile_frames, match_frames = [], [], []
    transforms: dict[str, RigidTransform] = {}
    for sid in sessions:
        if sid == reference.session_id:
            continue
        forward = register_rigid(ref_pre, prepared[sid], **rigid_kwargs)
        reverse = register_rigid(prepared[sid], ref_pre, **rigid_kwargs)
        transforms[sid] = forward
        shift = (forward.shift_y_px, forward.shift_x_px)
        aligned = warp_image(prepared[sid], forward.angle_deg, shift, order=1)
        tiles = pd.DataFrame(residual_tile_shifts(ref_pre, aligned, valid_support(shape, forward.angle_deg, shift), tile_size_px))
        tiles.insert(0, "plane", plane)
        tiles.insert(0, "session_id", sid)
        tile_frames.append(tiles)
        transform_rows.append({
            "session_id": sid, "plane": plane, **_transform_row(forward, reverse, normalized_correlation(ref_pre, prepared[sid])),
            "n_full_tiles": int(len(tiles)),
            "residual_median_px": float(tiles["residual_shift_px"].median()) if len(tiles) else np.nan,
            "residual_max_px": float(tiles["residual_shift_px"].max()) if len(tiles) else np.nan,
            "residual_median_um": float(tiles["residual_shift_px"].median() * px_um) if len(tiles) else np.nan,
            "depth_um": by_id[sid].metadata.get("depth_um", np.nan),
            "reference_depth_um": reference.metadata.get("depth_um", np.nan),
        })
    transforms_df = pd.DataFrame(transform_rows)

    ref_sparse = footprints_to_sparse(reference.footprints)
    ref_centroids = sparse_centroids(ref_sparse, shape)
    to_reference: dict[str, pd.DataFrame] = {}
    warped_footprints: dict[str, np.ndarray] = {reference.session_id: reference.footprints}
    for sid in sessions:
        if sid == reference.session_id:
            continue
        mov = by_id[sid]
        warped = warp_footprints(mov.footprints, transforms[sid])
        warped_footprints[sid] = warped
        mov_sparse = footprints_to_sparse(warped)
        matches = assign_matches(iou_matrix(ref_sparse, mov_sparse), reference.roi_ids, mov.roi_ids,
                                 ref_centroids, sparse_centroids(mov_sparse, shape), **gates)
        to_reference[sid] = matches
        frame = matches.copy()
        frame.insert(0, "comparison", "to_reference")
        frame.insert(0, "reference_session_id", reference.session_id)
        frame.insert(0, "moving_session_id", sid)
        frame.insert(0, "plane", plane)
        match_frames.append(frame)

    adjacent: dict[tuple[str, str], pd.DataFrame] = {}
    for a, b in zip(sessions[:-1], sessions[1:]):
        pair = register_rigid(prepared[a], prepared[b], **rigid_kwargs)
        warped_b = warp_footprints(by_id[b].footprints, pair)
        sparse_a, sparse_b = footprints_to_sparse(by_id[a].footprints), footprints_to_sparse(warped_b)
        matches = assign_matches(iou_matrix(sparse_a, sparse_b), by_id[a].roi_ids, by_id[b].roi_ids,
                                 sparse_centroids(sparse_a, shape), sparse_centroids(sparse_b, shape), **gates)
        adjacent[(a, b)] = matches
        frame = matches.copy()
        frame.insert(0, "comparison", "adjacent_day")
        frame.insert(0, "reference_session_id", a)
        frame.insert(0, "moving_session_id", b)
        frame.insert(0, "plane", plane)
        match_frames.append(frame)

    tracks = build_tracks(reference.roi_ids, sessions, reference.session_id, to_reference, adjacent)
    tracks.insert(0, "plane", plane)
    return {
        "plane": plane, "sessions": sessions, "transforms": transforms, "transforms_table": transforms_df,
        "tiles": pd.concat(tile_frames, ignore_index=True) if tile_frames else pd.DataFrame(),
        "matches": pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame(),
        "tracks": tracks, "to_reference": to_reference, "adjacent": adjacent,
        "prepared": prepared, "warped_footprints": warped_footprints, "reference_centroids": ref_centroids,
    }


def fov_correspondence(
    sessions_by_plane: Mapping[str, Mapping[str, SessionStructuralData]],
    reference_session: str,
    *,
    maximum_angle_deg: float = 3.0,
    angle_step_deg: float = 0.25,
    max_shift_px: float = 64.0,
    highpass_sigma_px: float = 20.0,
    smooth_sigma_px: float = 1.0,
) -> pd.DataFrame:
    """Rigid NCC of every moving (session, plane) against every reference plane; flags whether the diagonal wins.

    A plane whose best-matching reference plane is itself confirms plane identity
    across days; off-diagonal winners indicate a plane swap or large Z drift.
    """
    planes = list(sessions_by_plane)
    prepared = {(p, sid): preprocess_projection(s.projection, highpass_sigma_px, smooth_sigma_px)
                for p, by_sid in sessions_by_plane.items() for sid, s in by_sid.items()}
    rows = []
    for moving_plane in planes:
        for sid in sessions_by_plane[moving_plane]:
            if sid == reference_session:
                continue
            for reference_plane in planes:
                t = register_rigid(prepared[(reference_plane, reference_session)], prepared[(moving_plane, sid)],
                                   maximum_angle_deg=maximum_angle_deg, angle_step_deg=angle_step_deg, max_shift_px=max_shift_px)
                rows.append({"session_id": sid, "moving_plane": moving_plane, "reference_plane": reference_plane, **t.as_dict(),
                             "rigid_ncc": t.score, "identity_ncc": normalized_correlation(prepared[(reference_plane, reference_session)], prepared[(moving_plane, sid)])})
    fov = pd.DataFrame(rows)
    if fov.empty:
        return fov
    best = fov.loc[fov.groupby(["session_id", "moving_plane"])["rigid_ncc"].idxmax(), ["session_id", "moving_plane", "reference_plane"]]
    best = best.rename(columns={"reference_plane": "best_reference_plane"})
    best["diagonal_is_best"] = best["moving_plane"].eq(best["best_reference_plane"])
    return fov.merge(best, on=["session_id", "moving_plane"], how="left")


def run_structural_registration(
    sessions_by_plane: Mapping[str, Mapping[str, SessionStructuralData]],
    reference_session: str,
    numerical_dir: Path,
    *,
    session_order: list[str] | None = None,
    include_fov_correspondence: bool = True,
    overwrite: bool = False,
    extra_parameters: dict[str, Any] | None = None,
    **plane_kwargs: Any,
) -> dict[str, Any]:
    """Register every plane onto ``reference_session`` and write the pilot-compatible CSV set + manifest.

    ``sessions_by_plane[plane][session_id]`` is a validated ``SessionStructuralData``. Every plane must
    contain the reference session. Per-plane results are returned under ``"planes"`` so callers can
    draw figures (overlays, galleries) without re-running the registration.
    """
    numerical_dir = Path(numerical_dir)
    if numerical_dir.exists() and any(numerical_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty run root: {numerical_dir}")
    numerical_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    planes = list(sessions_by_plane)
    for p in planes:
        if reference_session not in sessions_by_plane[p]:
            raise ValueError(f"plane {p} lacks the reference session {reference_session}")

    inventory_rows = []
    for p in planes:
        for sid, s in sessions_by_plane[p].items():
            inventory_rows.append({"session_id": sid, "plane": p, "pixel_size_um": s.pixel_size_um, "n_rois": s.n_rois,
                                   "image_shape": "x".join(map(str, s.projection.shape)), **{k: v for k, v in s.metadata.items() if np.isscalar(v) or isinstance(v, str)}})
    pd.DataFrame(inventory_rows).to_csv(numerical_dir / "session_plane_inventory.csv", index=False)

    rigid_keys = {"maximum_angle_deg", "angle_step_deg", "max_shift_px", "highpass_sigma_px", "smooth_sigma_px"}
    outputs: dict[str, Path] = {"session_plane_inventory_csv": numerical_dir / "session_plane_inventory.csv"}
    if include_fov_correspondence and len(planes) > 1:
        fov = fov_correspondence(sessions_by_plane, reference_session, **{k: v for k, v in plane_kwargs.items() if k in rigid_keys})
        fov.to_csv(numerical_dir / "fov_correspondence_ncc.csv", index=False)
        outputs["fov_correspondence_csv"] = numerical_dir / "fov_correspondence_ncc.csv"
    else:
        fov = pd.DataFrame()

    results: dict[str, dict[str, Any]] = {}
    for p in planes:
        by_sid = sessions_by_plane[p]
        reference = by_sid[reference_session]
        moving = [s for sid, s in by_sid.items() if sid != reference_session]
        results[p] = register_plane(reference, moving, plane=p, session_order=session_order, **plane_kwargs)

    transforms = pd.concat([r["transforms_table"] for r in results.values()], ignore_index=True)
    tiles = pd.concat([r["tiles"] for r in results.values()], ignore_index=True)
    matches = pd.concat([r["matches"] for r in results.values()], ignore_index=True)
    tracks = pd.concat([r["tracks"] for r in results.values()], ignore_index=True)
    for name, df in (("registration_transforms", transforms), ("residual_tile_shifts", tiles),
                     ("roi_match_candidates", matches), ("candidate_track_coverage", tracks)):
        df.to_csv(numerical_dir / f"{name}.csv", index=False)
        outputs[f"{name}_csv"] = numerical_dir / f"{name}.csv"

    summary = {}
    for p, r in results.items():
        tf, tr = r["transforms_table"], r["tracks"]
        summary[p] = {
            "median_rigid_ncc": float(tf["rigid_ncc"].median()) if len(tf) else np.nan,
            "min_rigid_ncc": float(tf["rigid_ncc"].min()) if len(tf) else np.nan,
            "median_residual_um": float(tf["residual_median_um"].median()) if len(tf) else np.nan,
            "n_reference_rois": int(len(tr)),
            "n_full_coverage_tracks": int(tr["covers_all_sessions"].sum()),
            "n_full_coverage_fully_consistent": int((tr["covers_all_sessions"] & tr["adjacent_consistency_fraction"].eq(1.0)).sum()),
        }
        if not fov.empty:
            summary[p]["fov_diagonal_best_fraction"] = float(fov.loc[fov["moving_plane"].eq(p)].drop_duplicates(["session_id"])["diagonal_is_best"].mean())

    sessions = results[planes[0]]["sessions"]
    manifest = write_manifest(
        numerical_dir / "manifest.json",
        artifact_type="structural_registration",
        created_by=SOURCE,
        parameters={"reference_session": reference_session, "sessions": sessions, "planes": planes,
                    "include_fov_correspondence": include_fov_correspondence, **plane_kwargs, **(extra_parameters or {})},
        output_files={k: str(v) for k, v in outputs.items()},
        extra={
            "summary": summary, "runtime_s": round(time.time() - started, 1),
            "registration_image": "structural projection; percentile scaled, Gaussian high-pass, light smoothing; rotation grid search with Hann-windowed phase correlation bounded by max_shift_px and parabolic subpixel peak; scored by masked NCC on the overlap support",
            "limitations": [
                "Candidate thresholds (IoU, centroid distance, margin) are provisional and are not acceptance criteria.",
                "Candidate tracks are template-anchored to the reference session; no global identities are assigned.",
                "Rigid registration only; residual tile shifts indicate whether affine/nonrigid refinement is needed.",
                "Manual review and a frozen manifest are required before any downstream single-cell claim.",
            ],
        },
    )
    return {"manifest": manifest, "summary": summary, "planes": results, "fov": fov, "outputs": outputs}


def run_registration_diagnostics(
    reference: SessionStructuralData,
    moving_sessions: list[SessionStructuralData],
    numerical_dir: Path,
    figure_dir: Path,
    *,
    transforms: dict[str, RigidTransform] | None = None,
    candidate_iou: float = 0.5,
    max_centroid_px: float = 4.0,
    min_iou_margin: float = 0.2,
    overlap_iou: float = 0.2,
    n_null: int = 200,
    n_neighbors: int = 4,
    n_permutations: int = 1000,
    overlay_max_cells: int = 40,
    title_context: str = "",
    dpi: int = 300,
    registration_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full diagnostics battery; returns a summary dict and writes artifacts."""
    reference.validate()
    for session in moving_sessions:
        session.validate()
    numerical_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    pixel_size_um = reference.pixel_size_um
    shape = reference.projection.shape
    reference_prepared = preprocess_projection(reference.projection)
    ref_sparse = footprints_to_sparse(reference.footprints)
    ref_centroids_arr = sparse_centroids(ref_sparse, shape)
    ref_index = {int(rid): i for i, rid in enumerate(reference.roi_ids)}
    ref_centroid_of = {int(rid): ref_centroids_arr[i] for i, rid in enumerate(reference.roi_ids)}

    matches_by_session: dict[str, pd.DataFrame] = {}
    warped_by_session: dict[str, np.ndarray] = {}
    transforms_out: dict[str, dict[str, float]] = {}
    distance_frames, correlation_frames, residual_frames, size_frames = [], [], [], []
    ref_centroids_by_session: list[np.ndarray] = []
    mov_centroids_by_session: list[np.ndarray] = []

    for session in moving_sessions:
        transform = (transforms or {}).get(session.session_id)
        if transform is None:
            transform = register_rigid(reference_prepared, preprocess_projection(session.projection), **(registration_kwargs or {}))
        transforms_out[session.session_id] = transform.as_dict()
        warped = warp_footprints(session.footprints, transform)
        warped_by_session[session.session_id] = warped
        mov_sparse = footprints_to_sparse(warped)
        mov_centroids_arr = sparse_centroids(mov_sparse, shape)
        mov_index = {int(rid): i for i, rid in enumerate(session.roi_ids)}
        matches = assign_matches(
            iou_matrix(ref_sparse, mov_sparse), reference.roi_ids, session.roi_ids, ref_centroids_arr, mov_centroids_arr,
            candidate_iou=candidate_iou, max_centroid_px=max_centroid_px, min_iou_margin=min_iou_margin, overlap_iou=overlap_iou,
        )
        matches_by_session[session.session_id] = matches
        mov_centroid_of = {int(rid): mov_centroids_arr[i] for i, rid in enumerate(session.roi_ids)}
        distance_frames.append(centroid_distance_table(matches, pixel_size_um, session.session_id))
        correlation_frames.append(
            footprint_correlation_table(reference.footprints, warped, matches, ref_index, mov_index, session.session_id, n_null=n_null, seed=seed)
        )
        residual_frames.append(residual_displacement_table(matches, ref_centroid_of, mov_centroid_of, pixel_size_um, session.session_id))
        size_frames.append(
            footprint_size_change_table(reference.footprints, warped, matches, ref_index, mov_index, pixel_size_um, session.session_id)
        )
        session_ref_pts, session_mov_pts = [], []
        for _, m in matches.loc[matches["candidate"]].iterrows():
            session_ref_pts.append(ref_centroid_of[int(m["reference_roi_id"])])
            session_mov_pts.append(mov_centroid_of[int(m["moving_roi_id"])])
        ref_centroids_by_session.append(np.asarray(session_ref_pts))
        mov_centroids_by_session.append(np.asarray(session_mov_pts))

    distances = pd.concat(distance_frames, ignore_index=True)
    correlations = pd.concat(correlation_frames, ignore_index=True)
    residuals = pd.concat(residual_frames, ignore_index=True)
    sizes = pd.concat(size_frames, ignore_index=True)
    coverage = coverage_summary(matches_by_session, reference.roi_ids)
    geometry = local_geometry_change(
        ref_centroids_by_session, mov_centroids_by_session, pixel_size_um,
        n_neighbors=n_neighbors, n_permutations=n_permutations, seed=seed,
    )

    session_order = [s.session_id for s in moving_sessions]
    tables = {
        "matched_centroid_distances_csv": (numerical_dir / "matched_centroid_distances.csv", distances),
        "footprint_correlations_csv": (numerical_dir / "footprint_correlations.csv", correlations),
        "residual_displacements_csv": (numerical_dir / "residual_displacements.csv", residuals),
        "footprint_size_changes_csv": (numerical_dir / "footprint_size_changes.csv", sizes),
        "coverage_csv": (numerical_dir / "coverage_summary.csv", coverage),
        "match_candidates_csv": (
            numerical_dir / "match_candidates.csv",
            pd.concat([m.assign(moving_session_id=s) for s, m in matches_by_session.items()], ignore_index=True),
        ),
        "transforms_csv": (
            numerical_dir / "transforms.csv",
            pd.DataFrame([{"session_id": s, **t} for s, t in transforms_out.items()]),
        ),
    }
    for path, frame in tables.values():
        frame.to_csv(path, index=False)

    median_diameter = float(np.median(reference.footprint_equivalent_diameters_um()))
    full_cells = coverage.loc[coverage["n_sessions_matched"].eq(len(session_order)), "reference_roi_id"].astype(int)
    overlay_cells = full_cells.head(overlay_max_cells).tolist()
    outlines = {
        reference.session_id: [reference.footprints[ref_index[rid]] > 0 for rid in overlay_cells]
    }
    for session in moving_sessions:
        mov_index = {int(rid): i for i, rid in enumerate(session.roi_ids)}
        matches = matches_by_session[session.session_id]
        lookup = dict(zip(matches["reference_roi_id"].astype(int), matches["moving_roi_id"].astype(int)))
        outlines[session.session_id] = [
            warped_by_session[session.session_id][mov_index[lookup[rid]]] > 0 for rid in overlay_cells if rid in lookup
        ]

    figures = {
        "centroid_distance_histogram_png": figure_dir / "centroid_distance_histogram.png",
        "footprint_correlation_distribution_png": figure_dir / "footprint_correlation_distribution.png",
        "roi_overlays_by_session_png": figure_dir / "roi_overlays_by_session.png",
        "residual_displacement_map_png": figure_dir / "residual_displacement_map.png",
        "local_geometry_null_png": figure_dir / "local_geometry_null.png",
        "coverage_summary_png": figure_dir / "coverage_summary.png",
        "footprint_size_change_png": figure_dir / "footprint_size_change.png",
    }
    plot_centroid_distance_histogram(distances, median_diameter, figures["centroid_distance_histogram_png"], title_context=title_context, dpi=dpi)
    plot_footprint_correlation_distribution(correlations, figures["footprint_correlation_distribution_png"], title_context=title_context, dpi=dpi)
    gray = np.clip((reference_prepared + 2.0) / 6.0, 0.0, 1.0)
    plot_roi_overlays_by_session(gray, outlines, figures["roi_overlays_by_session_png"], title_context=f"{title_context}; first {len(overlay_cells)} full-coverage cells", dpi=dpi)
    plot_residual_displacement_map(residuals, shape, figures["residual_displacement_map_png"], title_context=title_context, dpi=dpi)
    plot_local_geometry_null(geometry, figures["local_geometry_null_png"], title_context=title_context, dpi=dpi)
    plot_coverage_summary(coverage, session_order, figures["coverage_summary_png"], title_context=title_context, dpi=dpi)
    plot_footprint_size_change(sizes, session_order, figures["footprint_size_change_png"], title_context=title_context, dpi=dpi)

    return {
        "transforms": transforms_out,
        "n_reference_cells": int(reference.n_rois),
        "n_full_coverage_cells": int(len(full_cells)),
        "median_match_distance_um": float(distances["centroid_distance_um"].median()),
        "median_footprint_diameter_um": median_diameter,
        "median_footprint_correlation_match": float(
            correlations.loc[correlations["kind"].eq("match"), "footprint_correlation"].median()
        ),
        "median_footprint_correlation_shuffled": float(
            correlations.loc[correlations["kind"].eq("shuffled"), "footprint_correlation"].median()
        ),
        "median_residual_um": float(residuals["residual_um"].median()),
        "local_geometry_p_value": geometry["p_value"],
        "thresholds_note": "candidate thresholds are preparation-specific pilot values, not acceptance criteria",
        "output_files": {k: str(p) for k, (p, _) in tables.items()} | {k: str(p) for k, p in figures.items()},
    }


def _add_correlations(matches: pd.DataFrame, reference: SessionStructuralData, warped: np.ndarray,
                      ref_index: dict[int, int], mov_index: dict[int, int]) -> pd.DataFrame:
    correlations = [
        weighted_footprint_correlation(reference.footprints[ref_index[int(r)]], warped[mov_index[int(m)]])
        for r, m in zip(matches["reference_roi_id"], matches["moving_roi_id"])
    ]
    out = matches.copy()
    out["footprint_correlation"] = correlations
    return out


def run_identity_refinement(
    reference: SessionStructuralData,
    moving_sessions: list[SessionStructuralData],
    numerical_dir: Path,
    figure_dir: Path,
    *,
    transforms: dict[str, RigidTransform],
    candidate_iou: float = 0.5,
    max_centroid_px: float = 4.0,
    min_iou_margin: float = 0.2,
    overlap_iou: float = 0.2,
    evidence_z: float = 3.0,
    n_evidence_null: int = 200,
    p_same_accept: float = 0.95,
    review_band: tuple[float, float] = (0.5, 0.9),
    gallery_max_cells: int = 20,
    gallery_half_px: int = 24,
    title_context: str = "",
    dpi: int = 300,
    seed: int = 0,
) -> dict[str, Any]:
    reference.validate()
    numerical_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    gates = dict(candidate_iou=candidate_iou, max_centroid_px=max_centroid_px,
                 min_iou_margin=min_iou_margin, overlap_iou=overlap_iou)
    shape = reference.projection.shape
    session_order = [s.session_id for s in moving_sessions]
    ref_index = {int(r): i for i, r in enumerate(reference.roi_ids)}
    ref_gray = np.clip((preprocess_projection(reference.projection) + 2.0) / 6.0, 0.0, 1.0)

    pair_frames, evidence_by_session = [], {}
    warped_all: dict[str, np.ndarray] = {}
    mov_index_by_session: dict[str, dict[int, int]] = {}
    aligned_gray_by_session: dict[str, np.ndarray] = {}
    for session in moving_sessions:
        session.validate()
        is_soma = np.asarray(session.metadata.get("is_soma"), dtype=bool)
        if is_soma.shape != session.roi_ids.shape:
            raise ValueError(f"{session.session_id}: metadata['is_soma'] must align with roi_ids")
        transform = transforms[session.session_id]
        warped = warp_footprints(session.footprints, transform)
        warped_all[session.session_id] = warped
        mov_index = {int(m): i for i, m in enumerate(session.roi_ids)}
        mov_index_by_session[session.session_id] = mov_index
        aligned = warp_image(preprocess_projection(session.projection),
                             transform.angle_deg, (transform.shift_y_px, transform.shift_x_px), order=1)
        aligned_gray_by_session[session.session_id] = np.clip((aligned + 2.0) / 6.0, 0.0, 1.0)

        soma_rows = np.flatnonzero(is_soma)
        soma_sparse = footprints_to_sparse(warped[soma_rows])
        ref_sparse = footprints_to_sparse(reference.footprints)
        base = assign_matches(
            iou_matrix(ref_sparse, soma_sparse), reference.roi_ids, session.roi_ids[soma_rows],
            sparse_centroids(ref_sparse, shape), sparse_centroids(soma_sparse, shape), **gates,
        )
        base = _add_correlations(base, reference, warped, ref_index, mov_index)
        base["mechanism"] = "base_soma"

        matched_ref = set(base.loc[base["candidate"], "reference_roi_id"].astype(int))
        unmatched_ref = np.asarray([r for r in reference.roi_ids if int(r) not in matched_ref], dtype=int)
        claimed_mov = set(base["moving_roi_id"].astype(int))
        eligible_mov = np.asarray([m for m, s in zip(session.roi_ids, is_soma) if not s and int(m) not in claimed_mov], dtype=int)
        rescue = classifier_override_rescue(
            reference.footprints, reference.roi_ids, unmatched_ref, warped, session.roi_ids, eligible_mov, **gates,
        )
        rescue = _add_correlations(rescue, reference, warped, ref_index, mov_index)
        rescue["mechanism"] = "classifier_override"

        rescued_ref = set(rescue.loc[rescue["candidate"], "reference_roi_id"].astype(int))
        still_unmatched = np.asarray([r for r in unmatched_ref if int(r) not in rescued_ref], dtype=int)
        evidence = backprojection_evidence(
            aligned, reference.footprints, reference.roi_ids, still_unmatched,
            n_null=n_evidence_null, evidence_z=evidence_z, seed=seed,
        )
        evidence.insert(0, "session_id", session.session_id)
        evidence_by_session[session.session_id] = evidence

        for frame in (base, rescue):
            frame.insert(0, "session_id", session.session_id)
            pair_frames.append(frame)

    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs, model = fit_p_same(pairs, reference.pixel_size_um)
    pairs["p_same_raw"] = pairs["p_same"]
    evidence_all = pd.concat(evidence_by_session.values(), ignore_index=True)

    # Adjacent-day transitive consistency (computed on raw posteriors, before any acceptance)
    adjacent_pairs = adjacent_direct_assignments(
        warped_all, {s: moving_sessions[session_order.index(s)].roi_ids for s in session_order}, session_order, shape, **gates)
    flag_series = transitive_consistency_flags(pairs, adjacent_pairs)
    consistency_flags: dict[int, bool] = {int(i): bool(v) for i, v in flag_series.dropna().items()}

    # Isotonic recalibration: fit on even-index sessions, validate (P4) on odd-index sessions
    calibration_sessions = set(session_order[0::2])
    validation_sessions = set(session_order[1::2])
    fit_idx = [i for i in consistency_flags if pairs.loc[i, "session_id"] in calibration_sessions]
    calibration_applied = False
    if len(fit_idx) >= 50 and len(validation_sessions) >= 1:
        x_knots, y_knots = fit_isotonic_calibration(
            pairs.loc[fit_idx, "p_same_raw"].to_numpy(dtype=float),
            np.asarray([consistency_flags[i] for i in fit_idx], dtype=float),
        )
        pairs["p_same"] = apply_isotonic(pairs["p_same_raw"].to_numpy(dtype=float), x_knots, y_knots)
        calibration_applied = True

    def accepted_sets(threshold: float) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
        ok = pairs["candidate"] & (pairs["p_same"] >= threshold)
        base_ok = pairs.loc[ok & pairs["mechanism"].eq("base_soma")]
        rescue_ok = pairs.loc[ok & pairs["mechanism"].eq("classifier_override")]
        return (
            {s: set(g["reference_roi_id"].astype(int)) for s, g in base_ok.groupby("session_id")},
            {s: set(g["reference_roi_id"].astype(int)) for s, g in rescue_ok.groupby("session_id")},
        )

    accepted_base, accepted_rescue = accepted_sets(p_same_accept)
    provenance = provenance_table(reference.roi_ids, session_order, accepted_base, accepted_rescue, evidence_by_session)

    # rescue-gain summary and threshold curve inputs
    def full_coverage(include_rescue: bool, include_evidence: bool, threshold: float) -> int:
        base_t, rescue_t = accepted_sets(threshold)
        count = 0
        for ref_id in reference.roi_ids.astype(int):
            ok_all = True
            for s in session_order:
                present = ref_id in base_t.get(s, set())
                if include_rescue:
                    present = present or ref_id in rescue_t.get(s, set())
                if include_evidence and not present:
                    ev = evidence_by_session[s]
                    hit = ev.loc[ev["reference_roi_id"].eq(ref_id), "label"]
                    present = len(hit) > 0 and hit.iloc[0] == "present_but_undetected"
                if not present:
                    ok_all = False
                    break
            count += ok_all
        return count

    gain = pd.DataFrame([
        {
            "label": f"P_same>={p_same_accept:g}",
            "base": full_coverage(False, False, p_same_accept),
            "plus_override": full_coverage(True, False, p_same_accept),
            "plus_backprojection": full_coverage(True, True, p_same_accept),
        }
    ])
    provenance_full_at = {t: full_coverage(True, False, t) for t in np.round(np.arange(0.5, 0.996, 0.05), 2)}

    # P4: consistency per calibrated-P_same bin, on validation sessions only (held out from the isotonic fit)
    calibration_rows = []
    bins = np.linspace(0.0, 1.0, 11)
    validation_idx = [i for i in consistency_flags
                      if (not calibration_applied) or pairs.loc[i, "session_id"] in validation_sessions]
    val_p = pairs.loc[validation_idx, "p_same"].to_numpy(dtype=float)
    val_flags = np.asarray([consistency_flags[i] for i in validation_idx], dtype=float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = val_flags[(val_p >= lo) & (val_p < (hi + (1e-9 if hi == 1.0 else 0)))]
        n = len(in_bin)
        frac = float(np.mean(in_bin)) if n else np.nan
        calibration_rows.append({"bin_center": 0.5 * (lo + hi), "fraction_consistent": frac,
                                 "se": float(np.sqrt(frac * (1 - frac) / n)) if n and 0 < frac < 1 else 0.0, "n": n})
    calibration = pd.DataFrame(calibration_rows).dropna(subset=["fraction_consistent"])

    # R5 gallery crops for rescued cells
    rescued_cells = sorted({int(r) for s in accepted_rescue.values() for r in s})[:gallery_max_cells]
    crops: dict[int, dict[str, tuple[np.ndarray, np.ndarray | None, str]]] = {}
    prov_lookup = provenance.set_index(["reference_roi_id", "session_id"])["category"]
    pair_mask_lookup = {
        (row["session_id"], int(row["reference_roi_id"])): int(row["moving_roi_id"])
        for _, row in pairs.loc[pairs["candidate"] & (pairs["p_same"] >= p_same_accept)].iterrows()
    }
    for ref_id in rescued_cells:
        footprint = reference.footprints[ref_index[ref_id]]
        ys, xs = np.nonzero(footprint > 0)
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, x0 = max(cy - gallery_half_px, 0), max(cx - gallery_half_px, 0)
        size = 2 * gallery_half_px
        crops[ref_id] = {}
        for s in session_order:
            image = aligned_gray_by_session[s][y0 : y0 + size, x0 : x0 + size]
            outline = None
            m = pair_mask_lookup.get((s, ref_id))
            if m is not None:
                outline = warped_all[s][mov_index_by_session[s][m]][y0 : y0 + size, x0 : x0 + size] > 0
            crops[ref_id][s] = (image, outline, str(prov_lookup.loc[(ref_id, s)]))

    tables = {
        "refined_pairs_csv": (numerical_dir / "refined_pairs.csv", pairs),
        "backprojection_evidence_csv": (numerical_dir / "backprojection_evidence.csv", evidence_all),
        "provenance_csv": (numerical_dir / "track_day_provenance.csv", provenance),
        "rescue_gain_csv": (numerical_dir / "rescue_gain.csv", gain),
        "calibration_csv": (numerical_dir / "p_same_calibration.csv", calibration),
        "transitive_consistency_pairs_csv": (numerical_dir / "transitive_consistency_pairs.csv",
                                             consistency_pairs_table(pairs, flag_series, calibration_sessions)),
    }
    for path, frame in tables.values():
        frame.to_csv(path, index=False)

    figures = {
        "r1_provenance_matrix_png": figure_dir / "r1_provenance_matrix.png",
        "r2_provenance_bars_png": figure_dir / "r2_provenance_bars.png",
        "r3_backprojection_evidence_png": figure_dir / "r3_backprojection_evidence.png",
        "r4_rescue_gain_png": figure_dir / "r4_rescue_gain.png",
        "r5_rescued_gallery_png": figure_dir / "r5_rescued_gallery.png",
        "p1_feature_scatter_png": figure_dir / "p1_p_same_feature_scatter.png",
        "p2_posterior_histogram_png": figure_dir / "p2_p_same_histogram.png",
        "p3_threshold_curve_png": figure_dir / "p3_threshold_curve.png",
        "p4_calibration_png": figure_dir / "p4_calibration.png",
    }
    plot_provenance_matrix(provenance, session_order, figures["r1_provenance_matrix_png"], title_context=title_context, dpi=dpi)
    plot_provenance_stacked_bars(provenance, session_order, figures["r2_provenance_bars_png"], title_context=title_context, dpi=dpi)
    plot_backprojection_evidence(evidence_all, figures["r3_backprojection_evidence_png"], title_context=title_context, dpi=dpi)
    plot_rescue_gain(gain, figures["r4_rescue_gain_png"], title_context=title_context, dpi=dpi)
    plot_rescued_gallery(crops, session_order, figures["r5_rescued_gallery_png"], title_context=title_context, dpi=dpi)
    plot_p_same_scatter(pairs, reference.pixel_size_um, figures["p1_feature_scatter_png"], title_context=title_context, dpi=dpi, model=model)
    plot_p_same_histogram(pairs, p_same_accept, review_band, figures["p2_posterior_histogram_png"], title_context=title_context, dpi=dpi)
    plot_threshold_curve(pairs, provenance_full_at, figures["p3_threshold_curve_png"], title_context=title_context, dpi=dpi)
    plot_calibration(calibration, figures["p4_calibration_png"], title_context=title_context, dpi=dpi)
    if not figures["r5_rescued_gallery_png"].exists():
        figures.pop("r5_rescued_gallery_png")

    n_accept, efm = expected_false_matches(pairs["p_same"].to_numpy(), p_same_accept)
    return {
        "p_same_model": model.as_dict(),
        "calibration_applied": calibration_applied,
        "n_calibration_points": len(fit_idx),
        "calibration_split": {"fit_sessions": sorted(calibration_sessions), "validation_sessions": sorted(validation_sessions)},
        "n_pairs": int(len(pairs)),
        "n_accepted_pairs": n_accept,
        "expected_false_matches": round(efm, 2),
        "p_same_accept": p_same_accept,
        "full_coverage_base": int(gain["base"].iloc[0]),
        "full_coverage_plus_override": int(gain["plus_override"].iloc[0]),
        "full_coverage_plus_backprojection": int(gain["plus_backprojection"].iloc[0]),
        "n_present_but_undetected_cell_days": int((evidence_all["label"] == "present_but_undetected").sum()),
        "n_rescued_cells_in_gallery": len(rescued_cells),
        "thresholds_note": "all gates, evidence_z, and p_same_accept are provisional pending manual review",
        "output_files": {k: str(p) for k, (p, _) in tables.items()} | {k: str(p) for k, p in figures.items()},
    }


def _gray(prepared: np.ndarray) -> np.ndarray:
    return np.clip((prepared + 2.0) / 6.0, 0.0, 1.0)


def select_review_tracks(pairs: pd.DataFrame, moving_sessions: list[str], *, p_same_threshold: float, max_tracks: int | None = None) -> list[int]:
    """Reference ROI ids with a candidate pair at ``P_same >= threshold`` in every moving session."""
    ok = pairs.loc[pairs["candidate"].astype(bool) & (pairs["p_same"] >= p_same_threshold)]
    per_session = {s: set(g["reference_roi_id"].astype(int)) for s, g in ok.groupby("session_id")}
    ids = sorted(set.intersection(*[per_session.get(s, set()) for s in moving_sessions])) if moving_sessions else []
    return ids[:max_tracks] if max_tracks else ids


def run_review_gui(
    reference: SessionStructuralData,
    moving_sessions: list[SessionStructuralData],
    transforms: Mapping[str, RigidTransform],
    refined_pairs: pd.DataFrame,
    track_day_provenance: pd.DataFrame,
    output_root: Path,
    *,
    plane: str,
    session_labels: Mapping[str, str] | None = None,
    p_same_threshold: float = 0.90,
    review_policy: str = "full",
    auto_accept_threshold: float = 0.95,
    audit_fraction: float = 0.1,
    audit_seed: int = 47,
    max_tracks: int = 250,
    crop_half_px: int = 20,
    title: str | None = None,
    export_slug: str | None = None,
    extra_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``manual_review_<plane>.html`` + manifest; returns ``{"review_html", "n_tracks", "manifest"}``."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reference.validate()
    moving_ids = [s.session_id for s in moving_sessions]
    labels = dict(session_labels or {s: s for s in [reference.session_id, *moving_ids]})
    ref_label = "ref " + labels[reference.session_id]
    track_ids = select_review_tracks(refined_pairs, moving_ids, p_same_threshold=p_same_threshold, max_tracks=max_tracks)
    ok = refined_pairs.loc[refined_pairs["candidate"].astype(bool) & (refined_pairs["p_same"] >= p_same_threshold)]
    per_session = {s: {int(r.reference_roi_id): r for r in g.itertuples()} for s, g in ok.groupby("session_id")}

    gray = {reference.session_id: _gray(preprocess_projection(reference.projection))}
    warped_fp: dict[str, np.ndarray] = {}
    mov_index: dict[str, dict[int, int]] = {}
    for s in moving_sessions:
        s.validate()
        t = transforms[s.session_id]
        gray[s.session_id] = _gray(warp_image(preprocess_projection(s.projection), t.angle_deg, (t.shift_y_px, t.shift_x_px), order=1))
        warped_fp[s.session_id] = warp_footprints(s.footprints, t)
        mov_index[s.session_id] = {int(m): i for i, m in enumerate(s.roi_ids)}
    ref_index = {int(r): i for i, r in enumerate(reference.roi_ids)}
    prov_lookup = track_day_provenance.set_index(["reference_roi_id", "session_id"])["category"] if len(track_day_provenance) else pd.Series(dtype=object)

    half, size = crop_half_px, 2 * crop_half_px
    tracks = []
    for ref_id in track_ids:
        footprint = reference.footprints[ref_index[ref_id]]
        ys, xs = np.nonzero(footprint > 0)
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, x0 = max(cy - half, 0), max(cx - half, 0)
        crop = lambda img: img[y0:y0 + size, x0:x0 + size]  # noqa: E731
        cells = {ref_label: {"image_b64": encode_crop_png(crop(gray[reference.session_id]), crop(footprint > 0)),
                             "category": "matched", "tooltip": f"reference ROI {ref_id}"}}
        p_values = []
        for sid in moving_ids:
            pair = per_session[sid][ref_id]
            outline = crop(warped_fp[sid][mov_index[sid][int(pair.moving_roi_id)]] > 0)
            category = str(prov_lookup.get((ref_id, sid), "matched"))
            p_values.append(float(pair.p_same))
            cells[labels[sid]] = {"image_b64": encode_crop_png(crop(gray[sid]), outline), "category": category,
                                  "tooltip": f"P_same {pair.p_same:.3f} | IoU {pair.mask_iou:.2f} | {category}"}
        n_rescued = int(track_day_provenance.loc[track_day_provenance["reference_roi_id"].eq(ref_id), "category"]
                        .eq("rescued_classifier_override").sum()) if len(track_day_provenance) else 0
        tracks.append({"track_id": int(ref_id), "metrics": {"min P_same": round(min(p_values), 3), "n rescued": n_rescued}, "cells": cells})

    tracks.sort(key=lambda t: t["metrics"]["min P_same"])  # worst-first for the reviewer
    if review_policy == "triage":
        rng = np.random.default_rng(audit_seed)
        auto = [t for t in tracks if t["metrics"]["min P_same"] >= auto_accept_threshold]
        n_audit = int(round(audit_fraction * len(auto)))
        audit_ids = set(rng.choice([t["track_id"] for t in auto], size=n_audit, replace=False).tolist()) if n_audit else set()
        for t in tracks:
            is_auto = t["metrics"]["min P_same"] >= auto_accept_threshold and t["track_id"] not in audit_ids
            t["review_required"] = not is_auto
            t["metrics"]["review"] = "audit" if t["track_id"] in audit_ids else ("auto" if is_auto else "required")
        policy_note = (f" Review policy: TRIAGE - tracks with min P_same >= {auto_accept_threshold:g} are auto-accepted (override by "
                       f"selecting and deciding); a seeded {audit_fraction:.0%} audit sample plus all lower-confidence tracks require explicit decisions.")
    elif review_policy == "full":
        policy_note = " Review policy: FULL - every track requires an explicit decision."
    else:
        raise ValueError("review_policy must be 'full' or 'triage'")

    title = title or f"Manual track review - {plane} (reference {labels[reference.session_id]})"
    html_path = output_root / f"manual_review_{plane.lower()}.html"
    build_manual_review_html(
        tracks, [ref_label, *[labels[s] for s in moving_ids]], html_path, title=title,
        export_slug=export_slug if export_slug is not None else plane.lower().replace("_", ""),
        context_note=(f"{len(tracks)} candidate tracks with calibrated P_same >= {p_same_threshold} in all {len(moving_ids)} retained sessions. "
                      "All identities are candidates until accepted here; the exported CSV feeds the frozen cell manifest." + policy_note),
    )
    manifest = write_manifest(
        output_root / f"manifest_{plane.lower()}.json",
        artifact_type="manual_review_gui", created_by=SOURCE,
        parameters={"plane": plane, "p_same_threshold": p_same_threshold, "review_policy": review_policy,
                    "auto_accept_threshold": auto_accept_threshold, "audit_fraction": audit_fraction, "audit_seed": audit_seed,
                    "max_tracks": max_tracks, "crop_half_px": crop_half_px, **(extra_parameters or {})},
        output_files={"review_html": str(html_path)},
        extra={"n_tracks": len(tracks), "n_review_required": int(sum(t.get("review_required", True) for t in tracks)),
               "gui_title": title, "reference_session": reference.session_id, "moving_sessions": moving_ids},
    )
    return {"review_html": html_path, "n_tracks": len(tracks), "manifest": manifest}


def run_freeze_manifest(
    gui_run: Path,
    refinement_runs: Mapping[str, Path],
    output_root: Path,
    *,
    subject_id: str | None = None,
    p_same_threshold: float = 0.90,
    reviewer_note: str | None = None,
    extra_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``refinement_runs`` maps plane -> refinement run directory. Returns the manifest dict."""
    gui_run, output_root = Path(gui_run), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    summary: dict[str, dict[str, int]] = {}
    excluded: list[str] = []
    for plane, run_dir in refinement_runs.items():
        run_dir = Path(run_dir)
        decisions, decisions_path = load_review_decisions(gui_run, plane, subject_id)
        pairs = pd.read_csv(run_dir / "refined_pairs.csv")
        prov_path = run_dir / "track_day_provenance.csv"
        provenance = pd.read_csv(prov_path) if prov_path.exists() else None
        frozen = freeze_plane(plane, decisions, pairs, provenance, p_same_threshold=p_same_threshold)
        slug = plane_slug(plane)
        out_path = output_root / f"accepted_tracks_{slug}.csv"
        frozen.accepted_tracks.to_csv(out_path, index=False)
        outputs[f"accepted_tracks_{slug}_csv"] = str(out_path)
        input_hashes[f"decisions_{slug}_csv"] = sha256(decisions_path)
        input_hashes[f"refined_pairs_{slug}_csv"] = sha256(run_dir / "refined_pairs.csv")
        summary[plane] = frozen.summary
        run_manifest = run_dir / "manifest.json"
        if run_manifest.exists() and not excluded:
            excluded = json.loads(run_manifest.read_text()).get("excluded_sessions", [])
    manifest = write_manifest(
        output_root / "manifest.json",
        artifact_type="frozen_cell_manifest",
        created_by=SOURCE,
        parameters={"subject_id": subject_id, "p_same_track_threshold": p_same_threshold,
                    "planes": list(refinement_runs), **(extra_parameters or {})},
        output_files=outputs,
        extra={
            "reviewer": reviewer_note or f"manual review via {gui_run.name} (all tracks decided; decision sources recorded per plane)",
            "excluded_sessions": excluded,
            "selection_rule": ("Tracks shown to the reviewer were all candidates with calibrated P_same >= threshold in every retained "
                               "session; acceptance is the reviewer's structural judgment only (no activity, RF response, or model "
                               "information was visible in the GUI)."),
            "input_files": {"gui_run": str(gui_run), "refinement_runs": {p: str(r) for p, r in refinement_runs.items()}},
            "input_sha256": input_hashes,
            "summary": summary,
            "status": "frozen",
        },
    )
    return manifest


# ----------------------------------------------------------------------------- public stage names
register = run_structural_registration
diagnose = run_registration_diagnostics
refine = run_identity_refinement
review = run_review_gui
freeze = run_freeze_manifest
