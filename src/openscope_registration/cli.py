"""Command line: ``osreg <stage> [options]`` for the five pipeline stages.

Every stage that reads imaging data accepts ``--source openscope`` (AIND S3
planar NWB zarr; requires the ``[openscope]`` extra) or ``--source npz
--npz-dir DIR`` (local ``<session_id>.npz`` files written by
``openscope_registration.io.save_npz_session``). Stages ``diagnose``,
``refine`` and ``review`` read the reference session, session order, and
transforms from the stage-1 run directory via ``--registration-root``.

Example (synthetic data, no network)::

    python examples/run_synthetic_pipeline.py out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline
from .io.npz import add_source_arguments, load_sessions, load_transforms, sessions_from_registration, short_label
from .manifests import write_manifest


def _reference_soma_only(args, reference_id: str, plane: str, loaded: dict):
    """For refine/review the reference must be soma-only while moving sessions carry all ROIs."""
    if args.source == "openscope":
        import s3fs

        from .io.openscope import load_session_structural_data

        return load_session_structural_data(s3fs.S3FileSystem(anon=True), reference_id, plane, soma_only=True)
    return loaded[reference_id]


# ----------------------------------------------------------------------------- register


def _add_register(sub) -> None:
    p = sub.add_parser("register", help="Stage 1-2: rigid alignment + one-to-one candidate matching for all planes of a subject")
    p.add_argument("--sessions", nargs="+", required=True, help="All session ids including the reference")
    p.add_argument("--reference-session", required=True)
    p.add_argument("--planes", nargs="+", required=True)
    p.add_argument("--numerical-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--maximum-angle-deg", type=float, default=3.0)
    p.add_argument("--angle-step-deg", type=float, default=0.25)
    p.add_argument("--max-shift-px", type=float, default=64.0)
    p.add_argument("--tile-size-px", type=int, default=64)
    p.add_argument("--candidate-iou", type=float, default=0.5, help="Provisional candidate gate, not an acceptance rule")
    p.add_argument("--max-centroid-px", type=float, default=4.0)
    p.add_argument("--min-iou-margin", type=float, default=0.2)
    p.add_argument("--overlap-iou", type=float, default=0.2)
    p.add_argument("--no-fov-correspondence", action="store_true")
    add_source_arguments(p)
    p.set_defaults(func=_run_register)


def _run_register(args) -> None:
    sessions = sorted(set(args.sessions))
    if args.reference_session not in sessions:
        raise SystemExit("--reference-session must be one of --sessions")
    sessions_by_plane = {plane: load_sessions(args, sessions, plane) for plane in args.planes}
    result = pipeline.run_structural_registration(
        sessions_by_plane, args.reference_session, args.numerical_dir, session_order=sessions,
        include_fov_correspondence=not args.no_fov_correspondence, overwrite=args.overwrite,
        maximum_angle_deg=args.maximum_angle_deg, angle_step_deg=args.angle_step_deg, max_shift_px=args.max_shift_px,
        tile_size_px=args.tile_size_px, candidate_iou=args.candidate_iou, max_centroid_px=args.max_centroid_px,
        min_iou_margin=args.min_iou_margin, overlap_iou=args.overlap_iou, extra_parameters={"source": args.source},
    )
    print(json.dumps(result["summary"], indent=2, default=str))


# ----------------------------------------------------------------------------- diagnose


def _add_diagnose(sub) -> None:
    p = sub.add_parser("diagnose", help="Stage 3a: diagnostics battery for one plane using a stage-1 run's transforms")
    p.add_argument("--registration-root", type=Path, required=True)
    p.add_argument("--plane", required=True)
    p.add_argument("--numerical-dir", type=Path, required=True)
    p.add_argument("--figure-dir", type=Path, required=True)
    p.add_argument("--exclude-sessions", nargs="*", default=[])
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--title-context", default="")
    p.add_argument("--dpi", type=int, default=300)
    add_source_arguments(p)
    p.set_defaults(func=_run_diagnose)


def _run_diagnose(args) -> None:
    reference_id, sessions = sessions_from_registration(args.registration_root, set(args.exclude_sessions))
    loaded = load_sessions(args, sessions, args.plane)
    transforms = load_transforms(args.registration_root, args.plane)
    moving = [loaded[s] for s in sessions if s != reference_id]
    summary = pipeline.run_registration_diagnostics(
        loaded[reference_id], moving, args.numerical_dir, args.figure_dir, transforms=transforms,
        n_null=args.n_null, n_permutations=args.n_permutations,
        title_context=args.title_context or f"{args.plane}, reference {reference_id}", dpi=args.dpi,
    )
    if "output_files" in summary:
        write_manifest(args.numerical_dir / "manifest.json", artifact_type="registration_diagnostics_battery", created_by="osreg diagnose",
                       parameters={"plane": args.plane, "registration_root": str(args.registration_root), "source": args.source,
                                   "n_null": args.n_null, "n_permutations": args.n_permutations},
                       output_files=summary["output_files"],
                       extra={"excluded_sessions": list(args.exclude_sessions), "reference_session": reference_id,
                              "summary": {k: v for k, v in summary.items() if k != "output_files"}})
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (dict, list))}, indent=2, default=str))


# ----------------------------------------------------------------------------- refine


def _add_refine(sub) -> None:
    p = sub.add_parser("refine", help="Stage 3b: P_same mixture + isotonic calibration + rescue pass for one plane")
    p.add_argument("--registration-root", type=Path, required=True)
    p.add_argument("--plane", required=True)
    p.add_argument("--numerical-dir", type=Path, required=True)
    p.add_argument("--figure-dir", type=Path, required=True)
    p.add_argument("--exclude-sessions", nargs="*", default=[])
    p.add_argument("--p-same-accept", type=float, default=0.95)
    p.add_argument("--review-band", type=float, nargs=2, default=(0.5, 0.9))
    p.add_argument("--evidence-z", type=float, default=3.0)
    p.add_argument("--title-context", default="")
    p.add_argument("--dpi", type=int, default=300)
    add_source_arguments(p)
    p.set_defaults(func=_run_refine)


def _run_refine(args) -> None:
    reference_id, sessions = sessions_from_registration(args.registration_root, set(args.exclude_sessions))
    args.all_rois = True  # moving sessions carry all ROIs (with is_soma flags) for the rescue pass
    loaded = load_sessions(args, sessions, args.plane)
    reference = _reference_soma_only(args, reference_id, args.plane, loaded)
    for s in loaded.values():
        s.metadata.setdefault("is_soma", np.ones(s.n_rois, dtype=bool))
    moving = [loaded[s] for s in sessions if s != reference_id]
    summary = pipeline.run_identity_refinement(
        reference, moving, args.numerical_dir, args.figure_dir, transforms=load_transforms(args.registration_root, args.plane),
        p_same_accept=args.p_same_accept, review_band=tuple(args.review_band), evidence_z=args.evidence_z,
        title_context=args.title_context or f"{args.plane}, reference {reference_id}", dpi=args.dpi,
    )
    write_manifest(args.numerical_dir / "manifest.json", artifact_type="identity_refinement", created_by="osreg refine",
                   parameters={"plane": args.plane, "registration_root": str(args.registration_root), "source": args.source,
                               "p_same_accept": args.p_same_accept, "review_band": list(args.review_band), "evidence_z": args.evidence_z},
                   output_files=summary["output_files"],
                   extra={"excluded_sessions": list(args.exclude_sessions), "reference_session": reference_id,
                          "summary": {k: v for k, v in summary.items() if k != "output_files"}})
    print(json.dumps({k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))}, indent=2, default=str))


# ----------------------------------------------------------------------------- review


def _add_review(sub) -> None:
    p = sub.add_parser("review", help="Stage 4: build the manual-review HTML GUI for one plane")
    p.add_argument("--registration-root", type=Path, required=True)
    p.add_argument("--refinement-run", type=Path, required=True)
    p.add_argument("--plane", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--p-same-track-threshold", type=float, default=0.90)
    p.add_argument("--review-policy", choices=["full", "triage"], default="full")
    p.add_argument("--auto-accept-threshold", type=float, default=0.95)
    p.add_argument("--audit-fraction", type=float, default=0.1)
    p.add_argument("--audit-seed", type=int, default=47)
    p.add_argument("--max-tracks", type=int, default=250)
    p.add_argument("--crop-half-px", type=int, default=20)
    p.add_argument("--export-slug", default=None, help="Suffix of the exported decisions CSV (default: plane slug)")
    add_source_arguments(p)
    p.set_defaults(func=_run_review)


def _run_review(args) -> None:
    refinement = json.loads((args.refinement_run / "manifest.json").read_text())
    reference_id, sessions = sessions_from_registration(args.registration_root, set(refinement.get("excluded_sessions", [])))
    args.all_rois = True
    loaded = load_sessions(args, sessions, args.plane)
    loaded[reference_id] = _reference_soma_only(args, reference_id, args.plane, loaded)
    moving = [loaded[s] for s in sessions if s != reference_id]
    result = pipeline.run_review_gui(
        loaded[reference_id], moving, load_transforms(args.registration_root, args.plane),
        pd.read_csv(args.refinement_run / "refined_pairs.csv"), pd.read_csv(args.refinement_run / "track_day_provenance.csv"),
        args.output_root, plane=args.plane, session_labels={s: short_label(s) for s in sessions},
        p_same_threshold=args.p_same_track_threshold, review_policy=args.review_policy,
        auto_accept_threshold=args.auto_accept_threshold, audit_fraction=args.audit_fraction, audit_seed=args.audit_seed,
        max_tracks=args.max_tracks, crop_half_px=args.crop_half_px, export_slug=args.export_slug,
        extra_parameters={"refinement_run": str(args.refinement_run), "registration_root": str(args.registration_root)},
    )
    print(json.dumps({"review_html": str(result["review_html"]), "n_tracks": result["n_tracks"]}, indent=2))


# ----------------------------------------------------------------------------- freeze


def _add_freeze(sub) -> None:
    p = sub.add_parser("freeze", help="Stage 5: freeze the reviewed cell manifest from GUI decisions + refinement runs")
    p.add_argument("--gui-run", type=Path, required=True, help="Directory holding manual_review_decisions_*.csv exports")
    p.add_argument("--planes", nargs="+", required=True)
    p.add_argument("--refinement-runs", nargs="+", type=Path, required=True, help="One per plane, same order")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--subject-id", default=None, help="Prefix used in the decisions CSV names, if any")
    p.add_argument("--p-same-track-threshold", type=float, default=0.90, help="Must equal the GUI inclusion threshold")
    p.set_defaults(func=_run_freeze)


def _run_freeze(args) -> None:
    if len(args.planes) != len(args.refinement_runs):
        raise SystemExit("--planes and --refinement-runs must have the same length")
    manifest = pipeline.run_freeze_manifest(args.gui_run, dict(zip(args.planes, args.refinement_runs)), args.output_root,
                                            subject_id=args.subject_id, p_same_threshold=args.p_same_track_threshold)
    print(json.dumps(manifest["summary"], indent=2))


# ----------------------------------------------------------------------------- entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osreg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    for add in (_add_register, _add_diagnose, _add_refine, _add_review, _add_freeze):
        add(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
