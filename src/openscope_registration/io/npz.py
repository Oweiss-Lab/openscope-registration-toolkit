"""Local .npz session format and helpers shared by the CLI (session loading, transforms, session order).

Sessions are specified as ``--source openscope`` (AIND OpenScope planar NWB zarr on
S3 via the adaptor) or ``--source npz`` (local ``<dir>/<session_id>.npz`` files with
``projection``, ``footprints``, ``roi_ids``, ``pixel_size_um`` and optional
``depth_um``), so every stage can run on public data or on a local export without
any study-specific path registry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import RigidTransform, SessionStructuralData


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["openscope", "npz"], default="openscope",
                        help="openscope: AIND S3 planar NWB zarr (needs s3fs, zarr); npz: local <npz-dir>/<session_id>.npz")
    parser.add_argument("--npz-dir", type=Path, default=None, help="Directory of per-session .npz files (source=npz)")
    parser.add_argument("--all-rois", action="store_true", help="Load all ROIs, not only is_soma (openscope source)")
    parser.add_argument("--n-io-workers", type=int, default=8)


def save_npz_session(path: Path, session: SessionStructuralData) -> Path:
    """Write a validated session to the local ``.npz`` format read by :func:`load_npz_session`."""
    session.validate()
    extra = {"depth_um": float(session.metadata["depth_um"])} if "depth_um" in session.metadata else {}
    np.savez_compressed(path, projection=session.projection, footprints=session.footprints, roi_ids=session.roi_ids,
                        pixel_size_um=session.pixel_size_um, **extra)
    return Path(path)


def load_npz_session(path: Path, session_id: str, plane: str) -> SessionStructuralData:
    with np.load(path, allow_pickle=False) as z:
        meta = {"plane": plane}
        if "depth_um" in z:
            meta["depth_um"] = float(z["depth_um"])
        return SessionStructuralData(session_id=session_id, projection=np.asarray(z["projection"], float),
                                     footprints=np.asarray(z["footprints"], float), roi_ids=np.asarray(z["roi_ids"], int),
                                     pixel_size_um=float(z["pixel_size_um"]), metadata=meta).validate()


def load_sessions(args: argparse.Namespace, session_ids: list[str], plane: str) -> dict[str, SessionStructuralData]:
    """Load every session of one plane in parallel according to ``--source``."""
    if args.source == "npz":
        if args.npz_dir is None:
            raise SystemExit("--npz-dir is required with --source npz")
        return {s: load_npz_session(args.npz_dir / f"{s}.npz", s, plane) for s in session_ids}
    import s3fs

    from .openscope import load_session_structural_data

    fs = s3fs.S3FileSystem(anon=True)
    with ThreadPoolExecutor(max_workers=args.n_io_workers) as pool:
        loaded = list(pool.map(lambda s: load_session_structural_data(fs, s, plane, soma_only=not args.all_rois), session_ids))
    return dict(zip(session_ids, loaded))


def load_transforms(registration_root: Path, plane: str) -> dict[str, RigidTransform]:
    """Per-session rigid transforms for one plane from a structural-registration run."""
    tf = pd.read_csv(Path(registration_root) / "registration_transforms.csv")
    tf = tf.loc[tf["plane"].eq(plane)]
    return {r.session_id: RigidTransform(float(r.angle_deg), float(r.shift_y_px), float(r.shift_x_px), float(getattr(r, "rigid_ncc", np.nan)))
            for r in tf.itertuples()}


def sessions_from_registration(registration_root: Path, excluded: set[str] | None = None) -> tuple[str, list[str]]:
    """(reference_session, ordered session list) recorded by a structural-registration run."""
    import json

    m = json.loads((Path(registration_root) / "manifest.json").read_text())
    params = m.get("parameters", m)
    ref = params.get("reference_session") or m.get("reference_session")
    sessions = params.get("sessions") or m.get("sessions")
    excluded = excluded or set()
    return ref, [s for s in sessions if s not in excluded]


def short_label(session_id: str) -> str:
    """Compact session label: the date token if the id looks like ``<name>_<subject>_<date>_<time>``."""
    parts = session_id.split("_")
    return parts[2][5:] if len(parts) >= 3 and len(parts[2]) >= 10 else session_id
