"""Adaptor: AIND OpenScope planar-ophys NWB zarr assets on S3 -> SessionStructuralData.

Owns all source-specific parsing: processed-asset resolution, zarr layout,
soma classification filter, pixel geometry, and depth metadata. Returns a
validated contract; no analysis or figures here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data import SessionStructuralData

BUCKET = "aind-open-data"


def resolve_planar_asset(fs: Any, session_id: str, bucket: str = BUCKET) -> str:
    """Latest processed asset containing a planar pophys.nwb.zarr for the session."""
    prefixes = sorted(p.split("/", 1)[1] for p in fs.glob(f"{bucket}/{session_id}_processed_*"))
    planar = [p for p in prefixes if fs.exists(f"{bucket}/{p}/pophys.nwb.zarr/.zgroup")]
    if not planar:
        raise FileNotFoundError(f"No planar pophys.nwb.zarr asset for {session_id}")
    return f"s3://{bucket}/{planar[-1]}"


def load_session_structural_data(
    fs: Any,
    session_id: str,
    plane: str,
    *,
    planar_asset: str | None = None,
    soma_only: bool = True,
    bucket: str = BUCKET,
) -> SessionStructuralData:
    """Load one plane's average projection + soma footprints into the contract."""
    import zarr

    asset = planar_asset or resolve_planar_asset(fs, session_id, bucket)
    root = asset.replace("s3://", "") + "/pophys.nwb.zarr"
    group = zarr.open(fs.get_mapper(f"{root}/processing/{plane}"), mode="r")
    meta = zarr.open(fs.get_mapper(f"{root}/general/optophysiology/{plane}"), mode="r")
    table = group["image_segmentation"]["roi_table"]
    roi_ids = np.asarray(table["id"][:], dtype=int)
    is_soma = np.asarray(table["is_soma"][:], dtype=bool)
    keep_rows = np.flatnonzero(is_soma) if soma_only else np.arange(len(roi_ids))
    footprints = np.asarray(table["image_mask"].get_orthogonal_selection((keep_rows, slice(None), slice(None))), dtype=float)
    grid_spacing = [float(v) for v in np.asarray(meta["grid_spacing"][()]).ravel()]
    location = str(np.asarray(meta["location"][()]).ravel()[0])
    depth_um = float(location.split("Depth:")[-1]) if "Depth:" in location else float("nan")
    return SessionStructuralData(
        session_id=session_id,
        projection=np.asarray(group["images"]["average_projection"][:], dtype=float),
        footprints=footprints,
        roi_ids=roi_ids[keep_rows],
        pixel_size_um=float(np.mean(grid_spacing)),
        metadata={
            "planar_asset": asset,
            "plane": plane,
            "location": location,
            "depth_um": depth_um,
            "roi_filter": "is_soma == 1" if soma_only else "all ROIs",
            "is_soma": is_soma[keep_rows],
            "projection_source": "processing/<plane>/images/average_projection",
            "imaging_rate_hz": float(np.asarray(meta["imaging_rate"][()]).ravel()[0]),
        },
    ).validate()
