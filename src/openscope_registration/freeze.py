"""Freeze a reviewed cross-session cell manifest (pipeline stage 5).

Joins reviewer decisions (exported from the manual-review GUI) with the
calibrated identity-refinement pairs to produce, per plane, the frozen table of
accepted tracks: reference ROI id -> moving-session ROI id for every retained
session, with per-session calibrated ``P_same``, mechanism, and provenance.

The frozen manifest is the registration-gate output every downstream analysis
must consume; nothing after this point may revisit a match in light of the
results it produces. Integrity rules enforced here:

* every reviewed track has a decision (no ``undecided`` rows);
* an accepted track keeps only pairs that were candidates with
  ``P_same >= threshold`` (the same threshold the GUI used for inclusion);
* an accepted track must then cover every retained session exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PAIR_COLUMNS = ["session_id", "reference_roi_id", "moving_roi_id", "p_same", "p_same_raw", "mask_iou",
                "centroid_distance_px", "footprint_correlation", "mechanism"]


def plane_slug(plane: str) -> str:
    """``VISp_0`` -> ``visp0``; the slug used in decision/accepted-track file names."""
    return plane.lower().replace("_", "")


def load_review_decisions(gui_run: Path, plane: str, subject_id: str | None = None) -> tuple[pd.DataFrame, Path]:
    """Read ``manual_review_decisions_<subject>_<slug>.csv`` (or the legacy un-prefixed name)."""
    slug = plane_slug(plane)
    candidates = []
    if subject_id is not None:
        candidates.append(gui_run / f"manual_review_decisions_{subject_id}_{slug}.csv")
    candidates.append(gui_run / f"manual_review_decisions_{slug}.csv")
    for path in candidates:
        if path.exists():
            decisions = pd.read_csv(path)
            if "source" not in decisions.columns:
                decisions["source"] = "manual"
            return decisions, path
    raise FileNotFoundError(f"no review decisions for {plane} under {gui_run}")


@dataclass
class FrozenPlane:
    plane: str
    accepted_tracks: pd.DataFrame
    summary: dict[str, int] = field(default_factory=dict)


def freeze_plane(
    plane: str,
    decisions: pd.DataFrame,
    refined_pairs: pd.DataFrame,
    track_day_provenance: pd.DataFrame | None,
    *,
    p_same_threshold: float = 0.90,
) -> FrozenPlane:
    """Apply the acceptance rule and integrity checks for one plane; raises on incomplete review or broken coverage."""
    decisions = decisions.copy()
    if "source" not in decisions.columns:
        decisions["source"] = "manual"
    if decisions["decision"].eq("undecided").any():
        raise ValueError(f"{plane}: undecided tracks remain; review incomplete")
    accepted_ids = set(decisions.loc[decisions["decision"].eq("accept"), "track_id"].astype(int))
    cols = [c for c in PAIR_COLUMNS if c in refined_pairs.columns]
    keep = refined_pairs.loc[
        refined_pairs["candidate"].astype(bool) & (refined_pairs["p_same"] >= p_same_threshold)
        & refined_pairs["reference_roi_id"].astype(int).isin(accepted_ids), cols].copy()
    if track_day_provenance is not None:
        keep = keep.merge(track_day_provenance, on=["reference_roi_id", "session_id"], how="left")
    keep.insert(0, "plane", plane)
    keep = keep.sort_values(["reference_roi_id", "session_id"]).reset_index(drop=True)
    n_sessions = int(keep["session_id"].nunique())
    counts = keep.groupby("reference_roi_id")["session_id"].nunique()
    incomplete = counts.index[counts != n_sessions].tolist()
    if incomplete:
        raise ValueError(f"{plane}: accepted tracks missing sessions at threshold: {incomplete}")
    missing_accept = sorted(accepted_ids - set(counts.index))
    if missing_accept:
        raise ValueError(f"{plane}: reviewer-accepted tracks with no qualifying pairs: {missing_accept}")
    summary = {
        "n_reviewed": int(len(decisions)),
        "n_accepted": int(len(accepted_ids)),
        "n_rejected": int(decisions["decision"].eq("reject").sum()),
        "n_unsure": int(decisions["decision"].eq("unsure").sum()),
        "n_accepted_manual": int((decisions["decision"].eq("accept") & decisions["source"].eq("manual")).sum()),
        "n_accepted_auto": int((decisions["decision"].eq("accept") & decisions["source"].eq("auto")).sum()),
        "n_sessions": n_sessions,
        "n_rescued_track_days": int(keep["mechanism"].eq("classifier_override").sum()) if "mechanism" in keep else 0,
    }
    return FrozenPlane(plane, keep, summary)
