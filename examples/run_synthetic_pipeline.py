#!/usr/bin/env python3
"""Run the whole five-stage pipeline on synthetic sessions, with no network access.

    python examples/run_synthetic_pipeline.py out/

Three synthetic sessions share one cell layout; two are rotated and shifted.
After the run, open ``out/review/manual_review_p0.html`` to see the review GUI;
this script plays the reviewer by accepting every shown track so that the
freeze step can complete, then reads the frozen manifest back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from openscope_registration import SessionStructuralData, accepted_tracks_wide, warp_image
from openscope_registration.cli import main as osreg
from openscope_registration.io import save_npz_session

SESSIONS = ["demo_000_2026-01-01_00-00-00", "demo_000_2026-01-03_00-00-00", "demo_000_2026-01-05_00-00-00"]
REFERENCE = SESSIONS[1]


def make_session(session_id: str, *, angle_deg: float = 0.0, shift=(0.0, 0.0), seed: int = 0, size: int = 192) -> SessionStructuralData:
    rng = np.random.default_rng(7)  # same cell layout for every session
    grid = np.array([(y, x) for y in range(30, size - 20, 26) for x in range(30, size - 20, 26)])
    centers = grid[rng.permutation(len(grid))[:30]] + rng.integers(-4, 5, size=(30, 2))
    yy, xx = np.mgrid[:size, :size]
    footprints = np.asarray([((yy - cy) ** 2 + (xx - cx) ** 2 <= 16).astype(float) for cy, cx in centers])
    projection = ndimage.gaussian_filter((footprints * rng.uniform(0.5, 1.5, (30, 1, 1))).sum(0), 1.0)
    projection = projection + 0.02 * np.random.default_rng(seed).standard_normal(projection.shape)
    if angle_deg or shift != (0.0, 0.0):
        projection = warp_image(projection, angle_deg, shift, order=1)
        footprints = np.asarray([warp_image(m, angle_deg, shift, order=0) for m in footprints])
    return SessionStructuralData(session_id, projection, np.clip(footprints, 0, None), np.arange(30), 0.78, {"depth_um": 200.0}).validate()


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "sessions"
    npz.mkdir(exist_ok=True)
    for sid, kw in {SESSIONS[0]: dict(angle_deg=1.0, shift=(2.0, -1.5), seed=2), SESSIONS[1]: dict(seed=1),
                    SESSIONS[2]: dict(angle_deg=-0.75, shift=(-1.0, 2.5), seed=3)}.items():
        save_npz_session(npz / f"{sid}.npz", make_session(sid, **kw))
    src = ["--source", "npz", "--npz-dir", str(npz)]

    osreg(["register", "--sessions", *SESSIONS, "--reference-session", REFERENCE, "--planes", "P0",
           "--numerical-dir", str(out / "registration"), "--overwrite", "--angle-step-deg", "0.5", "--tile-size-px", "48", *src])
    osreg(["diagnose", "--registration-root", str(out / "registration"), "--plane", "P0",
           "--numerical-dir", str(out / "battery"), "--figure-dir", str(out / "battery_figs"), "--n-null", "50", "--n-permutations", "100", *src])
    osreg(["refine", "--registration-root", str(out / "registration"), "--plane", "P0",
           "--numerical-dir", str(out / "refine"), "--figure-dir", str(out / "refine_figs"), *src])
    osreg(["review", "--registration-root", str(out / "registration"), "--refinement-run", str(out / "refine"), "--plane", "P0",
           "--output-root", str(out / "review"), *src])

    # play the reviewer: accept every track the GUI showed
    pairs = pd.read_csv(out / "refine" / "refined_pairs.csv")
    shown = (pairs.loc[pairs["candidate"] & (pairs["p_same"] >= 0.90)].groupby("reference_roi_id")["session_id"].nunique())
    shown = sorted(shown[shown == len(SESSIONS) - 1].index)
    pd.DataFrame({"track_id": shown, "decision": "accept", "source": "manual"}).to_csv(out / "review" / "manual_review_decisions_p0.csv", index=False)

    osreg(["freeze", "--gui-run", str(out / "review"), "--planes", "P0", "--refinement-runs", str(out / "refine"), "--output-root", str(out / "frozen")])
    manifest = json.loads((out / "frozen" / "manifest.json").read_text())
    wide = accepted_tracks_wide(out / "frozen", "p0", REFERENCE)
    print(f"\nfrozen manifest status={manifest['status']}; {len(wide)} accepted tracks across {wide.shape[1]} sessions")
    print(wide.head())


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "out"))
