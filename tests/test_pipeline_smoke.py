"""End-to-end smoke test: all five pipeline stages via the CLI entrypoints on synthetic sessions.

Stage 1-2 structural registration -> 3a diagnostics battery -> 3b identity
refinement -> 4 review GUI -> (synthetic reviewer accepts every shown track) ->
5 freeze -> downstream consumer reads the frozen manifest. Runs on local .npz
sessions (``--source npz``), so it needs no network access.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from openscope_registration.diagnostics import accepted_tracks_wide
from openscope_registration.cli import main as osreg
from .test_registration_diagnostics import make_session

SESSIONS = ["sub_000_2026-01-01_00-00-00", "sub_000_2026-01-03_00-00-00", "sub_000_2026-01-05_00-00-00"]
REFERENCE = SESSIONS[1]


def _write_npz(npz_dir: Path) -> None:
    npz_dir.mkdir(parents=True)
    specs = {SESSIONS[0]: dict(angle_deg=1.0, shift=(2.0, -1.5), seed=2),
             SESSIONS[1]: dict(seed=1),
             SESSIONS[2]: dict(angle_deg=-0.75, shift=(-1.0, 2.5), seed=3)}
    for sid, kw in specs.items():
        s = make_session(sid, **kw)
        np.savez(npz_dir / f"{sid}.npz", projection=s.projection, footprints=s.footprints, roi_ids=s.roi_ids,
                 pixel_size_um=s.pixel_size_um, depth_um=200.0)


class PipelineSmokeTest(unittest.TestCase):
    def test_five_stages_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root = Path(tmp)
            npz = root / "sessions"
            _write_npz(npz)
            src = ["--source", "npz", "--npz-dir", str(npz)]

            reg = root / "registration"
            osreg(["register", "--sessions", *SESSIONS, "--reference-session", REFERENCE, "--planes", "P0",
                                          "--numerical-dir", str(reg), "--angle-step-deg", "0.5", "--tile-size-px", "48", *src])
            self.assertTrue((reg / "registration_transforms.csv").exists())
            tf = pd.read_csv(reg / "registration_transforms.csv")
            self.assertEqual(len(tf), 2)
            self.assertTrue((tf["rigid_ncc"] > 0.8).all())

            battery = root / "battery"
            osreg(["diagnose", "--registration-root", str(reg), "--plane", "P0", "--numerical-dir", str(battery),
                                      "--figure-dir", str(root / "battery_figs"), "--n-null", "20", "--n-permutations", "20", "--dpi", "50", *src])
            self.assertTrue((battery / "residual_displacements.csv").exists())

            refine = root / "refine"
            osreg(["refine", "--registration-root", str(reg), "--plane", "P0", "--numerical-dir", str(refine),
                                      "--figure-dir", str(root / "refine_figs"), "--dpi", "50", *src])
            pairs = pd.read_csv(refine / "refined_pairs.csv")
            self.assertTrue((refine / "transitive_consistency_pairs.csv").exists())
            self.assertGreater(int(pairs["candidate"].sum()), 20)

            gui = root / "review"
            osreg(["review", "--registration-root", str(reg), "--refinement-run", str(refine), "--plane", "P0",
                             "--output-root", str(gui), "--crop-half-px", "10", *src])
            self.assertTrue((gui / "manual_review_p0.html").exists())
            n_tracks = json.loads((gui / "manifest_p0.json").read_text())["n_tracks"]
            self.assertGreater(n_tracks, 10)

            # synthetic reviewer: accept everything the GUI showed (the GUI export format)
            shown = sorted(set(pairs.loc[pairs["candidate"] & (pairs["p_same"] >= 0.90)]
                               .groupby("reference_roi_id")["session_id"].nunique().pipe(lambda c: c[c == 2]).index))
            pd.DataFrame({"track_id": shown, "decision": "accept", "source": "manual"}).to_csv(gui / "manual_review_decisions_p0.csv", index=False)

            frozen = root / "frozen"
            osreg(["freeze", "--gui-run", str(gui), "--planes", "P0", "--refinement-runs", str(refine), "--output-root", str(frozen)])
            manifest = json.loads((frozen / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "frozen")
            self.assertEqual(manifest["summary"]["P0"]["n_accepted"], len(shown))
            wide = accepted_tracks_wide(frozen, "p0", REFERENCE)
            self.assertEqual(set(wide.columns), set(SESSIONS))
            self.assertEqual(len(wide), len(shown))
            # identical cell layout across synthetic sessions -> identity mapping
            for sid in SESSIONS:
                self.assertTrue((wide[sid].to_numpy() == wide.index.to_numpy()).all(), sid)


if __name__ == "__main__":
    unittest.main()
