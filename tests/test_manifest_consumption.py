"""Tests for frozen-manifest consumption helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from openscope_registration.diagnostics import (
    accepted_tracks_wide,
    derive_retained_and_reference,
)


def write_fixture(root: Path, excluded: list[str]) -> None:
    long = pd.DataFrame({
        "plane": ["p0"] * 4,
        "session_id": ["sess_a", "sess_b", "sess_a", "sess_b"],
        "reference_roi_id": [10, 10, 11, 11],
        "moving_roi_id": [101, 201, 102, 202],
    })
    long.to_csv(root / "accepted_tracks_p0.csv", index=False)
    (root / "manifest.json").write_text(json.dumps({"excluded_sessions": excluded}))


class ManifestConsumptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_fixture(self.root, excluded=["sess_bad"])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_wide_pivot_includes_reference_identity(self) -> None:
        wide = accepted_tracks_wide(self.root, "p0", "sess_ref")
        self.assertEqual(list(wide.index), [10, 11])
        self.assertEqual(wide.loc[10, "sess_a"], 101)
        self.assertEqual(wide.loc[11, "sess_b"], 202)
        self.assertEqual(wide.loc[10, "sess_ref"], 10)

    def test_reference_derivation(self) -> None:
        eligible = {"sess_a", "sess_b", "sess_bad", "sess_ref"}
        sessions, reference = derive_retained_and_reference([self.root], eligible)
        self.assertEqual(reference, "sess_ref")
        self.assertEqual(sessions, ["sess_a", "sess_b", "sess_ref"])

    def test_ambiguous_reference_raises(self) -> None:
        eligible = {"sess_a", "sess_b", "sess_ref", "sess_ref2"}
        with self.assertRaises(ValueError):
            derive_retained_and_reference([self.root], eligible)

    def test_manifest_session_outside_eligible_raises(self) -> None:
        with self.assertRaises(ValueError):
            derive_retained_and_reference([self.root], {"sess_a", "sess_ref"})


if __name__ == "__main__":
    unittest.main()
