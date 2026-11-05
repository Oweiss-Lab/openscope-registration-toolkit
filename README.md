# openscope-registration-toolkit

Cross-session (longitudinal) cell registration for two-photon calcium imaging:
align each day's field of view to a reference day, match cells one-to-one by
pixel-mask overlap, score every match with a calibrated same-cell probability,
inspect a diagnostic battery with chance-level controls, review candidates by
eye, and freeze a manifest of accepted cell identities that downstream analyses
consume without ever revisiting a match.

Developed in the Oweiss Lab for the AIND OpenScope mesoscope receptive-field
dataset (10 mice, 8 planes, 5–9 sessions over ~5 weeks), but the core takes any
session as a projection image plus per-cell pixel masks.

```bash
pip install git+https://github.com/Oweiss-Lab/openscope-registration-toolkit.git
# or, for the AIND OpenScope S3 loader:
pip install "openscope-registration-toolkit[openscope] @ git+https://github.com/Oweiss-Lab/openscope-registration-toolkit.git"
```

Try it with no data and no network:

```bash
python examples/run_synthetic_pipeline.py out/      # runs all five stages on synthetic sessions
```

## The five stages

| # | `osreg` subcommand | What it does | Writes |
|---|---|---|---|
| 1–2 | `register` | For every moving session and plane: rigid transform onto the reference (rotation grid search + bounded phase correlation, scored by masked NCC on the overlap), inverse-consistency check, per-tile residual shifts; warp pixel masks, build the IoU table, Hungarian one-to-one assignment with provisional gates, consecutive-session assignments, template-anchored candidate tracks; cross-plane field-of-view correspondence | `registration_transforms.csv`, `roi_match_candidates.csv`, `candidate_track_coverage.csv`, `residual_tile_shifts.csv`, `fov_correspondence_ncc.csv`, `manifest.json` |
| 3a | `diagnose` | Diagnostic battery with nulls: centroid distances, mask correlation vs shuffled pairs, per-cell residual displacement map, local-geometry permutation test, coverage matrix, footprint-size change (Z-drift probe) | battery CSVs + one figure per diagnostic |
| 3b | `refine` | Two-component mixture on (mask correlation, log centroid distance) → raw posterior; adjacent-day **transitive consistency** as an independent signal; isotonic recalibration fit on even-indexed sessions and checked on odd-indexed ones → `P_same`; rescue pass for cells missed by soma segmentation; P1–P4 / R1–R5 figures | `refined_pairs.csv`, `transitive_consistency_pairs.csv`, `p_same_calibration.csv`, `track_day_provenance.csv` |
| 4 | `review` | Self-contained HTML GUI: every track with `P_same ≥ threshold` on all days, structural crops with the matched mask outline, keyboard accept/reject/unsure, CSV export; structural evidence only, no activity shown | `manual_review_<plane>.html` → reviewer exports `manual_review_decisions_<slug>.csv` |
| 5 | `freeze` | Accept = reviewer accept ∧ candidate ∧ `P_same ≥ threshold`; every accepted track must cover every retained session; input hashes recorded | `accepted_tracks_<slug>.csv` per plane, `manifest.json` with `status: frozen` |

Stages 3a–4 read the reference session, session order, and transforms from the
stage-1 run directory (`--registration-root`), so a full run needs only session
ids, plane names, and output paths.

```bash
osreg register --sessions S1 S2 S3 --reference-session S2 --planes VISp_0 VISl_4 --numerical-dir out/reg
for P in VISp_0 VISl_4; do
  osreg diagnose --registration-root out/reg --plane $P --numerical-dir out/battery_$P --figure-dir figs/battery_$P
  osreg refine   --registration-root out/reg --plane $P --numerical-dir out/refine_$P  --figure-dir figs/refine_$P
  osreg review   --registration-root out/reg --refinement-run out/refine_$P --plane $P --output-root out/review
done
# ... open out/review/manual_review_<plane>.html, decide, export the CSV into out/review ...
osreg freeze --gui-run out/review --planes VISp_0 VISl_4 --refinement-runs out/refine_VISp_0 out/refine_VISl_4 --output-root out/frozen
```

Every subcommand that reads imaging data takes `--source openscope` (AIND S3
planar NWB zarr, default) or `--source npz --npz-dir DIR`.

## Bringing your own data

A session is a `SessionStructuralData`:

```python
from openscope_registration import SessionStructuralData
from openscope_registration.io import save_npz_session

session = SessionStructuralData(
    session_id="mouseA_2026-01-05",
    projection=mean_image,            # (H, W) float; any structural projection
    footprints=masks,                 # (n_cells, H, W) >= 0; weighted or binary pixel masks
    roi_ids=ids,                      # (n_cells,) ints, unique and stable within the session
    pixel_size_um=0.78,
    metadata={"depth_um": 200.0},     # optional
).validate()
save_npz_session("sessions/mouseA_2026-01-05.npz", session)   # then: osreg ... --source npz --npz-dir sessions
```

Or call the stages directly:

```python
from openscope_registration import register, diagnose, refine, review, freeze
result = register({"VISp_0": {s.session_id: s for s in sessions}}, reference_session="mouseA_2026-01-05", numerical_dir="out/reg")
```

Downstream code should read **only** the frozen manifest:

```python
from openscope_registration import accepted_tracks_wide, derive_retained_and_reference
retained, reference = derive_retained_and_reference(["out/frozen"], set(all_session_ids))
wide = accepted_tracks_wide("out/frozen", "visp0", reference)   # index = reference ROI id, one column per session
```

## Method notes and limits

- **Candidate ≠ accepted.** The IoU / centroid-distance / margin gates in stage 1–2 define which pairs enter scoring. They are provisional, preparation-specific values (defaults: IoU ≥ 0.5, centroids ≤ 4 px, margin ≥ 0.2), never acceptance criteria. Identities exist only after stage 5.
- **Calibration is against an independent signal.** `P_same` is the mixture posterior recalibrated so that, on held-out sessions, a value of 0.9 means ~90 % of such pairs are confirmed by a direct assignment between adjacent days. Transitive consistency is necessary, not sufficient: a stable wrong match passes, so calibrated values at low scores bound rather than measure the true same-cell probability.
- **Rigid only.** Residual per-cell displacement after the rigid fit is reported (mean vector, systematic fraction, affine R²) so you can see whether a non-rigid step is warranted; the package does not perform one.
- **Nulls are part of the output.** Shuffled-pair correlations and the local-geometry permutation test are the chance references; a battery without them certifies nothing.
- **Thresholds must be validated locally.** Cell size, pixel size, and session spacing change the right gates and the mixture's separability.

## Package layout

```
src/openscope_registration/
  data.py          SessionStructuralData, RigidTransform
  rigid.py         preprocessing, phase correlation, register_rigid, warps, masked NCC, tile residuals
  matching.py      IoU table, Hungarian assignment + gates, weighted mask correlation, build_tracks
  psame.py         mixture model, transitive consistency, isotonic calibration, reliability bins
  rescue.py        classifier-override rescue, back-projection evidence
  diagnostics.py   battery metrics, residual summaries, frozen-manifest consumption
  freeze.py        acceptance rule and integrity checks
  pipeline.py      the five stage functions (register, diagnose, refine, review, freeze)
  plotting.py      every figure
  review_html.py   manual-review GUI builder
  manifests.py     manifest writer with hashes
  cli.py           `osreg` entry point
  io/              npz loader/saver and the optional AIND OpenScope S3 adaptor
tests/             synthetic ground-truth tests incl. the five-stage end-to-end run
examples/          run_synthetic_pipeline.py
```

Tests: `pip install -e ".[dev]" && pytest`.

## Citing

See [CITATION.cff](CITATION.cff). Method references: Sheintuch et al. 2017 (CellReg; probabilistic same-cell modeling), Pnevmatikakis & Giovannucci 2017 (NoRMCorre; rigid/piecewise-rigid registration), and the AIND OpenScope data documentation for the planar NWB layout.

## License

MIT. See [LICENSE](LICENSE).
