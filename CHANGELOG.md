# Changelog

## Unreleased

- Fix `plot_footprint_size_change` on matplotlib >= 3.11, where `Axes.boxplot(labels=...)` was removed; tick labels are now set explicitly. This broke three end-to-end tests on CI (matplotlib 3.11.2) while passing locally on 3.9.4.
- CI: run the Python 3.9 job on `ubuntu-22.04` (3.9 is not available on `ubuntu-latest`/26.04), add 3.13, disable fail-fast.

## 0.1.0 (2026-11-05)

Initial public release, extracted from the Oweiss Lab OpenScope receptive-field
drift study. Flat package layout (`openscope_registration.{data, rigid, matching,
psame, rescue, diagnostics, freeze, plotting, review_html, manifests, pipeline,
cli, io}`), one `osreg` command with five subcommands, 36 synthetic-ground-truth
tests including an end-to-end five-stage run on local `.npz` sessions.

Reproduction against the original OpenScope pilot (animal 846289, planes
VISp_0 and VISl_5, 8 sessions, identical parameters): rigid transforms, NCC,
inverse-consistency residuals, candidate pair sets, IoU/centroid values, and
candidate-track coverage are identical (atol 1e-6 / exact). Stages 3b-5 on
VISp_0 (`refine` with the new transforms, `freeze` with the study's actual
reviewer decisions) reproduce the frozen study run bit-for-bit: 2,196 scored
pairs with identical IoU, mask correlation, raw and calibrated P_same,
calibration bins and summary counts; 159 accepted tracks / 1,113 track-days
with an identical pair set. One intentional
difference: the per-tile residual-shift diagnostic now uses the same
subpixel phase-correlation estimator as the global registration; the pilot's
tile residuals were quantized to whole pixels. `tests/test_pilot_reproduction.py`
re-runs this comparison when `OSREG_PILOT_ROOT` / `OSREG_NEW_ROOT` are set.
