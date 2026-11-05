"""openscope-registration-toolkit: cross-session cell registration for calcium imaging.

Five stages, each a plain function in :mod:`openscope_registration.pipeline`:

1. :func:`register`  rigid alignment onto a reference session + one-to-one
   pixel-mask matching + candidate tracks (``run_structural_registration``)
2. :func:`diagnose`  the diagnostic battery with shuffled/permutation nulls
3. :func:`refine`    same-cell probability (mixture + transitive-consistency
   calibration) and the rescue pass
4. :func:`review`    manual-review HTML GUI (structural evidence only)
5. :func:`freeze`    reviewer decisions -> frozen manifest of accepted tracks

Inputs are :class:`SessionStructuralData` objects (projection, pixel masks,
stable ROI ids, pixel size). Loaders for local ``.npz`` files and the AIND
OpenScope planar NWB zarr assets live in :mod:`openscope_registration.io`.
Command line: ``osreg <stage> ...`` (see ``osreg --help``).
"""

from .data import RigidTransform, SessionStructuralData
from .diagnostics import (
    accepted_tracks_wide,
    derive_retained_and_reference,
    residual_displacement_table,
    residual_vector_summary_table,
    summarise_residual_vectors,
)
from .freeze import freeze_plane
from .matching import assign_matches, build_tracks, iou_matrix, weighted_footprint_correlation
from .pipeline import (
    diagnose,
    fov_correspondence,
    freeze,
    refine,
    register,
    register_plane,
    review,
    run_freeze_manifest,
    run_identity_refinement,
    run_registration_diagnostics,
    run_review_gui,
    run_structural_registration,
    select_review_tracks,
)
from .psame import (
    apply_isotonic,
    fit_isotonic_calibration,
    fit_p_same,
    reliability_bins,
    transitive_consistency_flags,
)
from .rigid import masked_ncc, preprocess_projection, register_rigid, warp_footprints, warp_image

__version__ = "0.1.0"

__all__ = [
    "RigidTransform", "SessionStructuralData",
    "register", "diagnose", "refine", "review", "freeze",
    "run_structural_registration", "run_registration_diagnostics", "run_identity_refinement",
    "run_review_gui", "run_freeze_manifest", "register_plane", "fov_correspondence", "select_review_tracks",
    "preprocess_projection", "register_rigid", "warp_image", "warp_footprints", "masked_ncc",
    "iou_matrix", "assign_matches", "weighted_footprint_correlation", "build_tracks",
    "fit_p_same", "fit_isotonic_calibration", "apply_isotonic", "transitive_consistency_flags", "reliability_bins",
    "residual_displacement_table", "summarise_residual_vectors", "residual_vector_summary_table",
    "freeze_plane", "accepted_tracks_wide", "derive_retained_and_reference",
    "__version__",
]
