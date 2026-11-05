"""Rigid image registration primitives.

Projection preprocessing (percentile scaling, Gaussian high-pass, light
smoothing, z-score), Hann-windowed phase correlation with a bounded search and
parabolic subpixel refinement, rotation grid search scored by masked NCC on the
overlap support, image/mask warping, and per-tile residual shifts.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .data import RigidTransform


def preprocess_projection(image: np.ndarray, highpass_sigma_px: float = 20.0, smooth_sigma_px: float = 1.0) -> np.ndarray:
    """Percentile-scale, Gaussian high-pass, lightly smooth, and z-score a projection."""
    values = np.asarray(image, dtype=float)
    low, high = np.nanpercentile(values, [1.0, 99.5])
    scaled = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    detail = scaled - ndimage.gaussian_filter(scaled, highpass_sigma_px)
    smoothed = ndimage.gaussian_filter(detail, smooth_sigma_px)
    std = smoothed.std()
    return (smoothed - smoothed.mean()) / std if std > 0 else smoothed


def warp_image(image: np.ndarray, angle_deg: float, shift_yx: tuple[float, float], *, order: int) -> np.ndarray:
    rotated = ndimage.rotate(image, angle_deg, reshape=False, order=order, mode="constant", cval=0.0, prefilter=order > 1)
    return ndimage.shift(rotated, shift_yx, order=order, mode="constant", cval=0.0, prefilter=order > 1)


def warp_footprints(footprints: np.ndarray, transform: RigidTransform) -> np.ndarray:
    """Warp weighted footprints into reference coordinates (nearest-neighbor)."""
    shift = (transform.shift_y_px, transform.shift_x_px)
    return np.asarray([warp_image(mask.astype(float), transform.angle_deg, shift, order=0) for mask in footprints])


def normalized_correlation(reference: np.ndarray, moving: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(moving)
    if valid.sum() < 2:
        return float("nan")
    a = reference[valid] - reference[valid].mean()
    b = moving[valid] - moving[valid].mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def windowed_phase_shift(reference: np.ndarray, moving: np.ndarray, max_shift_px: float) -> tuple[float, float]:
    """Hann-apodized phase correlation; peak restricted to |shift| <= max_shift_px; parabolic subpixel."""
    window = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))
    cross = np.fft.fft2(reference * window) * np.conj(np.fft.fft2(moving * window))
    magnitude = np.abs(cross)
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 0)
    surface = np.fft.fftshift(np.abs(np.fft.ifft2(cross)))
    center = np.array(surface.shape) // 2
    yy, xx = np.indices(surface.shape)
    surface = np.where(np.hypot(yy - center[0], xx - center[1]) <= max_shift_px, surface, -np.inf)
    peak = np.array(np.unravel_index(np.argmax(surface), surface.shape))
    refined = peak.astype(float)
    for axis in (0, 1):
        lo, hi = peak.copy(), peak.copy()
        lo[axis] -= 1
        hi[axis] += 1
        if lo[axis] < 0 or hi[axis] >= surface.shape[axis]:
            continue
        a, b, c = surface[tuple(lo)], surface[tuple(peak)], surface[tuple(hi)]
        denominator = a - 2 * b + c
        if np.isfinite(denominator) and denominator < 0:
            refined[axis] += 0.5 * (a - c) / denominator
    shift = refined - center
    return float(shift[0]), float(shift[1])


def valid_support(shape: tuple[int, int], angle_deg: float, shift_yx: tuple[float, float]) -> np.ndarray:
    return warp_image(np.ones(shape), angle_deg, shift_yx, order=0) > 0.5


def masked_ncc(reference: np.ndarray, moving: np.ndarray, support: np.ndarray) -> float:
    return normalized_correlation(np.where(support, reference, np.nan), np.where(support, moving, np.nan))


def register_rigid(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    maximum_angle_deg: float = 3.0,
    angle_step_deg: float = 0.25,
    max_shift_px: float = 64.0,
) -> RigidTransform:
    """Grid-search rotation + bounded phase-correlation translation; score = NCC on overlap support."""
    best = (-np.inf, 0.0, (0.0, 0.0))
    for angle_deg in np.arange(-maximum_angle_deg, maximum_angle_deg + 0.5 * angle_step_deg, angle_step_deg):
        rotated = ndimage.rotate(moving, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
        shift_yx = windowed_phase_shift(reference, rotated, max_shift_px)
        aligned = ndimage.shift(rotated, shift_yx, order=1, mode="constant", cval=0.0)
        score = masked_ncc(reference, aligned, valid_support(reference.shape, float(angle_deg), shift_yx))
        if score > best[0]:
            best = (score, float(angle_deg), shift_yx)
    return RigidTransform(angle_deg=best[1], shift_y_px=best[2][0], shift_x_px=best[2][1], score=best[0])


def residual_tile_shifts(reference: np.ndarray, aligned: np.ndarray, support: np.ndarray, tile_size_px: int) -> list[dict[str, float]]:
    """Leftover per-tile shifts after global alignment (local-failure probe)."""
    window = np.outer(np.hanning(tile_size_px), np.hanning(tile_size_px))
    rows = []
    for y0 in range(0, reference.shape[0] - tile_size_px + 1, tile_size_px):
        for x0 in range(0, reference.shape[1] - tile_size_px + 1, tile_size_px):
            if not support[y0 : y0 + tile_size_px, x0 : x0 + tile_size_px].all():
                continue
            ref_tile = reference[y0 : y0 + tile_size_px, x0 : x0 + tile_size_px]
            mov_tile = aligned[y0 : y0 + tile_size_px, x0 : x0 + tile_size_px]
            dy, dx = windowed_phase_shift(ref_tile * window, mov_tile * window, tile_size_px / 2)
            rows.append(
                {
                    "tile_y0": float(y0), "tile_x0": float(x0),
                    "residual_shift_y_px": dy, "residual_shift_x_px": dx,
                    "residual_shift_px": float(np.hypot(dy, dx)),
                    "tile_ncc": normalized_correlation(ref_tile, mov_tile),
                }
            )
    return rows
