"""
src/data/augmentations.py

Lightweight, anomaly-preserving augmentations for telemetry windows.

These functions operate on a single window of shape
``(window_size, num_channels)`` -- the same per-window granularity
``MissionDataset`` already works at in ``__getitem__`` -- and exist to
increase the diversity of *anomaly* windows seen during training under
severe class imbalance (~2% anomaly windows). They are intentionally
conservative: every transformation is small enough that it perturbs a
window without plausibly changing whether it should be labeled
anomalous, so the accompanying label window is never touched.

Design notes:
  * Every augmentation is a pure function: it returns a new array and
    never mutates its input in place, so the caller's original window
    (and anything else holding a reference to it) is unaffected.
  * Every augmentation takes an explicit ``numpy.random.Generator``
    rather than reading from global ``numpy`` random state. This keeps
    augmentation reproducible under a caller-supplied seed and safe to
    use from multiple ``DataLoader`` worker processes without
    cross-worker interference.
  * Nothing here reverses the sequence, flips/permutes channels, or
    applies large-magnitude perturbations -- all of which risk
    destroying or relabeling the anomaly signal.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

# A single-window augmentation: (window, rng, **kwargs) -> augmented window.
AugmentationFn = Callable[..., np.ndarray]


def add_gaussian_noise(
    window: np.ndarray,
    rng: np.random.Generator,
    std: float = 0.01,
) -> np.ndarray:
    """Add very small, zero-mean Gaussian noise to every value.

    Args:
        window: Array of shape ``(window_size, num_channels)``.
        rng: Seeded random generator used for reproducible draws.
        std: Standard deviation of the noise, in the same units as
            ``window``. Kept small by default so the underlying signal
            (and any anomaly within it) is preserved.

    Returns:
        np.ndarray: A new array, same shape and dtype as ``window``.
    """
    window = np.asarray(window)
    noise = rng.normal(loc=0.0, scale=std, size=window.shape)
    return (window + noise).astype(window.dtype, copy=False)


def random_scaling(
    window: np.ndarray,
    rng: np.random.Generator,
    scale_range: Tuple[float, float] = (0.95, 1.05),
) -> np.ndarray:
    """Multiply each channel by an independent factor close to 1.

    Args:
        window: Array of shape ``(window_size, num_channels)``.
        rng: Seeded random generator used for reproducible draws.
        scale_range: ``(min_factor, max_factor)`` the per-channel
            scaling factor is drawn from. Kept close to 1.0 by default
            (small scaling) so relative magnitudes -- and therefore
            the anomaly -- are preserved.

    Returns:
        np.ndarray: A new array, same shape and dtype as ``window``.
    """
    window = np.asarray(window)
    num_channels = window.shape[-1]
    factors = rng.uniform(scale_range[0], scale_range[1], size=(num_channels,))
    return (window * factors).astype(window.dtype, copy=False)


def temporal_shift(
    window: np.ndarray,
    rng: np.random.Generator,
    max_shift: int = 3,
) -> np.ndarray:
    """Shift the sequence left/right by a few timesteps, edge-padded.

    The window is rolled along the time axis and the timesteps
    exposed at the leading/trailing edge are filled by repeating the
    nearest original timestep (edge padding), rather than wrapping
    values from the opposite end of the window -- wrap-around would
    stitch unrelated timesteps together and could fabricate spurious
    transitions.

    Args:
        window: Array of shape ``(window_size, num_channels)``.
        rng: Seeded random generator used for reproducible draws.
        max_shift: Maximum number of timesteps to shift by, in either
            direction. Kept small by default so anomalous timesteps
            remain within the window and close to their original
            position.

    Returns:
        np.ndarray: A new array, same shape and dtype as ``window``.
    """
    window = np.asarray(window)
    window_size = window.shape[0]

    effective_max_shift = min(max_shift, max(window_size - 1, 0))
    if effective_max_shift <= 0:
        return window.copy()

    shift = int(rng.integers(-effective_max_shift, effective_max_shift + 1))
    if shift == 0:
        return window.copy()

    shifted = np.roll(window, shift, axis=0)
    if shift > 0:
        shifted[:shift] = window[0]
    else:
        shifted[shift:] = window[-1]
    return shifted


def random_mask(
    window: np.ndarray,
    rng: np.random.Generator,
    mask_fraction: float = 0.02,
) -> np.ndarray:
    """Randomly zero out a very small percentage of values.

    Simulates brief sensor dropout. Each ``(timestep, channel)`` cell
    is masked independently with probability ``mask_fraction``, so at
    the default rate only ~2% of values in the window are touched.

    Args:
        window: Array of shape ``(window_size, num_channels)``.
        rng: Seeded random generator used for reproducible draws.
        mask_fraction: Fraction of cells to mask, in ``[0, 1]``. Kept
            very small by default so the anomaly pattern remains
            largely intact.

    Returns:
        np.ndarray: A new array, same shape and dtype as ``window``.
    """
    window = np.asarray(window)
    masked = window.copy()
    mask = rng.random(window.shape) < mask_fraction
    masked[mask] = 0.0
    return masked


def sensor_drift(
    window: np.ndarray,
    rng: np.random.Generator,
    max_drift: float = 0.02,
) -> np.ndarray:
    """Add a gradual linear drift across the window, per channel.

    Simulates slow sensor calibration drift: each channel is offset by
    a linear ramp from ``0`` at the first timestep to a small,
    independently-drawn endpoint at the last timestep.

    Args:
        window: Array of shape ``(window_size, num_channels)``.
        rng: Seeded random generator used for reproducible draws.
        max_drift: Maximum magnitude of the drift endpoint, in the
            same units as ``window``. Kept small by default.

    Returns:
        np.ndarray: A new array, same shape and dtype as ``window``.
    """
    window = np.asarray(window)
    window_size, num_channels = window.shape[0], window.shape[-1]

    ramp = np.linspace(0.0, 1.0, num=window_size, dtype=np.float64).reshape(-1, 1)
    drift_endpoint = rng.uniform(-max_drift, max_drift, size=(num_channels,))
    drift = ramp * drift_endpoint

    return (window + drift).astype(window.dtype, copy=False)


# Registry of all available augmentations, keyed by name. Used by
# `apply_random_augmentations` to pick a random subset each call.
# Keeping this as the single source of truth avoids duplicating the
# list of augmentation names anywhere else in the module.
_AUGMENTATIONS: Dict[str, AugmentationFn] = {
    "gaussian_noise": add_gaussian_noise,
    "random_scaling": random_scaling,
    "temporal_shift": temporal_shift,
    "random_mask": random_mask,
    "sensor_drift": sensor_drift,
}


def apply_random_augmentations(
    window: np.ndarray,
    rng: np.random.Generator,
    min_augmentations: int = 1,
    max_augmentations: int = 2,
) -> np.ndarray:
    """Apply a randomly chosen subset of augmentations to one window.

    On each call, between ``min_augmentations`` and ``max_augmentations``
    distinct augmentations are drawn (without replacement) from the
    registry above and applied in sequence, e.g.::

        window -> gaussian_noise -> sensor_drift
        window -> random_scaling

    Not every augmentation is applied every time -- this is what keeps
    the augmented anomaly windows diverse rather than uniformly
    perturbed in the same way.

    Args:
        window: Array of shape ``(window_size, num_channels)``.
        rng: Seeded random generator used for reproducible draws. The
            same generator instance should be reused across calls
            (e.g. one per dataset) so that the augmentation sequence
            is deterministic under a fixed seed.
        min_augmentations: Minimum number of augmentations to apply
            (inclusive).
        max_augmentations: Maximum number of augmentations to apply
            (inclusive). Clamped to the number of available
            augmentations if larger.

    Returns:
        np.ndarray: The augmented window. Same shape and dtype as the
        input; the input array itself is never modified.
    """
    window = np.asarray(window)

    names = list(_AUGMENTATIONS.keys())
    upper_bound = min(max_augmentations, len(names))
    lower_bound = min(min_augmentations, upper_bound)

    num_to_apply = int(rng.integers(lower_bound, upper_bound + 1))
    chosen_names = rng.choice(names, size=num_to_apply, replace=False)

    augmented = window
    for name in chosen_names:
        augmented = _AUGMENTATIONS[str(name)](augmented, rng)

    return augmented
