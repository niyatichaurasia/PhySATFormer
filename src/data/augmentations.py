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

Each augmentation here is chosen to correspond to a real, documented
spacecraft telemetry failure/noise mode (sensor noise, gain error,
calibration bias, thermal drift, sample-and-hold dropout, EMI/SET noise
bursts) rather than a generic signal-processing perturbation. In
particular, this file intentionally avoids any transformation that
shifts data in time relative to its label (e.g. temporal shifting),
since that would desynchronize the anomaly signature from the
timestep it is labeled at.

Design notes:
  * Every augmentation is a pure function: it returns a new array and
    never mutates its input in place, so the caller's original window
    (and anything else holding a reference to it) is unaffected.
  * Every augmentation takes an explicit ``numpy.random.Generator``
    rather than reading from global ``numpy`` random state. This keeps
    augmentation reproducible under a caller-supplied seed and safe to
    use from multiple ``DataLoader`` worker processes without
    cross-worker interference.
  * Nothing here reverses the sequence, flips/permutes channels, shifts
    telemetry in time relative to its labels, or applies large-magnitude
    perturbations -- all of which risk destroying or relabeling the
    anomaly signal.
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

    Physical motivation: ambient sensor/ADC noise (thermal noise,
    quantization noise) is present in essentially all real telemetry
    and is well modeled as small, zero-mean, roughly Gaussian additive
    noise applied uniformly across the window.

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

    Physical motivation: this models a per-channel **gain calibration
    error** -- a real sensor/amplifier fault mode where the reported
    value is a slightly mis-scaled version of the true physical value.
    Keeping the factor close to 1.0 preserves relative magnitudes (and
    therefore the anomaly) within the window.

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


def calibration_offset(
    window: np.ndarray,
    rng: np.random.Generator,
    offset_fraction: float = 0.05,
) -> np.ndarray:
    """Add a constant per-channel bias/offset across the whole window.

    Physical motivation: this models a **calibration bias (offset)
    error** -- a real, distinct failure mode from gain error
    (``random_scaling``, multiplicative) and thermal drift
    (``sensor_drift``, a ramp): a fixed additive bias for the duration
    of the window, as would result from an uncorrected zero-point
    offset in a sensor or ADC channel.

    The offset magnitude is scaled to a small fraction of that
    channel's own within-window standard deviation, so the
    perturbation stays proportionate across channels with very
    different physical units/ranges without requiring any global,
    per-mission channel statistics.

    Args:
        window: Array of shape ``(window_size, num_channels)``.
        rng: Seeded random generator used for reproducible draws.
        offset_fraction: Maximum offset magnitude, expressed as a
            fraction of each channel's within-window standard
            deviation. Kept small by default so the anomaly signal is
            preserved.

    Returns:
        np.ndarray: A new array, same shape and dtype as ``window``.
    """
    window = np.asarray(window)
    num_channels = window.shape[-1]

    channel_std = window.std(axis=0)
    # Guard against zero-variance channels (e.g. constant/flat
    # channels within this window), where a std-relative offset would
    # otherwise collapse to zero and never perturb the channel.
    channel_std = np.where(channel_std > 0, channel_std, 1.0)

    offsets = rng.uniform(-offset_fraction, offset_fraction, size=(num_channels,))
    offsets = offsets * channel_std

    return (window + offsets).astype(window.dtype, copy=False)


def sensor_dropout_hold(
    window: np.ndarray,
    rng: np.random.Generator,
    channel_fraction: float = 0.05,
    max_run_length: int = 5,
) -> np.ndarray:
    """Freeze a short run of timesteps at their last valid value, per channel.

    Physical motivation: real telemetry/bus dropouts on spacecraft
    typically do not read back as zero -- they read back as **stale
    (sample-and-hold) data**, where the ground system keeps receiving
    the last successfully sampled value until the link/sensor
    recovers. Zero-masking (the previous implementation) is often
    physically implausible, since ``0.0`` in raw physical units (e.g.
    temperature, voltage, pressure) may be an out-of-range or
    meaningless value for a given channel. Holding the last value is
    the physically accurate analogue of this failure mode.

    A small, randomly chosen subset of channels each has one short,
    contiguous run of timesteps overwritten with the value immediately
    preceding the run (or the window's first value, if the run starts
    at index 0).

    Args:
        window: Array of shape ``(window_size, num_channels)``.
        rng: Seeded random generator used for reproducible draws.
        channel_fraction: Fraction of channels (rounded up to at least
            one) that receive a stale-data run. Kept small by default
            so most of the window/channels remain untouched.
        max_run_length: Maximum length, in timesteps, of a single
            stale-data run. Kept short by default so anomalous
            timesteps are unlikely to be entirely overwritten.

    Returns:
        np.ndarray: A new array, same shape and dtype as ``window``.
    """
    window = np.asarray(window)
    held = window.copy()
    window_size, num_channels = window.shape[0], window.shape[-1]

    num_affected = max(1, int(round(channel_fraction * num_channels)))
    num_affected = min(num_affected, num_channels)
    affected_channels = rng.choice(num_channels, size=num_affected, replace=False)

    effective_max_run = min(max_run_length, window_size)
    if effective_max_run <= 0:
        return held

    for channel_idx in affected_channels:
        run_length = int(rng.integers(1, effective_max_run + 1))
        start = int(rng.integers(0, window_size - run_length + 1))
        hold_value = held[start - 1, channel_idx] if start > 0 else held[start, channel_idx]
        held[start:start + run_length, channel_idx] = hold_value

    return held.astype(window.dtype, copy=False)


def sensor_drift(
    window: np.ndarray,
    rng: np.random.Generator,
    max_drift: float = 0.02,
) -> np.ndarray:
    """Add a gradual linear drift across the window, per channel.

    Physical motivation: this models slow **thermal/calibration
    drift**, a common and gradual effect (e.g. as a spacecraft moves
    in and out of eclipse, or as electronics slowly self-heat) that
    ramps up (or down) smoothly over an observation window, unlike the
    constant offset of ``calibration_offset``.

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

    ramp = np.linspace(
        0.0,
        1.0,
        num=window_size,
        dtype=np.float64
    ).reshape(-1,1)

    channel_std = window.std(axis=0)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)

    drift_endpoint = (
        rng.uniform(
            -max_drift,
            max_drift,
            size=(num_channels,)
        )
        * channel_std
    )

    drift = ramp * drift_endpoint

    return (window + drift).astype(window.dtype, copy=False)


def localized_noise_burst(
    window: np.ndarray,
    rng: np.random.Generator,
    channel_fraction: float = 0.1,
    max_burst_length: int = 6,
    burst_std: float = 0.05,
) -> np.ndarray:
    """Inject a short, high-amplitude noise burst on a few channels.

    Physical motivation: spacecraft telemetry is exposed to the space
    radiation environment (single-event transients/upsets in sensor
    electronics) and to electromagnetic interference (e.g. from
    thruster firings or nearby subsystem switching), both of which
    produce short, localized bursts of elevated noise on one or a few
    channels rather than a smooth, window-wide perturbation. This is
    distinct from ``add_gaussian_noise`` (ambient, uniform, low
    amplitude) in being spatially and temporally localized and
    higher-amplitude.

    Args:
        window: Array of shape ``(window_size, num_channels)``.
        rng: Seeded random generator used for reproducible draws.
        channel_fraction: Fraction of channels (rounded up to at least
            one) that receive a noise burst. Kept small by default.
        max_burst_length: Maximum length, in timesteps, of a single
            burst. Kept short by default so it stays localized.
        burst_std: Standard deviation of the burst noise, in the same
            units as ``window``. Larger than ``add_gaussian_noise``'s
            default to represent an elevated-noise event, but still
            bounded to avoid destroying the anomaly signal.

    Returns:
        np.ndarray: A new array, same shape and dtype as ``window``.
    """
    window = np.asarray(window)
    bursted = window.copy()
    window_size, num_channels = window.shape[0], window.shape[-1]

    num_affected = max(1, int(round(channel_fraction * num_channels)))
    num_affected = min(num_affected, num_channels)
    affected_channels = rng.choice(num_channels, size=num_affected, replace=False)

    effective_max_burst = min(max_burst_length, window_size)
    if effective_max_burst <= 0:
        return bursted

    for channel_idx in affected_channels:
        burst_length = int(rng.integers(1, effective_max_burst + 1))
        start = int(rng.integers(0, window_size - burst_length + 1))
        channel_std = window[:, channel_idx].std()

        channel_std = max(channel_std, 1e-6)

        noise = rng.normal(
            loc=0.0,
            scale=burst_std * channel_std,
            size=burst_length,
        )
        bursted[start:start + burst_length, channel_idx] += noise

    return bursted.astype(window.dtype, copy=False)


# Registry of all available augmentations, keyed by name. Used by
# `apply_random_augmentations` to pick a random subset each call.
# Keeping this as the single source of truth avoids duplicating the
# list of augmentation names anywhere else in the module.
#
# Every entry corresponds to a distinct, physically-motivated telemetry
# fault/noise mode:
#   gaussian_noise       -> ambient sensor/ADC noise
#   random_scaling       -> gain calibration error (multiplicative)
#   calibration_offset   -> bias calibration error (constant additive)
#   sensor_dropout_hold  -> stale/sample-and-hold data (bus/sensor dropout)
#   sensor_drift         -> slow thermal/calibration drift (ramp)
#   localized_noise_burst -> EMI / single-event-transient noise burst
_AUGMENTATIONS: Dict[str, AugmentationFn] = {
    "gaussian_noise": add_gaussian_noise,
    "random_scaling": random_scaling,
    "calibration_offset": calibration_offset,
    "sensor_dropout_hold": sensor_dropout_hold,
    "sensor_drift": sensor_drift,
    "localized_noise_burst": localized_noise_burst,
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
