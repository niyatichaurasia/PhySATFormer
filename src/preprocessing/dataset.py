"""
src/preprocessing/dataset.py

MissionDataset: a torch.utils.data.Dataset over a mission split's
telemetry/label windows.

*** IMPORTANT NOTE ON PROVENANCE ***
The original src/preprocessing/dataset.py was not provided alongside
window_generator.py and pipeline.py, so this file is a reconstruction
based strictly on MissionDataset's observable contract elsewhere in the
codebase:
  - architecture.md: "TelemetryPipeline ... Returns train_dataset,
    val_dataset, test_dataset ... No DataLoaders."
  - pipeline.py: `MissionDataset(telemetry_windows, label_windows)`
  - train.py: instances are passed directly into `torch.utils.data.
    DataLoader(...)`, which only requires `__len__` and `__getitem__`.

If your real MissionDataset already matches this contract (indexable +
len()) and does *not* eagerly convert its constructor arguments to a
torch.Tensor or numpy array up front (e.g. `self.data =
torch.from_numpy(telemetry_windows)`), then plugging a LazyWindowSet
into your existing MissionDataset unmodified should already work,
since LazyWindowSet duck-types as the sequence MissionDataset was
previously indexing. Please diff this file against yours rather than
overwriting blindly -- it exists to make two specific behaviors
explicit that the memory fix in window_generator.py / pipeline.py
depends on:
  1. Telemetry/label windows are only materialized when an index is
     actually requested (never in __init__).
  2. Normalization is applied to exactly one window at a time, inside
     __getitem__, rather than to a fully materialized split.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentations import apply_random_augmentations


class MissionDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapping one split's (telemetry, label)
    windows.

    `telemetry_windows` and `label_windows` may be either:
      - a `LazyWindowSet` (preferred, production-scale: each window is
        materialized on demand, one at a time), or
      - a dense numpy array of shape `(num_windows, window_size,
        num_channels)` (legacy/eager path, e.g. from
        `WindowGenerator.generate()`, or in tests).

    Both are accepted because both satisfy the same minimal contract
    this class relies on: `len(x)` and `x[i] -> array-like of shape
    (window_size, num_channels)`. No branching on the concrete type is
    required or performed.

    Args:
        telemetry_windows: Index-aligned with `label_windows`. Windows
            are NOT normalized in advance -- if `normalizer` is
            provided, it is applied to exactly one window at a time,
            inside `__getitem__`.
        label_windows: Index-aligned with `telemetry_windows`. Never
            normalized, regardless of `normalizer`.
        normalizer: An already-fitted normalizer exposing
            `.transform(array_like) -> array_like`, applied to a single
            telemetry window's array at retrieval time. `transform` on
            a per-channel affine normalizer is an elementwise
            operation, so applying it one window at a time is
            mathematically identical to applying it to a fully
            materialized `(num_windows, window_size, num_channels)`
            array -- it is simply never materialized. If `None`,
            telemetry windows are returned unnormalized (useful for
            tests or when normalization is handled elsewhere).

    Anomaly-only augmentation:
        Every dataset is constructed with augmentation *disabled* by
        default (`training=False`), which preserves the exact previous
        behavior for any existing caller. To opt a specific split into
        anomaly-only augmentation, call `enable_anomaly_augmentation(...)`
        on it explicitly *after* construction -- this should only ever
        be done for the training split. Validation and test datasets
        must be left untouched (i.e. never have that method called on
        them), so they always return raw, unaugmented windows.

        When enabled, a window is augmented in `__getitem__` if and
        only if all three conditions hold:
          1. `training=True` for this dataset instance (set by
             `enable_anomaly_augmentation`).
          2. Augmentation is enabled (also set by
             `enable_anomaly_augmentation` -- the method is a no-op
             gate itself, so simply not calling it keeps augmentation
             off).
          3. The window contains at least one anomalous label AND a
             per-window random draw falls below `probability`.

        Normal (non-anomaly) windows are never augmented, regardless of
        configuration. Augmentation is applied to the raw telemetry
        window *before* normalization, so the fitted normalizer always
        sees values in the same space it was fit on.
    """

    def __init__(
        self,
        telemetry_windows: Sequence[np.ndarray],
        label_windows: Sequence[np.ndarray],
        normalizer: Optional[object] = None,
        training: bool = False,
    ) -> None:
        if len(telemetry_windows) != len(label_windows):
            raise ValueError(
                "telemetry_windows and label_windows must have the same "
                f"length; got {len(telemetry_windows)} and "
                f"{len(label_windows)}."
            )

        # Stored by reference -- NOT converted to a torch.Tensor or a
        # concatenated numpy array here. Doing so would materialize
        # every window in the split up front, which is exactly the
        # allocation this design avoids. Each window is only touched in
        # __getitem__, on demand, one at a time.
        self._telemetry_windows = telemetry_windows
        self._label_windows = label_windows
        self._normalizer = normalizer

        # Training / augmentation state. `training` only marks this
        # instance as *eligible* for augmentation; augmentation itself
        # stays off until `enable_anomaly_augmentation` is called
        # explicitly (see class docstring). Kept as two separate flags
        # rather than one so that a dataset can be "training" (e.g. for
        # future training-only behavior unrelated to augmentation)
        # without silently turning augmentation on.
        self._training = training
        self._augmentation_enabled = False
        self._augmentation_probability = 0.0
        self._augmentation_rng: Optional[np.random.Generator] = None

    def enable_anomaly_augmentation(
        self,
        probability: float = 0.6,
        seed: Optional[int] = None,
    ) -> None:
        """Turn on anomaly-only augmentation for this dataset instance.

        Intended to be called exactly once, immediately after
        construction, and only on the *training* split. Calling this on
        a validation or test dataset would enable augmentation for it,
        which the caller must not do.

        Args:
            probability: Probability, in `[0, 1]`, that a given anomaly
                window is augmented on any particular `__getitem__`
                call. Normal windows are unaffected by this value --
                they are never augmented.
            seed: Seed for this dataset's own `numpy.random.Generator`.
                Passing the same seed used elsewhere in the run (e.g.
                the experiment's global seed) keeps augmentation
                deterministic across runs with that seed. If `None`,
                the generator is seeded from OS entropy (non-deterministic).
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability must be in [0, 1]; got {probability}."
            )

        self._training = True
        self._augmentation_enabled = True
        self._augmentation_probability = float(probability)
        self._augmentation_rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._telemetry_windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        telemetry_window = np.asarray(self._telemetry_windows[idx])
        label_window = np.asarray(self._label_windows[idx])

        # Anomaly-only, training-only augmentation. Order of checks is
        # deliberately cheapest-first: the boolean flags short-circuit
        # before we ever inspect `label_window` or draw from the RNG.
        if self._training and self._augmentation_enabled:
            is_anomaly_window = bool(np.any(label_window > 0))
            if is_anomaly_window:
                assert self._augmentation_rng is not None  # set together
                if self._augmentation_rng.random() < self._augmentation_probability:
                    # Augment the raw telemetry only -- labels describe
                    # *where* the anomaly is and must stay index-aligned
                    # with the (lightly perturbed) telemetry, so
                    # `label_window` is never modified here.
                    telemetry_window = apply_random_augmentations(
                        telemetry_window, self._augmentation_rng
                    )

        if self._normalizer is not None:
            # TelemetryNormalizer.transform() expects a batched array of
            # shape (num_windows, window_size, num_channels). A single
            # window is (window_size, num_channels), so we add a
            # temporary leading batch dimension of size 1, transform,
            # and then squeeze it back out -- semantics are identical
            # to normalizing the whole split at once (see class
            # docstring), just applied one window at a time.
            batched_window = telemetry_window[np.newaxis, ...]
            normalized_batch = self._normalizer.transform(batched_window)
            telemetry_window = np.asarray(normalized_batch)[0]

        telemetry_tensor = torch.as_tensor(
            np.asarray(telemetry_window), dtype=torch.float32
        )
        label_tensor = torch.as_tensor(
            np.asarray(label_window), dtype=torch.float32
        )

        return telemetry_tensor, label_tensor