"""
src/preprocessing/normalizer.py

TelemetryNormalizer: channel-wise normalization of telemetry windows.

This module is responsible ONLY for normalization. It must NEVER:
    - synchronize telemetry
    - generate windows
    - split datasets
    - perform machine learning

Part of the PhySATFormer preprocessing pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Literal

import numpy as np

logger = logging.getLogger(__name__)

SupportedMethod = Literal["zscore", "minmax", "robust"]
_SUPPORTED_METHODS = ("zscore", "minmax", "robust")

# Small constant to avoid division by zero when a channel is constant.
_EPS = 1e-8


class TelemetryNormalizer:
    """
    Performs channel-wise normalization of telemetry windows.

    Telemetry windows are expected as a numpy array of shape:
        (number_of_windows, window_size, number_of_channels)

    Normalization statistics are computed independently per channel
    (i.e., across the window and window-size axes, per channel index)
    during `fit`, and applied during `transform`.

    Supported methods
    ------------------
    zscore : (x - mean) / std
    minmax : (x - min) / (max - min)
    robust : (x - median) / IQR

    This class is stateless with respect to synchronization, windowing,
    dataset splitting, or modeling concerns -- it strictly normalizes
    already-windowed telemetry arrays.
    """

    def __init__(
        self,
        method: SupportedMethod = "zscore",
        eps: float = _EPS,
        allow_nan: bool = False,
    ) -> None:
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported normalization method: {method!r}. "
                f"Supported methods are: {_SUPPORTED_METHODS}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps!r}.")

        self.method: SupportedMethod = method
        self.eps: float = eps
        self.allow_nan: bool = allow_nan

        self._is_fitted: bool = False
        self._num_channels: Optional[int] = None
        self._stats: Dict[str, np.ndarray] = {}
        self.input_dtype: Optional[np.dtype] = None

        logger.debug("TelemetryNormalizer initialized with method=%s", method)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, windows: np.ndarray) -> "TelemetryNormalizer":
        """
        Compute per-channel normalization statistics from telemetry windows.

        Parameters
        ----------
        windows : np.ndarray
            Array of shape (number_of_windows, window_size, number_of_channels).

        Returns
        -------
        TelemetryNormalizer
            self, to allow method chaining.
        """
        self._validate_input(windows, expected_channels=None)

        self.input_dtype = windows.dtype
        num_channels = windows.shape[2]
        # Compute statistics in float64 for numerical stability, regardless
        # of the input dtype. The results are cast back to float64 storage
        # and later applied while preserving the *input* dtype at transform time.
        data = windows.astype(np.float64, copy=False)

        if self.method == "zscore":
            mean = data.mean(axis=(0, 1))
            std = data.std(axis=(0, 1))
            std_safe = np.where(std < self.eps, self.eps, std)
            self._stats = {"mean": mean, "std": std_safe}

        elif self.method == "minmax":
            data_min = data.min(axis=(0, 1))
            data_max = data.max(axis=(0, 1))
            data_range = data_max - data_min
            range_safe = np.where(data_range < self.eps, self.eps, data_range)
            self._stats = {"min": data_min, "max": data_max, "range": range_safe}

        elif self.method == "robust":
            reshaped = data.reshape(-1, num_channels)
            median = np.median(reshaped, axis=0)
            q1 = np.percentile(reshaped, 25, axis=0)
            q3 = np.percentile(reshaped, 75, axis=0)
            iqr = q3 - q1
            iqr_safe = np.where(iqr < self.eps, self.eps, iqr)
            self._stats = {"median": median, "iqr": iqr_safe}

        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"Unsupported normalization method: {self.method!r}.")

        self._num_channels = num_channels
        self._is_fitted = True

        logger.info(
            "TelemetryNormalizer fitted: method=%s, num_channels=%d, "
            "windows=%d, window_size=%d",
            self.method,
            num_channels,
            windows.shape[0],
            windows.shape[1],
        )
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """
        Apply previously fitted per-channel normalization statistics.

        Parameters
        ----------
        windows : np.ndarray
            Array of shape (number_of_windows, window_size, number_of_channels).
            The number of channels must match the array used in `fit`.

        Returns
        -------
        np.ndarray
            A new array with the same shape and dtype as `windows`, normalized
            channel-wise. The original array is never modified.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TelemetryNormalizer.transform() called before fit(). "
                "Call fit() or fit_transform() first."
            )

        self._validate_input(windows, expected_channels=self._num_channels)

        original_dtype = windows.dtype
        data = windows.astype(np.float64, copy=True)

        if self.method == "zscore":
            mean = self._stats["mean"]
            std = self._stats["std"]
            normalized = (data - mean) / std

        elif self.method == "minmax":
            data_min = self._stats["min"]
            data_range = self._stats["range"]
            normalized = (data - data_min) / data_range

        elif self.method == "robust":
            median = self._stats["median"]
            iqr = self._stats["iqr"]
            normalized = (data - median) / iqr

        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"Unsupported normalization method: {self.method!r}.")

        return normalized.astype(original_dtype, copy=False)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """
        Fit normalization statistics on `windows` and immediately apply them.

        Equivalent to calling `fit(windows)` followed by `transform(windows)`.

        Parameters
        ----------
        windows : np.ndarray
            Array of shape (number_of_windows, window_size, number_of_channels).

        Returns
        -------
        np.ndarray
            Normalized array with the same shape and dtype as `windows`.
        """
        self.fit(windows)
        return self.transform(windows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(
        self, windows: np.ndarray, expected_channels: Optional[int]
    ) -> None:
        """
        Validate that `windows` is a well-formed telemetry array.

        Raises
        ------
        TypeError
            If `windows` is not a numpy.ndarray.
        ValueError
            If `windows` does not have 3 dimensions, is empty, contains
            infinite values (or, when `allow_nan=False`, NaN values), or
            has a channel count mismatching a previously fitted normalizer.
        """
        if not isinstance(windows, np.ndarray):
            raise TypeError(
                f"windows must be a numpy.ndarray, got {type(windows).__name__}."
            )

        if windows.ndim != 3:
            raise ValueError(
                "windows must have shape "
                "(number_of_windows, window_size, number_of_channels); "
                f"got array with ndim={windows.ndim} and shape={windows.shape}."
            )

        num_windows, window_size, num_channels = windows.shape

        if num_windows == 0 or window_size == 0 or num_channels == 0:
            raise ValueError(
                f"windows must not have a zero-length dimension; got shape={windows.shape}."
            )

        if not np.issubdtype(windows.dtype, np.number):
            raise TypeError(
                f"windows must have a numeric dtype, got dtype={windows.dtype}."
            )

        if self.allow_nan:
            if np.any(np.isinf(windows)):
                raise ValueError("windows contains infinite values.")
        else:
            if not np.all(np.isfinite(windows)):
                raise ValueError("windows contains NaN or infinite values.")

        if expected_channels is not None and num_channels != expected_channels:
            raise ValueError(
                f"Channel mismatch: normalizer was fitted with "
                f"{expected_channels} channels, but received {num_channels}."
            )

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether `fit` has been called successfully."""
        return self._is_fitted

    @property
    def num_channels(self) -> Optional[int]:
        """Number of channels the normalizer was fitted on, or None if unfitted."""
        return self._num_channels

    def get_stats(self) -> Dict[str, np.ndarray]:
        """
        Return a copy of the internally stored normalization statistics.

        Raises
        ------
        RuntimeError
            If called before `fit()`.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TelemetryNormalizer.get_stats() called before fit(). "
                "Call fit() or fit_transform() first."
            )
        return {key: value.copy() for key, value in self._stats.items()}

    def __repr__(self) -> str:
        return (
            f"TelemetryNormalizer(method={self.method!r}, "
            f"is_fitted={self._is_fitted}, num_channels={self._num_channels}, "
            f"input_dtype={self.input_dtype!r})"
        )