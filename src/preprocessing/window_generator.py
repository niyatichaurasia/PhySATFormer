"""
src/preprocessing/window_generator.py

WindowGenerator: converts synchronized multivariate telemetry into
fixed-length temporal windows suitable for Transformer training.

This module is strictly responsible for sequence (window) generation.
It does not normalize, synchronize, interpolate, fill missing values,
engineer features, or perform any machine learning.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)


# =========================================================================
# Exceptions
# =========================================================================


class WindowGenerationError(Exception):
    """Base exception for window generation failures."""


class InvalidInputError(WindowGenerationError):
    """Raised when the input DataFrame fails validation."""


class InsufficientDataError(WindowGenerationError):
    """Raised when there is not enough data to produce any window."""


# =========================================================================
# WindowGenerator
# =========================================================================


class WindowGenerator:
    """
    Converts a synchronized multivariate telemetry DataFrame into
    fixed-length sliding windows.

    Responsibilities:
        - Validate the input DataFrame (DatetimeIndex, sorted, numeric).
        - Slide a fixed-size window of length `window_size` across the
          time axis, advancing by `stride` rows at a time.
        - Return the resulting windows as a single numpy array of shape
          (number_of_windows, window_size, number_of_channels).

    Explicitly NOT responsible for:
        - Normalization / scaling
        - Synchronization / merging of channels
        - Interpolation or gap filling
        - Feature engineering
        - Label generation or forecasting targets
        - Any machine learning logic
    """

    def __init__(
        self,
        window_size: int,
        stride: int,
        drop_incomplete: bool = True,
    ) -> None:
        """
        Args:
            window_size: Number of time steps per window. Must be a
                positive integer.
            stride: Number of rows to advance between consecutive
                window start positions. Must be a positive integer.
            drop_incomplete: If True, any trailing window that would
                extend past the end of the DataFrame is discarded. If
                False, the trailing window is kept and padded with NaN
                to reach `window_size` rows.

        Raises:
            InvalidInputError: If any argument fails validation.
        """
        self._validate_constructor_args(window_size, stride, drop_incomplete)

        self.window_size: Final[int] = window_size
        self.stride: Final[int] = stride
        self.drop_incomplete: Final[bool] = drop_incomplete

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def generate(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate fixed-length sliding windows from synchronized telemetry.

        Args:
            df: Synchronized multivariate telemetry. Must have a
                DatetimeIndex, time-ordered rows, and one column per
                telemetry channel. The DataFrame is never modified.

        Returns:
            numpy.ndarray of shape
                (number_of_windows, window_size, number_of_channels)

            If drop_incomplete is False and the final window is
            incomplete, it is padded with numpy.nan along the time axis.

        Raises:
            InvalidInputError: If `df` fails validation.
            InsufficientDataError: If no window can be produced from
                the given data under the current configuration.
            WindowGenerationError: For any other window construction
                failure.
        """
        self._validate_dataframe(df)

        total_rows = len(df)
        n_channels = df.shape[1]

        logger.info(
            "Generating windows: total_rows=%d, window_size=%d, stride=%d, "
            "drop_incomplete=%s, n_channels=%d",
            total_rows,
            self.window_size,
            self.stride,
            self.drop_incomplete,
            n_channels,
        )

        if total_rows < self.window_size:
            if self.drop_incomplete:
                raise InsufficientDataError(
                    f"DataFrame has {total_rows} row(s), which is fewer than "
                    f"window_size={self.window_size}; no complete window can "
                    f"be produced and drop_incomplete=True."
                )
            windows = self._generate_single_padded_window(df, n_channels)
            logger.info("Generated %d window(s).", windows.shape[0])
            return windows

        values = np.asarray(df.to_numpy(copy=False))

        full_windows = self._generate_full_windows(values)

        if self.drop_incomplete:
            windows = full_windows
        else:
            windows = self._append_trailing_partial_window(
                values, full_windows, total_rows, n_channels
            )

        if windows.shape[0] == 0:
            raise InsufficientDataError(
                f"No windows could be generated from {total_rows} row(s) with "
                f"window_size={self.window_size}, stride={self.stride}, "
                f"drop_incomplete={self.drop_incomplete}."
            )

        logger.info("Generated %d window(s).", windows.shape[0])
        return windows

    # ---------------------------------------------------------------
    # Internal helpers: validation
    # ---------------------------------------------------------------

    @staticmethod
    def _validate_constructor_args(
        window_size: int, stride: int, drop_incomplete: bool
    ) -> None:
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise InvalidInputError(
                f"window_size must be an int, got {type(window_size).__name__}."
            )
        if window_size <= 0:
            raise InvalidInputError(f"window_size must be > 0, got {window_size}.")

        if not isinstance(stride, int) or isinstance(stride, bool):
            raise InvalidInputError(
                f"stride must be an int, got {type(stride).__name__}."
            )
        if stride <= 0:
            raise InvalidInputError(f"stride must be > 0, got {stride}.")

        if not isinstance(drop_incomplete, bool):
            raise InvalidInputError(
                f"drop_incomplete must be a bool, got {type(drop_incomplete).__name__}."
            )

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise InvalidInputError(
                f"df must be a pandas DataFrame, got {type(df).__name__}."
            )

        if df.empty:
            raise InvalidInputError("df is empty; cannot generate windows.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise InvalidInputError(
                f"df must have a DatetimeIndex, got {type(df.index).__name__}."
            )

        if df.index.hasnans:
            raise InvalidInputError("df index contains null timestamp(s).")

        if not df.index.is_monotonic_increasing:
            raise InvalidInputError(
                "df index (timestamps) must be sorted in strictly increasing "
                "or non-decreasing order; found unsorted timestamps."
            )

        if df.shape[1] == 0:
            raise InvalidInputError("df must contain at least one channel column.")

        non_numeric = [
            col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise InvalidInputError(
                f"df contains non-numeric column(s): {non_numeric}. All telemetry "
                f"channel columns must be numeric."
            )

    # ---------------------------------------------------------------
    # Internal helpers: window construction
    # ---------------------------------------------------------------

    def _generate_full_windows(self, values: np.ndarray) -> np.ndarray:
        """
        Build all complete (fully in-bounds) windows using a strided,
        zero-copy view over `values`, then select windows at the
        requested stride.

        `values` has shape (total_rows, n_channels).
        Returns an array of shape (n_full_windows, window_size, n_channels).
        """
        # sliding_window_view over axis 0 yields shape:
        # (total_rows - window_size + 1, n_channels, window_size)
        view = sliding_window_view(values, self.window_size, axis=0)

        # Select window start positions according to stride (zero-copy view).
        strided_view = view[:: self.stride]

        # Reorder axes to (n_windows, window_size, n_channels) and copy
        # once here to produce a clean, contiguous, owned output array.
        windows = np.transpose(strided_view, (0, 2, 1)).copy()
        return windows

    def _append_trailing_partial_window(
        self,
        values: np.ndarray,
        full_windows: np.ndarray,
        total_rows: int,
        n_channels: int,
    ) -> np.ndarray:
        """
        When drop_incomplete=False, check whether there is remaining data
        past the last full window's start position (advanced by stride)
        that does not fill a complete window, and if so, append it,
        padded with NaN.
        """
        n_full_windows = full_windows.shape[0]

        if n_full_windows == 0:
            last_start = 0
        else:
            last_full_start = (n_full_windows - 1) * self.stride
            last_start = last_full_start + self.stride

        if last_start >= total_rows:
            # No leftover rows to form a trailing partial window.
            return full_windows

        remaining = values[last_start:total_rows]
        remaining_len = remaining.shape[0]

        if remaining_len == self.window_size:
            # Already a full window not caught by the stride selection
            # (can occur when total_rows aligns exactly); include as-is.
            padded = remaining[np.newaxis, ...]
        else:
            padded = np.full(
                (1, self.window_size, n_channels), np.nan, dtype=np.float64
            )
            padded[0, :remaining_len, :] = remaining

        return np.concatenate([full_windows, padded], axis=0)

    def _generate_single_padded_window(
        self, df: pd.DataFrame, n_channels: int
    ) -> np.ndarray:
        """
        Handle the edge case where total_rows < window_size and
        drop_incomplete=False: produce exactly one NaN-padded window
        containing all available rows.
        """
        values = np.asarray(df.to_numpy(copy=False))
        total_rows = values.shape[0]

        padded = np.full((1, self.window_size, n_channels), np.nan, dtype=np.float64)
        padded[0, :total_rows, :] = values
        return padded