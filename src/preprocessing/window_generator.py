"""
src/preprocessing/window_generator.py

WindowGenerator: converts synchronized multivariate telemetry, together
with its aligned dense channel-wise anomaly labels, into fixed-length
temporal windows suitable for Transformer training.

This module is strictly responsible for sequence (window) generation.
It does not normalize, synchronize, interpolate, fill missing values,
engineer features, collapse/aggregate labels, or perform any machine
learning.
"""

from __future__ import annotations

import logging
from typing import Final, Tuple

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
    Converts synchronized multivariate telemetry, together with its
    aligned dense channel-wise anomaly labels, into fixed-length sliding
    windows.

    Responsibilities:
        - Validate the input telemetry DataFrame (DatetimeIndex, sorted,
          numeric) and the dense label matrix (matching shape).
        - Slide a fixed-size window of length `window_size` across the
          time axis, advancing by `stride` rows at a time.
        - Return the resulting telemetry windows and label windows as two
          numpy arrays, each of shape
          (number_of_windows, window_size, number_of_channels), such that
          every telemetry window corresponds exactly to the same
          timestamp range in the label matrix.

    Explicitly NOT responsible for:
        - Normalization / scaling
        - Synchronization / merging of channels
        - Interpolation or gap filling
        - Feature engineering
        - Collapsing, aggregating, or majority-voting labels
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

    def generate(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fixed-length sliding windows from synchronized telemetry
        and its aligned dense channel-wise anomaly labels.

        Args:
            df: Synchronized multivariate telemetry. Must have a
                DatetimeIndex, time-ordered rows, and one column per
                telemetry channel. The DataFrame is never modified.
            labels: Dense channel-wise anomaly labels of shape
                ``(num_timestamps, num_channels)``, where
                ``labels[t, c] == 1`` iff channel ``c`` is anomalous at
                timestamp ``t``. Must have the same number of rows as
                `df` (row ``t`` corresponds to ``df.index[t]``) and the
                same number of columns as `df` (column ``c`` corresponds
                to ``df.columns[c]``). Never modified. Labels are never
                collapsed, aggregated, or reduced -- the full
                per-timestamp, per-channel label tensor is windowed
                exactly like the telemetry.

        Returns:
            Tuple of ``(telemetry_windows, label_windows)``:
                telemetry_windows : numpy.ndarray
                    Shape ``(number_of_windows, window_size,
                    number_of_channels)``.
                label_windows : numpy.ndarray
                    Shape ``(number_of_windows, window_size,
                    number_of_channels)``, aligned index-for-index with
                    `telemetry_windows` (i.e. `label_windows[i]` covers
                    exactly the same timestamp range as
                    `telemetry_windows[i]`).

            If drop_incomplete is False and the final window is
            incomplete, both the telemetry and label windows are padded
            with numpy.nan along the time axis.

        Raises:
            InvalidInputError: If `df` or `labels` fails validation, or
                if their shapes are inconsistent with each other.
            InsufficientDataError: If no window can be produced from
                the given data under the current configuration.
            WindowGenerationError: For any other window construction
                failure.
        """
        self._validate_dataframe(df)
        self._validate_labels(labels, df)

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

        telemetry_values = np.asarray(df.to_numpy(copy=False))
        label_values = np.asarray(labels)

        if total_rows < self.window_size:
            if self.drop_incomplete:
                raise InsufficientDataError(
                    f"DataFrame has {total_rows} row(s), which is fewer than "
                    f"window_size={self.window_size}; no complete window can "
                    f"be produced and drop_incomplete=True."
                )
            telemetry_windows = self._generate_single_padded_window(
                telemetry_values, total_rows, n_channels
            )
            label_windows = self._generate_single_padded_window(
                label_values, total_rows, n_channels
            )
            logger.info("Generated %d window(s).", telemetry_windows.shape[0])
            return telemetry_windows, label_windows

        full_telemetry_windows = self._generate_full_windows(telemetry_values)
        full_label_windows = self._generate_full_windows(label_values)

        if self.drop_incomplete:
            telemetry_windows = full_telemetry_windows
            label_windows = full_label_windows
        else:
            telemetry_windows = self._append_trailing_partial_window(
                telemetry_values, full_telemetry_windows, total_rows, n_channels
            )
            label_windows = self._append_trailing_partial_window(
                label_values, full_label_windows, total_rows, n_channels
            )

        if telemetry_windows.shape[0] == 0:
            raise InsufficientDataError(
                f"No windows could be generated from {total_rows} row(s) with "
                f"window_size={self.window_size}, stride={self.stride}, "
                f"drop_incomplete={self.drop_incomplete}."
            )

        logger.info("Generated %d window(s).", telemetry_windows.shape[0])
        return telemetry_windows, label_windows

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

    @staticmethod
    def _validate_labels(labels: np.ndarray, df: pd.DataFrame) -> None:
        """
        Validate the dense channel-wise label matrix against the
        telemetry DataFrame it must align with.

        Checks:
            - `labels` is a numpy.ndarray
            - `labels` is not empty
            - `labels` has exactly 2 dimensions
              (num_timestamps, num_channels)
            - `labels` has a numeric dtype
            - `labels` row count matches `df` row count (matching
              telemetry/label lengths)
            - `labels` column count matches `df` column count (matching
              number of channels)

        Raises:
            InvalidInputError: If any of the above checks fail.
        """
        if not isinstance(labels, np.ndarray):
            raise InvalidInputError(
                f"labels must be a numpy.ndarray, got {type(labels).__name__}."
            )

        if labels.size == 0:
            raise InvalidInputError("labels is empty; cannot generate windows.")

        if labels.ndim != 2:
            raise InvalidInputError(
                "labels must have shape (num_timestamps, num_channels); "
                f"got array with ndim={labels.ndim} and shape={labels.shape}."
            )

        if not np.issubdtype(labels.dtype, np.number):
            raise InvalidInputError(
                f"labels must have a numeric dtype, got dtype={labels.dtype}."
            )

        num_timestamps, num_channels = labels.shape

        if num_timestamps != len(df):
            raise InvalidInputError(
                "labels and df must have the same number of timestamps "
                f"(rows); got labels.shape[0]={num_timestamps} but "
                f"len(df)={len(df)}."
            )

        if num_channels != df.shape[1]:
            raise InvalidInputError(
                "labels and df must have the same number of channels "
                f"(columns); got labels.shape[1]={num_channels} but "
                f"df.shape[1]={df.shape[1]}."
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
        self, values: np.ndarray, total_rows: int, n_channels: int
    ) -> np.ndarray:
        """
        Handle the edge case where total_rows < window_size and
        drop_incomplete=False: produce exactly one NaN-padded window
        containing all available rows.

        `values` has shape (total_rows, n_channels) and may correspond
        to either the telemetry array or the dense label array; both are
        padded identically so that the resulting windows remain aligned.
        """
        padded = np.full((1, self.window_size, n_channels), np.nan, dtype=np.float64)
        padded[0, :total_rows, :] = values
        return padded