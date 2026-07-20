"""
src/preprocessing/window_generator.py

WindowGenerator: converts synchronized multivariate telemetry, together
with its aligned dense channel-wise anomaly labels, into fixed-length
temporal windows suitable for Transformer training.

This module is strictly responsible for sequence (window) generation.
It does not normalize, synchronize, interpolate, fill missing values,
engineer features, collapse/aggregate labels, or perform any machine
learning.

MEMORY DESIGN
==============
For overlapping windows (stride < window_size, the common case),
`num_windows` grows much faster than the number of underlying rows.
Materializing every window into a single owned (num_windows,
window_size, n_channels) array duplicates every row up to
`window_size / stride` times. On Mission1 this produced ~229,976
windows of shape (128, 76) -- ~16.7 GiB for a single array (telemetry
*or* labels; both together is worse).

`generate()` (eager, materializing) is kept for small missions,
notebooks, and tests where convenience matters more than memory.

`generate_lazy()` is the production-scale path: it performs the exact
same validation and returns two `LazyWindowSet` objects instead of
materialized arrays. A `LazyWindowSet` does not copy or view the
overlapping window structure at all -- it stores only the underlying
2D `(total_rows, n_channels)` array (whose memory footprint is
independent of `stride` and typically orders of magnitude smaller than
the windowed array) plus O(1) bookkeeping, and computes a single
window, on demand, as an independent small copy in `__getitem__`.

`LazyWindowSet` is intentionally a plain, indexable, `len()`-able
sequence (duck-typed like a `Sequence[np.ndarray]`) so it is a drop-in
replacement anywhere a materialized `(N, window_size, n_channels)`
array was previously being indexed one window at a time -- most
importantly, inside a `torch.utils.data.Dataset.__getitem__`.
"""

from __future__ import annotations

import logging
from typing import Final, Sequence, Tuple

import numpy as np
import pandas as pd

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
# LazyWindowSet
# =========================================================================


class LazyWindowSet(Sequence):
    """
    A memory-efficient, read-only, indexable sequence of fixed-length
    sliding windows over a 2D `(total_rows, n_channels)` array.

    No window is ever materialized until it is actually indexed, and
    only the single requested window is materialized at that point --
    never the full collection. This bounds peak extra memory to
    `O(window_size * n_channels)` regardless of how many windows exist,
    versus `O(num_windows * window_size * n_channels)` for an eagerly
    materialized array.

    Each `__getitem__` call returns an independently owned numpy array
    (a fresh `.copy()` of just that one window). This is deliberate --
    see "Why copy a single window?" below.

    Why copy a single window?
    --------------------------
    An alternative zero-copy design would return a *view* into the
    shared underlying array (e.g. via `numpy.lib.stride_tricks.
    sliding_window_view`) for every window. That was rejected because
    overlapping windows would then share the same underlying memory:
    an in-place mutation to one window (e.g. a normalizer doing
    `window *= scale` in place) would silently corrupt every other
    window that overlaps the same source rows. Consumers would have to
    uphold a fragile "never mutate in place" invariant across the
    entire codebase (normalizer, augmentations, collation, etc.) to
    stay correct. Copying a single window is cheap (tens of KB) and
    removes that hazard entirely, while still avoiding the actual
    memory blowup, which comes from materializing *all* windows at
    once, not from copying one window at a time.

    Not a numpy array: it does not support numpy fancy indexing,
    broadcasting, or `.shape` on the whole collection. It supports
    `len()`, integer indexing (`ws[i]`), and iteration, which is all a
    `torch.utils.data.Dataset` or a manual training loop needs.
    """

    __slots__ = (
        "_values",
        "window_size",
        "stride",
        "n_channels",
        "dtype",
        "_num_full_windows",
        "_has_padded_trailing_window",
        "_trailing_start",
        "_trailing_len",
        "_num_windows",
    )

    def __init__(
        self,
        values: np.ndarray,
        window_size: int,
        stride: int,
        num_full_windows: int,
        has_padded_trailing_window: bool,
        trailing_start: int,
        trailing_len: int,
    ) -> None:
        """
        Args:
            values: The underlying, un-windowed `(total_rows,
                n_channels)` array. Stored by reference (not copied);
                this is the entire memory footprint of the
                LazyWindowSet beyond a few scalars.
            window_size: Length of each window along the time axis.
            stride: Row offset between consecutive window starts.
            num_full_windows: Number of complete, in-bounds windows
                (i.e. windows requiring no NaN padding).
            has_padded_trailing_window: Whether there is one
                additional, NaN-padded trailing window beyond the full
                windows (only possible when `drop_incomplete=False`).
            trailing_start: Row index at which the trailing partial
                window begins. Unused if `has_padded_trailing_window`
                is False.
            trailing_len: Number of real (non-padded) rows available
                for the trailing partial window. Unused if
                `has_padded_trailing_window` is False.
        """
        self._values = values
        self.window_size = window_size
        self.stride = stride
        self.n_channels = values.shape[1]
        self.dtype = values.dtype
        self._num_full_windows = num_full_windows
        self._has_padded_trailing_window = has_padded_trailing_window
        self._trailing_start = trailing_start
        self._trailing_len = trailing_len
        self._num_windows = num_full_windows + (
            1 if has_padded_trailing_window else 0
        )

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._num_windows

    def __getitem__(self, idx: int) -> np.ndarray:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        n = len(self)
        original_idx = idx
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(
                f"LazyWindowSet index {original_idx} out of range for "
                f"{n} window(s)."
            )

        if idx < self._num_full_windows:
            start = idx * self.stride
            # Independent copy: see class docstring ("Why copy a
            # single window?") for why this is required for safety.
            return self._values[start : start + self.window_size].copy()

        # The single, NaN-padded trailing window (drop_incomplete=False).
        window = np.full(
            (self.window_size, self.n_channels), np.nan, dtype=np.float64
        )
        remaining = self._values[
            self._trailing_start : self._trailing_start + self._trailing_len
        ]
        window[: self._trailing_len, :] = remaining
        return window

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"LazyWindowSet(num_windows={len(self)}, "
            f"window_size={self.window_size}, stride={self.stride}, "
            f"n_channels={self.n_channels}, dtype={self.dtype})"
        )

    # ------------------------------------------------------------------
    # Escape hatch
    # ------------------------------------------------------------------

    def materialize(self) -> np.ndarray:
        """
        Eagerly build the full `(num_windows, window_size, n_channels)`
        array. Provided for parity with the legacy eager API, small
        missions, and tests -- NOT recommended for production-scale
        datasets, since this reintroduces the exact memory cost this
        class exists to avoid.
        """
        logger.warning(
            "LazyWindowSet.materialize() called: allocating a dense "
            "(%d, %d, %d) array (%s). Prefer indexing the LazyWindowSet "
            "lazily for large datasets.",
            len(self),
            self.window_size,
            self.n_channels,
            self.dtype,
        )
        out = np.empty((len(self), self.window_size, self.n_channels), dtype=np.float64)
        for i in range(len(self)):
            out[i] = self[i]
        return out


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
        - Return the resulting telemetry windows and label windows,
          index-aligned, such that window `i` of the telemetry set
          corresponds exactly to the same timestamp range as window `i`
          of the label set.

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
    # Public API: lazy (production-scale) path
    # ---------------------------------------------------------------

    def generate_lazy(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
    ) -> Tuple[LazyWindowSet, LazyWindowSet]:
        """
        Generate fixed-length sliding windows lazily: no
        `(num_windows, window_size, n_channels)` array is ever
        allocated. Use this for any dataset where `num_windows *
        window_size * n_channels` would not comfortably fit in RAM.

        Args:
            df: Synchronized multivariate telemetry. Must have a
                DatetimeIndex, time-ordered rows, and one column per
                telemetry channel. The DataFrame is never modified.
            labels: Dense channel-wise anomaly labels of shape
                `(num_timestamps, num_channels)`, row/column-aligned
                with `df`. Never modified.

        Returns:
            Tuple of `(telemetry_windows, label_windows)`, each a
            `LazyWindowSet` of length `number_of_windows`, index-aligned
            with each other.

        Raises:
            InvalidInputError: If `df` or `labels` fails validation, or
                if their shapes are inconsistent with each other.
            InsufficientDataError: If no window can be produced from
                the given data under the current configuration.
        """
        self._validate_dataframe(df)
        self._validate_labels(labels, df)

        total_rows = len(df)
        n_channels = df.shape[1]

        logger.info(
            "Generating windows (lazy): total_rows=%d, window_size=%d, "
            "stride=%d, drop_incomplete=%s, n_channels=%d",
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
            # Exactly one NaN-padded window containing all available rows.
            telemetry_windows = LazyWindowSet(
                telemetry_values,
                self.window_size,
                self.stride,
                num_full_windows=0,
                has_padded_trailing_window=True,
                trailing_start=0,
                trailing_len=total_rows,
            )
            label_windows = LazyWindowSet(
                label_values,
                self.window_size,
                self.stride,
                num_full_windows=0,
                has_padded_trailing_window=True,
                trailing_start=0,
                trailing_len=total_rows,
            )
            logger.info("Generated %d window(s) (lazy).", len(telemetry_windows))
            return telemetry_windows, label_windows

        num_full_windows = (total_rows - self.window_size) // self.stride + 1
        last_full_start = (num_full_windows - 1) * self.stride
        next_start = last_full_start + self.stride

        has_padded_trailing_window = False
        trailing_start = 0
        trailing_len = 0

        if not self.drop_incomplete and next_start < total_rows:
            remaining_len = total_rows - next_start
            if remaining_len == self.window_size:
                # A full window the stride selection didn't land on
                # exactly (can happen when total_rows aligns evenly);
                # counting it as "full" keeps behavior identical to
                # the eager implementation, and no NaN padding is used.
                num_full_windows += 1
            else:
                has_padded_trailing_window = True
                trailing_start = next_start
                trailing_len = remaining_len

        telemetry_windows = LazyWindowSet(
            telemetry_values,
            self.window_size,
            self.stride,
            num_full_windows,
            has_padded_trailing_window,
            trailing_start,
            trailing_len,
        )
        label_windows = LazyWindowSet(
            label_values,
            self.window_size,
            self.stride,
            num_full_windows,
            has_padded_trailing_window,
            trailing_start,
            trailing_len,
        )

        if len(telemetry_windows) == 0:
            raise InsufficientDataError(
                f"No windows could be generated from {total_rows} row(s) with "
                f"window_size={self.window_size}, stride={self.stride}, "
                f"drop_incomplete={self.drop_incomplete}."
            )

        logger.info("Generated %d window(s) (lazy).", len(telemetry_windows))
        return telemetry_windows, label_windows

    # ---------------------------------------------------------------
    # Public API: eager (legacy / small-data / test) path
    # ---------------------------------------------------------------

    def generate(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fixed-length sliding windows, materialized eagerly as
        dense `(number_of_windows, window_size, number_of_channels)`
        numpy arrays.

        WARNING -- memory: this allocates `number_of_windows *
        window_size * number_of_channels` elements for telemetry, and
        again for labels. When `stride < window_size` (overlapping
        windows), this can be many times larger than the underlying
        DataFrame. For production-scale datasets, use
        `generate_lazy()` instead, which returns index-aligned
        `LazyWindowSet` objects that never materialize more than one
        window at a time.

        This method is implemented in terms of `generate_lazy()`
        followed by `LazyWindowSet.materialize()`, so both paths share
        one validation/window-boundary implementation and always
        produce identical windows in identical order.

        Returns:
            Tuple of `(telemetry_windows, label_windows)` as dense
            numpy arrays. See `generate_lazy()` for argument and
            validation details.
        """
        telemetry_lazy, label_lazy = self.generate_lazy(df, labels)
        telemetry_windows = telemetry_lazy.materialize()
        label_windows = label_lazy.materialize()
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
