"""
analyzer.py

Part of the PhySATFormer research project.

This module defines the TelemetryAnalyzer class, whose sole responsibility
is to analyze raw telemetry DataFrames and produce a structured, read-only
report describing their structure and temporal characteristics.

TelemetryAnalyzer MUST NEVER:
    - modify data
    - normalize
    - interpolate
    - synchronize
    - generate windows
    - perform machine learning

It is a pure, read-only analysis component.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingIntervalStatistics:
    """Container for sampling interval statistics, expressed in seconds."""

    mean_seconds: Optional[float]
    median_seconds: Optional[float]
    min_seconds: Optional[float]
    max_seconds: Optional[float]
    std_seconds: Optional[float]

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "std_seconds": self.std_seconds,
        }


@dataclass(frozen=True)
class TelemetryReport:
    """Structured report describing a telemetry DataFrame."""

    number_of_rows: int
    number_of_columns: int
    start_timestamp: Optional[pd.Timestamp]
    end_timestamp: Optional[pd.Timestamp]
    duration: Optional[pd.Timedelta]
    sampling_interval_statistics: SamplingIntervalStatistics
    irregular_sampling_detected: bool
    duplicate_timestamps: int
    monotonic_time_index: bool
    inferred_sampling_frequency: Optional[str]
    memory_usage_mb: float
    column_names: list = field(default_factory=list)
    channel_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number_of_rows": self.number_of_rows,
            "number_of_columns": self.number_of_columns,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration": self.duration,
            "sampling_interval_statistics": self.sampling_interval_statistics.as_dict(),
            "irregular_sampling_detected": self.irregular_sampling_detected,
            "duplicate_timestamps": self.duplicate_timestamps,
            "monotonic_time_index": self.monotonic_time_index,
            "inferred_sampling_frequency": self.inferred_sampling_frequency,
            "memory_usage_mb": self.memory_usage_mb,
            "column_names": self.column_names,
            "channel_name": self.channel_name,
        }


class TelemetryAnalyzer:
    """
    Analyzes raw telemetry DataFrames without modifying them in any way.

    This class is strictly read-only. It inspects structural and temporal
    properties of a telemetry DataFrame (assumed to be indexed by a
    DatetimeIndex) and returns a structured report as a dictionary.

    Responsibilities explicitly EXCLUDED from this class:
        - data modification
        - normalization
        - interpolation
        - synchronization
        - window generation
        - machine learning
    """

    def __init__(self) -> None:
        self._logger = logger

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a telemetry DataFrame and return a structured report.

        Parameters
        ----------
        df : pandas.DataFrame
            Telemetry data indexed by a DatetimeIndex. This DataFrame is
            never modified.

        Returns
        -------
        dict
            A structured report containing metadata about the DataFrame's
            shape, time range, sampling characteristics, and memory usage.

        Raises
        ------
        TypeError
            If `df` is not a pandas.DataFrame, or if its index is not a
            pandas.DatetimeIndex.
        """
        self._validate_input(df)

        self._logger.debug(
            "Starting telemetry analysis: rows=%d, columns=%d",
            df.shape[0],
            df.shape[1],
        )

        number_of_rows, number_of_columns = df.shape

        start_timestamp, end_timestamp, duration = self._analyze_time_range(df.index)
        sampling_stats = self._compute_sampling_interval_statistics(df.index)
        irregular_sampling_detected = self._detect_irregular_sampling(df.index)
        duplicate_timestamps = self._count_duplicate_timestamps(df.index)
        monotonic_time_index = bool(df.index.is_monotonic_increasing)
        inferred_frequency = self._infer_sampling_frequency(df.index)
        memory_usage_mb = self._compute_memory_usage_mb(df)
        channel_name = str(df.columns[0]) if number_of_columns == 1 else None

        report = TelemetryReport(
            number_of_rows=number_of_rows,
            number_of_columns=number_of_columns,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            duration=duration,
            sampling_interval_statistics=sampling_stats,
            irregular_sampling_detected=irregular_sampling_detected,
            duplicate_timestamps=duplicate_timestamps,
            monotonic_time_index=monotonic_time_index,
            inferred_sampling_frequency=inferred_frequency,
            memory_usage_mb=memory_usage_mb,
            column_names=list(df.columns),
            channel_name=channel_name,
        )

        self._logger.debug("Completed telemetry analysis: %s", report.to_dict())

        return report.to_dict()

    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate that the input is a DataFrame with a DatetimeIndex.

        Raises
        ------
        TypeError
            If `df` is not a pandas.DataFrame, or its index is not a
            pandas.DatetimeIndex.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas.DataFrame, got {type(df).__name__!r}."
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "Expected the DataFrame index to be a pandas.DatetimeIndex, "
                f"got {type(df.index).__name__!r}."
            )

    def _analyze_time_range(
        self, index: pd.DatetimeIndex
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timedelta]]:
        """Compute the start timestamp, end timestamp, and duration."""
        if len(index) == 0:
            return None, None, None

        start_timestamp = index.min()
        end_timestamp = index.max()
        duration = end_timestamp - start_timestamp

        return start_timestamp, end_timestamp, duration

    def _compute_sampling_interval_statistics(
        self, index: pd.DatetimeIndex
    ) -> SamplingIntervalStatistics:
        """
        Compute descriptive statistics of the time deltas between
        consecutive (sorted) timestamps, expressed in seconds.
        """
        if len(index) < 2:
            return SamplingIntervalStatistics(
                mean_seconds=None,
                median_seconds=None,
                min_seconds=None,
                max_seconds=None,
                std_seconds=None,
            )

        sorted_index = index.sort_values()
        deltas = sorted_index.to_series().diff().dropna()
        deltas_seconds = deltas.dt.total_seconds()

        if deltas_seconds.empty:
            return SamplingIntervalStatistics(
                mean_seconds=None,
                median_seconds=None,
                min_seconds=None,
                max_seconds=None,
                std_seconds=None,
            )

        return SamplingIntervalStatistics(
            mean_seconds=float(deltas_seconds.mean()),
            median_seconds=float(deltas_seconds.median()),
            min_seconds=float(deltas_seconds.min()),
            max_seconds=float(deltas_seconds.max()),
            std_seconds=float(deltas_seconds.std()),
        )

    def _detect_irregular_sampling(self, index: pd.DatetimeIndex) -> bool:
        """
        Detect whether the sampling interval between consecutive sorted
        timestamps is irregular (i.e. not all intervals are approximately
        equal).
        """
        if len(index) < 3:
            return False

        sorted_index = index.sort_values()
        deltas = sorted_index.to_series().diff().dropna().dt.total_seconds()

        if deltas.empty:
            return False

        median_delta = deltas.median()
        if median_delta <= 0:
            return False

        # Allow small floating point / jitter tolerance (1% of median interval).
        tolerance = max(median_delta * 0.01, 1e-9)
        irregular = (deltas - median_delta).abs() > tolerance

        return bool(irregular.any())

    def _count_duplicate_timestamps(self, index: pd.DatetimeIndex) -> int:
        """Count the number of duplicate timestamp entries in the index."""
        return int(index.duplicated().sum())

    def _infer_sampling_frequency(self, index: pd.DatetimeIndex) -> Optional[str]:
        """
        Attempt to infer a pandas frequency alias (e.g. 'S', 'min', 'H')
        for the given DatetimeIndex. Returns None if no frequency can be
        confidently inferred.
        """
        if len(index) < 3:
            return None

        sorted_index = index.sort_values()

        try:
            inferred = pd.infer_freq(sorted_index)
        except (ValueError, TypeError) as exc:
            self._logger.debug("Could not infer sampling frequency: %s", exc)
            inferred = None

        return inferred

    def _compute_memory_usage_mb(self, df: pd.DataFrame) -> float:
        """Compute the deep memory usage of the DataFrame in megabytes."""
        total_bytes = df.memory_usage(index=True, deep=True).sum()
        return round(float(total_bytes) / (1024 ** 2), 2)