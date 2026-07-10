"""Diagnostics utility for inspecting telemetry DataFrames.

This module provides :class:`ChannelInspector`, a lightweight,
read-only inspection tool for telemetry data. It is strictly limited
to producing a structured diagnostic report and must never mutate,
preprocess, normalize, synchronize, plot, or otherwise transform the
input data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class ChannelInspector:
    """Inspects a telemetry DataFrame and produces a structured report.

    ChannelInspector is a read-only diagnostics utility. It only
    reads and summarizes the contents of a DataFrame; it never
    modifies, preprocesses, normalizes, synchronizes, or plots data.
    """

    def inspect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect a telemetry DataFrame and return a diagnostic report.

        Args:
            df: The pandas DataFrame to inspect. This DataFrame is
                never modified.

        Returns:
            A dictionary containing the following keys:
                rows: Number of rows in the DataFrame.
                columns: Number of columns in the DataFrame.
                column_names: List of column names.
                dtypes: Mapping of column name to dtype string.
                missing_values: Mapping of column name to count of
                    missing (NaN/null) values.
                duplicate_rows: Number of fully duplicated rows.
                memory_usage_mb: Deep memory usage of the DataFrame in
                    megabytes, rounded to two decimals.
                summary_statistics: Serializable dictionary form of
                    ``df.describe(include="all")``.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas.DataFrame, got {type(df).__name__!r}."
            )

        logger.debug("Inspecting DataFrame with shape %s", df.shape)

        report: Dict[str, Any] = {
            "rows": self._get_row_count(df),
            "columns": self._get_column_count(df),
            "column_names": self._get_column_names(df),
            "dtypes": self._get_dtypes(df),
            "missing_values": self._get_missing_values(df),
            "duplicate_rows": self._get_duplicate_row_count(df),
            "memory_usage_mb": self._get_memory_usage_mb(df),
            "summary_statistics": self._get_summary_statistics(df),
        }

        logger.debug("Inspection complete.")
        return report

    @staticmethod
    def _get_row_count(df: pd.DataFrame) -> int:
        """Return the number of rows in the DataFrame."""
        return int(df.shape[0])

    @staticmethod
    def _get_column_count(df: pd.DataFrame) -> int:
        """Return the number of columns in the DataFrame."""
        return int(df.shape[1])

    @staticmethod
    def _get_column_names(df: pd.DataFrame) -> list[str]:
        """Return the list of column names as strings."""
        return [str(col) for col in df.columns]

    @staticmethod
    def _get_dtypes(df: pd.DataFrame) -> Dict[str, str]:
        """Return a mapping of column name to dtype string."""
        return {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    @staticmethod
    def _get_missing_values(df: pd.DataFrame) -> Dict[str, int]:
        """Return a mapping of column name to missing value count."""
        return {str(col): int(count) for col, count in df.isna().sum().items()}

    @staticmethod
    def _get_duplicate_row_count(df: pd.DataFrame) -> int:
        """Return the number of fully duplicated rows."""
        return int(df.duplicated(keep="first").sum())

    @staticmethod
    def _get_memory_usage_mb(df: pd.DataFrame) -> float:
        """Return the DataFrame's deep memory usage in megabytes."""
        total_bytes = int(df.memory_usage(deep=True, index=True).sum())
        return round(total_bytes / (1024 ** 2), 2)

    @staticmethod
    def _get_summary_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Return a serializable dict form of df.describe(include="all").

        Any NaN values produced by describe() (e.g. for statistics
        that are not applicable to a given column) are converted to
        None to keep the result JSON-serializable.
        """
        try:
            described = df.describe(include="all")
        except ValueError:
            # Occurs if df has no columns at all.
            return {}

        described_dict = described.to_dict()

        serializable: Dict[str, Dict[str, Any]] = {}
        for col, stats in described_dict.items():
            serializable[str(col)] = {
                str(stat_name): (None if pd.isna(value) else value)
                for stat_name, value in stats.items()
            }

        return serializable