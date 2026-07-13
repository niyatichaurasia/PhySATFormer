"""
src/preprocessing/assembler.py

TelemetryAssembler: combines multiple raw telemetry channels from an ESA
Mission into a single synchronized multivariate telemetry DataFrame.

This module is strictly responsible for telemetry assembly. It does not
normalize, interpolate, fill missing values, generate windows, or engineer
features. It returns raw, synchronized telemetry only.
"""

from __future__ import annotations

import logging

import pandas as pd

# NOTE: Import path assumes the project's canonical Mission/Channel
# definitions live in src.data.mission. Update this import if the
# project structure differs.
from src.core.channel import Channel
from src.core.mission import Mission
logger = logging.getLogger(__name__)


# =========================================================================
# Exceptions
# =========================================================================


class TelemetryAssemblyError(Exception):
    """Base exception for telemetry assembly failures."""


class ChannelNotFoundError(TelemetryAssemblyError):
    """Raised when a requested channel does not exist on the Mission."""


class TelemetryLoadError(TelemetryAssemblyError):
    """Raised when a channel's telemetry could not be loaded."""


class InvalidTimestampError(TelemetryAssemblyError):
    """Raised when a channel's telemetry has invalid/unusable timestamps."""


# =========================================================================
# TelemetryAssembler
# =========================================================================


class TelemetryAssembler:
    """
    Combines multiple raw telemetry channels from a Mission into a single
    synchronized multivariate telemetry DataFrame.

    Responsibilities:
        - Load raw telemetry per requested channel.
        - Rename each channel's value column to its channel identifier.
        - Merge all channels into one DataFrame, time-aligned via
          pandas.merge_asof(), preserving a DatetimeIndex.

    Explicitly NOT responsible for:
        - Normalization / scaling
        - Interpolation or gap filling
        - Sliding window generation
        - Feature engineering
        - Any machine learning logic
    """

    _VALID_DIRECTIONS = frozenset({"backward", "forward", "nearest"})

    def __init__(
        self,
        tolerance: pd.Timedelta | str | None = None,
        direction: str = "nearest",
    ) -> None:
        """
        Args:
            tolerance: Maximum allowed distance between matched timestamps
                for pandas.merge_asof(). If None, merge_asof matches the
                nearest prior timestamp with no distance limit. Accepts a
                pandas.Timedelta or a pandas-parsable offset string
                (e.g. "1s", "500ms").
            direction: Direction passed to pandas.merge_asof() for
                timestamp alignment. Must be one of "backward", "forward",
                or "nearest".
        """
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid direction '{direction}'. Must be one of "
                f"{sorted(self._VALID_DIRECTIONS)}."
            )
        self._tolerance = pd.Timedelta(tolerance) if tolerance is not None else None
        self._direction = direction

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def assemble(self, mission: Mission, channel_ids: list[str]) -> pd.DataFrame:
        """
        Assemble raw telemetry for the given channel_ids into a single
        synchronized multivariate DataFrame.

        Args:
            mission: Mission object providing access to Channel objects.
            channel_ids: List of channel identifiers to assemble, e.g.
                ["channel_1", "channel_2", "channel_5"].

        Returns:
            A pandas DataFrame with a DatetimeIndex and one column per
            requested channel, containing raw (unmodified) telemetry
            values, time-aligned via merge_asof.

        Raises:
            ValueError: If channel_ids is empty.
            ChannelNotFoundError: If a requested channel does not exist.
            TelemetryLoadError: If a channel's telemetry cannot be loaded.
            InvalidTimestampError: If a channel's telemetry has invalid
                or unusable timestamps.
        """
        if not channel_ids:
            raise ValueError("channel_ids must be a non-empty list of channel identifiers.")

        logger.info("Assembling telemetry for channels: %s", channel_ids)

        series_frames: list[pd.DataFrame] = []
        for channel_id in channel_ids:
            channel = self._get_channel(mission, channel_id)
            raw = self._load_channel_telemetry(channel, channel_id)
            prepared = self._prepare_channel_frame(raw, channel_id)
            series_frames.append(prepared)

        assembled = self._merge_all(series_frames, channel_ids)

        logger.info(
            "Assembled telemetry DataFrame: shape=%s, span=%s -> %s",
            assembled.shape,
            assembled.index.min() if not assembled.empty else None,
            assembled.index.max() if not assembled.empty else None,
        )
        return assembled

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _get_channel(self, mission: Mission, channel_id: str) -> Channel:
        try:
            channel = mission.get_channel(channel_id)
        except Exception as exc:  # noqa: BLE001 - re-raise as domain error
            raise ChannelNotFoundError(
                f"Failed to resolve channel '{channel_id}' from mission: {exc}"
            ) from exc

        if channel is None:
            raise ChannelNotFoundError(f"Channel '{channel_id}' does not exist on this mission.")

        return channel

    def _load_channel_telemetry(self, channel: Channel, channel_id: str) -> pd.DataFrame:
        try:
            telemetry = channel.load()
        except Exception as exc:  # noqa: BLE001 - re-raise as domain error
            raise TelemetryLoadError(
                f"Failed to load telemetry for channel '{channel_id}': {exc}"
            ) from exc

        if telemetry is None:
            raise TelemetryLoadError(f"Telemetry for channel '{channel_id}' is None.")

        if not isinstance(telemetry, pd.DataFrame):
            raise TelemetryLoadError(
                f"Telemetry for channel '{channel_id}' must be a pandas DataFrame, "
                f"got {type(telemetry).__name__}."
            )

        if telemetry.empty:
            raise TelemetryLoadError(f"Telemetry for channel '{channel_id}' is empty.")

        return telemetry

    def _prepare_channel_frame(self, raw: pd.DataFrame, channel_id: str) -> pd.DataFrame:
        """
        Produce a single-value-column frame with a DatetimeIndex named
        'timestamp', sorted ascending, with the value column renamed to
        channel_id. Telemetry values are preserved unchanged; duplicate
        timestamps are logged but not removed.
        """
        self._validate_channel_frame(raw, channel_id)
        frame = self._normalize_timestamp_index(raw, channel_id)

        original_col = frame.columns[0]
        frame = frame.rename(columns={original_col: channel_id})
        frame = frame.sort_index()

        if frame.index.has_duplicates:
            n_dupes = int(frame.index.duplicated().sum())
            logger.warning(
                "Channel '%s' has %d duplicate timestamp(s); telemetry preserved unchanged.",
                channel_id,
                n_dupes,
            )

        return frame

    def _find_timestamp_column(self, frame: pd.DataFrame, channel_id: str) -> str:
        for candidate in ("timestamp", "time", "datetime", "date"):
            if candidate in frame.columns:
                return candidate

        raise InvalidTimestampError(
            f"Channel '{channel_id}' telemetry has no DatetimeIndex and no "
            f"recognizable timestamp column (expected one of: "
            f"'timestamp', 'time', 'datetime', 'date')."
        )

    def _validate_channel_frame(self, frame: pd.DataFrame, channel_id: str) -> None:
        """
        Validate a channel's raw telemetry frame without modifying it.

        Checks:
          - a valid DatetimeIndex, or a recognizable timestamp column
            whose values are all parseable and non-null
          - exactly one telemetry value column
        """
        if isinstance(frame.index, pd.DatetimeIndex):
            if frame.index.hasnans:
                raise InvalidTimestampError(
                    f"Channel '{channel_id}' has null timestamp(s) in its DatetimeIndex."
                )
            value_columns = list(frame.columns)
        else:
            timestamp_col = self._find_timestamp_column(frame, channel_id)
            try:
                parsed = pd.to_datetime(frame[timestamp_col], errors="raise", utc=False)
            except Exception as exc:  # noqa: BLE001 - re-raise as domain error
                raise InvalidTimestampError(
                    f"Channel '{channel_id}' has unparseable timestamps in column "
                    f"'{timestamp_col}': {exc}"
                ) from exc

            if parsed.isna().any():
                raise InvalidTimestampError(
                    f"Channel '{channel_id}' has null/unparseable timestamp value(s) "
                    f"in column '{timestamp_col}'."
                )
            value_columns = [c for c in frame.columns if c != timestamp_col]

        if len(value_columns) == 0:
            raise TelemetryLoadError(
                f"Telemetry for channel '{channel_id}' has no value column."
            )
        if len(value_columns) > 1:
            raise TelemetryLoadError(
                f"Telemetry for channel '{channel_id}' must contain exactly one "
                f"value column, found {len(value_columns)}: {value_columns}."
            )

    def _normalize_timestamp_index(self, frame: pd.DataFrame, channel_id: str) -> pd.DataFrame:
        """
        Return `frame` indexed by a DatetimeIndex named 'timestamp'.
        Assumes `frame` has already passed `_validate_channel_frame`.

        Accepts either:
          - a frame already indexed by a DatetimeIndex, or
          - a frame with a 'timestamp' (or 'time') column to be set as index.
        """
        if isinstance(frame.index, pd.DatetimeIndex):
            return frame.rename_axis("timestamp")

        timestamp_col = self._find_timestamp_column(frame, channel_id)
        parsed = pd.to_datetime(frame[timestamp_col], errors="raise", utc=False)
        frame = frame.drop(columns=[timestamp_col])
        frame.index = pd.DatetimeIndex(parsed, name="timestamp")
        return frame

    def _merge_all(
        self,
        series_frames: list[pd.DataFrame],
        channel_ids: list[str],
    ) -> pd.DataFrame:
        """
        Merge single-column channel frames into one multivariate DataFrame
        using pandas.merge_asof(), aligned on timestamps.
        """
        if len(series_frames) == 1:
            return series_frames[0]

        base = series_frames[0].reset_index()
        for i in range(1, len(series_frames)):
            right = series_frames[i].reset_index()
            try:
                base = pd.merge_asof(
                    base,
                    right,
                    on="timestamp",
                    direction=self._direction,
                    tolerance=self._tolerance,
                )
            except Exception as exc:  # noqa: BLE001 - re-raise as domain error
                raise TelemetryAssemblyError(
                    f"Failed to merge channel '{channel_ids[i]}' into assembled "
                    f"telemetry via merge_asof: {exc}"
                ) from exc

        base = base.set_index("timestamp")
        base = base.sort_index()
        base = base[channel_ids]
        return base