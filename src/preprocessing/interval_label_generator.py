"""
src/preprocessing/interval_label_generator.py

IntervalLabelGenerator: converts ESA Mission interval anomaly annotations
(``Mission.labels``) into dense, per-timestamp, per-channel binary labels
aligned with an already-synchronized multivariate telemetry DataFrame.

This module is strictly responsible for interval -> dense-label conversion.
It does not:
    - load telemetry
    - synchronize telemetry streams
    - normalize / standardize data
    - generate sliding windows
    - split datasets
    - perform any modeling or learning logic

All such responsibilities belong to other modules in the preprocessing
pipeline (TelemetryAssembler, WindowGenerator, TelemetryNormalizer,
MissionDataset, TelemetryPipeline). This module sits conceptually between
TelemetryAssembler and WindowGenerator, but is intentionally NOT wired
into TelemetryPipeline as part of this change -- it is a standalone,
independently usable component.

Research formulation
---------------------
The model performs channel-level spatio-temporal anomaly localization.
Ground truth is therefore a dense ``(num_timestamps, num_channels)``
binary matrix, where ``labels[t, c] == 1`` iff telemetry channel ``c`` is
anomalous at timestamp ``t``. Each anomaly interval only ever affects the
single column corresponding to the channel it was annotated against; if
multiple channels are anomalous at overlapping timestamps, every
corresponding column is independently set to ``1``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.core.mission import Mission
from src.utils.constants import (
    CHANNEL_ID_COLUMN,
    END_TIME_COLUMN,
    LABEL_CHANNEL_COLUMN,
    START_TIME_COLUMN,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Exceptions
# =========================================================================


class IntervalLabelGenerationError(Exception):
    """Base exception for interval label generation failures."""


class InvalidTelemetryError(IntervalLabelGenerationError):
    """Raised when the supplied telemetry DataFrame fails validation."""


class InvalidLabelsError(IntervalLabelGenerationError):
    """Raised when ``Mission.labels`` fails validation."""


# =========================================================================
# IntervalLabelGenerator
# =========================================================================


class IntervalLabelGenerator:
    """Converts anomaly intervals into dense, per-timestamp/per-channel labels.

    The ESA Mission dataset stores anomaly annotations as intervals, each
    identifying a telemetry channel together with a start and end
    timestamp. Downstream Transformer models require one binary label per
    ``(timestamp, channel)`` pair (0 = normal, 1 = anomaly) so that
    channel-level spatio-temporal anomaly localization can be trained
    directly.

    ``IntervalLabelGenerator`` performs exactly this conversion. Given a
    :class:`~src.core.mission.Mission` (for its interval annotations and
    channel ordering) and an already-synchronized telemetry
    :class:`pandas.DataFrame` (for its timestamp axis and channel
    membership), it produces a dense binary label matrix of shape
    ``(num_timestamps, num_channels)``.

    Channel ordering (i.e. column order) is determined by
    ``Mission.channels``, restricted to the channels that are actually
    present in ``telemetry.columns`` (channels that are not part of the
    synchronized telemetry cannot be labeled and are therefore excluded
    from the output entirely).

    A ``(timestamp, channel)`` pair is labeled anomalous (``1``) if the
    timestamp falls within ``[StartTime, EndTime]`` (inclusive, by
    default) of at least one anomaly interval annotated against that
    channel. Intervals referencing channels absent from
    ``telemetry.columns`` are ignored, since they cannot affect any
    output column.

    Parameters
    ----------
    inclusive : bool, optional
        Whether interval boundaries (``StartTime`` and ``EndTime``) are
        treated as inclusive when assigning labels, by default True.

    Attributes
    ----------
    inclusive : bool
        Whether interval boundaries are inclusive.

    Examples
    --------
    >>> generator = IntervalLabelGenerator()
    >>> labels = generator.generate(mission, telemetry)
    >>> labels.shape
    (len(telemetry), num_channels)
    """

    def __init__(self, inclusive: bool = True) -> None:
        self.inclusive = inclusive
        logger.debug(
            "IntervalLabelGenerator initialized (inclusive=%s)", self.inclusive
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        mission: Mission,
        telemetry: pd.DataFrame,
    ) -> np.ndarray:
        """Generate dense per-channel binary anomaly labels for ``telemetry``.

        Parameters
        ----------
        mission : Mission
            Mission object providing access to interval anomaly
            annotations via ``mission.labels`` and to channel ordering
            via ``mission.channels``. Telemetry is never loaded from
            ``mission``; only its metadata is read.
        telemetry : pandas.DataFrame
            Synchronized multivariate telemetry whose index is a
            :class:`pandas.DatetimeIndex` representing the timestamp
            axis to label, and whose columns identify the telemetry
            channels to label. Never modified.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(num_timestamps, num_channels)`` and dtype
            ``int64`` containing ``0`` (normal) or ``1`` (anomaly) for
            every ``(timestamp, channel)`` pair. Rows follow
            ``telemetry.index`` order; columns follow the order in which
            each telemetry channel appears in ``mission.channels``.

        Raises
        ------
        InvalidTelemetryError
            If ``telemetry`` fails validation (wrong type, empty, not
            indexed by a sorted ``DatetimeIndex``).
        InvalidLabelsError
            If ``mission.labels`` fails validation (wrong type, missing
            required columns, unparseable timestamps).
        """
        self._validate_telemetry(telemetry)

        num_timestamps = len(telemetry)

        channel_order = self._get_channel_order(mission, telemetry)
        num_channels = len(channel_order)
        channel_to_index = {
            channel_id: idx for idx, channel_id in enumerate(channel_order)
        }

        labels = np.zeros((num_timestamps, num_channels), dtype=np.int64)

        if num_channels == 0:
            logger.info(
                "No telemetry channels present in mission.channels; "
                "returning empty dense labels (num_timestamps=%d, "
                "num_channels=0).",
                num_timestamps,
            )
            return labels

        raw_labels = self._get_mission_labels(mission)
        prepared_labels = self._prepare_labels(
            raw_labels, channel_to_index, telemetry
        )

        if prepared_labels.empty:
            logger.info(
                "No applicable anomaly intervals found for telemetry "
                "channels %s; returning all-normal dense labels "
                "(num_timestamps=%d, num_channels=%d).",
                channel_order,
                num_timestamps,
                num_channels,
            )
            return labels

        timestamps = telemetry.index

        num_intervals_applied = 0
        num_intervals_out_of_range = 0

        for start, end, channel_id in zip(
            prepared_labels[START_TIME_COLUMN],
            prepared_labels[END_TIME_COLUMN],
            prepared_labels[LABEL_CHANNEL_COLUMN],
        ):
            start_idx, end_idx = self._interval_to_index_bounds(
                timestamps, start, end
            )

            if start_idx >= end_idx:
                num_intervals_out_of_range += 1
                logger.debug(
                    "Interval for channel '%s' [%s, %s] does not overlap "
                    "any telemetry timestamp; skipped.",
                    channel_id,
                    start,
                    end,
                )
                continue

            column_idx = channel_to_index[channel_id]
            labels[start_idx:end_idx, column_idx] = 1
            num_intervals_applied += 1

        num_anomalous = int(labels.sum())
        logger.info(
            "Dense per-channel label generation complete: "
            "num_timestamps=%d, num_channels=%d, num_anomalous_entries=%d, "
            "intervals_applied=%d, intervals_out_of_range=%d.",
            num_timestamps,
            num_channels,
            num_anomalous,
            num_intervals_applied,
            num_intervals_out_of_range,
        )

        return labels

    # ------------------------------------------------------------------
    # Internal helpers: validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_telemetry(telemetry: pd.DataFrame) -> None:
        """Validate that ``telemetry`` is a well-formed synchronized frame.

        Raises
        ------
        InvalidTelemetryError
            If ``telemetry`` is not a DataFrame, is empty, is not indexed
            by a DatetimeIndex, contains null timestamps, or is not
            sorted in non-decreasing order.
        """
        if not isinstance(telemetry, pd.DataFrame):
            raise InvalidTelemetryError(
                f"telemetry must be a pandas DataFrame, got "
                f"{type(telemetry).__name__}."
            )

        if telemetry.empty:
            raise InvalidTelemetryError(
                "telemetry is empty; cannot generate dense labels."
            )

        if not isinstance(telemetry.index, pd.DatetimeIndex):
            raise InvalidTelemetryError(
                f"telemetry must have a DatetimeIndex, got "
                f"{type(telemetry.index).__name__}."
            )

        if telemetry.index.hasnans:
            raise InvalidTelemetryError(
                "telemetry index contains null timestamp(s)."
            )

        if not telemetry.index.is_monotonic_increasing:
            raise InvalidTelemetryError(
                "telemetry index (timestamps) must be sorted in "
                "non-decreasing order."
            )

    @staticmethod
    def _get_mission_labels(mission: Mission) -> pd.DataFrame:
        """Retrieve and validate ``mission.labels``.

        Raises
        ------
        InvalidLabelsError
            If ``mission.labels`` is not a DataFrame or is missing
            required columns.
        """
        try:
            raw_labels = mission.labels
        except Exception as exc:  # noqa: BLE001 - re-raise as domain error
            raise InvalidLabelsError(
                f"Failed to retrieve labels from mission: {exc}"
            ) from exc

        if not isinstance(raw_labels, pd.DataFrame):
            raise InvalidLabelsError(
                f"mission.labels must be a pandas DataFrame, got "
                f"{type(raw_labels).__name__}."
            )

        required_columns = (
            LABEL_CHANNEL_COLUMN,
            START_TIME_COLUMN,
            END_TIME_COLUMN,
        )
        missing_columns = [
            col for col in required_columns if col not in raw_labels.columns
        ]
        if missing_columns:
            raise InvalidLabelsError(
                f"mission.labels is missing required columns: "
                f"{missing_columns}. Found columns: {list(raw_labels.columns)}"
            )

        return raw_labels

    @staticmethod
    def _get_channel_order(
        mission: Mission,
        telemetry: pd.DataFrame,
    ) -> list:
        """Determine output column ordering from ``mission.channels``.

        The full channel ordering is read from ``mission.channels`` (via
        ``CHANNEL_ID_COLUMN``), then restricted to the channels that are
        actually present in ``telemetry.columns``, preserving the
        ``mission.channels`` order rather than ``telemetry.columns``
        order.

        Raises
        ------
        InvalidLabelsError
            If ``mission.channels`` is not a DataFrame or is missing the
            channel identifier column.
        """
        try:
            channels_metadata = mission.channels
        except Exception as exc:  # noqa: BLE001 - re-raise as domain error
            raise InvalidLabelsError(
                f"Failed to retrieve channel metadata from mission: {exc}"
            ) from exc

        if not isinstance(channels_metadata, pd.DataFrame):
            raise InvalidLabelsError(
                f"mission.channels must be a pandas DataFrame, got "
                f"{type(channels_metadata).__name__}."
            )

        if CHANNEL_ID_COLUMN not in channels_metadata.columns:
            raise InvalidLabelsError(
                f"mission.channels is missing required column "
                f"'{CHANNEL_ID_COLUMN}'. Found columns: "
                f"{list(channels_metadata.columns)}"
            )

        available_channels = set(telemetry.columns)
        ordered_channel_ids = channels_metadata[CHANNEL_ID_COLUMN].tolist()

        channel_order = [
            channel_id
            for channel_id in ordered_channel_ids
            if channel_id in available_channels
        ]

        return channel_order

    # ------------------------------------------------------------------
    # Internal helpers: label preparation
    # ------------------------------------------------------------------

    def _prepare_labels(
        self,
        raw_labels: pd.DataFrame,
        channel_to_index: dict,
        telemetry: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter, parse, and de-duplicate anomaly intervals.

        Handles the following edge cases:
            - intervals referencing channels absent from the output
              channel ordering (dropped, since they cannot affect any
              output column)
            - duplicated intervals (dropped)
            - unsorted labels (result is sorted by start time)
            - unparseable / null timestamps (dropped, with a warning)

        Parameters
        ----------
        raw_labels : pandas.DataFrame
            Raw label metadata as returned by ``mission.labels``.
        channel_to_index : dict
            Mapping of channel identifier to output column index, as
            produced by :meth:`_get_channel_order`. Only intervals whose
            channel appears in this mapping are retained.
        telemetry : pandas.DataFrame
            Synchronized telemetry; only ``telemetry.index`` is used
            here, as a timezone reference for timestamp parsing.

        Returns
        -------
        pandas.DataFrame
            A cleaned copy of the label metadata containing only the
            columns ``LABEL_CHANNEL_COLUMN``, ``START_TIME_COLUMN``, and
            ``END_TIME_COLUMN``, with parsed timestamps, sorted by
            ``START_TIME_COLUMN``.
        """
        labels = raw_labels[
            [LABEL_CHANNEL_COLUMN, START_TIME_COLUMN, END_TIME_COLUMN]
        ].copy()

        relevant_mask = labels[LABEL_CHANNEL_COLUMN].isin(channel_to_index.keys())
        num_irrelevant = int((~relevant_mask).sum())
        if num_irrelevant:
            logger.debug(
                "Dropping %d anomaly interval(s) referencing channel(s) "
                "not present in the output channel ordering.",
                num_irrelevant,
            )
        labels = labels.loc[relevant_mask]

        if labels.empty:
            return labels

        labels[START_TIME_COLUMN] = self._parse_timestamps(
            labels[START_TIME_COLUMN], telemetry.index
        )
        labels[END_TIME_COLUMN] = self._parse_timestamps(
            labels[END_TIME_COLUMN], telemetry.index
        )

        null_mask = (
            labels[START_TIME_COLUMN].isna() | labels[END_TIME_COLUMN].isna()
        )
        num_null = int(null_mask.sum())
        if num_null:
            logger.warning(
                "Dropping %d anomaly interval(s) with unparseable or null "
                "timestamp(s).",
                num_null,
            )
            labels = labels.loc[~null_mask]

        if labels.empty:
            return labels

        invalid_order_mask = labels[START_TIME_COLUMN] > labels[END_TIME_COLUMN]
        num_invalid_order = int(invalid_order_mask.sum())
        if num_invalid_order:
            logger.warning(
                "Dropping %d anomaly interval(s) where StartTime is after "
                "EndTime.",
                num_invalid_order,
            )
            labels = labels.loc[~invalid_order_mask]

        num_before_dedup = len(labels)
        labels = labels.drop_duplicates(
            subset=[LABEL_CHANNEL_COLUMN, START_TIME_COLUMN, END_TIME_COLUMN]
        )
        num_duplicates = num_before_dedup - len(labels)
        if num_duplicates:
            logger.debug(
                "Dropped %d duplicate anomaly interval(s).", num_duplicates
            )

        labels = labels.sort_values(START_TIME_COLUMN).reset_index(drop=True)

        return labels

    @staticmethod
    def _parse_timestamps(
        series: pd.Series,
        reference_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Parse a Series of timestamp strings to match ``reference_index``.

        Timestamps are parsed with UTC awareness (ESA interval timestamps
        include a ``Z`` suffix). The result's timezone-awareness is then
        aligned with ``reference_index``: if the telemetry index is
        timezone-naive, parsed timestamps are converted to UTC and the
        timezone is dropped; if the telemetry index is timezone-aware,
        parsed timestamps are converted to that same timezone.

        Unparseable values become ``NaT`` rather than raising, so that
        malformed rows can be dropped by the caller instead of aborting
        the entire conversion.
        """
        parsed = pd.to_datetime(series, errors="coerce", utc=True)

        if reference_index.tz is None:
            parsed = parsed.dt.tz_localize(None)
        else:
            parsed = parsed.dt.tz_convert(reference_index.tz)

        return parsed

    # ------------------------------------------------------------------
    # Internal helpers: interval -> index conversion
    # ------------------------------------------------------------------

    def _interval_to_index_bounds(
        self,
        timestamps: pd.DatetimeIndex,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[int, int]:
        """Convert a ``[start, end]`` interval into a half-open index range.

        Uses binary search (``searchsorted``) over the sorted telemetry
        timestamp index rather than looping over every timestamp.

        Parameters
        ----------
        timestamps : pandas.DatetimeIndex
            Sorted telemetry timestamp axis.
        start : pandas.Timestamp
            Interval start timestamp.
        end : pandas.Timestamp
            Interval end timestamp.

        Returns
        -------
        tuple[int, int]
            ``(start_idx, end_idx)`` such that
            ``timestamps[start_idx:end_idx]`` covers every telemetry
            timestamp lying within the interval (empty if the interval
            does not overlap the telemetry range at all).
        """
        if self.inclusive:
            start_idx = int(timestamps.searchsorted(start, side="left"))
            end_idx = int(timestamps.searchsorted(end, side="right"))
        else:
            start_idx = int(timestamps.searchsorted(start, side="right"))
            end_idx = int(timestamps.searchsorted(end, side="left"))

        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(timestamps))

        return start_idx, end_idx