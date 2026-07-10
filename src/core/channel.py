"""Channel representation for PhySATFormer.

This module defines the :class:`Channel` dataclass, which represents a
single telemetry channel belonging to an ESA mission. A ``Channel``
instance is a thin, metadata-aware wrapper around one channel's telemetry
data. It knows *what* it is (via metadata) and *where* its data lives (via
``mission_root``), but it delegates all I/O and diagnostic work to
dedicated collaborators (:class:`~src.data.loader.ChannelLoader` and
:class:`~src.data.inspector.ChannelInspector`).

Design boundaries (see project specification):
    * Channel never parses metadata CSVs.
    * Channel never opens archive/ZIP files.
    * Channel never normalizes, preprocesses, or otherwise mutates
      telemetry values.
    * Channel never constructs tensors or performs machine learning.

Typical usage::

    channel = Channel(metadata=row, mission_root=Path("/data/mission"))
    df = channel.load()
    report = channel.inspect()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.inspector import ChannelInspector
from src.data.loader import ChannelLoader
from src.utils.constants import (
    CATEGORICAL_FLAG_COLUMN,
    CHANNEL_ID_COLUMN,
    GROUP_COLUMN,
    SUBSYSTEM_COLUMN,
    TARGET_FLAG_COLUMN,
    UNIT_COLUMN,
)

logger = logging.getLogger(__name__)


@dataclass
class Channel:
    """Represents a single telemetry channel within an ESA mission.

    A ``Channel`` holds the metadata describing one telemetry channel
    (e.g. its identifier, subsystem, engineering group, physical unit,
    and modeling flags) together with a reference to the mission root
    directory on disk. Telemetry data itself is loaded lazily, on
    demand, and cached in memory until explicitly cleared or reloaded.

    Attributes:
        metadata: A pandas Series containing the metadata row for this
            channel, as sourced from the mission's channel metadata
            table.
        mission_root: Path to the root directory of the mission this
            channel belongs to. Used only to locate data for delegation
            to :class:`ChannelLoader`; Channel never inspects or opens
            files within it directly.
        _data: Cached telemetry DataFrame. ``None`` until :meth:`load`
            is called for the first time.
    """

    metadata: pd.Series
    mission_root: Path
    _data: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate constructor inputs.

        Raises:
            TypeError: If ``metadata`` is not a pandas Series or
                ``mission_root`` is not a ``Path``.
            ValueError: If ``metadata`` does not contain the channel
                identifier column.
        """
        if not isinstance(self.metadata, pd.Series):
            raise TypeError(
                f"metadata must be a pandas Series, got {type(self.metadata).__name__}"
            )
        if not isinstance(self.mission_root, Path):
            raise TypeError(
                f"mission_root must be a pathlib.Path, got {type(self.mission_root).__name__}"
            )
        if CHANNEL_ID_COLUMN not in self.metadata:
            raise ValueError(
                f"metadata is missing required column '{CHANNEL_ID_COLUMN}'"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Return the channel identifier.

        Returns:
            The value stored under ``CHANNEL_ID_COLUMN`` in ``metadata``.

        Raises:
            KeyError: If the identifier column is missing from metadata.
        """
        return self._get_metadata_value(CHANNEL_ID_COLUMN)

    @property
    def subsystem(self) -> object:
        """Return the engineering subsystem this channel belongs to.

        Returns:
            The value stored under ``SUBSYSTEM_COLUMN`` in ``metadata``.

        Raises:
            KeyError: If the subsystem column is missing from metadata.
        """
        return self._get_metadata_value(SUBSYSTEM_COLUMN)

    @property
    def group(self) -> object:
        """Return the engineering group this channel belongs to.

        Returns:
            The value stored under ``GROUP_COLUMN`` in ``metadata``.

        Raises:
            KeyError: If the group column is missing from metadata.
        """
        return self._get_metadata_value(GROUP_COLUMN)

    @property
    def unit(self) -> object:
        """Return the physical unit of this channel's telemetry values.

        Returns:
            The value stored under ``UNIT_COLUMN`` in ``metadata``.

        Raises:
            KeyError: If the unit column is missing from metadata.
        """
        return self._get_metadata_value(UNIT_COLUMN)

    @property
    def is_target(self) -> bool:
        """Return whether this channel is a modeling target.

        Returns:
            The boolean value stored under ``TARGET_FLAG_COLUMN`` in
            ``metadata``.

        Raises:
            KeyError: If the target flag column is missing from metadata.
        """
        return bool(self._get_metadata_value(TARGET_FLAG_COLUMN))

    @property
    def is_categorical(self) -> bool:
        """Return whether this channel's telemetry is categorical.

        Returns:
            The boolean value stored under ``CATEGORICAL_FLAG_COLUMN`` in
            ``metadata``.

        Raises:
            KeyError: If the categorical flag column is missing from
                metadata.
        """
        return bool(self._get_metadata_value(CATEGORICAL_FLAG_COLUMN))

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Return this channel's telemetry data, loading it if needed.

        If telemetry has already been loaded and cached, the cached
        DataFrame is returned immediately. Otherwise, loading is
        delegated to :class:`ChannelLoader`, which is responsible for
        locating and reading the underlying data (including any archive
        handling). Channel itself never opens files directly.

        Returns:
            The telemetry data for this channel as a pandas DataFrame.

        Raises:
            RuntimeError: If ``ChannelLoader`` fails to load the
                telemetry data.
        """
        if self._data is not None:
            logger.debug("Channel %s: returning cached telemetry.", self.id)
            return self._data

        logger.info("Channel %s: loading telemetry via ChannelLoader.", self.id)
        try:
            loader = ChannelLoader(root=self.mission_root)
            self._data = loader.load(self.metadata)
        except Exception as exc:
            raise RuntimeError(
                f"Channel {self.id}: failed to load telemetry via ChannelLoader."
            ) from exc

        return self._data

    def inspect(self) -> object:
        """Run diagnostics on this channel and return the report.

        Delegates inspection to :class:`ChannelInspector`. If telemetry
        has not yet been loaded, it is loaded first so the inspector has
        data to analyze.

        Returns:
            The inspection report produced by ``ChannelInspector``.

        Raises:
            RuntimeError: If ``ChannelInspector`` fails to produce a
                report.
        """
        data = self.load()

        logger.info("Channel %s: running inspection via ChannelInspector.", self.id)
        try:
            data = self.load()
            inspector = ChannelInspector()
            return inspector.inspect(data)
        except Exception as exc:
            raise RuntimeError(
                f"Channel {self.id}: failed to inspect telemetry via ChannelInspector."
            ) from exc

    def reload(self) -> pd.DataFrame:
        """Clear the cached telemetry and reload it from source.

        Returns:
            The freshly loaded telemetry data as a pandas DataFrame.
        """
        logger.info("Channel %s: reloading telemetry.", self.id)
        self.clear_cache()
        return self.load()

    def clear_cache(self) -> None:
        """Remove any cached telemetry data for this channel.

        After calling this method, the next call to :meth:`load` will
        trigger a fresh load via ``ChannelLoader``.
        """
        if self._data is not None:
            logger.debug("Channel %s: clearing cached telemetry.", self.id)
        self._data = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_metadata_value(self, column: str) -> object:
        """Retrieve a single value from this channel's metadata.

        Args:
            column: The metadata column name to retrieve.

        Returns:
            The value stored under ``column`` in ``metadata``.

        Raises:
            KeyError: If ``column`` is not present in ``metadata``.
        """
        if column not in self.metadata:
            raise KeyError(
                f"Channel metadata is missing required column '{column}'."
            )
        return self.metadata[column]

    # ------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return an unambiguous, human-readable representation.

        Returns:
            A string of the form
            ``Channel(id=<id>, subsystem='<subsystem>')``.
        """
        return f"Channel(\n    id={self.id!r},\n    subsystem={self.subsystem!r}\n)"
