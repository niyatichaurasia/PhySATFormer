"""Mission module for PhySATFormer.

This module defines the Mission dataclass, the root object of the data
layer. A Mission represents a single ESA mission and is responsible only
for organizing and exposing access to mission metadata. It performs no
preprocessing, normalization, synchronization, feature engineering, or
machine learning operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.core.channel import Channel
from src.data.parser import MetadataParser
from src.utils.constants import CHANNEL_ID_COLUMN

logger = logging.getLogger(__name__)


@dataclass
class Mission:
    """Represents a single ESA mission.

    The Mission object is the entry point to the dataset. It lazily loads
    metadata and provides access to channels, anomaly labels, and
    telecommands without loading telemetry data into memory.
    """

    root: Path

    _channels: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _labels: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _anomaly_types: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _telecommands: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the mission root directory."""
        self.root = Path(self.root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Mission directory does not exist: {self.root}"
            )

    def load_metadata(self) -> None:
        """Load mission metadata lazily.

        Metadata is loaded only once and cached for future access.
        """
        if self._channels is not None:
            return

        logger.info("Loading metadata from %s", self.root)

        parser = MetadataParser(self.root)

        self._channels = parser.parse_channels()
        self._labels = parser.parse_labels()
        self._anomaly_types = parser.parse_anomaly_types()
        self._telecommands = parser.parse_telecommands()

        logger.info("Mission metadata loaded successfully.")

    @property
    def channels(self) -> pd.DataFrame:
        """Return channel metadata."""
        self.load_metadata()

        if self._channels is None:
            raise RuntimeError("Channel metadata could not be loaded.")

        return self._channels.copy()

    @property
    def labels(self) -> pd.DataFrame:
        """Return anomaly labels."""
        self.load_metadata()

        if self._labels is None:
            raise RuntimeError("Label metadata could not be loaded.")

        return self._labels.copy()

    @property
    def telecommands(self) -> pd.DataFrame:
        """Return telecommand metadata."""
        self.load_metadata()

        if self._telecommands is None:
            raise RuntimeError("Telecommand metadata could not be loaded.")

        return self._telecommands.copy()

    @property
    def anomaly_types(self) -> pd.DataFrame:
        """Return anomaly type metadata."""
        self.load_metadata()

        if self._anomaly_types is None:
            raise RuntimeError("Anomaly type metadata could not be loaded.")

        return self._anomaly_types.copy()

    @property
    def num_channels(self) -> int:
        """Return the number of telemetry channels."""
        return len(self.channels)

    def get_channel(self, channel_id: int | str) -> Channel:
        """Return a Channel object.

        This method does not load telemetry values. It only constructs
        the corresponding Channel object using metadata.
        """

        # Allow users to request channels using integers.
        # Convert 15 -> "channel_15" to match the ESA dataset.
        if isinstance(channel_id, int):
            channel_id = f"channel_{channel_id}"

        channels = self.channels

        if CHANNEL_ID_COLUMN not in channels.columns:
            raise ValueError(
                f"Expected column '{CHANNEL_ID_COLUMN}' not found in channels metadata."
            )

        channel_metadata = channels.loc[
            channels[CHANNEL_ID_COLUMN] == channel_id
        ]

        if channel_metadata.empty:
            raise ValueError(
                f"Channel '{channel_id}' does not exist."
            )

        return Channel(
            metadata=channel_metadata.iloc[0],
            mission_root=self.root,
        )

    def __repr__(self) -> str:
        """Return a concise string representation."""
        try:
            n_channels = self.num_channels
        except Exception:
            n_channels = "?"

        return (
            f"Mission("
            f"name='{self.root.name}', "
            f"channels={n_channels})"
        )