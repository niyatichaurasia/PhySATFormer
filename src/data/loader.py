"""Loader for ESA Mission telemetry channel archives.

This module defines :class:`ChannelLoader`, whose sole responsibility is
to locate and load raw telemetry data from the ESA Mission dataset's
per-channel ZIP archives. It performs no preprocessing, normalization,
synchronization, timestamp manipulation, dtype inference, or column
renaming of any kind. It returns telemetry data exactly as stored by
ESA, wrapped in a pandas DataFrame.
"""

from __future__ import annotations

import logging
import pickle
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.constants import (
    CHANNEL_ID_COLUMN,
    CHANNELS_DIRECTORY,
)

logger = logging.getLogger(__name__)


class ChannelLoader:
    """Loads raw telemetry channel data from the ESA Mission dataset.

    This class is strictly a data-access component. It opens the ZIP
    archive corresponding to a given channel, locates the pickle file
    contained within it, and deserializes it into a pandas DataFrame.
    It performs no preprocessing, normalization, synchronization,
    timestamp modification, or dtype inference.

    Attributes:
        root: Root directory of the ESA Mission dataset. The loader
            expects channel archives to live under
            ``root / "channels" / "<channel_name>.zip"``.
    """

    def __init__(self, root: Path) -> None:
        """Initializes the ChannelLoader.

        Args:
            root: Root directory of the ESA Mission dataset.
        """
        self.root = Path(root)

    def load(self, metadata: pd.Series) -> pd.DataFrame:
        """Loads raw telemetry for a single channel described by metadata.

        Args:
            metadata: A row from ``channels.csv`` (as a pandas Series).
                Must contain a ``Channel`` field identifying which
                telemetry archive to open (e.g. ``"channel_15"``).

        Returns:
            The raw telemetry data exactly as stored by ESA, as a
            pandas DataFrame.

        Raises:
            KeyError: If the ``Channel`` field is missing from
                ``metadata``.
            FileNotFoundError: If the corresponding archive does not
                exist on disk.
            ValueError: If the ZIP archive is corrupted, contains no
                pickle file, or the pickle does not deserialize to a
                pandas DataFrame.
        """
        channel_name = self._get_channel_name(metadata)
        archive_path = self._resolve_archive_path(channel_name)

        logger.debug("Loading channel '%s' from %s", channel_name, archive_path)

        pickle_bytes = self._read_pickle_bytes_from_archive(archive_path)
        obj = self._deserialize_pickle(pickle_bytes, archive_path)

        if not isinstance(obj, pd.DataFrame):
            raise ValueError(
                f"Object loaded from pickle in '{archive_path}' is not a "
                f"pandas DataFrame (got {type(obj).__name__})."
            )

        logger.info(
            "Loaded telemetry for %s (%d rows)",
            channel_name,
            len(obj),
        )

        return obj

    @staticmethod
    def _get_channel_name(metadata: pd.Series) -> str:
        """Extracts the channel name from a metadata row.

        Args:
            metadata: A row from ``channels.csv``.

        Returns:
            The channel identifier (e.g. ``"channel_15"``).

        Raises:
            KeyError: If the ``Channel`` field is missing from
                ``metadata``.
        """
        try:
            channel_name = metadata[CHANNEL_ID_COLUMN]
        except KeyError as exc:
            raise KeyError(
                "Metadata row is missing required 'Channel' field."
            ) from exc

        return str(channel_name)

    def _resolve_archive_path(self, channel_name: str) -> Path:
        """Resolves the path to a channel's ZIP archive.

        Args:
            channel_name: The channel identifier (e.g. ``"channel_15"``).

        Returns:
            Path to the expected ZIP archive.

        Raises:
            FileNotFoundError: If the archive does not exist.
        """
        archive_path = self.root / CHANNELS_DIRECTORY / f"{channel_name}.zip"

        if not archive_path.exists():
            raise FileNotFoundError(
                f"Telemetry archive not found for channel '{channel_name}': "
                f"{archive_path}"
            )

        return archive_path

    @staticmethod
    def _read_pickle_bytes_from_archive(archive_path: Path) -> bytes:
        """Opens a ZIP archive and reads the bytes of its pickle member.

        Args:
            archive_path: Path to the ZIP archive.

        Returns:
            Raw bytes of the pickle file found inside the archive.

        Raises:
            ValueError: If the ZIP archive is corrupted or contains no
                pickle file.
        """
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                pickle_member = ChannelLoader._find_pickle_member(
                    archive, archive_path
                )
                with archive.open(pickle_member) as fh:
                    return fh.read()
        except zipfile.BadZipFile as exc:
            raise ValueError(
                f"Archive '{archive_path}' is not a valid ZIP file or is "
                f"corrupted."
            ) from exc

    @staticmethod
    @staticmethod
    def _find_pickle_member(
        archive: zipfile.ZipFile,
        archive_path: Path,
    ) -> str:
        """Locate the telemetry file inside an ESA channel archive.

        ESA channel archives contain exactly one telemetry file
        (typically with no file extension). Hidden directories are ignored.
        """

        candidates = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
        ]

        if not candidates:
            raise ValueError(
                f"No telemetry file found inside archive '{archive_path}'."
            )

        if len(candidates) > 1:
            logger.warning(
                "Multiple files found in '%s'; using the first: %s",
                archive_path,
                candidates[0],
            )

        return candidates[0]

    @staticmethod
    def _deserialize_pickle(pickle_bytes: bytes, archive_path: Path) -> object:
        """Deserializes raw pickle bytes.

        Args:
            pickle_bytes: Raw bytes read from the pickle member.
            archive_path: Path to the archive (used for error messages).

        Returns:
            The deserialized Python object.

        Raises:
            ValueError: If the bytes cannot be unpickled.
        """
        try:
            return pickle.loads(pickle_bytes)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
            raise ValueError(
                f"Failed to unpickle telemetry data from archive "
                f"'{archive_path}': {exc}"
            ) from exc