"""Metadata CSV parser for the PhySATFormer project.

This module defines :class:`MetadataParser`, which is responsible
exclusively for reading and validating ESA metadata CSV files
(channels, labels, anomaly types, and telecommands). It performs no
preprocessing, normalization, feature engineering, or value
modification of any kind. Its sole responsibility is I/O and schema
validation.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils.constants import (
    ANOMALY_TYPES_METADATA_FILE,
    CHANNELS_METADATA_FILE,
    CHANNELS_REQUIRED_COLUMNS,
    LABELS_METADATA_FILE,
    LABELS_REQUIRED_COLUMNS,
    TELECOMMANDS_METADATA_FILE,
)

logger = logging.getLogger(__name__)


class MetadataParser:
    """Reads and validates ESA metadata CSV files.

    This class is strictly limited to reading metadata CSV files from
    disk and validating that they contain the expected columns. It
    must never perform preprocessing, normalization, feature
    engineering, machine learning, or any modification of the
    underlying data values.

    Attributes:
        root (Path): Root directory containing the metadata CSV
            files.
    """

    def __init__(self, root: Path) -> None:
        """Initializes the MetadataParser.

        Args:
            root: Root directory containing the metadata CSV files.

        Raises:
            FileNotFoundError: If `root` does not exist or is not a
                directory.
        """
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Metadata root directory does not exist: {root}")
        if not root.is_dir():
            raise FileNotFoundError(f"Metadata root path is not a directory: {root}")

        self.root: Path = root
        logger.debug("Initialized MetadataParser with root: %s", self.root)

    def parse_channels(self) -> pd.DataFrame:
        """Reads and validates the channels metadata CSV file.

        Returns:
            A pandas DataFrame containing the raw channels metadata.

        Raises:
            FileNotFoundError: If the channels CSV file does not
                exist.
            ValueError: If the CSV file is malformed or missing
                required columns.
        """
        df = self._read_csv(CHANNELS_METADATA_FILE)
        self._validate_columns(df, CHANNELS_REQUIRED_COLUMNS)
        logger.info(
            "Parsed channels metadata: %d rows, %d columns",
            df.shape[0],
            df.shape[1],
        )
        return df

    def parse_labels(self) -> pd.DataFrame:
        """Reads and validates the labels metadata CSV file.

        Returns:
            A pandas DataFrame containing the raw labels metadata.

        Raises:
            FileNotFoundError: If the labels CSV file does not exist.
            ValueError: If the CSV file is malformed or missing
                required columns.
        """
        df = self._read_csv(LABELS_METADATA_FILE)
        self._validate_columns(df, LABELS_REQUIRED_COLUMNS)
        logger.info(
            "Parsed labels metadata: %d rows, %d columns",
            df.shape[0],
            df.shape[1],
        )
        return df

    def parse_anomaly_types(self) -> pd.DataFrame:
        """Reads the anomaly types metadata CSV file.

        Returns:
            A pandas DataFrame containing the raw anomaly types
            metadata.

        Raises:
            FileNotFoundError: If the anomaly types CSV file does not
                exist.
            ValueError: If the CSV file is malformed.
        """
        df = self._read_csv(ANOMALY_TYPES_METADATA_FILE)
        logger.info(
            "Parsed anomaly types metadata: %d rows, %d columns",
            df.shape[0],
            df.shape[1],
        )
        return df

    def parse_telecommands(self) -> pd.DataFrame:
        """Reads the telecommands metadata CSV file.

        Returns:
            A pandas DataFrame containing the raw telecommands
            metadata.

        Raises:
            FileNotFoundError: If the telecommands CSV file does not
                exist.
            ValueError: If the CSV file is malformed.
        """
        df = self._read_csv(TELECOMMANDS_METADATA_FILE)
        logger.info(
            "Parsed telecommands metadata: %d rows, %d columns",
            df.shape[0],
            df.shape[1],
        )
        return df

    def _read_csv(self, filename: str) -> pd.DataFrame:
        """Safely reads a CSV file from the root directory.

        Args:
            filename: Name of the CSV file to read, relative to
                `self.root`.

        Returns:
            A pandas DataFrame containing the raw CSV contents.

        Raises:
            FileNotFoundError: If the file does not exist at the
                expected path.
            ValueError: If the file is empty, malformed, or cannot be
                parsed as a valid CSV.
        """
        file_path = self.root / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required metadata file not found: {file_path}"
            )
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Expected a file but found something else at: {file_path}"
            )

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Metadata CSV file is empty: {file_path}") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Metadata CSV file is malformed and could not be parsed: "
                f"{file_path}"
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Metadata CSV file could not be read: {file_path}"
            ) from exc

        if df.empty:
            raise ValueError(f"Metadata CSV file contains no data rows: {file_path}")

        logger.debug("Read CSV file: %s (%d rows)", file_path, df.shape[0])
        return df

    def _validate_columns(
        self, df: pd.DataFrame, expected_columns: list[str]
    ) -> None:
        """Validates that a DataFrame contains all required columns.

        Args:
            df: DataFrame to validate.
            expected_columns: List of column names that must be
                present in `df`.

        Raises:
            ValueError: If one or more expected columns are missing
                from `df`.
        """
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                "Metadata CSV is missing required columns: "
                f"{missing_columns}. Found columns: {list(df.columns)}"
            )