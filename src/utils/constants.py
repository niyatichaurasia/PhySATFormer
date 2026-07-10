"""Project-wide constants for PhySATFormer.

This module centralizes dataset schema, directory names,
default configuration values, and project-wide constants.
"""

# ==========================================================
# Dataset Directories
# ==========================================================

CHANNELS_DIRECTORY = "channels"
TELECOMMANDS_DIRECTORY = "telecommands"

# ==========================================================
# Metadata Files
# ==========================================================

CHANNELS_METADATA_FILE = "channels.csv"
LABELS_METADATA_FILE = "labels.csv"
ANOMALY_TYPES_METADATA_FILE = "anomaly_types.csv"
TELECOMMANDS_METADATA_FILE = "telecommands.csv"

# ==========================================================
# Channel Metadata Columns
# (Verified against ESA Mission1 dataset)
# ==========================================================

CHANNEL_ID_COLUMN = "Channel"
SUBSYSTEM_COLUMN = "Subsystem"
UNIT_COLUMN = "Physical Unit"
GROUP_COLUMN = "Group"
TARGET_FLAG_COLUMN = "Target"
CATEGORICAL_FLAG_COLUMN = "Categorical"



# ==========================================================
# Label Metadata Columns
# ==========================================================

LABEL_ID_COLUMN = "ID"
LABEL_CHANNEL_COLUMN = "Channel"
START_TIME_COLUMN = "StartTime"
END_TIME_COLUMN = "EndTime"

# ==========================================================
# Metadata Values
# ==========================================================

YES = "YES"
NO = "NO"

TARGET_TRUE = YES
TARGET_FALSE = NO

CATEGORICAL_TRUE = YES
CATEGORICAL_FALSE = NO


# ==========================================================
# Default Configuration
# ==========================================================

DEFAULT_RANDOM_SEED = 42
DEFAULT_WINDOW_SIZE = 60
DEFAULT_WINDOW_STRIDE = 1

# ==========================================================
# Logging
# ==========================================================

LOGGER_NAME = "physatformer"

#===========================================================
# Required Columns
#===========================================================

CHANNELS_REQUIRED_COLUMNS = [
    CHANNEL_ID_COLUMN,
    SUBSYSTEM_COLUMN,
    UNIT_COLUMN,
    GROUP_COLUMN,
    TARGET_FLAG_COLUMN,
    CATEGORICAL_FLAG_COLUMN,
]

LABELS_REQUIRED_COLUMNS = [
    LABEL_ID_COLUMN,
    LABEL_CHANNEL_COLUMN,
    START_TIME_COLUMN,
    END_TIME_COLUMN,
]

CHANNEL_FILE_PREFIX = "channel_"
TELECOMMAND_FILE_PREFIX = "telecommand_"

CHANNEL_FILE_EXTENSION = ".zip"

#=======================================================
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

MISSION_ROOT = Path(
    os.getenv(
        "PHYSATFORMER_DATASET",
        "data/raw/Mission1",
    )
)