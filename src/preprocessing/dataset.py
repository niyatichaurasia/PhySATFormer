"""
dataset.py

PhySATFormer - src/preprocessing/dataset.py

Defines MissionDataset, a thin PyTorch Dataset wrapper around
pre-computed telemetry windows (and optional labels).

Responsibilities of MissionDataset are strictly limited to:
    - validating input shapes/types
    - exposing __len__ and __getitem__
    - lazily converting numpy arrays to torch tensors per-item

MissionDataset MUST NOT:
    - synchronize telemetry streams
    - normalize / standardize data
    - construct windows from raw sequences
    - split data into train/val/test
    - perform any modeling or learning logic

All such responsibilities belong to other modules in the
preprocessing pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MissionDataset(Dataset):
    """PyTorch Dataset exposing pre-computed telemetry windows.

    Parameters
    ----------
    windows : numpy.ndarray
        Array of shape (number_of_windows, window_size, number_of_channels)
        containing telemetry windows. Must already be synchronized,
        normalized, and windowed by upstream preprocessing steps.
    labels : numpy.ndarray, optional
        Array of shape (number_of_windows, window_size, number_of_channels)
        containing per-window label windows, aligned with ``windows``.
        If omitted, __getitem__ returns only the telemetry window tensor.

    Notes
    -----
    - Input arrays are never copied or modified during construction.
    - Conversion to torch.Tensor happens lazily, per-item, inside
      __getitem__. Telemetry and label windows are converted to
      torch.float32.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self._validate_windows(windows)
        if labels is not None:
            self._validate_labels(windows, labels)

        self.windows = windows
        self.labels = labels

        logger.info(
            "MissionDataset created: num_windows=%d, window_size=%d, "
            "num_channels=%d, has_labels=%s",
            self.windows.shape[0],
            self.windows.shape[1],
            self.windows.shape[2],
            self.labels is not None,
        )

    @staticmethod
    def _validate_windows(windows: np.ndarray) -> None:
        if not isinstance(windows, np.ndarray):
            raise TypeError(
                f"'windows' must be a numpy.ndarray, got {type(windows).__name__}."
            )
        if windows.ndim != 3:
            raise ValueError(
                "'windows' must have exactly 3 dimensions "
                "(number_of_windows, window_size, number_of_channels), "
                f"got shape {windows.shape} with {windows.ndim} dimensions."
            )
        if windows.shape[0] == 0:
            raise ValueError("'windows' must contain at least one window.")

    @staticmethod
    def _validate_labels(windows: np.ndarray, labels: np.ndarray) -> None:
        if not isinstance(labels, np.ndarray):
            raise TypeError(
                f"'labels' must be a numpy.ndarray, got {type(labels).__name__}."
            )
        if labels.ndim != 3:
            raise ValueError(
                "'labels' must have exactly 3 dimensions "
                "(number_of_windows, window_size, number_of_channels), "
                f"got shape {labels.shape} with {labels.ndim} dimensions."
            )
        if labels.shape[0] != windows.shape[0]:
            raise ValueError(
                "'labels' number of windows must match 'windows': "
                f"got labels.shape[0]={labels.shape[0]} but "
                f"windows.shape[0]={windows.shape[0]}."
            )
        if labels.shape[1] != windows.shape[1]:
            raise ValueError(
                "'labels' window size must match 'windows': "
                f"got labels.shape[1]={labels.shape[1]} but "
                f"windows.shape[1]={windows.shape[1]}."
            )
        if labels.shape[2] != windows.shape[2]:
            raise ValueError(
                "'labels' number of channels must match 'windows': "
                f"got labels.shape[2]={labels.shape[2]} but "
                f"windows.shape[2]={windows.shape[2]}."
            )

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[int, torch.Tensor], Tuple[int, torch.Tensor, torch.Tensor]]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError(
                f"Index must be an int, got {type(index).__name__}."
            )
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}."
            )

        window = torch.from_numpy(self.windows[index]).to(torch.float32)

        if self.labels is None:
            return index, window

        label = torch.from_numpy(self.labels[index]).to(torch.float32)
        return index, window, label