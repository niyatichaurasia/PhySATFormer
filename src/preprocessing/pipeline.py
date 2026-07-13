"""
src/preprocessing/pipeline.py

TelemetryPipeline: orchestrates the complete preprocessing workflow for
PhySATFormer. This module is a pure orchestrator -- it coordinates existing
preprocessing components (TelemetryAssembler, WindowGenerator,
TelemetryNormalizer, MissionDataset) and MUST NOT implement any
preprocessing algorithms itself (no assembly, windowing, or normalization
logic lives here).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from src.core.mission import Mission
from src.preprocessing.assembler import TelemetryAssembler
from src.preprocessing.window_generator import WindowGenerator
from src.preprocessing.normalizer import TelemetryNormalizer
from src.preprocessing.dataset import MissionDataset

logger = logging.getLogger(__name__)


class TelemetryPipeline:
    """
    Orchestrates the end-to-end preprocessing pipeline:

        Mission
          -> TelemetryAssembler            (synchronize raw telemetry)
          -> chronological train/val/test split of synchronized telemetry
          -> WindowGenerator                (produce sliding windows per split)
          -> TelemetryNormalizer            (fit on train, transform all)
          -> MissionDataset objects

    This class contains no preprocessing algorithms of its own; it only
    instantiates and coordinates the existing project components.
    """

    def __init__(
        self,
        window_size: int,
        stride: int,
        normalization_method: str,
        train_ratio: float,
        validation_ratio: float,
        random_seed: int,
        direction: str,
    ) -> None:
        self._validate_constructor_args(
            window_size=window_size,
            stride=stride,
            normalization_method=normalization_method,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            random_seed=random_seed,
        )

        self.window_size = window_size
        self.stride = stride
        self.normalization_method = normalization_method
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.random_seed = random_seed
        self.direction = direction

        self._assembler = TelemetryAssembler(
            direction=self.direction
        )
        self._window_generator = WindowGenerator(
            window_size=self.window_size,
            stride=self.stride,
        )
        self._normalizer = TelemetryNormalizer(method=self.normalization_method)

        logger.debug(
            "TelemetryPipeline initialized "
            "(window_size=%d, stride=%d, normalization_method=%s, "
            "train_ratio=%.3f, validation_ratio=%.3f, random_seed=%d, "
            "direction=%s)",
            self.window_size,
            self.stride,
            self.normalization_method,
            self.train_ratio,
            self.validation_ratio,
            self.random_seed,
            self.direction,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self, mission: Mission, channel_ids: List[str], max_rows: int | None = None
    ) -> Tuple[MissionDataset, MissionDataset, MissionDataset]:
        """
        Run the full preprocessing pipeline for a given mission.

        Args:
            mission: Mission object to be preprocessed.
            channel_ids: Telemetry channel identifiers to assemble.
            max_rows: Optional cap on the number of synchronized telemetry
                rows to process, applied immediately after assembly and
                before the chronological train/validation/test split. This
                is intended ONLY for development and debugging on
                resource-constrained machines (e.g. quickly iterating on a
                small slice of data). Leaving it as None (the default)
                processes the full dataset and preserves existing
                production behavior. Must be a positive int if provided.

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset).
        """
        logger.info("Building preprocessing pipeline for mission: %s", mission)

        synchronized_df = self._assembler.assemble(mission, channel_ids)
        logger.debug(
            "Telemetry assembled: %d synchronized samples", len(synchronized_df)
        )

        if max_rows is not None:
            if max_rows <= 0:
                raise ValueError(
                    f"max_rows must be positive, got {max_rows}."
                )
            synchronized_df = synchronized_df.iloc[:max_rows]
            logger.debug(
                "Synchronized telemetry truncated to max_rows=%d "
                "(development/debugging mode)",
                max_rows,
            )

        train_df, validation_df, test_df = self._split_synchronized_telemetry(
            synchronized_df
        )

        train_windows = self._window_generator.generate(train_df)
        validation_windows = self._window_generator.generate(validation_df)
        test_windows = self._window_generator.generate(test_df)

        num_windows = len(train_windows) + len(validation_windows) + len(test_windows)
        logger.info(
            "Generated %d sliding windows (train=%d validation=%d test=%d)",
            num_windows,
            len(train_windows),
            len(validation_windows),
            len(test_windows),
        )

        self._normalizer.fit(train_windows)

        train_windows = self._normalizer.transform(train_windows)
        validation_windows = self._normalizer.transform(validation_windows)
        test_windows = self._normalizer.transform(test_windows)

        train_dataset = MissionDataset(train_windows)
        validation_dataset = MissionDataset(validation_windows)
        test_dataset = MissionDataset(test_windows)

        logger.info(
            "Pipeline build complete: total_windows=%d train=%d validation=%d test=%d",
            num_windows,
            len(train_windows),
            len(validation_windows),
            len(test_windows),
        )

        return train_dataset, validation_dataset, test_dataset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _split_synchronized_telemetry(self, synchronized_df):
        """
        Split the synchronized telemetry DataFrame chronologically into
        train/validation/test segments using train_ratio and
        validation_ratio, BEFORE window generation. Windows are then
        generated independently per segment so that no window straddles a
        split boundary, preventing temporal data leakage between splits.
        """
        num_samples = len(synchronized_df)
        if num_samples == 0:
            raise ValueError(
                "TelemetryAssembler produced 0 synchronized samples; cannot "
                "split empty telemetry into train/validation/test."
            )

        train_end = int(np.floor(self.train_ratio * num_samples))
        validation_end = train_end + int(np.floor(self.validation_ratio * num_samples))

        train_df = synchronized_df.iloc[:train_end]
        validation_df = synchronized_df.iloc[train_end:validation_end]
        test_df = synchronized_df.iloc[validation_end:]

        if len(train_df) == 0:
            raise ValueError(
                "Computed an empty training split; increase train_ratio or "
                "the number of synchronized samples."
            )
        if len(test_df) == 0:
            raise ValueError(
                "Computed an empty test split; decrease train_ratio/"
                "validation_ratio or increase the number of synchronized samples."
            )

        return train_df, validation_df, test_df

    @staticmethod
    def _validate_constructor_args(
        window_size: int,
        stride: int,
        normalization_method: str,
        train_ratio: float,
        validation_ratio: float,
        random_seed: int,
    ) -> None:
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size must be a positive int, got {window_size!r}")

        if not isinstance(stride, int) or stride <= 0:
            raise ValueError(f"stride must be a positive int, got {stride!r}")

        if not isinstance(normalization_method, str) or not normalization_method:
            raise ValueError(
                f"normalization_method must be a non-empty str, got "
                f"{normalization_method!r}"
            )

        if not isinstance(train_ratio, (int, float)) or not (0.0 < train_ratio < 1.0):
            raise ValueError(
                f"train_ratio must be a float in (0, 1), got {train_ratio!r}"
            )

        if not isinstance(validation_ratio, (int, float)) or not (
            0.0 <= validation_ratio < 1.0
        ):
            raise ValueError(
                f"validation_ratio must be a float in [0, 1), got "
                f"{validation_ratio!r}"
            )

        if train_ratio + validation_ratio >= 1.0:
            raise ValueError(
                "train_ratio + validation_ratio must be < 1.0 so that a "
                f"non-empty test set remains (got train_ratio={train_ratio!r}, "
                f"validation_ratio={validation_ratio!r})"
            )

        if not isinstance(random_seed, int):
            raise ValueError(f"random_seed must be an int, got {random_seed!r}")