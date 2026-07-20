"""
src/preprocessing/pipeline.py

TelemetryPipeline: orchestrates the complete preprocessing workflow for
PhySATFormer. This module is a pure orchestrator -- it coordinates existing
preprocessing components (TelemetryAssembler, IntervalLabelGenerator,
WindowGenerator, TelemetryNormalizer, MissionDataset) and MUST NOT
implement any preprocessing algorithms itself (no assembly, labeling,
windowing, or normalization logic lives here).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.core.mission import Mission
from src.preprocessing.assembler import TelemetryAssembler
from src.preprocessing.interval_label_generator import IntervalLabelGenerator
from src.preprocessing.window_generator import WindowGenerator, LazyWindowSet
from src.preprocessing.normalizer import TelemetryNormalizer
from src.preprocessing.dataset import MissionDataset

logger = logging.getLogger(__name__)

# Upper bound on how many *train* windows are materialized at once to fit
# the normalizer. This is an orchestration-level memory/accuracy tradeoff
# (how much data to hand to an existing, unmodified fit() call), not a
# normalization algorithm -- TelemetryNormalizer.fit() itself is untouched
# and still receives a plain (N, window_size, num_channels) array as its
# documented contract requires. See TelemetryPipeline.build() docstring
# for the justification.
DEFAULT_MAX_FIT_WINDOWS = 20_000


class TelemetryPipeline:
    """
    Orchestrates the end-to-end preprocessing pipeline:

        Mission
          -> TelemetryAssembler            (synchronize raw telemetry)
          -> chronological train/val/test split of synchronized telemetry
          -> IntervalLabelGenerator         (dense per-channel labels per split,
                                              computed on the raw, unwindowed
                                              telemetry DataFrame for each split)
          -> WindowGenerator                (LAZY sliding telemetry/label
                                              windows per split -- see NOTE
                                              on memory below)
          -> TelemetryNormalizer            (fit on a bounded sample of
                                              train windows; transform is
                                              applied lazily, per window, by
                                              MissionDataset -- never to a
                                              fully materialized split)
          -> MissionDataset objects

    NOTE on ordering: TelemetryNormalizer operates strictly on already
    -windowed telemetry arrays of shape
    (num_windows, window_size, num_channels) -- it never accepts a
    DataFrame. Normalization therefore happens *after* WindowGenerator,
    not before it. Splitting still happens first (on the raw,
    synchronized DataFrame) so that windows are generated independently
    per split and never straddle a split boundary; this is what actually
    prevents temporal leakage. Fitting the normalizer only on the
    resulting train windows (rather than on the train DataFrame) is
    equivalent in this regard -- the train windows are already fully
    disjoint, in time, from validation/test -- and it is the only order
    compatible with TelemetryNormalizer's documented contract.

    NOTE on memory (production-scale datasets): WindowGenerator windows
    can overlap heavily (stride < window_size), so the total number of
    windows can be orders of magnitude larger than the number of
    underlying telemetry rows. Materializing every split's windows as a
    single dense array does not scale (this is what previously caused an
    ~16.7 GiB allocation on the full Mission1 dataset). This pipeline
    therefore:

        1. Calls `WindowGenerator.generate_lazy()` instead of
           `generate()`, receiving `LazyWindowSet` objects that never
           hold more than one window's worth of extra memory, for every
           split.
        2. Fits `TelemetryNormalizer` on at most `max_fit_windows` train
           windows (a bounded, seeded, evenly-spaced sample), rather than
           the full, potentially-huge train window set. This is a
           standard practice for computing normalization statistics at
           scale: per-channel mean/std estimates converge quickly with
           sample size, and a deterministic evenly-spaced sample avoids
           both the memory cost of full materialization and the bias of
           taking only a contiguous prefix (which could
           under-represent later mission phases). `TelemetryNormalizer`
           itself is not modified -- it still receives a plain dense
           array via its existing `fit()` contract, just a smaller one.
        3. Never eagerly calls `TelemetryNormalizer.transform()` on an
           entire split. Instead, the fitted normalizer is handed to
           `MissionDataset`, which applies `.transform()` to exactly one
           window at a time, inside `__getitem__` -- so no split's full
           windowed telemetry is ever resident in memory simultaneously.
           `TelemetryNormalizer.transform()` is a per-element affine
           operation, so applying it window-by-window is mathematically
           identical to applying it to the whole array at once.

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
        max_fit_windows: int = DEFAULT_MAX_FIT_WINDOWS,
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
        self.max_fit_windows = max_fit_windows

        self._assembler = TelemetryAssembler(
            direction=self.direction
        )
        self._normalizer = TelemetryNormalizer(method=self.normalization_method)
        self._label_generator = IntervalLabelGenerator()
        self._window_generator = WindowGenerator(
            window_size=self.window_size,
            stride=self.stride,
        )

        logger.debug(
            "TelemetryPipeline initialized "
            "(window_size=%d, stride=%d, normalization_method=%s, "
            "train_ratio=%.3f, validation_ratio=%.3f, random_seed=%d, "
            "direction=%s, max_fit_windows=%d)",
            self.window_size,
            self.stride,
            self.normalization_method,
            self.train_ratio,
            self.validation_ratio,
            self.random_seed,
            self.direction,
            self.max_fit_windows,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self, mission: Mission, channel_ids: List[str], max_rows: int | None = None
    ) -> Tuple[MissionDataset, MissionDataset, MissionDataset]:
        """
        Run the full preprocessing pipeline for a given mission.

        Pipeline stages (in order): telemetry assembly (loading +
        synchronization), optional development truncation, chronological
        train/validation/test split, dense channel-wise label generation
        via IntervalLabelGenerator on each split's (unwindowed) telemetry
        DataFrame, LAZY sliding-window generation via
        `WindowGenerator.generate_lazy()` (which produces index-aligned
        `LazyWindowSet` telemetry/label pairs per split without
        materializing a dense windows array), and finally normalization
        via TelemetryNormalizer -- fit on a bounded sample of train
        telemetry windows, and applied lazily per-window inside each
        `MissionDataset` rather than eagerly to a fully materialized
        split. Label windows are never normalized.

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

        # Label generation operates on the raw (unnormalized) per-split
        # telemetry DataFrame. IntervalLabelGenerator only needs the
        # timestamp index and channel/column identity to align interval
        # annotations to rows -- it never reads telemetry values -- so it
        # is indifferent to normalization and must run on a DataFrame,
        # before WindowGenerator, per its documented contract.
        train_labels = self._label_generator.generate(mission, train_df)
        validation_labels = self._label_generator.generate(mission, validation_df)
        test_labels = self._label_generator.generate(mission, test_df)

        self._validate_telemetry_label_alignment(train_df, train_labels, "train")
        self._validate_telemetry_label_alignment(
            validation_df, validation_labels, "validation"
        )
        self._validate_telemetry_label_alignment(test_df, test_labels, "test")

        # WindowGenerator consumes a telemetry DataFrame plus its aligned
        # dense label matrix and emits two index-aligned LazyWindowSet
        # objects (telemetry, labels) instead of materialized 3D numpy
        # arrays. Running this per split (on data that was already split
        # chronologically above) guarantees no window straddles a split
        # boundary, which is what actually prevents temporal leakage.
        # LazyWindowSet preserves window ordering exactly -- see
        # WindowGenerator's equivalence with its own eager generate().
        train_telemetry_windows, train_label_windows = (
            self._window_generator.generate_lazy(train_df, train_labels)
        )
        (
            validation_telemetry_windows,
            validation_label_windows,
        ) = self._window_generator.generate_lazy(validation_df, validation_labels)
        (
            test_telemetry_windows,
            test_label_windows,
        ) = self._window_generator.generate_lazy(test_df, test_labels)

        num_windows = (
            len(train_telemetry_windows)
            + len(validation_telemetry_windows)
            + len(test_telemetry_windows)
        )
        logger.info(
            "Generated %d sliding windows (train=%d validation=%d test=%d), "
            "lazily -- no dense windows array was allocated.",
            num_windows,
            len(train_telemetry_windows),
            len(validation_telemetry_windows),
            len(test_telemetry_windows),
        )

        # Fit on a bounded, deterministic sample of train windows rather
        # than the (potentially huge) full train window set. See the
        # "NOTE on memory" section of this class's docstring for the
        # justification. TelemetryNormalizer.fit() is unmodified: it
        # still receives a plain dense (N, window_size, num_channels)
        # array via its existing contract.
        fit_sample = self._sample_windows_for_fit(
            train_telemetry_windows, self.max_fit_windows, self.random_seed
        )
        self._normalizer.fit(fit_sample)
        del fit_sample

        # Label windows are binary anomaly indicators, not telemetry
        # measurements, and are never normalized. Telemetry normalization
        # is applied lazily, per window, inside MissionDataset -- never
        # eagerly to a fully materialized split -- so no
        # (num_windows, window_size, num_channels) array is ever
        # allocated for telemetry either.
        train_dataset = MissionDataset(
            train_telemetry_windows, train_label_windows, normalizer=self._normalizer
        )
        validation_dataset = MissionDataset(
            validation_telemetry_windows,
            validation_label_windows,
            normalizer=self._normalizer,
        )
        test_dataset = MissionDataset(
            test_telemetry_windows, test_label_windows, normalizer=self._normalizer
        )

        logger.info(
            "Pipeline build complete: total_windows=%d train=%d validation=%d test=%d",
            num_windows,
            len(train_dataset),
            len(validation_dataset),
            len(test_dataset),
        )

        return train_dataset, validation_dataset, test_dataset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_windows_for_fit(
        lazy_windows: LazyWindowSet, max_fit_windows: int, random_seed: int
    ) -> np.ndarray:
        """
        Materialize at most `max_fit_windows` windows from `lazy_windows`
        for the sole purpose of fitting the normalizer, and return them
        as a dense array (TelemetryNormalizer.fit()'s existing contract).

        If `lazy_windows` already has at most `max_fit_windows` windows,
        all of them are used (identical statistics to the pre-existing
        "fit on all train windows" behavior). Otherwise, an evenly-spaced
        (not merely a contiguous prefix) sample of indices is drawn so
        the sample represents the full time range of the split, and the
        sample is deterministic given `random_seed` for reproducibility.

        This bounds the *peak* memory of the fit step to `max_fit_windows
        * window_size * num_channels`, independent of how many train
        windows actually exist.
        """
        n = len(lazy_windows)
        if n <= max_fit_windows:
            return lazy_windows.materialize()

        rng = np.random.default_rng(random_seed)
        # Evenly-spaced base grid across the full split, jittered slightly
        # and sorted, so the sample is both representative of the whole
        # time range and reproducible.
        indices = np.linspace(0, n - 1, num=max_fit_windows, dtype=np.int64)
        indices = np.unique(indices)
        if len(indices) < max_fit_windows:
            extra_needed = max_fit_windows - len(indices)
            remaining_pool = np.setdiff1d(
                np.arange(n, dtype=np.int64), indices, assume_unique=True
            )
            extra = rng.choice(
                remaining_pool, size=min(extra_needed, len(remaining_pool)), replace=False
            )
            indices = np.sort(np.concatenate([indices, extra]))

        logger.info(
            "Fitting normalizer on a sample of %d/%d train windows "
            "(max_fit_windows=%d) to bound peak memory during fit.",
            len(indices),
            n,
            max_fit_windows,
        )

        sample = np.empty(
            (len(indices), lazy_windows.window_size, lazy_windows.n_channels),
            dtype=np.float64,
        )
        for out_i, window_i in enumerate(indices):
            sample[out_i] = lazy_windows[int(window_i)]
        return sample

    def _split_synchronized_telemetry(self, synchronized_df):
        """
        Split the synchronized telemetry DataFrame chronologically into
        train/validation/test segments using train_ratio and
        validation_ratio, BEFORE normalization, label generation, and
        window generation. Windows are generated independently per
        segment so that no window straddles a split boundary, preventing
        temporal data leakage between splits.
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
    def _validate_telemetry_label_alignment(
        telemetry_df: pd.DataFrame, labels: np.ndarray, split_name: str
    ) -> None:
        """
        Validate that dense labels produced by IntervalLabelGenerator are
        row-aligned with the normalized telemetry split they were
        generated from, before handing both off to WindowGenerator.

        This is a defensive orchestration-level check: it does not
        implement any labeling or windowing logic itself, it only
        guards against silently feeding misaligned telemetry/label pairs
        into WindowGenerator, which would otherwise corrupt every
        resulting window for the affected split.

        Args:
            telemetry_df: Raw (unwindowed, unnormalized) telemetry for a
                single split.
            labels: Dense channel-wise labels generated for the same
                split, expected shape (num_timestamps, num_channels).
            split_name: Human-readable split identifier used in error
                messages (e.g. "train", "validation", "test").

        Raises:
            ValueError: If the number of label rows does not match the
                number of telemetry rows for this split.
        """
        num_telemetry_rows = len(telemetry_df)
        num_label_rows = labels.shape[0] if labels.ndim > 0 else 0

        if num_telemetry_rows != num_label_rows:
            raise ValueError(
                f"Telemetry/label row misalignment in '{split_name}' split: "
                f"telemetry has {num_telemetry_rows} row(s) but "
                f"IntervalLabelGenerator produced {num_label_rows} label "
                f"row(s). Refusing to generate windows from misaligned "
                f"telemetry and labels."
            )

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
