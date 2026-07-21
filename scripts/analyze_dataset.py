"""
scripts/analyze_dataset.py
===========================

Pre-training dataset analysis for PhySATFormer.

This is a read-only, standalone analysis script. It builds the exact same
Mission -> TelemetryPipeline -> (train, validation, test) MissionDataset
objects that ``train.py`` builds for real training, by importing and
calling the *existing, unmodified* orchestration functions from
``train.py`` (``load_configs``, ``setup_logging``, ``set_random_seed``,
``build_mission``, ``build_pipeline``, ``build_datasets``). No
preprocessing algorithm (assembly, labeling, windowing, normalization) is
reimplemented here -- this script only reads what those components
already produce.

Labels are never assumed. They are read directly from each
``MissionDataset``'s ``__getitem__`` output -- i.e. exactly the
``(telemetry, label)`` tensor pairs that ``Trainer`` receives during
training -- via a plain ``torch.utils.data.DataLoader`` used only for
efficient iteration (standard library usage, not a project component).

What this script deliberately does NOT do:
    * modify Trainer, Model, Pipeline, Dataset, or any other project file
    * duplicate assembly / labeling / windowing / normalization logic
    * train, validate gradients, or touch the model/optimizer/trainer at all

Outputs (all created automatically if missing):
    outputs/figures/class_distribution.png
    outputs/figures/split_distribution.png
    outputs/figures/telemetry_example.png
    outputs/figures/fault_window_examples.png
    outputs/figures/channel_correlation_heatmap.png   (optional, best-effort)
    outputs/metrics/dataset_summary.json
    outputs/metrics/dataset_statistics.csv
    outputs/metrics/split_statistics.csv
    outputs/reports/dataset_report.txt

Usage:
    python scripts/analyze_dataset.py
    python scripts/analyze_dataset.py --dataset-config configs/dataset.yaml
    python scripts/analyze_dataset.py --max-search-windows 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Make the project root importable regardless of the current working
# directory this script is invoked from (mirrors scripts/test_one_epoch.py).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Reuse the real orchestration functions from train.py. Nothing about
# pipeline construction is reimplemented here -- every step below calls
# into the existing, unmodified code, exactly as train.py itself does.
# ---------------------------------------------------------------------------
from train import (  # noqa: E402
    build_datasets,
    build_mission,
    build_pipeline,
    load_configs,
    set_random_seed,
    setup_logging,
)
from src.utils.constants import CHANNEL_ID_COLUMN  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_SEARCH_WINDOWS = 20_000  # cap on how many windows to scan when
# hunting for representative normal/anomaly example windows to plot -- this
# bounds runtime on very large datasets without changing any computed
# dataset-wide statistic (those are always computed over every window).
DEFAULT_CORRELATION_SAMPLE_WINDOWS = 200

FIGURE_DPI = 150
NORMAL_COLOR = "#4C72B0"
ANOMALY_COLOR = "#C44E52"

SPLIT_NAMES: Tuple[str, str, str] = ("train", "validation", "test")

logger = logging.getLogger("analyze_dataset")


# =============================================================================
# Data containers
# =============================================================================


class SplitLabelSummary:
    """Window-level label summary for a single dataset split.

    Attributes:
        split_name: Name of the split ("train", "validation", "test").
        num_windows: Total number of windows in this split.
        num_anomaly_windows: Number of windows containing at least one
            anomalous (channel, timestep) label.
        num_normal_windows: Number of windows containing no anomalous
            labels at all.
        channel_anomaly_window_counts: Per-channel count of how many
            windows in this split contain at least one anomalous
            timestep for that channel. Shape (num_channels,).
        window_is_anomaly: Boolean array, shape (num_windows,), aligned
            with dataset iteration order.
    """

    def __init__(
        self,
        split_name: str,
        num_windows: int,
        window_is_anomaly: np.ndarray,
        channel_anomaly_window_counts: np.ndarray,
    ) -> None:
        self.split_name = split_name
        self.num_windows = num_windows
        self.window_is_anomaly = window_is_anomaly
        self.channel_anomaly_window_counts = channel_anomaly_window_counts

    @property
    def num_anomaly_windows(self) -> int:
        return int(self.window_is_anomaly.sum())

    @property
    def num_normal_windows(self) -> int:
        return self.num_windows - self.num_anomaly_windows

    @property
    def anomaly_percentage(self) -> float:
        if self.num_windows == 0:
            return 0.0
        return 100.0 * self.num_anomaly_windows / self.num_windows

    @property
    def normal_percentage(self) -> float:
        if self.num_windows == 0:
            return 0.0
        return 100.0 * self.num_normal_windows / self.num_windows

    @property
    def ratio_normal_to_anomaly(self) -> str:
        if self.num_anomaly_windows == 0:
            return f"{self.num_normal_windows}:0 (no anomaly windows)"
        ratio = self.num_normal_windows / self.num_anomaly_windows
        return f"{ratio:.2f}:1"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split_name,
            "total_windows": self.num_windows,
            "normal_windows": self.num_normal_windows,
            "anomaly_windows": self.num_anomaly_windows,
            "normal_percentage": round(self.normal_percentage, 4),
            "anomaly_percentage": round(self.anomaly_percentage, 4),
            "ratio_normal_to_anomaly": self.ratio_normal_to_anomaly,
        }


# =============================================================================
# Stage 1: configuration, logging, pipeline construction (all reused)
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments containing config paths and analysis knobs.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the PhySATFormer dataset before training, using the "
            "existing, unmodified preprocessing pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "train.yaml",
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "model.yaml",
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "dataset.yaml",
        help="Path to the dataset configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs",
        help="Root directory under which figures/metrics/reports are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used only for iterating datasets during analysis.",
    )
    parser.add_argument(
        "--max-search-windows",
        type=int,
        default=DEFAULT_MAX_SEARCH_WINDOWS,
        help=(
            "Max windows to scan (from the start of the train split) when "
            "searching for representative normal/anomaly example windows "
            "to plot. Does not affect any computed statistic."
        ),
    )
    return parser.parse_args()


def build_pipeline_outputs(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Any, Any, Dataset, Dataset, Dataset]:
    """Load configs and build Mission / TelemetryPipeline / datasets.

    Reuses ``train.py``'s own orchestration functions exactly as
    ``train.py`` and ``scripts/test_one_epoch.py`` do. No preprocessing
    logic is reimplemented.

    Returns:
        Tuple of (train_cfg, model_cfg, dataset_cfg, mission, pipeline,
        train_dataset, validation_dataset, test_dataset).
    """
    train_cfg, model_cfg, dataset_cfg = load_configs(
        train_config=args.train_config,
        model_config=args.model_config,
        dataset_config=args.dataset_config,
    )
    logger.info("Configs loaded.")

    experiment_cfg = train_cfg.get("experiment", {}) or {}
    seed = experiment_cfg.get("seed", train_cfg.get("random_seed", 42))
    deterministic = experiment_cfg.get("deterministic", True)
    set_random_seed(seed=seed, deterministic=deterministic)
    logger.info("Random seed set to %s (deterministic=%s).", seed, deterministic)

    mission = build_mission(dataset_cfg, logger)
    pipeline = build_pipeline(dataset_cfg, logger)
    logger.info("Mission and TelemetryPipeline constructed.")

    train_dataset, validation_dataset, test_dataset = build_datasets(
        pipeline=pipeline,
        mission=mission,
        dataset_cfg=dataset_cfg,
        logger=logger,
    )
    logger.info(
        "Datasets built: train=%d validation=%d test=%d",
        len(train_dataset),
        len(validation_dataset),
        len(test_dataset),
    )

    return (
        train_cfg,
        model_cfg,
        dataset_cfg,
        mission,
        pipeline,
        train_dataset,
        validation_dataset,
        test_dataset,
    )


# =============================================================================
# Stage 2: window-level label analysis (read, never assumed)
# =============================================================================


def analyze_split_labels(
    dataset: Dataset,
    split_name: str,
    num_channels: int,
    batch_size: int,
) -> SplitLabelSummary:
    """Compute window-level normal/anomaly counts for one split.

    Labels are read exactly as ``Trainer`` receives them: by iterating
    the ``MissionDataset`` through a plain ``torch.utils.data.DataLoader``
    (no shuffling, so results are deterministic) and inspecting the
    label tensor returned by ``__getitem__``. A window is considered
    "anomaly" if any (timestep, channel) label in that window is 1;
    otherwise it is "normal". No assumption about label semantics is
    made beyond reading the values already produced by the pipeline.

    Args:
        dataset: A ``MissionDataset`` (or any ``Dataset`` with the same
            ``(telemetry, label)`` contract) for one split.
        split_name: Human-readable split name, used only for logging.
        num_channels: Number of telemetry channels, used to size the
            per-channel anomaly counters.
        batch_size: Batch size for the analysis DataLoader.

    Returns:
        A ``SplitLabelSummary`` for this split.
    """
    num_windows = len(dataset)
    if num_windows == 0:
        logger.warning("Split '%s' has zero windows; skipping.", split_name)
        return SplitLabelSummary(
            split_name, 0, np.array([], dtype=bool), np.zeros(num_channels, dtype=np.int64)
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    window_flags: List[np.ndarray] = []
    channel_counts = np.zeros(num_channels, dtype=np.int64)

    for telemetry_batch, label_batch in loader:
        del telemetry_batch  # not needed for label analysis
        labels_np = label_batch.numpy()  # (B, window_size, num_channels)
        per_channel_any = labels_np.sum(axis=1) > 0  # (B, num_channels)
        window_any = per_channel_any.any(axis=1)  # (B,)
        window_flags.append(window_any)
        channel_counts += per_channel_any.sum(axis=0).astype(np.int64)

    window_is_anomaly = np.concatenate(window_flags)
    logger.info(
        "Split '%s': %d/%d windows contain at least one anomalous label.",
        split_name,
        int(window_is_anomaly.sum()),
        num_windows,
    )
    return SplitLabelSummary(split_name, num_windows, window_is_anomaly, channel_counts)


def combine_overall_summary(
    summaries: Sequence[SplitLabelSummary], num_channels: int
) -> SplitLabelSummary:
    """Combine per-split summaries into a single "overall" summary."""
    total_windows = sum(s.num_windows for s in summaries)
    if total_windows == 0:
        combined_flags = np.array([], dtype=bool)
    else:
        combined_flags = np.concatenate(
            [s.window_is_anomaly for s in summaries if s.num_windows > 0]
        )
    combined_channel_counts = np.zeros(num_channels, dtype=np.int64)
    for s in summaries:
        combined_channel_counts += s.channel_anomaly_window_counts
    return SplitLabelSummary("overall", total_windows, combined_flags, combined_channel_counts)


# =============================================================================
# Stage 3: general dataset statistics
# =============================================================================


def estimate_total_telemetry_rows(dataset: Dataset) -> Optional[int]:
    """Best-effort estimate of underlying (pre-windowing) telemetry rows.

    ``MissionDataset`` and ``LazyWindowSet`` intentionally expose no
    public accessor for the raw row count (by design, per
    architecture.md, to keep windows lazily materialized and avoid
    duplicating any windowing/assembly logic in this analysis script).
    This helper introspects the already-built objects on a best-effort
    basis; if the underlying structure ever changes, this simply
    degrades to ``None`` rather than raising, per the "fail gracefully"
    requirement -- it never re-loads or re-assembles telemetry itself.

    Args:
        dataset: A ``MissionDataset`` for one split.

    Returns:
        The estimated number of underlying telemetry rows for this
        split, or ``None`` if it could not be determined.
    """
    try:
        telemetry_windows = getattr(dataset, "_telemetry_windows", None)
        values = getattr(telemetry_windows, "_values", None)
        if values is None:
            return None
        return int(len(values))
    except Exception:  # noqa: BLE001 - best-effort, must never crash analysis
        logger.debug("Could not estimate raw telemetry row count.", exc_info=True)
        return None


def compute_general_statistics(
    mission: Any,
    pipeline: Any,
    datasets: Dict[str, Dataset],
) -> Dict[str, Any]:
    """Compute the general (non-label) dataset statistics.

    Args:
        mission: The constructed ``Mission`` object.
        pipeline: The constructed ``TelemetryPipeline`` object.
        datasets: Mapping of split name -> ``MissionDataset``.

    Returns:
        Dict of general statistics.
    """
    num_channels = int(mission.num_channels)

    row_estimates = {
        split: estimate_total_telemetry_rows(ds) for split, ds in datasets.items()
    }
    known_rows = [v for v in row_estimates.values() if v is not None]
    total_rows: Optional[int] = sum(known_rows) if known_rows else None

    window_counts = {split: len(ds) for split, ds in datasets.items()}

    return {
        "num_channels": num_channels,
        "window_size": int(pipeline.window_size),
        "stride": int(pipeline.stride),
        "normalization_method": pipeline.normalization_method,
        "total_telemetry_rows": total_rows,
        "telemetry_rows_per_split": row_estimates,
        "num_train_windows": window_counts.get("train", 0),
        "num_validation_windows": window_counts.get("validation", 0),
        "num_test_windows": window_counts.get("test", 0),
        "total_windows": sum(window_counts.values()),
    }


# =============================================================================
# Stage 4: example window retrieval (for plotting only)
# =============================================================================


def find_example_windows(
    dataset: Dataset,
    channel_index: int,
    num_normal: int,
    num_anomaly: int,
    max_search_windows: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Scan a dataset (bounded) for representative normal/anomaly windows.

    Args:
        dataset: A ``MissionDataset`` to scan.
        channel_index: Channel used to decide normal vs. anomaly for a
            given window (any anomalous timestep on this channel makes
            the window an "anomaly" example).
        num_normal: Desired number of normal example windows.
        num_anomaly: Desired number of anomaly example windows.
        max_search_windows: Upper bound on how many windows to scan.

    Returns:
        Tuple of (normal_examples, anomaly_examples), each a list of
        (telemetry_window, label_window) numpy array pairs.
    """
    normal_examples: List[Tuple[np.ndarray, np.ndarray]] = []
    anomaly_examples: List[Tuple[np.ndarray, np.ndarray]] = []

    search_limit = min(len(dataset), max_search_windows)
    for idx in range(search_limit):
        telemetry, label = dataset[idx]
        telemetry_np = telemetry.numpy()
        label_np = label.numpy()
        is_anomaly = bool(label_np[:, channel_index].sum() > 0)

        if is_anomaly and len(anomaly_examples) < num_anomaly:
            anomaly_examples.append((telemetry_np, label_np))
        elif not is_anomaly and len(normal_examples) < num_normal:
            normal_examples.append((telemetry_np, label_np))

        if len(normal_examples) >= num_normal and len(anomaly_examples) >= num_anomaly:
            break

    return normal_examples, anomaly_examples


def sample_windows_for_correlation(
    dataset: Dataset, max_windows: int
) -> Optional[np.ndarray]:
    """Sample a bounded number of telemetry windows for a correlation plot.

    Args:
        dataset: A ``MissionDataset`` to sample from.
        max_windows: Maximum number of windows to materialize.

    Returns:
        Array of shape (num_sampled_windows * window_size, num_channels),
        or ``None`` if the dataset is empty.
    """
    n = len(dataset)
    if n == 0:
        return None

    count = min(n, max_windows)
    indices = np.linspace(0, n - 1, num=count, dtype=np.int64)
    indices = np.unique(indices)

    rows = []
    for idx in indices:
        telemetry, _ = dataset[int(idx)]
        rows.append(telemetry.numpy())
    return np.concatenate(rows, axis=0)  # (count * window_size, num_channels)


# =============================================================================
# Stage 5: visualizations
# =============================================================================


def _apply_publication_style() -> None:
    """Apply a consistent, publication-quality matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "figure.autolayout": True,
        }
    )


def plot_class_distribution(overall: SplitLabelSummary, output_path: Path) -> None:
    """Bar chart of overall normal vs. anomaly window counts."""
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = [overall.num_normal_windows, overall.num_anomaly_windows]
    bars = ax.bar(
        ["Normal", "Anomaly"], counts, color=[NORMAL_COLOR, ANOMALY_COLOR], width=0.55
    )
    ax.set_ylabel("Number of windows")
    ax.set_title("Overall Class Distribution (Window-Level)")
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / max(overall.num_windows, 1)
        ax.annotate(
            f"{count:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_split_distribution(summaries: Sequence[SplitLabelSummary], output_path: Path) -> None:
    """Grouped bar chart of normal/anomaly counts per split."""
    labels = [s.split_name.capitalize() for s in summaries]
    normal_counts = [s.num_normal_windows for s in summaries]
    anomaly_counts = [s.num_anomaly_windows for s in summaries]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, normal_counts, width, label="Normal", color=NORMAL_COLOR)
    ax.bar(x + width / 2, anomaly_counts, width, label="Anomaly", color=ANOMALY_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of windows")
    ax.set_title("Class Distribution by Split")
    ax.legend()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_telemetry_example(
    dataset: Dataset,
    channel_index: int,
    channel_label: str,
    max_search_windows: int,
    output_path: Path,
) -> None:
    """Plot one representative telemetry channel with anomaly regions.

    Windows are concatenated in dataset order (non-overlapping stride
    through the search range) purely for visualization; no new
    telemetry is loaded or synchronized -- values come directly from
    the dataset's own ``__getitem__`` output.
    """
    n = len(dataset)
    if n == 0:
        logger.warning("Cannot plot telemetry example: dataset is empty.")
        return

    window_size = dataset[0][0].shape[0]
    max_windows_to_plot = max(1, min(n, max_search_windows // max(window_size, 1), 50))
    indices = np.linspace(0, n - 1, num=max_windows_to_plot, dtype=np.int64)
    indices = np.unique(indices)

    values: List[np.ndarray] = []
    anomaly_mask: List[np.ndarray] = []
    for idx in indices:
        telemetry, label = dataset[int(idx)]
        values.append(telemetry.numpy()[:, channel_index])
        anomaly_mask.append(label.numpy()[:, channel_index] > 0)

    series = np.concatenate(values)
    mask = np.concatenate(anomaly_mask)
    timesteps = np.arange(len(series))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(timesteps, series, color=NORMAL_COLOR, linewidth=0.8, label=channel_label)

    if mask.any():
        ax.fill_between(
            timesteps,
            series.min(),
            series.max(),
            where=mask,
            color=ANOMALY_COLOR,
            alpha=0.25,
            label="Anomaly region",
        )

    ax.set_xlabel("Timestep (concatenated sampled windows)")
    ax.set_ylabel("Normalized value")
    ax.set_title(f"Representative Telemetry Channel: {channel_label}")
    ax.legend(loc="upper right")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fault_window_examples(
    normal_examples: Sequence[Tuple[np.ndarray, np.ndarray]],
    anomaly_examples: Sequence[Tuple[np.ndarray, np.ndarray]],
    channel_index: int,
    channel_label: str,
    output_path: Path,
) -> None:
    """Plot a grid of example normal and anomaly windows for one channel."""
    examples = [(w, l, "Normal") for w, l in normal_examples] + [
        (w, l, "Anomaly") for w, l in anomaly_examples
    ]
    if not examples:
        logger.warning("No example windows available to plot.")
        return

    n_cols = min(len(examples), 4)
    n_rows = int(np.ceil(len(examples) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for i, (telemetry, label, kind) in enumerate(examples):
        ax = axes[i // n_cols][i % n_cols]
        series = telemetry[:, channel_index]
        mask = label[:, channel_index] > 0
        color = NORMAL_COLOR if kind == "Normal" else ANOMALY_COLOR
        ax.plot(series, color=color, linewidth=1.0)
        if mask.any():
            ax.fill_between(
                np.arange(len(series)),
                series.min(),
                series.max(),
                where=mask,
                color=ANOMALY_COLOR,
                alpha=0.25,
            )
        ax.set_title(kind, fontsize=10, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(examples), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"Example Windows -- Channel: {channel_label}", fontweight="bold")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_channel_correlation_heatmap(
    windows: Optional[np.ndarray], channel_labels: Sequence[str], output_path: Path
) -> bool:
    """Plot a channel-to-channel correlation heatmap, if data is available.

    This is an optional, best-effort figure: if correlation cannot be
    computed (e.g. no data, degenerate/constant channels), the figure is
    skipped and ``False`` is returned rather than raising.

    Returns:
        True if the figure was produced, False if it was skipped.
    """
    if windows is None or windows.shape[0] < 2:
        logger.info("Skipping channel_correlation_heatmap.png: insufficient data.")
        return False

    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(windows, rowvar=False)
        if not np.isfinite(corr).all():
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:  # noqa: BLE001 - optional figure, must not abort analysis
        logger.warning("Could not compute channel correlation matrix.", exc_info=True)
        return False

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Channel-to-Channel Correlation (sampled windows)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")

    step = max(1, len(channel_labels) // 20)
    tick_positions = np.arange(0, len(channel_labels), step)
    tick_labels = [channel_labels[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


# =============================================================================
# Stage 6: saving statistics / reports
# =============================================================================


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Write ``data`` as pretty-printed, deterministic JSON."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=str)


def save_dataset_statistics_csv(general_stats: Dict[str, Any], path: Path) -> None:
    """Write general dataset statistics as a flat metric/value CSV."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in sorted(general_stats.keys()):
            value = general_stats[key]
            if isinstance(value, dict):
                for sub_key in sorted(value.keys()):
                    writer.writerow([f"{key}.{sub_key}", value[sub_key]])
            else:
                writer.writerow([key, value])


def save_split_statistics_csv(summaries: Sequence[SplitLabelSummary], path: Path) -> None:
    """Write per-split (and overall) class-distribution statistics as CSV."""
    fieldnames = [
        "split",
        "total_windows",
        "normal_windows",
        "anomaly_windows",
        "normal_percentage",
        "anomaly_percentage",
        "ratio_normal_to_anomaly",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.as_dict())


def write_text_report(
    general_stats: Dict[str, Any],
    summaries: Sequence[SplitLabelSummary],
    figures_written: Sequence[str],
    path: Path,
) -> None:
    """Write a human-readable plain-text summary report."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("PhySATFormer -- Pre-Training Dataset Analysis Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("GENERAL DATASET STATISTICS")
    lines.append("-" * 70)
    lines.append(f"Number of channels        : {general_stats['num_channels']}")
    lines.append(f"Window size                : {general_stats['window_size']}")
    lines.append(f"Stride                     : {general_stats['stride']}")
    lines.append(f"Normalization method       : {general_stats['normalization_method']}")
    total_rows = general_stats["total_telemetry_rows"]
    lines.append(
        f"Total telemetry rows (est.): {total_rows if total_rows is not None else 'unavailable'}"
    )
    lines.append(f"Train windows              : {general_stats['num_train_windows']:,}")
    lines.append(f"Validation windows         : {general_stats['num_validation_windows']:,}")
    lines.append(f"Test windows               : {general_stats['num_test_windows']:,}")
    lines.append(f"Total windows              : {general_stats['total_windows']:,}")
    lines.append("")

    lines.append("CLASS DISTRIBUTION")
    lines.append("-" * 70)
    for summary in summaries:
        lines.append(f"[{summary.split_name.upper()}]")
        lines.append(f"  Normal windows   : {summary.num_normal_windows:,} "
                      f"({summary.normal_percentage:.2f}%)")
        lines.append(f"  Anomaly windows  : {summary.num_anomaly_windows:,} "
                      f"({summary.anomaly_percentage:.2f}%)")
        lines.append(f"  Ratio normal:anom: {summary.ratio_normal_to_anomaly}")
        lines.append("")

    lines.append("FIGURES GENERATED")
    lines.append("-" * 70)
    for name in figures_written:
        lines.append(f"  - {name}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("End of report.")
    lines.append("=" * 70)

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the full, read-only dataset analysis end to end."""
    args = parse_arguments()

    global logger
    try:
        logger = setup_logging()
    except Exception:  # noqa: BLE001 - logging setup must never block analysis
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("analyze_dataset")
        logger.warning("Falling back to basicConfig logging.", exc_info=True)

    print("=" * 70)
    print("PhySATFormer -- Pre-Training Dataset Analysis")
    print("=" * 70)

    figures_dir = args.output_dir / "figures"
    metrics_dir = args.output_dir / "metrics"
    reports_dir = args.output_dir / "reports"
    for directory in (figures_dir, metrics_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        (
            _train_cfg,
            _model_cfg,
            _dataset_cfg,
            mission,
            pipeline,
            train_dataset,
            validation_dataset,
            test_dataset,
        ) = build_pipeline_outputs(args)
    except Exception:
        logger.error("Failed to build the preprocessing pipeline.", exc_info=True)
        traceback.print_exc()
        sys.exit(1)

    datasets: Dict[str, Dataset] = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    try:
        num_channels = int(mission.num_channels)
    except Exception:
        logger.error("Could not determine num_channels from Mission.", exc_info=True)
        sys.exit(1)

    try:
        channel_labels = [str(cid) for cid in mission.channels[CHANNEL_ID_COLUMN].tolist()]
    except Exception:
        logger.warning("Could not read channel identifiers; using indices.", exc_info=True)
        channel_labels = [f"channel_{i}" for i in range(num_channels)]

    # ------------------------------------------------------------------
    # General statistics
    # ------------------------------------------------------------------
    general_stats = compute_general_statistics(mission, pipeline, datasets)
    logger.info("General statistics computed: %s", general_stats)

    # ------------------------------------------------------------------
    # Class distribution per split (and overall)
    # ------------------------------------------------------------------
    split_summaries: List[SplitLabelSummary] = []
    for split_name in SPLIT_NAMES:
        try:
            summary = analyze_split_labels(
                datasets[split_name], split_name, num_channels, args.batch_size
            )
        except Exception:
            logger.error(
                "Failed to analyze labels for split '%s'; recording as empty.",
                split_name,
                exc_info=True,
            )
            summary = SplitLabelSummary(
                split_name, 0, np.array([], dtype=bool), np.zeros(num_channels, dtype=np.int64)
            )
        split_summaries.append(summary)

    overall_summary = combine_overall_summary(split_summaries, num_channels)
    all_summaries = split_summaries + [overall_summary]

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    _apply_publication_style()
    figures_written: List[str] = []

    try:
        plot_class_distribution(overall_summary, figures_dir / "class_distribution.png")
        figures_written.append("class_distribution.png")
    except Exception:
        logger.warning("Failed to produce class_distribution.png.", exc_info=True)

    try:
        plot_split_distribution(split_summaries, figures_dir / "split_distribution.png")
        figures_written.append("split_distribution.png")
    except Exception:
        logger.warning("Failed to produce split_distribution.png.", exc_info=True)

    # Choose the most anomalous channel in train (falls back to channel 0)
    # purely to make the example figures informative.
    train_summary = split_summaries[0]
    if train_summary.channel_anomaly_window_counts.sum() > 0:
        representative_channel = int(np.argmax(train_summary.channel_anomaly_window_counts))
    else:
        representative_channel = 0
    representative_label = channel_labels[representative_channel]

    try:
        plot_telemetry_example(
            train_dataset,
            representative_channel,
            representative_label,
            args.max_search_windows,
            figures_dir / "telemetry_example.png",
        )
        figures_written.append("telemetry_example.png")
    except Exception:
        logger.warning("Failed to produce telemetry_example.png.", exc_info=True)

    try:
        normal_examples, anomaly_examples = find_example_windows(
            train_dataset,
            representative_channel,
            num_normal=4,
            num_anomaly=4,
            max_search_windows=args.max_search_windows,
        )
        plot_fault_window_examples(
            normal_examples,
            anomaly_examples,
            representative_channel,
            representative_label,
            figures_dir / "fault_window_examples.png",
        )
        figures_written.append("fault_window_examples.png")
    except Exception:
        logger.warning("Failed to produce fault_window_examples.png.", exc_info=True)

    try:
        correlation_windows = sample_windows_for_correlation(
            train_dataset, DEFAULT_CORRELATION_SAMPLE_WINDOWS
        )
        produced = plot_channel_correlation_heatmap(
            correlation_windows, channel_labels, figures_dir / "channel_correlation_heatmap.png"
        )
        if produced:
            figures_written.append("channel_correlation_heatmap.png")
    except Exception:
        logger.warning("Failed to produce channel_correlation_heatmap.png.", exc_info=True)

    # ------------------------------------------------------------------
    # Save metrics / reports
    # ------------------------------------------------------------------
    summary_json = {
        "general_statistics": general_stats,
        "class_distribution": {s.split_name: s.as_dict() for s in all_summaries},
        "figures_generated": figures_written,
    }

    try:
        save_json(summary_json, metrics_dir / "dataset_summary.json")
        save_dataset_statistics_csv(general_stats, metrics_dir / "dataset_statistics.csv")
        save_split_statistics_csv(all_summaries, metrics_dir / "split_statistics.csv")
        write_text_report(
            general_stats, all_summaries, figures_written, reports_dir / "dataset_report.txt"
        )
    except Exception:
        logger.error("Failed to write one or more output files.", exc_info=True)
        traceback.print_exc()
        sys.exit(1)

    print("-" * 70)
    print(f"Channels           : {general_stats['num_channels']}")
    print(f"Total windows       : {general_stats['total_windows']:,}")
    print(
        f"Overall class split : "
        f"{overall_summary.num_normal_windows:,} normal / "
        f"{overall_summary.num_anomaly_windows:,} anomaly "
        f"({overall_summary.anomaly_percentage:.2f}% anomaly)"
    )
    print(f"Figures written to  : {figures_dir}")
    print(f"Metrics written to  : {metrics_dir}")
    print(f"Report written to   : {reports_dir / 'dataset_report.txt'}")
    print("=" * 70)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()