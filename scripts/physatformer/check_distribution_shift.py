"""
check_distribution_shift.py

Standalone diagnostic: quantifies how different the train/validation/test
splits actually are, BEFORE spending a full training run on it.

It does NOT train a model. It reuses the exact same components train.py
uses (TelemetryAssembler, IntervalLabelGenerator, TelemetryPipeline's own
chronological split) so the comparison reflects the real pipeline, then
reports:

  1. Anomaly rate (overall and per-channel) in train vs. validation vs.
     test. A large drop/rise across splits means the model is being
     asked to generalize across genuinely different operating regimes,
     not just noise.
  2. Per-channel raw telemetry mean/std shift: how many standard
     deviations (using TRAIN statistics) each split's mean differs from
     train's mean. Channels with |z-shift| > 3 are flagged -- these are
     the channels most likely responsible for validation/test looking
     different from train, e.g. due to a mission-phase change,
     recalibration, or a sensor fault confined to one time period.

Run from the project root (so `src.*` imports resolve):

    python check_distribution_shift.py --dataset-config dataset.yaml

Output: a printed summary plus a CSV report written next to this script,
`distribution_shift_report.csv`, with one row per channel.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.core.mission import Mission
from src.preprocessing.assembler import TelemetryAssembler
from src.preprocessing.interval_label_generator import IntervalLabelGenerator
from src.preprocessing.pipeline import TelemetryPipeline
from src.utils.constants import CHANNEL_ID_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("distribution_shift_check")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=str, default="dataset.yaml")
    parser.add_argument(
        "--report-path", type=str, default="distribution_shift_report.csv"
    )
    parser.add_argument(
        "--z-shift-flag-threshold",
        type=float,
        default=3.0,
        help="Flag a channel if |mean shift| exceeds this many train std-devs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.dataset_config, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    mission = Mission(root=dataset_cfg["root"])
    channel_ids = mission.channels[CHANNEL_ID_COLUMN].tolist()

    assembler = TelemetryAssembler(direction=dataset_cfg["direction"])
    label_generator = IntervalLabelGenerator()

    logger.info("Assembling telemetry for %d channels...", len(channel_ids))
    synchronized_df = assembler.assemble(mission, channel_ids)

    if dataset_cfg.get("max_rows"):
        synchronized_df = synchronized_df.iloc[: dataset_cfg["max_rows"]]

    # Reuse the exact same chronological split logic train.py's pipeline
    # uses, so this diagnostic reflects the real split boundaries.
    pipeline = TelemetryPipeline(
        window_size=dataset_cfg["window_size"],
        stride=dataset_cfg["stride"],
        normalization_method=dataset_cfg["normalization_method"],
        train_ratio=dataset_cfg["train_ratio"],
        validation_ratio=dataset_cfg["validation_ratio"],
        random_seed=dataset_cfg["random_seed"],
        direction=dataset_cfg["direction"],
    )
    train_df, val_df, test_df = pipeline._split_synchronized_telemetry(synchronized_df)

    splits = {"train": train_df, "validation": val_df, "test": test_df}

    logger.info(
        "Split sizes (rows): train=%d validation=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info(
        "Split time ranges: train=[%s, %s] validation=[%s, %s] test=[%s, %s]",
        train_df.index.min(), train_df.index.max(),
        val_df.index.min(), val_df.index.max(),
        test_df.index.min(), test_df.index.max(),
    )

    # ------------------------------------------------------------------
    # 1. Anomaly-rate comparison across splits
    # ------------------------------------------------------------------
    split_labels = {
        name: label_generator.generate(mission, df) for name, df in splits.items()
    }

    print("\n=== Overall anomaly rate per split ===")
    overall_rates = {}
    for name, labels in split_labels.items():
        rate = float(labels.mean())
        overall_rates[name] = rate
        print(f"  {name:12s}: {rate:.4%}  ({int(labels.sum())} anomalous cells)")

    if overall_rates["train"] > 0:
        val_ratio = overall_rates["validation"] / overall_rates["train"]
        test_ratio = overall_rates["test"] / overall_rates["train"]
        print(
            f"\n  validation/train anomaly-rate ratio: {val_ratio:.2f}x   "
            f"test/train ratio: {test_ratio:.2f}x"
        )
        if val_ratio < 0.3 or val_ratio > 3.0:
            print(
                "  >>> WARNING: validation anomaly rate is very different from "
                "train's. Even a perfect model would show a different "
                "precision/recall balance here just from prevalence alone."
            )

    # ------------------------------------------------------------------
    # 2. Per-channel telemetry mean/std shift (z-shift vs. TRAIN stats)
    # ------------------------------------------------------------------
    train_mean = train_df.mean(axis=0)
    train_std = train_df.std(axis=0).replace(0.0, np.nan)

    rows = []
    for channel in channel_ids:
        if channel not in train_df.columns:
            continue
        row = {"channel": channel}
        for name, df in splits.items():
            split_mean = df[channel].mean()
            z_shift = (split_mean - train_mean[channel]) / train_std[channel]
            row[f"{name}_mean"] = split_mean
            if name != "train":
                row[f"{name}_z_shift_vs_train"] = z_shift
        # Per-channel anomaly rate per split, for cross-reference.
        for name, labels in split_labels.items():
            col_idx = list(splits[name].columns).index(channel)
            row[f"{name}_anomaly_rate"] = float(labels[:, col_idx].mean())
        rows.append(row)

    report_df = pd.DataFrame(rows).set_index("channel")
    report_df.to_csv(args.report_path)
    logger.info("Full per-channel report written to %s", args.report_path)

    flagged = report_df[
        (report_df["validation_z_shift_vs_train"].abs() > args.z_shift_flag_threshold)
        | (report_df["test_z_shift_vs_train"].abs() > args.z_shift_flag_threshold)
    ]

    print(
        f"\n=== Channels with |z-shift| > {args.z_shift_flag_threshold} "
        f"in validation or test (vs. train statistics) ==="
    )
    if flagged.empty:
        print("  None. Raw telemetry statistics look consistent across splits.")
    else:
        print(flagged[["validation_z_shift_vs_train", "test_z_shift_vs_train"]])
        print(
            f"\n  {len(flagged)}/{len(report_df)} channels flagged. If this list "
            "is large, that's real distribution shift from the chronological "
            "split (different mission phase / calibration), not a training bug -- "
            "consider blocked/rolling-window validation instead of one hold-out."
        )

    print(f"\nDone. See {args.report_path} for the full per-channel table.")


if __name__ == "__main__":
    main()
