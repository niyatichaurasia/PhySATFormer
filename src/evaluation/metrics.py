"""
src/evaluation/metrics.py

Figure-producing evaluation metrics for a finished `EvaluationResult`:
confusion matrix, ROC curve, and precision-recall curve, each saved as a
PNG. This module is intentionally decoupled from `Evaluator` (no
inference logic lives here) and from `src/training/metrics.py`'s
`PhySATMetrics` (which is a stateless, per-batch training-loop metric,
not a report-figure generator).

All functions here take already-computed flat arrays (as produced by
`Evaluator.evaluate`) and are pure I/O: given arrays in, PNG file on
disk out.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)


def _ensure_parent_dir(output_path: Union[str, pathlib.Path]) -> pathlib.Path:
    """Resolve `output_path` and create its parent directory if needed."""
    resolved = pathlib.Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_confusion_matrix(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    output_path: Union[str, pathlib.Path] = "results/figures/confusion_matrix.png",
) -> pathlib.Path:
    """
    Compute and save a confusion-matrix figure for binary channel-level
    anomaly predictions.

    Args:
        ground_truth: Flat array of binary ground-truth labels.
        predictions: Flat array of binary (already-thresholded)
            predictions, index-aligned with `ground_truth`.
        output_path: Destination PNG path. Parent directories are
            created automatically.

    Returns:
        The resolved path the figure was saved to.
    """
    resolved_path = _ensure_parent_dir(output_path)

    cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Anomaly"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(resolved_path, dpi=150)
    plt.close(fig)

    logger.info("Saved confusion matrix to '%s'.", resolved_path)
    return resolved_path


def save_roc_curve(
    ground_truth: np.ndarray,
    probabilities: np.ndarray,
    output_path: Union[str, pathlib.Path] = "results/figures/roc_curve.png",
) -> pathlib.Path:
    """
    Compute and save an ROC curve figure with the AUC annotated in the
    legend.

    Args:
        ground_truth: Flat array of binary ground-truth labels.
        probabilities: Flat array of predicted probabilities in
            `[0, 1]`, index-aligned with `ground_truth`.
        output_path: Destination PNG path.

    Returns:
        The resolved path the figure was saved to.
    """
    resolved_path = _ensure_parent_dir(output_path)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        ground_truth, probabilities
    )
    roc_auc = auc(false_positive_rate, true_positive_rate)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
        linewidth=2,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(resolved_path, dpi=150)
    plt.close(fig)

    logger.info(
        "Saved ROC curve to '%s' (AUC=%.4f).", resolved_path, roc_auc
    )
    return resolved_path


def save_pr_curve(
    ground_truth: np.ndarray,
    probabilities: np.ndarray,
    output_path: Union[str, pathlib.Path] = "results/figures/pr_curve.png",
) -> pathlib.Path:
    """
    Compute and save a precision-recall curve figure with average
    precision (AP) annotated in the legend.

    Args:
        ground_truth: Flat array of binary ground-truth labels.
        probabilities: Flat array of predicted probabilities in
            `[0, 1]`, index-aligned with `ground_truth`.
        output_path: Destination PNG path.

    Returns:
        The resolved path the figure was saved to.
    """
    resolved_path = _ensure_parent_dir(output_path)

    precision, recall, _ = precision_recall_curve(ground_truth, probabilities)
    average_precision = average_precision_score(ground_truth, probabilities)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        recall,
        precision,
        label=f"PR curve (AP = {average_precision:.4f})",
        linewidth=2,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(resolved_path, dpi=150)
    plt.close(fig)

    logger.info(
        "Saved PR curve to '%s' (AP=%.4f).", resolved_path, average_precision
    )
    return resolved_path


def save_all_figures(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    figures_dir: Union[str, pathlib.Path] = "results/figures",
) -> dict:
    """
    Convenience wrapper: save the confusion matrix, ROC curve, and PR
    curve in one call, all under `figures_dir`.

    Args:
        ground_truth: Flat array of binary ground-truth labels.
        predictions: Flat array of binary (thresholded) predictions.
        probabilities: Flat array of predicted probabilities.
        figures_dir: Directory under which the three PNGs are saved.

    Returns:
        A dict mapping each figure name to its saved `pathlib.Path`.
    """
    figures_dir = pathlib.Path(figures_dir)
    return {
        "confusion_matrix": save_confusion_matrix(
            ground_truth, predictions, figures_dir / "confusion_matrix.png"
        ),
        "roc_curve": save_roc_curve(
            ground_truth, probabilities, figures_dir / "roc_curve.png"
        ),
        "pr_curve": save_pr_curve(
            ground_truth, probabilities, figures_dir / "pr_curve.png"
        ),
    }
