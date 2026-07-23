"""
src/evaluation/plots.py

Renders training-history curves (loss, F1, precision, recall) from the
`results/history/history.json` file produced at the end of training
(see `src/utils/history.py` / the `train.py` integration described in
the accompanying integration notes).

Uses matplotlib only, per spec -- no seaborn, no sklearn.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, List, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# (history_key_prefix, y_axis_label, output_filename)
_CURVE_SPECS = (
    ("loss", "Loss", "loss_curve.png"),
    ("f1", "F1 Score", "f1_curve.png"),
    ("precision", "Precision", "precision_curve.png"),
    ("recall", "Recall", "recall_curve.png"),
)


def load_history(
    history_path: Union[str, pathlib.Path] = "results/history/history.json",
) -> Dict[str, List[float]]:
    """
    Load the training-history dict saved by `train.py` at the end of
    training.

    Args:
        history_path: Path to `history.json`.

    Returns:
        A dict with keys `train_loss`, `val_loss`, `train_precision`,
        `val_precision`, `train_recall`, `val_recall`, `train_f1`,
        `val_f1`, each mapping to a list of per-epoch floats.

    Raises:
        FileNotFoundError: If `history_path` does not exist.
    """
    resolved_path = pathlib.Path(history_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"No history file found at '{resolved_path}'. Make sure "
            "training has completed and history saving is enabled."
        )

    with resolved_path.open("r", encoding="utf-8") as history_file:
        history = json.load(history_file)

    logger.info("Loaded training history from '%s'.", resolved_path)
    return history


def _plot_curve(
    history: Dict[str, List[float]],
    metric_prefix: str,
    y_label: str,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Plot one train/val curve pair (e.g. `train_loss` vs `val_loss`)."""
    train_key = f"train_{metric_prefix}"
    val_key = f"val_{metric_prefix}"

    if train_key not in history or val_key not in history:
        raise KeyError(
            f"history.json is missing expected keys '{train_key}' and/or "
            f"'{val_key}'."
        )

    epochs = range(1, len(history[train_key]) + 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history[train_key], label=f"Train {y_label}", linewidth=2)
    ax.plot(epochs, history[val_key], label=f"Validation {y_label}", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs. Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved %s curve to '%s'.", y_label.lower(), output_path)
    return output_path


def generate_history_plots(
    history_path: Union[str, pathlib.Path] = "results/history/history.json",
    figures_dir: Union[str, pathlib.Path] = "results/figures",
) -> Dict[str, pathlib.Path]:
    """
    Load `history.json` and render all four training-history curves
    (loss, F1, precision, recall) into `figures_dir`.

    Args:
        history_path: Path to `history.json`, as produced at the end of
            training.
        figures_dir: Directory the four PNGs are saved into.

    Returns:
        A dict mapping each curve name (`"loss"`, `"f1"`, `"precision"`,
        `"recall"`) to its saved `pathlib.Path`.
    """
    history = load_history(history_path)
    figures_dir = pathlib.Path(figures_dir)

    saved_paths: Dict[str, pathlib.Path] = {}
    for metric_prefix, y_label, filename in _CURVE_SPECS:
        saved_paths[metric_prefix] = _plot_curve(
            history, metric_prefix, y_label, figures_dir / filename
        )

    return saved_paths
