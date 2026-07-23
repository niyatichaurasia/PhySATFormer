"""
src/utils/history.py

Tiny, dependency-free utility for persisting the training-history dict
returned by `Trainer.fit(...)` to `results/history/history.json`.

This is the ONLY new logic touching the training path (see integration
notes). `Trainer.fit` itself is unmodified -- it already returns the
exact dict this function expects; this module just serializes it.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

REQUIRED_HISTORY_KEYS = (
    "train_loss",
    "val_loss",
    "train_precision",
    "val_precision",
    "train_recall",
    "val_recall",
    "train_f1",
    "val_f1",
)


def save_history(
    history: Dict[str, List[float]],
    output_path: Union[str, pathlib.Path] = "results/history/history.json",
) -> pathlib.Path:
    """
    Serialize a `Trainer.fit(...)` history dict to JSON.

    Args:
        history: The dict returned by `Trainer.fit(...)`, expected to
            contain (at least) all keys in `REQUIRED_HISTORY_KEYS`, each
            mapping to a list of per-epoch floats.
        output_path: Destination JSON path. Parent directories are
            created automatically.

    Returns:
        The resolved path the history was written to.

    Raises:
        KeyError: If `history` is missing any of `REQUIRED_HISTORY_KEYS`.
    """
    missing_keys = [key for key in REQUIRED_HISTORY_KEYS if key not in history]
    if missing_keys:
        raise KeyError(
            f"history is missing required key(s): {missing_keys}. Got keys: "
            f"{list(history.keys())}."
        )

    resolved_path = pathlib.Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    logger.info("Saved training history to '%s'.", resolved_path)
    return resolved_path
