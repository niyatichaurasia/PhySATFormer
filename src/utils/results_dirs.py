"""
src/utils/results_dirs.py

Ensures the `results/` output-directory tree exists before any
evaluation/explainability artifact is written. Called automatically by
`scripts/evaluate_model.py` and `scripts/generate_explanations.py`, and
safe to call repeatedly (idempotent).
"""

from __future__ import annotations

import pathlib
from typing import Union

_SUBDIRECTORIES = (
    "checkpoints",
    "history",
    "figures",
    "predictions",
    "shap",
    "attention_maps",
)


def ensure_results_directories(
    results_root: Union[str, pathlib.Path] = "results",
) -> pathlib.Path:
    """
    Create `results/` and all expected subdirectories if missing.

    Args:
        results_root: Root results directory.

    Returns:
        The resolved `results_root` path.
    """
    root = pathlib.Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    for subdirectory in _SUBDIRECTORIES:
        (root / subdirectory).mkdir(parents=True, exist_ok=True)
    return root
