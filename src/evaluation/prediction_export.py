"""
src/evaluation/prediction_export.py

Exports raw per-(window, timestep, channel) predictions from an
`EvaluationResult` to a flat CSV file, for downstream analysis or
inclusion as supplementary material in a paper submission.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Union

import pandas as pd

from src.evaluation.evaluator import EvaluationResult

logger = logging.getLogger(__name__)


def export_predictions_csv(
    evaluation_result: EvaluationResult,
    output_path: Union[str, pathlib.Path] = "results/predictions/predictions.csv",
) -> pathlib.Path:
    """
    Write `GroundTruth`, `Prediction`, `Probability` columns to CSV.

    Args:
        evaluation_result: The result of `Evaluator.evaluate(...)`.
        output_path: Destination CSV path. Parent directories are
            created automatically.

    Returns:
        The resolved path the CSV was written to.
    """
    resolved_path = pathlib.Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(
        {
            "GroundTruth": evaluation_result.ground_truth,
            "Prediction": evaluation_result.predictions,
            "Probability": evaluation_result.probabilities,
        }
    )

    predictions_df.to_csv(resolved_path, index=False)

    logger.info(
        "Exported %d predictions to '%s'.",
        len(predictions_df),
        resolved_path,
    )
    return resolved_path
