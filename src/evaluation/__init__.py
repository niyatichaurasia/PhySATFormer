"""
src/evaluation

Post-training evaluation utilities: checkpoint-driven inference
(`Evaluator`), report figures (`metrics`), training-history curves
(`plots`), and raw prediction export (`prediction_export`).
"""

from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import (
    save_all_figures,
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
)
from src.evaluation.plots import generate_history_plots, load_history
from src.evaluation.prediction_export import (
    StreamingPredictionWriter,
    NullPredictionWriter,
    get_prediction_writer,
)
__all__ = [
    "Evaluator",
    "save_confusion_matrix",
    "save_roc_curve",
    "save_pr_curve",
    "save_all_figures",
    "load_history",
    "generate_history_plots",
    "StreamingPredictionWriter",
    "NullPredictionWriter",
    "get_prediction_writer",
]
