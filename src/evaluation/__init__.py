"""
src/evaluation

Post-training evaluation utilities: checkpoint-driven inference
(`Evaluator`), report figures (`metrics`), training-history curves
(`plots`), and raw prediction export (`prediction_export`).
"""

from src.evaluation.evaluator import EvaluationResult, Evaluator
from src.evaluation.metrics import (
    save_all_figures,
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
)
from src.evaluation.plots import generate_history_plots, load_history
from src.evaluation.prediction_export import export_predictions_csv

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "save_confusion_matrix",
    "save_roc_curve",
    "save_pr_curve",
    "save_all_figures",
    "load_history",
    "generate_history_plots",
    "export_predictions_csv",
]
