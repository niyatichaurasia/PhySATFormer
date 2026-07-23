"""
src/explainability

Post-hoc explainability utilities: SHAP-based feature attribution
(`shap_explainer`), model-agnostic attention capture/visualization
(`attention_visualizer`), and PhySATFormer-specific physics-prior vs.
learned-attention comparison (`physics_visualizer`).
"""

from src.explainability.attention_visualizer import (
    AttentionCapture,
    capture_attention,
    save_attention_heatmaps,
)
from src.explainability.physics_visualizer import generate_physics_comparison
from src.explainability.shap_explainer import compute_and_save_shap

__all__ = [
    "AttentionCapture",
    "capture_attention",
    "save_attention_heatmaps",
    "generate_physics_comparison",
    "compute_and_save_shap",
]
