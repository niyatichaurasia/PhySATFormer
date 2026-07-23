"""
src/explainability/physics_visualizer.py

PhySATFormer-specific explainability: compares the static physics prior
(`PhysicsRelationshipMatrix`'s `(C, C)` output) against the model's
*learned* channel attention, to show how much the physics-guided
attention actually deviates from -- or reinforces -- the engineering
prior.

Unlike `attention_visualizer.py` (model-agnostic), this module is
specific to `PhySATFormer`: `BaselineTransformer` has no physics prior to
compare against.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_numpy(matrix: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def save_physics_matrix_heatmap(
    physics_matrix: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, pathlib.Path] = "results/attention_maps/physics_matrix.png",
) -> pathlib.Path:
    """Save a heatmap of the static physics relationship matrix."""
    resolved_path = pathlib.Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = _to_numpy(physics_matrix)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Physics Relationship Matrix (prior)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(resolved_path, dpi=150)
    plt.close(fig)

    logger.info("Saved physics matrix heatmap to '%s'.", resolved_path)
    return resolved_path


def save_learned_attention_heatmap(
    learned_attention: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, pathlib.Path] = "results/attention_maps/learned_attention.png",
) -> pathlib.Path:
    """Save a heatmap of the model's learned (head/batch-averaged) channel attention."""
    resolved_path = pathlib.Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = _to_numpy(learned_attention)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title("Learned Channel Attention")
    ax.set_xlabel("Channel (key)")
    ax.set_ylabel("Channel (query)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(resolved_path, dpi=150)
    plt.close(fig)

    logger.info("Saved learned attention heatmap to '%s'.", resolved_path)
    return resolved_path


def save_difference_heatmap(
    physics_matrix: Union[torch.Tensor, np.ndarray],
    learned_attention: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, pathlib.Path] = "results/attention_maps/physics_vs_learned_diff.png",
) -> pathlib.Path:
    """
    Save a heatmap of `learned_attention - physics_matrix`, highlighting
    where the model's learned attention diverges from the engineering
    prior.

    Both inputs are min-max normalized to `[0, 1]` independently before
    differencing, since the physics matrix is binary (0/1) while learned
    attention weights are softmax outputs (sum to 1 per row) -- comparing
    raw values would conflate scale differences with genuine structural
    divergence.
    """
    resolved_path = pathlib.Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    physics = _to_numpy(physics_matrix).astype(np.float64)
    learned = _to_numpy(learned_attention).astype(np.float64)

    if physics.shape != learned.shape:
        raise ValueError(
            f"physics_matrix shape {physics.shape} does not match "
            f"learned_attention shape {learned.shape}."
        )

    def _normalize(m: np.ndarray) -> np.ndarray:
        m_min, m_max = m.min(), m.max()
        if m_max - m_min < 1e-12:
            return np.zeros_like(m)
        return (m - m_min) / (m_max - m_min)

    difference = _normalize(learned) - _normalize(physics)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(difference, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Learned Attention − Physics Prior (normalized)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(resolved_path, dpi=150)
    plt.close(fig)

    logger.info("Saved physics-vs-learned difference heatmap to '%s'.", resolved_path)
    return resolved_path


def save_side_by_side_comparison(
    physics_matrix: Union[torch.Tensor, np.ndarray],
    learned_attention: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, pathlib.Path] = "results/attention_maps/physics_vs_learned_comparison.png",
) -> pathlib.Path:
    """Save a single figure with physics prior, learned attention, and their difference side by side."""
    resolved_path = pathlib.Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    physics = _to_numpy(physics_matrix).astype(np.float64)
    learned = _to_numpy(learned_attention).astype(np.float64)

    def _normalize(m: np.ndarray) -> np.ndarray:
        m_min, m_max = m.min(), m.max()
        if m_max - m_min < 1e-12:
            return np.zeros_like(m)
        return (m - m_min) / (m_max - m_min)

    difference = _normalize(learned) - _normalize(physics)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = axes[0].imshow(physics, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Physics Prior")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(learned, cmap="viridis", aspect="auto")
    axes[1].set_title("Learned Attention")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(difference, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_title("Difference (Learned − Physics)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")

    fig.tight_layout()
    fig.savefig(resolved_path, dpi=150)
    plt.close(fig)

    logger.info("Saved side-by-side physics comparison to '%s'.", resolved_path)
    return resolved_path


def generate_physics_comparison(
    physics_matrix: Union[torch.Tensor, np.ndarray],
    learned_attention: Union[torch.Tensor, np.ndarray],
    output_dir: Union[str, pathlib.Path] = "results/attention_maps",
) -> dict:
    """
    Convenience wrapper generating all four physics-comparison figures in
    one call.

    Args:
        physics_matrix: `(C, C)` static physics relationship matrix.
        learned_attention: `(C, C)` learned channel attention (already
            batch- and, typically, head-averaged -- e.g. via
            `attention_visualizer.save_attention_heatmaps`'s underlying
            arrays).
        output_dir: Directory the four PNGs are saved into.

    Returns:
        Dict mapping figure name to its saved path.
    """
    output_dir = pathlib.Path(output_dir)
    return {
        "physics_matrix": save_physics_matrix_heatmap(
            physics_matrix, output_dir / "physics_matrix.png"
        ),
        "learned_attention": save_learned_attention_heatmap(
            learned_attention, output_dir / "learned_attention.png"
        ),
        "difference": save_difference_heatmap(
            physics_matrix, learned_attention, output_dir / "physics_vs_learned_diff.png"
        ),
        "comparison": save_side_by_side_comparison(
            physics_matrix,
            learned_attention,
            output_dir / "physics_vs_learned_comparison.png",
        ),
    }
