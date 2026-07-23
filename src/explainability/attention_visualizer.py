"""
src/explainability/attention_visualizer.py

Captures and visualizes channel-attention weights from a trained
PhySATFormer / BaselineTransformer model.

============================================================================
INTEGRATION NOTE -- the one place this task may require a model change
============================================================================
This visualizer captures attention weights via a `forward hook`
registered on every `torch.nn.MultiheadAttention` submodule inside the
model. That means:

  * If `ChannelAttentionBlock` / `StandardChannelAttentionBlock`
    internally use `nn.MultiheadAttention` (the standard, idiomatic
    choice, and consistent with their public docstrings), this module
    works completely unmodified -- no changes to `channel_attention_
    block.py` are needed. `nn.MultiheadAttention.forward(...)` returns
    `(attn_output, attn_weights)` whenever `need_weights=True` (PyTorch's
    default), and a forward hook can read `attn_weights` straight off the
    module's output tuple without touching the block's source at all.

  * The ONLY modification that may be required (per the task's "modify
    the attention modules only if necessary" instruction) is: if your
    `ChannelAttentionBlock.forward` calls
    `self.attention(q, k, v, need_weights=False)` (skipping weight
    computation for a small speed win), change that single call site to
    `need_weights=True, average_attn_weights=False` so per-head weights
    are actually computed and returned. No other logic changes.

This module deliberately does NOT assume the internal attribute name of
the attention submodule; it discovers all `nn.MultiheadAttention`
instances by walking `model.named_modules()`.
============================================================================
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


class AttentionCapture:
    """
    Registers forward hooks on every `nn.MultiheadAttention` submodule of
    `model` and records the attention-weight tensor each one produces on
    its next forward pass.

    Usage:
        >>> capture = AttentionCapture(model)
        >>> with capture:
        ...     _ = model(sample_batch)
        >>> capture.attention_weights  # {layer_name: tensor}
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attention_weights: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._attention_layer_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.MultiheadAttention)
        ]
        if not self._attention_layer_names:
            logger.warning(
                "No nn.MultiheadAttention submodules found in model. "
                "AttentionCapture will record nothing -- see the "
                "integration note at the top of attention_visualizer.py."
            )

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: tuple, output: tuple) -> None:
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            # when need_weights=True (the default).
            if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                self.attention_weights[layer_name] = output[1].detach().cpu()
            else:
                logger.warning(
                    "Layer '%s' returned no attention weights (need_weights "
                    "may be False). See integration note.",
                    layer_name,
                )

        return hook

    def __enter__(self) -> "AttentionCapture":
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def layer_names(self) -> List[str]:
        return list(self._attention_layer_names)


def capture_attention(
    model: nn.Module, sample_input: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Run one forward pass and capture every layer's attention weights.

    Args:
        model: The trained model (in `eval()` mode is recommended by the
            caller, though not enforced here).
        sample_input: A single input batch, shape `(B, T, C)` (or
            `(B, T, C, input_dim)`), matching the model's forward
            contract.

    Returns:
        Dict mapping `nn.MultiheadAttention` submodule name to its
        attention-weight tensor, shape `(B, num_heads, C, C)` if
        `average_attn_weights=False` was used, or `(B, C, C)` if
        head-averaged.
    """
    with torch.no_grad(), AttentionCapture(model) as capture:
        model(sample_input)
    return dict(capture.attention_weights)


def _plot_heatmap(
    matrix: np.ndarray, title: str, output_path: pathlib.Path
) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Channel (key)")
    ax.set_ylabel("Channel (query)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved attention heatmap to '%s'.", output_path)
    return output_path


def save_attention_heatmaps(
    attention_weights: Dict[str, torch.Tensor],
    output_dir: Union[str, pathlib.Path] = "results/attention_maps",
    layer: Optional[str] = None,
    head: Optional[int] = None,
    average_heads: bool = True,
    batch_index: int = 0,
) -> List[pathlib.Path]:
    """
    Save attention heatmaps for one or all layers/heads.

    Args:
        attention_weights: Output of `capture_attention(...)`, mapping
            layer name to a `(B, num_heads, C, C)` or `(B, C, C)` tensor.
        output_dir: Directory heatmaps are saved into, as
            `{layer}_head{n}.png` (or `{layer}_avg.png` if
            `average_heads=True`).
        layer: If given, restrict to this layer name only (must be a key
            of `attention_weights`). Otherwise all captured layers are
            plotted.
        head: If given (and `average_heads=False`), plot only this head
            index. Otherwise all heads are plotted individually.
        average_heads: If True, average across heads before plotting one
            heatmap per layer, ignoring `head`.
        batch_index: Which sample in the batch to visualize.

    Returns:
        List of saved figure paths.
    """
    resolved_dir = pathlib.Path(output_dir)
    layer_names = [layer] if layer is not None else list(attention_weights.keys())

    saved_paths: List[pathlib.Path] = []

    for layer_name in layer_names:
        if layer_name not in attention_weights:
            raise KeyError(
                f"Layer '{layer_name}' not found in captured attention "
                f"weights. Available layers: {list(attention_weights.keys())}"
            )

        weights = attention_weights[layer_name][batch_index]  # (num_heads, C, C) or (C, C)
        safe_layer_name = layer_name.replace(".", "_")

        if weights.dim() == 2:
            # Already head-averaged (e.g. average_attn_weights=True at
            # call time); nothing further to average.
            matrix = weights.numpy()
            path = resolved_dir / f"{safe_layer_name}_avg.png"
            saved_paths.append(
                _plot_heatmap(matrix, f"{layer_name} (attention)", path)
            )
            continue

        if average_heads:
            matrix = weights.mean(dim=0).numpy()
            path = resolved_dir / f"{safe_layer_name}_avg.png"
            saved_paths.append(
                _plot_heatmap(matrix, f"{layer_name} (head-averaged)", path)
            )
        else:
            head_indices = [head] if head is not None else range(weights.shape[0])
            for head_idx in head_indices:
                matrix = weights[head_idx].numpy()
                path = resolved_dir / f"{safe_layer_name}_head{head_idx}.png"
                saved_paths.append(
                    _plot_heatmap(
                        matrix, f"{layer_name} (head {head_idx})", path
                    )
                )

    return saved_paths
