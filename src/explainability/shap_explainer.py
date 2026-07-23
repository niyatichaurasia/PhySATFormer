"""
src/explainability/shap_explainer.py

SHAP-based explainability for a trained PhySATFormer / BaselineTransformer
model, using `shap.GradientExplainer` (works directly with PyTorch
models, no architecture modification required).

Because the model produces per-timestep, per-channel logits of shape
`(B, T, C)` rather than a single scalar, SHAP values are computed against
a scalar reduction of the output (by default, the mean logit over the
`target_timestep`/`target_channel` selection, or over the full output if
neither is given). This is a standard approach for explaining
sequence-to-sequence / dense-prediction models with SHAP, since
`GradientExplainer` (like all SHAP explainers) attributes a single
scalar output at a time.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class _ScalarOutputWrapper(nn.Module):
    """
    Wraps a `(B, T, C)`-output model so it exposes a single scalar output
    per sample, as required by `shap.GradientExplainer`.

    If `target_timestep`/`target_channel` are given, that single
    (timestep, channel) logit is used. Otherwise the mean over the full
    `(T, C)` output is used, which explains "what drives the model's
    overall anomaly propensity for this window" rather than any single
    cell.
    """

    def __init__(
        self,
        model: nn.Module,
        target_timestep: Optional[int] = None,
        target_channel: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_timestep = target_timestep
        self.target_channel = target_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)  # (B, T, C)

        if self.target_timestep is not None:
            logits = logits[:, self.target_timestep, :]  # (B, C)
        if self.target_channel is not None:
            logits = (
                logits[:, self.target_channel]
                if logits.dim() == 2
                else logits[:, :, self.target_channel]
            )
        if logits.dim() > 1:
            logits = logits.reshape(logits.shape[0], -1).mean(dim=1)

        return logits.unsqueeze(-1)  # (B, 1), scalar output per sample


def _sample_background_and_test(
    test_loader: DataLoader,
    num_background: int,
    num_test_windows: int,
    device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    Materialize a small background set and a small explanation set from
    `test_loader`, both as stacked `(N, T, C)` tensors on `device`.

    A bounded number of windows is materialized (never the full test
    set) so this stays memory-safe on production-scale datasets, mirroring
    the lazy-window philosophy already established in
    `TelemetryPipeline`/`LazyWindowSet`.
    """
    windows = []
    total_needed = num_background + num_test_windows

    for batch_inputs, _ in test_loader:
        windows.append(batch_inputs)
        if sum(w.shape[0] for w in windows) >= total_needed:
            break

    all_windows = torch.cat(windows, dim=0)[:total_needed]
    if all_windows.shape[0] < total_needed:
        raise ValueError(
            f"test_loader yielded only {all_windows.shape[0]} windows, but "
            f"{total_needed} (num_background + num_test_windows) were "
            "requested."
        )

    background = all_windows[:num_background].to(device)
    test_windows = all_windows[num_background:total_needed].to(device)
    return background, test_windows


def compute_and_save_shap(
    model: nn.Module,
    test_loader: DataLoader,
    output_dir: Union[str, pathlib.Path] = "results/shap",
    num_background: int = 50,
    num_test_windows: int = 20,
    target_timestep: Optional[int] = None,
    target_channel: Optional[int] = None,
    channel_names: Optional[list] = None,
    device: Union[str, torch.device] = "cpu",
) -> pathlib.Path:
    """
    Compute SHAP values for a sample of test windows and save summary and
    bar plots.

    Args:
        model: A trained `PhySATFormer` or `BaselineTransformer` instance
            (already loaded with trained weights -- this function does
            not load checkpoints itself; load weights beforehand, e.g.
            via `Evaluator.load_checkpoint`).
        test_loader: `DataLoader` wrapping the test-split `MissionDataset`.
        output_dir: Directory `shap_summary.png` and `shap_bar.png` are
            saved into.
        num_background: Number of windows used as the SHAP background
            (reference) distribution. Kept small and configurable since
            `GradientExplainer`'s cost scales with this.
        num_test_windows: Number of test windows to explain.
        target_timestep: If given, explain only this timestep's logits
            (averaged over channels). Mutually composable with
            `target_channel`.
        target_channel: If given, explain only this channel's logits
            (averaged over timesteps if `target_timestep` is also None).
        channel_names: Optional list of length `num_channels` used to
            label the channel axis in the summary plot. Defaults to
            `channel_0`, `channel_1`, ...
        device: Device to run the explainer on.

    Returns:
        The resolved `output_dir` the two PNGs were saved into.
    """
    resolved_dir = pathlib.Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    model = model.to(torch_device).eval()

    wrapped_model = _ScalarOutputWrapper(
        model, target_timestep=target_timestep, target_channel=target_channel
    ).to(torch_device)

    background, test_windows = _sample_background_and_test(
        test_loader, num_background, num_test_windows, torch_device
    )

    logger.info(
        "Computing SHAP values: %d background windows, %d explained windows.",
        background.shape[0],
        test_windows.shape[0],
    )

    explainer = shap.GradientExplainer(wrapped_model, background)
    shap_values = explainer.shap_values(test_windows)

    # GradientExplainer returns a list (one array per output) for
    # multi-output models; our wrapper has exactly one scalar output.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)  # (N, T, C[, 1])
    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]

    # Reduce the temporal axis (mean |SHAP| over time) to get one
    # importance value per channel per sample, matching the (N, C) shape
    # SHAP's summary/bar plots expect.
    channel_shap_values = shap_values.mean(axis=1)  # (N, C)
    channel_inputs = test_windows.detach().cpu().numpy().mean(axis=1)  # (N, C)

    num_channels = channel_shap_values.shape[1]
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(num_channels)]

    # --- Summary plot ---
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        channel_shap_values,
        channel_inputs,
        feature_names=channel_names,
        show=False,
    )
    summary_path = resolved_dir / "shap_summary.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP summary plot to '%s'.", summary_path)

    # --- Bar plot ---
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        channel_shap_values,
        channel_inputs,
        feature_names=channel_names,
        plot_type="bar",
        show=False,
    )
    bar_path = resolved_dir / "shap_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP bar plot to '%s'.", bar_path)

    return resolved_dir
