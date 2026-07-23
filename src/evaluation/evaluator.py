"""
src/evaluation/evaluator.py

Evaluator: loads a trained checkpoint and runs inference over a test
DataLoader, returning raw ground-truth labels, thresholded predictions,
and prediction probabilities.

This module deliberately contains NO plotting and NO metric computation.
Those responsibilities live in `src/evaluation/metrics.py` and
`src/evaluation/plots.py` respectively. `Evaluator` is a pure inference
utility that reuses the project's existing `CheckpointManager` for
checkpoint loading, so checkpoint file format never needs to be
duplicated or reinvented here.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Runs inference for an already-trained model over a test DataLoader.

    `Evaluator` is intentionally thin: it is responsible only for (1)
    restoring model weights from a checkpoint via the existing
    `CheckpointManager`, and (2) executing a no-grad forward pass over
    every batch of a `DataLoader` (typically wrapping the test-split
    `MissionDataset`), collecting logits and labels. It performs no
    plotting, no metric aggregation, and no file I/O for results --
    those are the responsibilities of `src/evaluation/metrics.py`,
    `src/evaluation/plots.py`, and `src/evaluation/prediction_export.py`.

    Batches are expected in the same `(window_batch, label_batch)` shape
    produced by a `DataLoader` wrapping `MissionDataset`, matching
    `Trainer._unpack_batch`'s contract.

    Attributes:
        model: The model (`PhySATFormer` or `BaselineTransformer`) to
            evaluate. Moved to `device` and set to `eval()` mode.
        device: The `torch.device` used for inference.
        threshold: Decision threshold applied to sigmoid probabilities
            to obtain binary predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[torch.device, str] = "cpu",
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            model: An already-constructed (but not necessarily
                weight-loaded) model instance. Weights are restored via
                `load_checkpoint`.
            device: Device to run inference on.
            threshold: Sigmoid decision threshold used to binarize
                probabilities into predicted labels. Must be in (0, 1).

        Raises:
            TypeError: If `model` is not an `nn.Module`.
            ValueError: If `threshold` is not in (0, 1).
        """
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"`model` must be a torch.nn.Module, got {type(model).__name__}."
            )
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"`threshold` must be in (0, 1), got {threshold!r}.")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.threshold = threshold

        self._checkpoint_manager: Optional[CheckpointManager] = None

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """
        Restore model weights from a checkpoint saved by
        `CheckpointManager.save_checkpoint`.

        Only the model's `state_dict` is restored; optimizer/scheduler
        state is irrelevant for evaluation and is intentionally not
        requested.

        Args:
            checkpoint_path: Path to a `.pt` checkpoint file (e.g.
                `checkpoints/physatformer/checkpoint_epoch_12.pt`, or
                whatever the training run's "best" checkpoint file is
                named).

        Returns:
            The checkpoint metadata dict (contains at least `"epoch"`
            and `"metric"`, when present in the checkpoint file).

        Note:
            This intentionally does NOT call
            `CheckpointManager.load_checkpoint(...)` with
            `optimizer=None, scheduler=None`. Every other call site in
            the codebase (`train.py`'s `resume_checkpoint`) always
            passes real, already-constructed optimizer/scheduler
            instances, so it is unverified whether
            `CheckpointManager.load_checkpoint` tolerates `None` for
            those arguments (e.g. it may unconditionally call
            `optimizer.load_state_dict(...)`). Since evaluation only
            ever needs the model's weights, the checkpoint file is
            loaded directly here and only `model_state_dict` is
            restored, without exercising that code path at all.
        """
        import pathlib

        resolved_path = pathlib.Path(checkpoint_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: '{resolved_path}'.")

        checkpoint = torch.load(resolved_path, map_location=self.device)

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint '{resolved_path}' does not contain a "
                "'model_state_dict' key."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        metadata = {
            "epoch": checkpoint.get("epoch"),
            "metric": checkpoint.get("metric") or checkpoint.get("best_metric"),
        }

        logger.info(
            "Loaded checkpoint '%s' (epoch=%s, metric=%s).",
            resolved_path,
            metadata.get("epoch"),
            metadata.get("metric"),
        )
        return metadata

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self, test_loader: DataLoader
    ) -> "EvaluationResult":
        """
        Run a full no-grad inference pass over `test_loader`.

        Args:
            test_loader: A `DataLoader` (typically wrapping the test
                split `MissionDataset`) yielding `(window, label)`
                batches, matching `Trainer`'s batch contract.

        Returns:
            An `EvaluationResult` bundling `ground_truth`,
            `predictions`, and `probabilities`, each a flat 1-D
            `np.ndarray` (all batches, timesteps, and channels
            concatenated together), suitable for direct consumption by
            `sklearn.metrics` functions.
        """
        self.model.eval()

        all_targets = []
        all_probabilities = []

        for batch_idx, batch in enumerate(test_loader):
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                raise ValueError(
                    "Expected a batch of (window, label) as produced by "
                    "MissionDataset; got "
                    f"{type(batch).__name__} of length "
                    f"{len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}."
                )
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(inputs)
            probabilities = torch.sigmoid(logits)

            all_targets.append(targets.detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())

            if batch_idx % 200 == 0:
                logger.debug("Evaluated batch %d/%d", batch_idx, len(test_loader))

        targets_tensor = torch.cat(all_targets, dim=0)
        probabilities_tensor = torch.cat(all_probabilities, dim=0)
        predictions_tensor = (probabilities_tensor >= self.threshold).float()

        ground_truth = targets_tensor.numpy().reshape(-1)
        predictions = predictions_tensor.numpy().reshape(-1)
        prediction_probabilities = probabilities_tensor.numpy().reshape(-1)

        logger.info(
            "Evaluation complete: %d total (window, timestep, channel) "
            "predictions.",
            ground_truth.shape[0],
        )

        return EvaluationResult(
            ground_truth=ground_truth,
            predictions=predictions,
            probabilities=prediction_probabilities,
        )


class EvaluationResult:
    """
    Simple container for the outputs of `Evaluator.evaluate`.

    Kept as a plain container (rather than a dict) so downstream
    consumers (metrics.py, prediction_export.py) get attribute access
    and basic shape validation, without pulling in a dependency like
    `dataclasses.dataclass` unnecessarily deep into typed contracts.

    Attributes:
        ground_truth: Flat `np.ndarray` of binary ground-truth labels.
        predictions: Flat `np.ndarray` of binary (thresholded)
            predictions.
        probabilities: Flat `np.ndarray` of sigmoid probabilities in
            `[0, 1]`, index-aligned with `ground_truth`/`predictions`.
    """

    def __init__(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        if not (ground_truth.shape == predictions.shape == probabilities.shape):
            raise ValueError(
                "ground_truth, predictions, and probabilities must share the "
                f"same shape; got {ground_truth.shape}, {predictions.shape}, "
                f"{probabilities.shape}."
            )
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.probabilities = probabilities
