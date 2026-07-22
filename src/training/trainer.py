"""
src/training/trainer.py

Core training engine for the PhySATFormer project.

This module implements the `Trainer` class, which orchestrates the full
training and validation lifecycle of a PhySATFormer model. The `Trainer`
is a thin orchestration layer: it does NOT reimplement any domain logic
that already lives in the following collaborator modules:

    * PhySATLoss            (src/training/losses.py)       -> loss computation
    * PhySATMetrics         (src/training/metrics.py)       -> precision / recall / f1
    * OptimizerFactory                                      -> optimizer construction
                              (the constructed `Optimizer` instance is injected here)
    * SchedulerFactory                                      -> scheduler construction
                              (the constructed `LRScheduler` instance is injected here)
    * CheckpointManager     (src/training/checkpoint.py)    -> checkpoint persistence
    * EarlyStopping         (src/training/early_stopping.py) -> early-stopping logic

Batches are expected to come from `MissionDataset`
(src/preprocessing/dataset.py), whose `__getitem__` returns
`(window, label)` pairs; the default `DataLoader` collation then
yields batches shaped as `(window_batch, label_batch)`.

The `Trainer` is intentionally free of mixed-precision training, distributed
training, and logging-framework integrations. It relies on plain Python
`print` statements for epoch summaries, as specified.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.training.checkpoint import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.losses import PhySATLoss
from src.training.metrics import PhySATMetrics


class Trainer:
    """
    Orchestrates training and validation for a PhySATFormer model.

    The `Trainer` composes together an already-constructed model, loss
    function, metrics module, optimizer, scheduler, checkpoint manager,
    and early-stopping monitor. It is responsible only for the training
    loop mechanics (device placement, forward/backward passes, gradient
    clipping, metric aggregation, checkpointing triggers, and early-stopping
    triggers) -- not for the internal logic of any of its collaborators.

    Attributes:
        model: The PhySATFormer model (a `torch.nn.Module`) to train.
        loss_fn: A `PhySATLoss` instance used to compute the training and
            validation loss from raw logits and binary targets.
        metrics: A `PhySATMetrics` instance used to compute precision,
            recall, and F1 (and confusion-matrix counts) for a single
            batch of raw logits and binary targets. `PhySATMetrics` is
            stateless (a pure function per call), so the `Trainer`
            accumulates the per-batch confusion-matrix counts it returns
            and re-derives epoch-level precision/recall/F1 from the
            accumulated counts using `PhySATMetrics`'s own formula.
        optimizer: A constructed `torch.optim.Optimizer` (e.g. produced by
            `OptimizerFactory`).
        scheduler: A constructed `torch.optim.lr_scheduler.LRScheduler`
            (e.g. produced by `SchedulerFactory`). Its `.step()` is called
            once per epoch, after validation.
        checkpoint_manager: A `CheckpointManager` instance responsible for
            persisting checkpoints to disk via `save_checkpoint(...)`.
        early_stopping: An `EarlyStopping` instance responsible for
            deciding when training should terminate, monitored on
            validation F1 (constructed by the caller with `mode="max"`).
        device: The `torch.device` (or device string) to move data and the
            model to.
        gradient_clip_value: Maximum gradient norm used for gradient
            clipping. Defaults to 1.0. Set to a non-positive value to
            disable clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: PhySATLoss,
        metrics: PhySATMetrics,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        checkpoint_manager: CheckpointManager,
        early_stopping: EarlyStopping,
        device: torch.device | str,
        gradient_clip_value: float = 1.0,
    ) -> None:
        self._validate_constructor_args(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_manager=checkpoint_manager,
            early_stopping=early_stopping,
            device=device,
            gradient_clip_value=gradient_clip_value,
        )

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        self.gradient_clip_value = gradient_clip_value

        # Tracks the best validation F1 observed so far, used to decide
        # whether a new checkpoint should be saved.
        self._best_val_f1: float = -math.inf

    # ------------------------------------------------------------------ #
    # Construction-time validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_constructor_args(
        model: nn.Module,
        loss_fn: Any,
        metrics: Any,
        optimizer: Any,
        scheduler: Any,
        checkpoint_manager: Any,
        early_stopping: Any,
        device: Any,
        gradient_clip_value: float,
    ) -> None:
        """Validate all constructor arguments, raising `TypeError` /
        `ValueError` with descriptive messages on failure."""

        if not isinstance(model, nn.Module):
            raise TypeError(
                f"`model` must be a torch.nn.Module, got {type(model).__name__}."
            )

        if not isinstance(loss_fn, PhySATLoss):
            raise TypeError(
                f"`loss_fn` must be a PhySATLoss instance, got {type(loss_fn).__name__}."
            )

        if not isinstance(metrics, PhySATMetrics):
            raise TypeError(
                "`metrics` must be a PhySATMetrics instance, "
                f"got {type(metrics).__name__}."
            )

        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "`optimizer` must be a torch.optim.Optimizer instance, "
                f"got {type(optimizer).__name__}."
            )

        if not isinstance(scheduler, LRScheduler):
            raise TypeError(
                "`scheduler` must be a torch.optim.lr_scheduler.LRScheduler "
                f"instance, got {type(scheduler).__name__}."
            )

        if not isinstance(checkpoint_manager, CheckpointManager):
            raise TypeError(
                "`checkpoint_manager` must be a CheckpointManager instance, "
                f"got {type(checkpoint_manager).__name__}."
            )

        if not isinstance(early_stopping, EarlyStopping):
            raise TypeError(
                "`early_stopping` must be an EarlyStopping instance, "
                f"got {type(early_stopping).__name__}."
            )

        try:
            torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"`device` is not a valid torch device: {device!r}") from exc

        if not isinstance(gradient_clip_value, (int, float)):
            raise TypeError(
                "`gradient_clip_value` must be a number, "
                f"got {type(gradient_clip_value).__name__}."
            )

    # ------------------------------------------------------------------ #
    # Batch-level helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split a collated `MissionDataset` batch into `(inputs, targets)`.

        `MissionDataset.__getitem__` returns `(window, label)` pairs; the
        default `DataLoader` collation therefore yields batches shaped as
        `(window_batch, label_batch)`.

        Args:
            batch: A `(window_batch, label_batch)` tuple/list, as produced
                by a `DataLoader` wrapping a `MissionDataset` constructed
                with labels.

        Returns:
            A `(inputs, targets)` tuple of tensors, both of shape
            `(batch_size, sequence_length, num_channels)`.

        Raises:
            ValueError: if the batch does not contain exactly two
                elements (window, label).
        """
        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise ValueError(
                "Expected a batch of (window, label) as produced by "
                "MissionDataset with labels; got "
                f"{type(batch).__name__} of length "
                f"{len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}."
            )

        window, label = batch
        return window, label

    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a single tensor to `self.device`."""
        return tensor.to(self.device, non_blocking=True)

    def _clip_gradients(self) -> None:
        """Apply gradient-norm clipping if `gradient_clip_value` is positive."""
        if self.gradient_clip_value and self.gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_clip_value
            )

    # ------------------------------------------------------------------ #
    # Metric accumulation helpers
    # ------------------------------------------------------------------ #

    class _MetricAccumulator:
        """
        Accumulates confusion-matrix counts across batches and derives
        epoch-level precision/recall/F1 using `PhySATMetrics`'s own
        formula, since `PhySATMetrics` itself is stateless (one call per
        batch, no cross-batch state).
        """

        def __init__(self, metrics: PhySATMetrics) -> None:
            self._metrics = metrics
            self.running_loss: float = 0.0
            self.num_batches: int = 0
            self.true_positives: float = 0.0
            self.false_positives: float = 0.0
            self.false_negatives: float = 0.0

        def update(
            self,
            batch_loss: float,
            batch_metrics: Dict[str, torch.Tensor],
        ) -> None:
            self.running_loss += batch_loss
            self.num_batches += 1
            self.true_positives += batch_metrics["true_positives"].item()
            self.false_positives += batch_metrics["false_positives"].item()
            self.false_negatives += batch_metrics["false_negatives"].item()

        def compute(self) -> Dict[str, float]:
            if self.num_batches == 0:
                raise ValueError("No batches were accumulated.")

            eps = self._metrics.eps
            precision = self.true_positives / (
                self.true_positives + self.false_positives + eps
            )
            recall = self.true_positives / (
                self.true_positives + self.false_negatives + eps
            )
            f1 = (2.0 * precision * recall) / (precision + recall + eps)

            return {
                "loss": self.running_loss / self.num_batches,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    # ------------------------------------------------------------------ #
    # Core train / validate step
    # ------------------------------------------------------------------ #

    def _training_step(self, batch: Any) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Execute a single training step: forward, loss, backward, clip,
        optimizer step, zero_grad.

        Returns:
            A `(loss_value, batch_metrics)` tuple, where `loss_value` is
            the scalar training loss (float) for this batch and
            `batch_metrics` is the raw dict returned by `PhySATMetrics`.
        """
        inputs, targets = self._unpack_batch(batch)
        inputs = self._move_to_device(inputs)
        targets = self._move_to_device(targets)

        self.optimizer.zero_grad()

        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)

        loss.backward()
        self._clip_gradients()
        self.optimizer.step()

        with torch.no_grad():
            batch_metrics = self.metrics(predictions.detach(), targets)

        return loss.item(), batch_metrics

    @torch.no_grad()
    def _validation_step(self, batch: Any) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Execute a single validation step: forward and loss computation only
        (no gradients, no optimizer step).

        Returns:
            A `(loss_value, batch_metrics)` tuple, where `loss_value` is
            the scalar validation loss (float) for this batch and
            `batch_metrics` is the raw dict returned by `PhySATMetrics`.
        """
        inputs, targets = self._unpack_batch(batch)
        inputs = self._move_to_device(inputs)
        targets = self._move_to_device(targets)

        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)

        batch_metrics = self.metrics(predictions, targets)

        return loss.item(), batch_metrics

    # ------------------------------------------------------------------ #
    # Epoch-level public API
    # ------------------------------------------------------------------ #

    def train_epoch(self, train_loader: Iterable) -> Dict[str, float]:
        """
        Run one full training epoch over `train_loader`.

        The model is expected to already be in training mode (`fit()`
        guarantees this, but calling `train_epoch` directly will also set
        it defensively).

        Args:
            train_loader: An iterable (typically a `torch.utils.data.DataLoader`
                wrapping a `MissionDataset`) yielding
                `(window, label)` batches.

        Returns:
            A dict with keys: "loss", "precision", "recall", "f1", aggregated
            over the entire epoch.
        """
        self.model.train()
        accumulator = self._MetricAccumulator(self.metrics)

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 2000 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")

            batch_loss, batch_metrics = self._training_step(batch)

            accumulator.update(batch_loss, batch_metrics)

        return accumulator.compute()

    def validate_epoch(self, validation_loader: Iterable) -> Dict[str, float]:
        """
        Run one full validation epoch over `validation_loader`.

        Gradients are disabled for the entire epoch and the model is set to
        evaluation mode.

        Args:
            validation_loader: An iterable (typically a
                `torch.utils.data.DataLoader` wrapping a `MissionDataset`)
                yielding `(window, label)` batches.

        Returns:
            A dict with keys: "loss", "precision", "recall", "f1", aggregated
            over the entire epoch.
        """
        self.model.eval()
        accumulator = self._MetricAccumulator(self.metrics)

        with torch.no_grad():
            for batch in validation_loader:
                batch_loss, batch_metrics = self._validation_step(batch)
                accumulator.update(batch_loss, batch_metrics)

        return accumulator.compute()

    # ------------------------------------------------------------------ #
    # Checkpointing / summary helpers
    # ------------------------------------------------------------------ #

    def _maybe_save_checkpoint(self, val_f1: float, epoch: int) -> bool:
        """
        Save a checkpoint via `self.checkpoint_manager.save_checkpoint(...)`
        if `val_f1` is a new best value, using validation F1 as the
        monitored metric.

        Returns:
            True if a checkpoint was saved, False otherwise.
        """
        if val_f1 > self._best_val_f1:
            self._best_val_f1 = val_f1
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metric=val_f1,
            )
            return True
        return False

    @staticmethod
    def _print_epoch_summary(
        epoch: int,
        num_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        checkpoint_saved: bool,
    ) -> None:
        """Print a concise, human-readable epoch summary."""
        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"train_precision={train_metrics['precision']:.4f} "
            f"train_recall={train_metrics['recall']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
            + (" | checkpoint saved" if checkpoint_saved else "")
        )

    @staticmethod
    def _init_history() -> Dict[str, List[float]]:
        """Initialize an empty training-history dict with all required keys."""
        return {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "train_precision": [],
            "val_precision": [],
            "train_recall": [],
            "val_recall": [],
        }

    @staticmethod
    def _append_history(
        history: Dict[str, List[float]],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Append one epoch's metrics onto the running `history` dict."""
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])

    # ------------------------------------------------------------------ #
    # Top-level training loop
    # ------------------------------------------------------------------ #
    def set_best_validation_f1(self, best_val_f1: float) -> None:
        """
        Restore the best validation F1 from a checkpoint when resuming training.
        """
        self._best_val_f1 = best_val_f1


    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        num_epochs: int,
        start_epoch: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Run the full training loop for up to `num_epochs` epochs.

        For each epoch:
            1. Set the model to training mode and run `train_epoch`.
            2. Set the model to evaluation mode and run `validate_epoch`.
            3. Step the learning-rate scheduler.
            4. Save a checkpoint whenever validation F1 improves.
            5. Update the early-stopping monitor on validation F1.
            6. Print an epoch summary.
            7. Stop early if `early_stopping.should_stop` becomes True.

        Args:
            train_loader: DataLoader (wrapping a `MissionDataset`) yielding
                training batches.
            validation_loader: DataLoader (wrapping a `MissionDataset`)
                yielding validation batches.
            num_epochs: Maximum number of epochs to train for.

        Returns:
            A history dict with keys:
                "train_loss", "val_loss", "train_f1", "val_f1",
                "train_precision", "val_precision",
                "train_recall", "val_recall"
            Each value is a list of per-epoch floats, in epoch order.
        """
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError(f"`num_epochs` must be a positive int, got {num_epochs!r}.")

        history = self._init_history()

        for epoch in range(start_epoch, num_epochs + 1):
            self.model.train()
            train_metrics = self.train_epoch(train_loader)

            self.model.eval()
            val_metrics = self.validate_epoch(validation_loader)

            self.scheduler.step()

            checkpoint_saved = self._maybe_save_checkpoint(val_metrics["f1"], epoch)

            self.early_stopping.update(val_metrics["f1"])

            self._append_history(history, train_metrics, val_metrics)
            self._print_epoch_summary(
                epoch, num_epochs, train_metrics, val_metrics, checkpoint_saved
            )

            if self.early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        return history