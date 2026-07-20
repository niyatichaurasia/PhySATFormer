"""Checkpoint management for PhySATFormer training.

This module implements :class:`CheckpointManager`, a small, self-contained
utility responsible for persisting and restoring training state (model
parameters, optimizer state, learning-rate scheduler state, and training
metadata such as epoch and metric) to and from disk.

This module intentionally contains no training-loop logic (no forward
passes, no loss computation, no epoch iteration). Its sole responsibility
is serialization and deserialization of checkpoint artifacts via
``torch.save`` / ``torch.load``.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CheckpointManager:
    """Save and load PhySATFormer training checkpoints.

    This class manages a single directory on disk into which training
    checkpoints are written and from which they may later be read. Each
    checkpoint bundles the model's parameters, the optimizer's state, the
    (optional) learning-rate scheduler's state, and a small amount of
    training metadata (epoch number and an arbitrary evaluation metric).

    This class performs no training-loop logic itself; it is strictly
    responsible for checkpoint serialization and deserialization.

    Parameters
    ----------
    checkpoint_dir : str or pathlib.Path
        Directory in which checkpoints will be saved and from which they
        will be loaded. Created (including any missing parent
        directories) if it does not already exist.

    Attributes
    ----------
    checkpoint_dir : pathlib.Path
        Resolved directory used for checkpoint storage.

    Examples
    --------
    >>> manager = CheckpointManager("checkpoints/physatformer")
    >>> path = manager.save_checkpoint(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     scheduler=scheduler,
    ...     epoch=3,
    ...     metric=0.912,
    ... )
    >>> state = manager.load_checkpoint(
    ...     checkpoint_path=path,
    ...     model=model,
    ...     optimizer=optimizer,
    ...     scheduler=scheduler,
    ... )
    >>> state["epoch"]
    3
    """

    #: Default filename template used when `filename` is not provided to
    #: :meth:`save_checkpoint`. ``{epoch}`` is substituted with the
    #: integer epoch number.
    _DEFAULT_FILENAME_TEMPLATE: str = "checkpoint_epoch_{epoch}.pt"

    def __init__(self, checkpoint_dir: Union[str, pathlib.Path]) -> None:
        if not isinstance(checkpoint_dir, (str, pathlib.Path)):
            raise TypeError(
                "`checkpoint_dir` must be a str or pathlib.Path, got "
                f"{type(checkpoint_dir).__name__}."
            )

        self.checkpoint_dir: pathlib.Path = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_model(model: nn.Module) -> None:
        """Validate that `model` is an ``nn.Module``.

        Parameters
        ----------
        model : nn.Module
            Candidate model instance.

        Raises
        ------
        TypeError
            If `model` is not an ``nn.Module``.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"`model` must be a torch.nn.Module, got {type(model).__name__}."
            )

    @staticmethod
    def _validate_optimizer(optimizer: Optional[Optimizer]) -> None:
        """Validate that `optimizer`, if provided, is an ``Optimizer``.

        Parameters
        ----------
        optimizer : Optimizer or None
            Candidate optimizer instance.

        Raises
        ------
        TypeError
            If `optimizer` is provided but is not a
            ``torch.optim.Optimizer``.
        """
        if optimizer is not None and not isinstance(optimizer, Optimizer):
            raise TypeError(
                "`optimizer` must be a torch.optim.Optimizer, got "
                f"{type(optimizer).__name__}."
            )

    @staticmethod
    def _validate_scheduler(scheduler: Optional[LRScheduler]) -> None:
        """Validate that `scheduler`, if provided, is an ``LRScheduler``.

        Parameters
        ----------
        scheduler : LRScheduler or None
            Candidate learning-rate scheduler instance.

        Raises
        ------
        TypeError
            If `scheduler` is provided but is not a
            ``torch.optim.lr_scheduler.LRScheduler``.
        """
        if scheduler is not None and not isinstance(scheduler, LRScheduler):
            raise TypeError(
                "`scheduler` must be a torch.optim.lr_scheduler.LRScheduler, "
                f"got {type(scheduler).__name__}."
            )

    @staticmethod
    def _validate_epoch(epoch: int) -> None:
        """Validate that `epoch` is a non-negative integer.

        Parameters
        ----------
        epoch : int
            Candidate epoch number.

        Raises
        ------
        TypeError
            If `epoch` is not an ``int``.
        ValueError
            If `epoch` is negative.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise TypeError(f"`epoch` must be an int, got {type(epoch).__name__}.")
        if epoch < 0:
            raise ValueError(f"`epoch` must be non-negative, got {epoch}.")

    def _resolve_checkpoint_path(self, checkpoint_path: Union[str, pathlib.Path]) -> pathlib.Path:
        """Validate and resolve a checkpoint path prior to loading.

        Parameters
        ----------
        checkpoint_path : str or pathlib.Path
            Candidate path to an existing checkpoint file.

        Returns
        -------
        pathlib.Path
            The resolved checkpoint path.

        Raises
        ------
        TypeError
            If `checkpoint_path` is not a ``str`` or ``pathlib.Path``.
        FileNotFoundError
            If no file exists at the resolved path.
        """
        if not isinstance(checkpoint_path, (str, pathlib.Path)):
            raise TypeError(
                "`checkpoint_path` must be a str or pathlib.Path, got "
                f"{type(checkpoint_path).__name__}."
            )

        resolved_path = pathlib.Path(checkpoint_path)

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"No checkpoint file found at '{resolved_path}'."
            )

        return resolved_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        epoch: int,
        metric: Any,
        filename: Optional[str] = None,
    ) -> pathlib.Path:
        """Save a training checkpoint to disk.

        Serializes the model's parameters, the optimizer's state, the
        (optional) scheduler's state, the current epoch, and an arbitrary
        evaluation metric into a single file within :attr:`checkpoint_dir`.

        Parameters
        ----------
        model : nn.Module
            Model whose ``state_dict()`` will be saved.
        optimizer : Optimizer
            Optimizer whose ``state_dict()`` will be saved.
        scheduler : LRScheduler or None
            Learning-rate scheduler whose ``state_dict()`` will be saved,
            if provided. If ``None``, no scheduler state is saved.
        epoch : int
            Current (completed) epoch number. Must be non-negative.
        metric : Any
            Evaluation metric (e.g. validation loss or accuracy)
            associated with this checkpoint. Stored as-is; no type
            constraints are imposed.
        filename : str, optional
            Name of the checkpoint file to create within
            :attr:`checkpoint_dir`. If not provided, defaults to
            ``"checkpoint_epoch_{epoch}.pt"``.

        Returns
        -------
        pathlib.Path
            The full path of the saved checkpoint file.

        Raises
        ------
        TypeError
            If `model` is not an ``nn.Module``, `optimizer` is not an
            ``Optimizer``, `scheduler` is provided but is not an
            ``LRScheduler``, `epoch` is not an ``int``, or `filename` is
            provided but is not a ``str``.
        ValueError
            If `epoch` is negative or `filename` is an empty string.
        """
        self._validate_model(model)
        self._validate_optimizer(optimizer)
        self._validate_scheduler(scheduler)
        self._validate_epoch(epoch)

        if filename is not None:
            if not isinstance(filename, str):
                raise TypeError(
                    f"`filename` must be a str, got {type(filename).__name__}."
                )
            if filename.strip() == "":
                raise ValueError("`filename` must not be empty.")
        else:
            filename = self._DEFAULT_FILENAME_TEMPLATE.format(epoch=epoch)

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "epoch": epoch,
            "metric": metric,
        }

        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, pathlib.Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Load a training checkpoint from disk.

        Restores the model's parameters in-place via
        ``model.load_state_dict()``, and, if provided, restores the
        optimizer's and scheduler's state in-place as well.

        Parameters
        ----------
        checkpoint_path : str or pathlib.Path
            Path to an existing checkpoint file previously produced by
            :meth:`save_checkpoint`.
        model : nn.Module
            Model into which the saved parameters will be loaded.
        optimizer : Optimizer, optional
            Optimizer into which the saved optimizer state will be
            loaded, by default None. If ``None``, optimizer state is not
            restored, even if present in the checkpoint.
        scheduler : LRScheduler, optional
            Learning-rate scheduler into which the saved scheduler state
            will be loaded, by default None. If ``None``, scheduler
            state is not restored, even if present in the checkpoint.
        device : str or torch.device, optional
            Device onto which tensors in the checkpoint are mapped, by
            default None (uses ``torch.load``'s default map location,
            i.e. the device(s) tensors were saved from).

        Returns
        -------
        dict
            Dictionary containing at least the keys ``"epoch"`` and
            ``"metric"``, plus any other top-level metadata present in
            the checkpoint (excluding the state-dict entries that were
            consumed to restore model/optimizer/scheduler state).

        Raises
        ------
        TypeError
            If `checkpoint_path` is not a ``str`` or ``pathlib.Path``,
            `model` is not an ``nn.Module``, `optimizer` is provided but
            is not an ``Optimizer``, or `scheduler` is provided but is
            not an ``LRScheduler``.
        FileNotFoundError
            If no file exists at `checkpoint_path`.
        KeyError
            If the checkpoint is missing a required ``"model_state_dict"``
            entry, or if `optimizer`/`scheduler` is provided but the
            corresponding state is absent from the checkpoint.
        """
        self._validate_model(model)
        self._validate_optimizer(optimizer)
        self._validate_scheduler(scheduler)

        resolved_path = self._resolve_checkpoint_path(checkpoint_path)

        checkpoint: Dict[str, Any] = torch.load(
            resolved_path, map_location=device
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint at '{resolved_path}' is missing required key "
                "'model_state_dict'."
            )
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            if (
                "optimizer_state_dict" not in checkpoint
                or checkpoint["optimizer_state_dict"] is None
            ):
                raise KeyError(
                    f"Checkpoint at '{resolved_path}' does not contain "
                    "optimizer state, but `optimizer` was provided."
                )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None:
            if (
                "scheduler_state_dict" not in checkpoint
                or checkpoint["scheduler_state_dict"] is None
            ):
                raise KeyError(
                    f"Checkpoint at '{resolved_path}' does not contain "
                    "scheduler state, but `scheduler` was provided."
                )
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        consumed_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        }
        metadata: Dict[str, Any] = {
            key: value
            for key, value in checkpoint.items()
            if key not in consumed_keys
        }

        if "epoch" not in metadata:
            raise KeyError(
                f"Checkpoint at '{resolved_path}' is missing required key "
                "'epoch'."
            )
        if "metric" not in metadata:
            raise KeyError(
                f"Checkpoint at '{resolved_path}' is missing required key "
                "'metric'."
            )

        return metadata