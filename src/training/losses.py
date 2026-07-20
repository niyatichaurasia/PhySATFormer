"""Loss functions for training PhySATFormer.

This module defines the loss function(s) used to train PhySATFormer for
channel-level spatio-temporal anomaly localization. The model produces raw
(pre-sigmoid) per-timestep, per-channel anomaly logits of shape
``(batch_size, sequence_length, num_channels)``, where every element is an
independent binary prediction (i.e. "is channel c anomalous at timestep t?").

The loss implemented here, :class:`PhySATLoss`, wraps
``torch.nn.BCEWithLogitsLoss`` and is intentionally the *only* place where
sigmoid-related numerics are handled: the model must never apply a sigmoid
to its output before this loss consumes it, since ``BCEWithLogitsLoss``
combines the sigmoid and the binary cross-entropy computation into a single,
numerically stable operation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PhySATLoss(nn.Module):
    """Binary cross-entropy loss for per-channel, per-timestep anomaly logits.

    This module wraps :class:`torch.nn.BCEWithLogitsLoss` to train
    PhySATFormer on its channel-level spatio-temporal anomaly localization
    task. Predictions are expected to be *raw logits* (i.e. no sigmoid
    applied); the sigmoid is applied internally, in a numerically stable
    fashion, by ``BCEWithLogitsLoss``.

    Both predictions and targets are expected to share the shape
    ``(batch_size, sequence_length, num_channels)``, with every element
    representing an independent binary anomaly label/prediction for a
    given (timestep, channel) pair.

    Parameters
    ----------
    reduction : str, optional
        Reduction mode applied to the per-element loss. Must be one of
        ``"mean"`` or ``"sum"``, by default ``"mean"``. ``"mean"`` returns
        the average loss over all elements; ``"sum"`` returns the total
        loss over all elements.
    pos_weight : torch.Tensor or None, optional
        Optional per-channel positive-class weight, forwarded unchanged to
        ``BCEWithLogitsLoss`` to counteract class imbalance (anomalies are
        typically rare). If provided, must be broadcastable against the
        trailing ``num_channels`` dimension of the input, by default None.

    Attributes
    ----------
    reduction : str
        The configured reduction mode (``"mean"`` or ``"sum"``).
    criterion : torch.nn.BCEWithLogitsLoss
        The underlying binary cross-entropy-with-logits loss module.

    Examples
    --------
    >>> import torch
    >>> loss_fn = PhySATLoss()
    >>> predictions = torch.randn(8, 32, 5)  # raw logits
    >>> targets = torch.randint(0, 2, (8, 32, 5)).float()
    >>> loss = loss_fn(predictions, targets)
    >>> loss.shape
    torch.Size([])
    """

    _VALID_REDUCTIONS = ("mean", "sum")

    def __init__(
        self,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        if not isinstance(reduction, str):
            raise TypeError(
                f"`reduction` must be a str, got {type(reduction).__name__}."
            )
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                "`reduction` must be one of "
                f"{self._VALID_REDUCTIONS}, got {reduction!r}."
            )

        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            raise TypeError(
                "`pos_weight` must be a torch.Tensor or None, got "
                f"{type(pos_weight).__name__}."
            )

        self.reduction = reduction

        self.criterion = nn.BCEWithLogitsLoss(
            reduction=reduction,
            pos_weight=pos_weight,
        )

    @staticmethod
    def _validate_inputs(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Validate the shapes, dtypes, and types of `predictions`/`targets`.

        Parameters
        ----------
        predictions : torch.Tensor
            Raw (pre-sigmoid) anomaly logits of shape
            ``(batch_size, sequence_length, num_channels)``.
        targets : torch.Tensor
            Ground-truth binary labels of shape
            ``(batch_size, sequence_length, num_channels)``.

        Raises
        ------
        TypeError
            If `predictions` or `targets` is not a ``torch.Tensor``, or if
            either has a non-floating-point dtype.
        ValueError
            If `predictions` and `targets` do not have matching shapes, or
            if either is not 3-D.
        """
        if not isinstance(predictions, torch.Tensor):
            raise TypeError(
                "`predictions` must be a torch.Tensor, got "
                f"{type(predictions).__name__}."
            )
        if not isinstance(targets, torch.Tensor):
            raise TypeError(
                f"`targets` must be a torch.Tensor, got {type(targets).__name__}."
            )

        if predictions.dim() != 3:
            raise ValueError(
                "Expected `predictions` of shape (batch_size, "
                "sequence_length, num_channels), got tensor with shape "
                f"{tuple(predictions.shape)}."
            )
        if targets.dim() != 3:
            raise ValueError(
                "Expected `targets` of shape (batch_size, sequence_length, "
                f"num_channels), got tensor with shape {tuple(targets.shape)}."
            )

        if predictions.shape != targets.shape:
            raise ValueError(
                "`predictions` and `targets` must have the same shape, got "
                f"{tuple(predictions.shape)} and {tuple(targets.shape)}."
            )

        if not predictions.dtype.is_floating_point:
            raise TypeError(
                "`predictions` must have a floating-point dtype, got "
                f"{predictions.dtype}."
            )
        if not targets.dtype.is_floating_point:
            raise TypeError(
                f"`targets` must have a floating-point dtype, got {targets.dtype}."
            )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the binary cross-entropy-with-logits loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Raw (pre-sigmoid) anomaly logits produced by PhySATFormer, of
            shape ``(batch_size, sequence_length, num_channels)``. Sigmoid
            must NOT be applied to this tensor before calling this method.
        targets : torch.Tensor
            Ground-truth binary anomaly labels (0.0 or 1.0) of shape
            ``(batch_size, sequence_length, num_channels)``.

        Returns
        -------
        torch.Tensor
            A scalar tensor holding the reduced loss, suitable for calling
            ``.backward()`` on directly.

        Raises
        ------
        TypeError
            If `predictions` or `targets` is not a ``torch.Tensor``, or if
            either has a non-floating-point dtype.
        ValueError
            If `predictions` and `targets` do not have matching shapes, or
            if either is not 3-D.
        """
        self._validate_inputs(predictions, targets)

        targets = targets.to(dtype=predictions.dtype, device=predictions.device)

        loss = self.criterion(predictions, targets)

        return loss