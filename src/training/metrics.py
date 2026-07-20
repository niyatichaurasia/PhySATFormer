"""Evaluation metrics for PhySATFormer channel-level anomaly localization.

This module implements :class:`PhySATMetrics`, a stateless PyTorch module
that converts raw per-timestep, per-channel anomaly logits into binary
predictions and computes both overall (micro-averaged, flattened across
batch, time, and channel axes) and channel-wise precision, recall, and
F1-score, along with the underlying confusion-matrix counts.

No external metric libraries (e.g. scikit-learn) are used; all
computation is implemented directly with PyTorch tensor operations so
that the metrics can be computed on-device (CPU or GPU) without any
host-device synchronization beyond what is required to return Python
scalars.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class PhySATMetrics(nn.Module):
    """Compute precision, recall, and F1-score for channel-level anomaly detection.

    Given raw (pre-sigmoid) logits and binary ground-truth labels of
    shape ``(batch_size, sequence_length, num_channels)``, this module:

    1. Converts logits to probabilities via :func:`torch.sigmoid`.
    2. Thresholds probabilities into binary predictions using a
       configurable ``threshold`` (default ``0.5``).
    3. Computes true positives (TP), false positives (FP), true
       negatives (TN), and false negatives (FN), both:

       - **Overall**: flattened across every element of the batch
         (i.e. micro-averaged across batch, sequence, and channel
         axes).
       - **Channel-wise**: independently for each of the
         ``num_channels`` telemetry channels, flattened across the
         batch and sequence axes.

    4. Derives precision, recall, and F1-score from these counts, both
       overall and per-channel.

    This module holds no learnable parameters and no persistent state;
    each call to :meth:`forward` is a pure function of its inputs.

    Parameters
    ----------
    threshold : float, optional
        Probability threshold above which a prediction is considered
        positive (anomalous), by default ``0.5``. Must lie in the
        closed interval ``[0, 1]``.
    eps : float, optional
        Small constant added to denominators to avoid division by zero
        when a class has no predicted or actual positives, by default
        ``1e-8``.

    Attributes
    ----------
    threshold : float
        Configured decision threshold.
    eps : float
        Configured numerical-stability epsilon.

    Examples
    --------
    >>> import torch
    >>> metrics = PhySATMetrics(threshold=0.5)
    >>> logits = torch.randn(4, 10, 5)
    >>> targets = torch.randint(0, 2, (4, 10, 5)).float()
    >>> results = metrics(logits, targets)
    >>> sorted(results.keys())
    ['channel_f1', 'channel_precision', 'channel_recall', 'false_negatives', 'false_positives', 'overall_f1', 'overall_precision', 'overall_recall', 'true_negatives', 'true_positives']
    >>> results["overall_f1"].shape
    torch.Size([])
    >>> results["channel_f1"].shape
    torch.Size([5])
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-8) -> None:
        super().__init__()

        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError(
                f"`threshold` must be a float, got {type(threshold).__name__}."
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                f"`threshold` must lie in [0, 1], got {threshold!r}."
            )

        if not isinstance(eps, (int, float)) or isinstance(eps, bool):
            raise TypeError(f"`eps` must be a float, got {type(eps).__name__}.")
        if float(eps) <= 0.0:
            raise ValueError(f"`eps` must be positive, got {eps!r}.")

        self.threshold: float = float(threshold)
        self.eps: float = float(eps)

    @staticmethod
    def _validate_inputs(logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Validate shapes, dtypes, and value ranges of the inputs.

        Parameters
        ----------
        logits : torch.Tensor
            Raw (pre-sigmoid) prediction logits of shape
            ``(batch_size, sequence_length, num_channels)``.
        targets : torch.Tensor
            Binary ground-truth labels of shape
            ``(batch_size, sequence_length, num_channels)``.

        Raises
        ------
        TypeError
            If `logits` or `targets` is not a ``torch.Tensor``, if
            `logits` is not a floating-point tensor, or if `targets`
            is not binary-valued.
        ValueError
            If `logits` and `targets` do not have the same shape, or if
            either tensor is not 3-D.
        """
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"`logits` must be a torch.Tensor, got {type(logits).__name__}."
            )
        if not isinstance(targets, torch.Tensor):
            raise TypeError(
                f"`targets` must be a torch.Tensor, got {type(targets).__name__}."
            )

        if not torch.is_floating_point(logits):
            raise TypeError(
                "`logits` must be a floating-point tensor, got dtype "
                f"{logits.dtype}."
            )

        if logits.dim() != 3:
            raise ValueError(
                "Expected `logits` of shape (batch_size, sequence_length, "
                f"num_channels), got tensor with shape {tuple(logits.shape)}."
            )
        if targets.dim() != 3:
            raise ValueError(
                "Expected `targets` of shape (batch_size, sequence_length, "
                f"num_channels), got tensor with shape {tuple(targets.shape)}."
            )

        if logits.shape != targets.shape:
            raise ValueError(
                "`logits` and `targets` must have matching shapes, got "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}."
            )

        unique_values = torch.unique(targets)
        if not torch.all(
            (unique_values == 0.0) | (unique_values == 1.0)
        ):
            raise ValueError(
                "`targets` must be binary (containing only 0 and 1 values), "
                f"got unique values {unique_values.tolist()}."
            )

    @staticmethod
    def _precision_recall_f1(
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        false_negatives: torch.Tensor,
        eps: float,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Derive precision, recall, and F1-score from confusion counts.

        Parameters
        ----------
        true_positives : torch.Tensor
            Count(s) of true positives.
        false_positives : torch.Tensor
            Count(s) of false positives.
        false_negatives : torch.Tensor
            Count(s) of false negatives.
        eps : float
            Small constant added to denominators to avoid division by
            zero.

        Returns
        -------
        precision : torch.Tensor
            Precision, elementwise over the input tensors.
        recall : torch.Tensor
            Recall, elementwise over the input tensors.
        f1 : torch.Tensor
            F1-score, elementwise over the input tensors.
        """
        precision = true_positives / (true_positives + false_positives + eps)
        recall = true_positives / (true_positives + false_negatives + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)

        return precision, recall, f1

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute overall and channel-wise detection metrics.

        Parameters
        ----------
        logits : torch.Tensor
            Raw (pre-sigmoid) prediction logits of shape
            ``(batch_size, sequence_length, num_channels)``.
        targets : torch.Tensor
            Binary ground-truth labels of shape
            ``(batch_size, sequence_length, num_channels)``, containing
            only ``0`` and ``1`` values.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with the following keys:

            - ``"overall_precision"`` : 0-D tensor, micro-averaged
              precision across all elements.
            - ``"overall_recall"`` : 0-D tensor, micro-averaged recall
              across all elements.
            - ``"overall_f1"`` : 0-D tensor, micro-averaged F1-score
              across all elements.
            - ``"channel_precision"`` : tensor of shape
              ``(num_channels,)``, precision computed independently
              per channel.
            - ``"channel_recall"`` : tensor of shape
              ``(num_channels,)``, recall computed independently per
              channel.
            - ``"channel_f1"`` : tensor of shape ``(num_channels,)``,
              F1-score computed independently per channel.
            - ``"true_positives"`` : 0-D tensor, overall true positive
              count.
            - ``"false_positives"`` : 0-D tensor, overall false
              positive count.
            - ``"true_negatives"`` : 0-D tensor, overall true negative
              count.
            - ``"false_negatives"`` : 0-D tensor, overall false
              negative count.

        Raises
        ------
        TypeError
            If `logits` or `targets` is not a ``torch.Tensor``, if
            `logits` is not floating-point, or if `targets` is not
            binary-valued.
        ValueError
            If `logits` and `targets` do not share the same shape, or
            if either is not 3-D.
        """
        self._validate_inputs(logits, targets)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= self.threshold).to(dtype=logits.dtype)
        targets = targets.to(dtype=logits.dtype)

        is_true_positive = predictions * targets
        is_false_positive = predictions * (1.0 - targets)
        is_true_negative = (1.0 - predictions) * (1.0 - targets)
        is_false_negative = (1.0 - predictions) * targets

        # Overall (micro-averaged) counts, flattened across every
        # element of the batch, sequence, and channel axes.
        overall_true_positives = is_true_positive.sum()
        overall_false_positives = is_false_positive.sum()
        overall_true_negatives = is_true_negative.sum()
        overall_false_negatives = is_false_negative.sum()

        overall_precision, overall_recall, overall_f1 = self._precision_recall_f1(
            overall_true_positives,
            overall_false_positives,
            overall_false_negatives,
            self.eps,
        )

        # Channel-wise counts: reduce over the batch (dim=0) and
        # sequence (dim=1) axes, leaving one value per channel.
        channel_true_positives = is_true_positive.sum(dim=(0, 1))
        channel_false_positives = is_false_positive.sum(dim=(0, 1))
        channel_false_negatives = is_false_negative.sum(dim=(0, 1))

        channel_precision, channel_recall, channel_f1 = self._precision_recall_f1(
            channel_true_positives,
            channel_false_positives,
            channel_false_negatives,
            self.eps,
        )

        return {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "channel_precision": channel_precision,
            "channel_recall": channel_recall,
            "channel_f1": channel_f1,
            "true_positives": overall_true_positives,
            "false_positives": overall_false_positives,
            "true_negatives": overall_true_negatives,
            "false_negatives": overall_false_negatives,
        }