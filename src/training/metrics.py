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

from typing import Dict, Optional

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

    @staticmethod
    def _confusion_counts(
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        threshold: float,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """Compute overall (flattened) TP/FP/TN/FN counts at a given threshold.

        Shared by :meth:`forward` (at the configured ``self.threshold``)
        and by :meth:`sweep_thresholds` (at each candidate threshold in a
        grid), so the exact same counting logic is used in both places.

        Parameters
        ----------
        probabilities : torch.Tensor
            Post-sigmoid probabilities, any shape.
        targets : torch.Tensor
            Binary ground-truth labels, same shape as `probabilities`.
        threshold : float
            Decision threshold in ``[0, 1]``.

        Returns
        -------
        tuple of torch.Tensor
            ``(true_positives, false_positives, true_negatives,
            false_negatives)``, each a 0-D tensor.
        """
        predictions = (probabilities >= threshold).to(dtype=probabilities.dtype)
        targets = targets.to(dtype=probabilities.dtype)

        true_positives = (predictions * targets).sum()
        false_positives = (predictions * (1.0 - targets)).sum()
        true_negatives = ((1.0 - predictions) * (1.0 - targets)).sum()
        false_negatives = ((1.0 - predictions) * targets).sum()

        return true_positives, false_positives, true_negatives, false_negatives

    def sweep_thresholds(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute overall precision/recall/F1 at each of several thresholds.

        Intended for POST-HOC threshold calibration on a validation set:
        rather than trusting a fixed ``self.threshold`` (default 0.5),
        this sweeps a grid of candidate thresholds so the operating
        point can be chosen from data (e.g. the threshold maximizing
        validation F1), then reused unchanged at test time.

        Parameters
        ----------
        logits : torch.Tensor
            Raw (pre-sigmoid) logits, any shape. Sigmoid is applied
            once, internally, before sweeping.
        targets : torch.Tensor
            Binary ground-truth labels, same shape as `logits`.
        thresholds : torch.Tensor, optional
            1-D tensor of candidate thresholds in ``[0, 1]``. Defaults
            to ``torch.linspace(0.01, 0.99, 99)``.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``"thresholds"``: the thresholds swept, shape ``(K,)``.
            ``"precision"``, ``"recall"``, ``"f1"``: overall
            (micro-averaged) scores at each threshold, shape ``(K,)``.
        """
        if thresholds is None:
            thresholds = torch.linspace(0.01, 0.99, 99)

        probabilities = torch.sigmoid(logits).detach()
        targets = targets.to(dtype=probabilities.dtype)

        precisions = torch.empty(len(thresholds))
        recalls = torch.empty(len(thresholds))
        f1s = torch.empty(len(thresholds))

        for i, t in enumerate(thresholds.tolist()):
            tp, fp, _, fn = self._confusion_counts(probabilities, targets, t)
            precision, recall, f1 = self._precision_recall_f1(tp, fp, fn, self.eps)
            precisions[i] = precision
            recalls[i] = recall
            f1s[i] = f1

        return {
            "thresholds": thresholds,
            "precision": precisions,
            "recall": recalls,
            "f1": f1s,
        }

    def best_threshold(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> "tuple[float, float]":
        """Find the threshold maximizing overall F1 on the given data.

        Meant to be called ONCE, on a validation set, after training
        completes. The returned threshold should then be reused
        unchanged for test-set evaluation -- never re-fit on test data,
        or the resulting metrics would no longer be an honest estimate
        of generalization.

        Parameters
        ----------
        logits : torch.Tensor
            Raw (pre-sigmoid) logits from the validation set.
        targets : torch.Tensor
            Binary ground-truth labels, same shape as `logits`.
        thresholds : torch.Tensor, optional
            Candidate thresholds to sweep; see :meth:`sweep_thresholds`.

        Returns
        -------
        tuple[float, float]
            ``(best_threshold, best_f1)``.
        """
        results = self.sweep_thresholds(logits, targets, thresholds)
        best_idx = int(torch.argmax(results["f1"]).item())
        return (
            float(results["thresholds"][best_idx].item()),
            float(results["f1"][best_idx].item()),
        )

    @staticmethod
    def auroc(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute overall (flattened) AUROC via the rank-sum (Mann-Whitney
        U) formula, without any external metrics library.

        Threshold-independent: measures whether the model ranks
        anomalous timestep-channels above normal ones, regardless of
        where the decision threshold is set. A model that is well-
        ranked but poorly calibrated will still score well here even if
        F1 at threshold=0.5 looks terrible -- which is exactly the
        distinction this metric exists to draw.

        Parameters
        ----------
        probabilities : torch.Tensor
            Post-sigmoid probabilities, any shape.
        targets : torch.Tensor
            Binary ground-truth labels, same shape as `probabilities`.

        Returns
        -------
        torch.Tensor
            0-D tensor holding the AUROC, or ``nan`` if `targets`
            contains only one class.

        Notes
        -----
        Ranks are computed via a double ``argsort`` and are NOT
        tie-averaged; for continuous sigmoid outputs (as opposed to
        e.g. rounded probabilities) ties are rare enough that this is
        a negligible approximation in practice.
        """
        probabilities = probabilities.detach().reshape(-1).double()
        targets = targets.detach().reshape(-1).double()

        n_pos = targets.sum()
        n_neg = targets.numel() - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(float("nan"))

        order = torch.argsort(probabilities)
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(
            1, probabilities.numel() + 1, dtype=torch.float64
        )

        sum_ranks_pos = ranks[targets.bool()].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return auc

    @staticmethod
    def auprc(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute overall (flattened) AUPRC (average precision), without
        any external metrics library.

        Uses the standard step-function average-precision formula:
        sort by descending score, accumulate TP/FP, and sum
        ``precision[i] * (recall[i] - recall[i-1])`` at every point
        where recall increases (i.e. at every true positive).
        Threshold-independent, like :meth:`auroc`, but more informative
        than AUROC under heavy class imbalance (anomaly rate here is
        roughly 10-15%), since AUPRC is sensitive to the absolute
        false-positive rate rather than only the true/false-positive
        rate ratio.

        Parameters
        ----------
        probabilities : torch.Tensor
            Post-sigmoid probabilities, any shape.
        targets : torch.Tensor
            Binary ground-truth labels, same shape as `probabilities`.

        Returns
        -------
        torch.Tensor
            0-D tensor holding the AUPRC, or ``nan`` if `targets`
            contains no positive samples.
        """
        probabilities = probabilities.detach().reshape(-1).double()
        targets = targets.detach().reshape(-1).double()

        total_pos = targets.sum()
        if total_pos == 0:
            return torch.tensor(float("nan"))

        order = torch.argsort(probabilities, descending=True)
        targets_sorted = targets[order]

        tps = torch.cumsum(targets_sorted, dim=0)
        fps = torch.cumsum(1.0 - targets_sorted, dim=0)

        precision = tps / (tps + fps)
        recall = tps / total_pos

        recall_prev = torch.cat([torch.zeros(1, dtype=recall.dtype), recall[:-1]])
        delta_recall = recall - recall_prev

        ap = (precision * delta_recall).sum()
        return ap

    def compute_ranking_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute overall AUROC and AUPRC from raw logits in one call.

        Convenience wrapper applying sigmoid once and calling
        :meth:`auroc` / :meth:`auprc` on the result. Intended to be
        called on a bounded/subsampled batch of validation predictions
        (see the caller in ``trainer.py``/``train.py``), since both
        metrics require sorting every element and are therefore not
        cheap to compute on an entire, fully-flattened split.

        Parameters
        ----------
        logits : torch.Tensor
            Raw (pre-sigmoid) logits, any shape.
        targets : torch.Tensor
            Binary ground-truth labels, same shape as `logits`.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``{"auroc": ..., "auprc": ...}``, each a 0-D tensor
            (possibly ``nan`` if only one class is present).
        """
        probabilities = torch.sigmoid(logits).detach()
        return {
            "auroc": self.auroc(probabilities, targets),
            "auprc": self.auprc(probabilities, targets),
        }

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