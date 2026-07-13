"""Standard multi-head temporal self-attention.

This module implements vanilla scaled dot-product multi-head self-attention
exactly as described in Vaswani et al. (2017), "Attention Is All You Need".
It operates purely along the temporal (sequence) axis of the input and does
not include any physics-informed or otherwise domain-specific bias. Channel
(feature) attention is intentionally out of scope for this module.

The class is written as a reusable base class: subclasses (e.g. a physics-
informed channel attention variant) can inherit from it and override only
`_compute_attention_scores` to change how attention scores are produced,
while reusing the projections, head splitting/merging, softmax, dropout,
and output projection defined here.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Multi-head self-attention applied along the temporal dimension.

    This module implements the standard scaled dot-product multi-head
    attention mechanism from Vaswani et al. (2017). Given an input tensor of
    shape ``(batch_size, sequence_length, d_model)``, it computes query,
    key, and value projections, splits them into multiple heads, applies
    scaled dot-product attention independently per head, merges the heads
    back together, and applies a final output projection.

    This class is designed as a reusable base class. Subclasses may override
    `_compute_attention_scores` to change how raw attention scores are
    computed (e.g. to inject a domain-specific bias) while reusing all other
    behavior unchanged.

    Parameters
    ----------
    d_model : int
        Dimensionality of the input and output feature space. Must be
        divisible by `num_heads`.
    num_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability applied to the attention weights and to the
        output projection, by default 0.0.
    bias : bool, optional
        Whether the linear projections (query, key, value, output) include
        a bias term, by default True.

    Attributes
    ----------
    d_model : int
        Dimensionality of the input and output feature space.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimensionality of each individual attention head, ``d_model // num_heads``.
    q_proj : nn.Linear
        Linear projection producing query vectors.
    k_proj : nn.Linear
        Linear projection producing key vectors.
    v_proj : nn.Linear
        Linear projection producing value vectors.
    out_proj : nn.Linear
        Linear projection applied after merging attention heads.
    attn_dropout : nn.Dropout
        Dropout applied to attention probabilities.
    resid_dropout : nn.Dropout
        Dropout applied to the final output projection.

    Examples
    --------
    >>> import torch
    >>> attn = TemporalAttention(d_model=64, num_heads=8, dropout=0.1)
    >>> x = torch.randn(2, 10, 64)  # (batch_size, sequence_length, d_model)
    >>> out, weights = attn(x)
    >>> out.shape
    torch.Size([2, 10, 64])
    >>> weights.shape
    torch.Size([2, 8, 10, 10])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"`d_model` must be a positive integer, got {d_model!r}.")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"`num_heads` must be a positive integer, got {num_heads!r}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"`d_model` ({d_model}) must be divisible by `num_heads` ({num_heads})."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout!r}.")

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape a tensor into multiple attention heads.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch_size, sequence_length, d_model)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, num_heads, sequence_length, head_dim)``.
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge multiple attention heads back into a single tensor.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch_size, num_heads, sequence_length, head_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, sequence_length, d_model)``.
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def _compute_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw (pre-softmax) scaled dot-product attention scores.

        Subclasses may override this method to modify how attention scores
        are derived from `query` and `key` (for example, to add a
        domain-specific bias term), while reusing all other behavior of
        `forward` unchanged.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape
            ``(batch_size, num_heads, sequence_length, head_dim)``.
        key : torch.Tensor
            Key tensor of shape
            ``(batch_size, num_heads, sequence_length, head_dim)``.

        Returns
        -------
        torch.Tensor
            Raw attention scores of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        return torch.matmul(query, key.transpose(-2, -1)) * scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head temporal self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, sequence_length, d_model)``.

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape ``(batch_size, sequence_length, d_model)``.
        attention_weights : torch.Tensor
            Attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected input of shape (batch_size, sequence_length, d_model), "
                f"got tensor with shape {tuple(x.shape)}."
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension of input to be d_model={self.d_model}, "
                f"got {x.shape[-1]}."
            )

        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        scores = self._compute_attention_scores(query, key)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        context = torch.matmul(attention_weights, value)
        context = self._merge_heads(context)

        output = self.out_proj(context)
        output = self.resid_dropout(output)

        return output, attention_weights