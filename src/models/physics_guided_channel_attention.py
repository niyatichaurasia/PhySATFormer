"""Physics-guided channel attention for PhySATFormer.

This module implements the core novelty of the PhySATFormer architecture:
a multi-head self-attention mechanism applied across the *channel* axis
of telemetry embeddings, biased by a static, domain-derived physics
relationship matrix (see :class:`~src.models.physics_matrix.
PhysicsRelationshipMatrix`).

The pipeline this module participates in is::

    (B, T, C)
        -> TelemetryChannelEncoder
    (B, T, C, d_c)
        -> reshape (performed OUTSIDE this module)
    (B * T, C, d_c)
        -> PhysicsGuidedChannelAttention
    (B * T, C, d_c)

Because the reshape folds the batch and temporal axes together before
this module is invoked, the "sequence" axis consumed by the inherited
:class:`~src.models.temporal_attention.TemporalAttention` machinery is,
in this context, the *channel* axis. Consequently the raw attention
score tensor has shape ``(batch, num_heads, num_channels, num_channels)``,
which aligns exactly with the physics relationship matrix produced by
``PhysicsRelationshipMatrix``.

This module intentionally does not implement its own projections, head
splitting/merging, softmax, dropout, or output projection: all of that
behavior is inherited unchanged from ``TemporalAttention``. Only the
computation of raw attention scores is overridden, to inject a
learnable, per-head physics bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.temporal_attention import TemporalAttention


class PhysicsGuidedChannelAttention(TemporalAttention):
    """Multi-head channel attention biased by a static physics prior.

    This class subclasses :class:`TemporalAttention` and reuses all of
    its projection, head-splitting/merging, softmax, dropout, and output
    projection logic. The only behavior that is overridden is
    :meth:`_compute_attention_scores`, which augments the standard
    scaled dot-product attention scores with a learnable, per-head
    physics bias derived from a static channel-relationship matrix.

    The physics bias encodes domain knowledge (e.g. subsystem
    membership) about which telemetry channels are physically related.
    Each attention head learns its own scalar weight controlling how
    strongly it relies on this physics prior, allowing the model to
    discover, per head, how much domain knowledge to trust.

    Parameters
    ----------
    d_model : int
        Dimensionality of the input and output feature space (i.e. the
        per-channel embedding dimension ``d_c``). Must be divisible by
        `num_heads`.
    num_heads : int
        Number of attention heads.
    physics_matrix : torch.Tensor
        Static, non-trainable channel-relationship matrix of shape
        ``(num_channels, num_channels)``, as produced by
        :class:`~src.models.physics_matrix.PhysicsRelationshipMatrix`.
        Must be a non-empty, square 2-D tensor. It is converted to
        ``torch.float32`` and registered as a non-trainable buffer.
    dropout : float, optional
        Dropout probability applied to the attention weights and to the
        output projection, by default 0.0.
    bias : bool, optional
        Whether the linear projections (query, key, value, output)
        include a bias term, by default True.

    Attributes
    ----------
    physics_matrix : torch.Tensor
        Registered buffer holding the static physics-relationship
        matrix, of shape ``(num_channels, num_channels)``.
    physics_weight : nn.Parameter
        Learnable per-head scalar weights of shape ``(num_heads,)``
        controlling the contribution of the physics bias for each
        attention head.

    Examples
    --------
    >>> import torch
    >>> num_channels = 5
    >>> physics_matrix = torch.eye(num_channels)
    >>> attn = PhysicsGuidedChannelAttention(
    ...     d_model=16,
    ...     num_heads=4,
    ...     physics_matrix=physics_matrix,
    ... )
    >>> x = torch.randn(2, num_channels, 16)  # (batch, channels, d_c)
    >>> out, weights = attn(x)
    >>> out.shape
    torch.Size([2, 5, 16])
    >>> weights.shape
    torch.Size([2, 4, 5, 5])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        physics_matrix: torch.Tensor,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        validated_physics_matrix = self._validate_physics_matrix(physics_matrix)

        self.register_buffer("physics_matrix", validated_physics_matrix)

        self.physics_weight = nn.Parameter(torch.ones(self.num_heads))

    @staticmethod
    def _validate_physics_matrix(physics_matrix: torch.Tensor) -> torch.Tensor:
        """Validate and normalize a physics-relationship matrix.

        Parameters
        ----------
        physics_matrix : torch.Tensor
            Candidate physics-relationship matrix.

        Returns
        -------
        torch.Tensor
            The validated matrix, converted to ``torch.float32``.

        Raises
        ------
        TypeError
            If `physics_matrix` is not a ``torch.Tensor``.
        ValueError
            If `physics_matrix` is empty, is not 2-D, or is not square.
        """
        if not isinstance(physics_matrix, torch.Tensor):
            raise TypeError(
                "`physics_matrix` must be a torch.Tensor, got "
                f"{type(physics_matrix).__name__}."
            )

        if physics_matrix.numel() == 0:
            raise ValueError("`physics_matrix` must be non-empty.")

        if physics_matrix.dim() != 2:
            raise ValueError(
                "`physics_matrix` must be a 2-D (square) tensor, got a "
                f"tensor with {physics_matrix.dim()} dimension(s) and shape "
                f"{tuple(physics_matrix.shape)}."
            )

        if physics_matrix.shape[0] != physics_matrix.shape[1]:
            raise ValueError(
                "`physics_matrix` must be square, got shape "
                f"{tuple(physics_matrix.shape)}."
            )

        return physics_matrix.to(dtype=torch.float32)

    def _compute_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics-biased scaled dot-product attention scores.

        First computes the standard scaled dot-product attention scores
        via the parent implementation, then adds a learnable, per-head
        physics bias derived from the static physics-relationship
        matrix.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape
            ``(batch_size, num_heads, num_channels, head_dim)``.
        key : torch.Tensor
            Key tensor of shape
            ``(batch_size, num_heads, num_channels, head_dim)``.

        Returns
        -------
        torch.Tensor
            Physics-biased attention scores of shape
            ``(batch_size, num_heads, num_channels, num_channels)``.

        Raises
        ------
        ValueError
            If the channel dimension of `query`/`key` does not match
            the size of the registered physics matrix.
        """
        scores = super()._compute_attention_scores(query, key)

        num_channels = query.shape[-2]
        if self.physics_matrix.shape[0] != num_channels:
            raise ValueError(
                "`physics_matrix` size "
                f"({self.physics_matrix.shape[0]}) does not match the "
                f"number of channels in the input ({num_channels})."
            )

        # (num_heads,) -> (1, num_heads, 1, 1) for broadcasting.
        head_weights = self.physics_weight.view(1, self.num_heads, 1, 1)

        # (num_channels, num_channels) -> (1, 1, num_channels, num_channels).
        matrix = self.physics_matrix.view(
            1, 1, num_channels, num_channels
        ).to(dtype=scores.dtype, device=scores.device)

        physics_bias = matrix * head_weights

        return scores + physics_bias