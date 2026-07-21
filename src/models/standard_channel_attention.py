"""Standard (non-physics-guided) channel attention for the baseline model.

This module implements the baseline counterpart of
:class:`~src.models.physics_guided_channel_attention.
PhysicsGuidedChannelAttention`. It applies vanilla scaled dot-product
multi-head self-attention across the *channel* axis of telemetry
embeddings, with no physics-derived bias of any kind.

The pipeline this module participates in is::

    (B, T, C)
        -> TelemetryChannelEncoder
    (B, T, C, d_c)
        -> reshape (performed OUTSIDE this module)
    (B * T, C, d_c)
        -> StandardChannelAttention
    (B * T, C, d_c)

Because the reshape folds the batch and temporal axes together before
this module is invoked, the "sequence" axis consumed by the inherited
:class:`~src.models.temporal_attention.TemporalAttention` machinery is,
in this context, the *channel* axis.

This module intentionally contains no logic of its own: it exists
purely to give the baseline's channel-attention stage its own class
identity (mirroring ``PhysicsGuidedChannelAttention``) for architectural
clarity and future extensibility, while reusing the exact vanilla
scaled dot-product attention computation already implemented in
``TemporalAttention`` unmodified.
"""

from __future__ import annotations

from src.models.temporal_attention import TemporalAttention


class StandardChannelAttention(TemporalAttention):
    """Vanilla multi-head channel attention with no physics bias.

    This class subclasses :class:`TemporalAttention` and reuses all of
    its projection, head-splitting/merging, attention-score
    computation, softmax, dropout, and output projection logic
    unchanged. It exists as a distinctly named class so that the
    baseline architecture's channel-attention stage mirrors the
    ``PhysicsGuidedChannelAttention`` used by PhySATFormer at the class
    level, without introducing any physics-derived bias.

    Parameters
    ----------
    d_model : int
        Dimensionality of the input and output feature space (i.e. the
        per-channel embedding dimension ``d_c``). Must be divisible by
        `num_heads`.
    num_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability applied to the attention weights and to the
        output projection, by default 0.0.
    bias : bool, optional
        Whether the linear projections (query, key, value, output)
        include a bias term, by default True.

    Examples
    --------
    >>> import torch
    >>> num_channels = 5
    >>> attn = StandardChannelAttention(d_model=16, num_heads=4)
    >>> x = torch.randn(2, num_channels, 16)  # (batch, channels, d_c)
    >>> out, weights = attn(x)
    >>> out.shape
    torch.Size([2, 5, 16])
    >>> weights.shape
    torch.Size([2, 4, 5, 5])
    """
