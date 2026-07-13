"""Channel-attention Transformer encoder block for the PhySATFormer architecture."""

import torch
from torch import nn

from src.models.physics_guided_channel_attention import PhysicsGuidedChannelAttention


class ChannelAttentionBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block for channel-axis attention.

    This is the channel-attention counterpart of
    :class:`~src.models.encoder_block.TransformerEncoderBlock`. Instead of
    attending across the temporal axis, it applies
    :class:`~src.models.physics_guided_channel_attention.
    PhysicsGuidedChannelAttention` across the *channel* axis of telemetry
    embeddings, biased by a static, domain-derived physics relationship
    matrix. Each sub-layer (attention and feed-forward) is wrapped in a
    residual connection and preceded by layer normalization (Pre-LN
    architecture).

    Parameters
    ----------
    channel_embedding_dim : int
        Dimensionality of the per-channel embedding, i.e. the size of the
        last dimension of the input and output tensors. Must be positive
        and divisible by ``num_heads``.
    num_heads : int
        Number of attention heads used by the channel attention module.
        Must be positive.
    ff_dim : int
        Dimensionality of the hidden layer in the feed-forward network.
        Must be positive.
    physics_matrix : torch.Tensor
        Static, non-trainable channel-relationship matrix of shape
        ``(num_channels, num_channels)``, as produced by
        :class:`~src.models.physics_matrix.PhysicsRelationshipMatrix`.
        Forwarded unchanged to ``PhysicsGuidedChannelAttention``.
    dropout : float, optional
        Dropout probability applied within attention and the feed-forward
        network, by default 0.1.

    Attributes
    ----------
    attention : PhysicsGuidedChannelAttention
        Physics-guided multi-head channel attention module.
    attention_norm : nn.LayerNorm
        Layer normalization applied before the attention sub-layer.
    feed_forward_norm : nn.LayerNorm
        Layer normalization applied before the feed-forward sub-layer.
    feed_forward : nn.Sequential
        Position-wise feed-forward network.
    dropout : nn.Dropout
        Dropout layer applied to sub-layer outputs before the residual
        connection.

    Examples
    --------
    >>> import torch
    >>> num_channels = 5
    >>> physics_matrix = torch.eye(num_channels)
    >>> block = ChannelAttentionBlock(
    ...     channel_embedding_dim=16,
    ...     num_heads=4,
    ...     ff_dim=32,
    ...     physics_matrix=physics_matrix,
    ... )
    >>> x = torch.randn(2, num_channels, 16)  # (batch, channels, d_c)
    >>> out = block(x)
    >>> out.shape
    torch.Size([2, 5, 16])
    """

    def __init__(
        self,
        channel_embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        physics_matrix: torch.Tensor,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if not isinstance(channel_embedding_dim, int) or channel_embedding_dim <= 0:
            raise ValueError(
                "`channel_embedding_dim` must be a positive integer, got "
                f"{channel_embedding_dim!r}."
            )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"`num_heads` must be a positive integer, got {num_heads!r}."
            )
        if not isinstance(ff_dim, int) or ff_dim <= 0:
            raise ValueError(f"`ff_dim` must be a positive integer, got {ff_dim!r}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout!r}.")

        self.channel_embedding_dim = channel_embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.attention = PhysicsGuidedChannelAttention(
            d_model=channel_embedding_dim,
            num_heads=num_heads,
            physics_matrix=physics_matrix,
            dropout=dropout,
        )

        self.attention_norm = nn.LayerNorm(channel_embedding_dim)
        self.feed_forward_norm = nn.LayerNorm(channel_embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(channel_embedding_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, channel_embedding_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Pre-LN physics-guided channel attention block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape
            ``(batch_size, num_channels, channel_embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape
            ``(batch_size, num_channels, channel_embedding_dim)``.

        Raises
        ------
        ValueError
            If `x` is not 3-D or its last dimension does not match
            ``channel_embedding_dim``.
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"x must be a torch.Tensor, got {type(x).__name__}."
            )

        if x.dim() != 3:
            raise ValueError(
                "Expected input of shape (batch_size, num_channels, "
                f"channel_embedding_dim), got tensor with shape {tuple(x.shape)}."
            )
        if x.shape[-1] != self.channel_embedding_dim:
            raise ValueError(
                "Expected last dimension of input to be "
                f"channel_embedding_dim={self.channel_embedding_dim}, got "
                f"{x.shape[-1]}."
            )

        normalized = self.attention_norm(x)
        attention_output, _ = self.attention(normalized)
        x = x + self.dropout(attention_output)

        normalized = self.feed_forward_norm(x)
        feed_forward_output = self.feed_forward(normalized)
        x = x + feed_forward_output

        return x