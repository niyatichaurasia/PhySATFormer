"""Transformer encoder block for the PhySATFormer architecture."""

import torch
from torch import nn

from src.models.attention import MultiHeadSelfAttention


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block.

    Combines multi-head self-attention with a position-wise feed-forward
    network, each wrapped in a residual connection and preceded by layer
    normalization (Pre-LN architecture).

    Parameters
    ----------
    d_model : int
        Dimensionality of the input and output embeddings.
    num_heads : int
        Number of attention heads used by the self-attention module.
    ff_dim : int
        Dimensionality of the hidden layer in the feed-forward network.
    dropout : float, optional
        Dropout probability applied within attention and the feed-forward
        network, by default 0.1.

    Attributes
    ----------
    attention : MultiHeadSelfAttention
        Multi-head self-attention module.
    attention_norm : nn.LayerNorm
        Layer normalization applied before the attention sub-layer.
    feed_forward_norm : nn.LayerNorm
        Layer normalization applied before the feed-forward sub-layer.
    feed_forward : nn.Sequential
        Position-wise feed-forward network.
    dropout : nn.Dropout
        Dropout layer applied to sub-layer outputs before the residual
        connection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Pre-LN Transformer encoder block to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, sequence_length, d_model)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, sequence_length, d_model)``.
        """
        normalized = self.attention_norm(x)
        attention_output = self.attention(normalized)
        x = x + self.dropout(attention_output)

        normalized = self.feed_forward_norm(x)
        feed_forward_output = self.feed_forward(normalized)
        x = x + self.dropout(feed_forward_output)

        return x