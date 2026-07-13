"""Channel pooling module for PhySATFormer.

This module bridges the channel-attention stage and the temporal
Transformer stage of the PhySATFormer pipeline. It reduces per-channel
embeddings down to a single per-timestep representation via mean
pooling over the channel dimension, then projects that pooled
representation into the model's hidden dimensionality.

This module performs no attention, no positional encoding, and no
flattening; it is strictly responsible for channel-dimension pooling
followed by a linear projection.
"""

import torch
import torch.nn as nn


class ChannelPool(nn.Module):
    """Pool per-channel embeddings and project into model space.

    Given a tensor of per-channel embeddings produced by an upstream
    channel-attention (or channel-encoding) stage, this module
    collapses the channel dimension via mean pooling and then applies
    a shared linear projection to map the pooled representation into
    ``d_model``-dimensional space, suitable for consumption by a
    temporal Transformer stage.

    Parameters
    ----------
    channel_embedding_dim : int
        Dimensionality of each channel's embedding vector at the
        input. Must be positive.
    d_model : int
        Dimensionality of the model's internal (hidden) representation
        produced at the output. Must be positive.

    Attributes
    ----------
    channel_embedding_dim : int
        Dimensionality of each channel's embedding vector.
    d_model : int
        Dimensionality of the projected output representation.
    projection : nn.Linear
        Linear layer mapping ``channel_embedding_dim`` to ``d_model``,
        applied after mean pooling over the channel dimension.

    Examples
    --------
    >>> pool = ChannelPool(channel_embedding_dim=16, d_model=64)
    >>> x = torch.randn(8, 32, 10, 16)  # (batch, seq_len, channels, d_c)
    >>> pooled = pool(x)
    >>> pooled.shape
    torch.Size([8, 32, 64])
    """

    def __init__(
        self,
        channel_embedding_dim: int,
        d_model: int,
    ) -> None:
        super().__init__()

        if not isinstance(channel_embedding_dim, int) or isinstance(
            channel_embedding_dim, bool
        ):
            raise TypeError(
                "channel_embedding_dim must be an int, got "
                f"{type(channel_embedding_dim).__name__}"
            )
        if channel_embedding_dim <= 0:
            raise ValueError(
                "channel_embedding_dim must be positive, got "
                f"{channel_embedding_dim}"
            )

        if not isinstance(d_model, int) or isinstance(d_model, bool):
            raise TypeError(f"d_model must be an int, got {type(d_model).__name__}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.channel_embedding_dim = channel_embedding_dim
        self.d_model = d_model

        self.projection = nn.Linear(channel_embedding_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool channel embeddings and project into model space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, sequence_length,
            num_channels, channel_embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, sequence_length,
            d_model)``.

        Raises
        ------
        TypeError
            If ``x`` is not a ``torch.Tensor``.
        ValueError
            If ``x`` does not have exactly 4 dimensions, if its
            trailing dimension does not match ``channel_embedding_dim``,
            or if the channel dimension is empty.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")

        if x.dim() != 4:
            raise ValueError(
                "Expected input of shape (batch_size, sequence_length, "
                f"num_channels, channel_embedding_dim), got tensor with "
                f"shape {tuple(x.shape)}."
            )

        if x.shape[-1] != self.channel_embedding_dim:
            raise ValueError(
                "Expected last dimension of input to be "
                f"channel_embedding_dim={self.channel_embedding_dim}, "
                f"got {x.shape[-1]}."
            )

        if x.shape[2] == 0:
            raise ValueError(
                "Expected a non-empty channel dimension (dim=2), got "
                f"shape {tuple(x.shape)}."
            )

        pooled = x.mean(dim=2)
        output = self.projection(pooled)

        return output