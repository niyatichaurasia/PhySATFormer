"""Per-channel telemetry embedding module for PhySATFormer.

This module implements the first stage of the Physics-Guided
Spatio-Temporal Transformer pipeline: converting raw scalar telemetry
readings into learnable embedding vectors. Each telemetry value, for
every channel at every timestep, is independently projected into a
``channel_embedding_dim``-dimensional space using a single shared
linear layer.

This module performs no positional encoding, no attention, and no
pooling; it is strictly responsible for the scalar-to-vector channel
embedding step.
"""

import torch
import torch.nn as nn


class TelemetryChannelEncoder(nn.Module):
    """Project scalar telemetry readings into per-channel embeddings.

    Each scalar telemetry value in the input tensor is treated as an
    independent ``input_dim``-dimensional vector (by default a single
    scalar) and passed through a shared ``nn.Linear`` layer to produce
    a ``channel_embedding_dim``-dimensional embedding. The same linear
    projection is applied identically to every (batch, timestep,
    channel) position; no interaction between channels or timesteps
    occurs in this module.

    Parameters
    ----------
    channel_embedding_dim : int
        Dimensionality of the output embedding produced for each
        telemetry value. Must be positive.
    input_dim : int, optional
        Dimensionality of each individual telemetry reading before
        embedding, by default 1 (a single scalar value per channel per
        timestep). Must be positive.

    Attributes
    ----------
    input_dim : int
        Dimensionality of each individual telemetry reading.
    channel_embedding_dim : int
        Dimensionality of the produced channel embeddings.
    projection : nn.Linear
        Shared linear layer mapping ``input_dim`` to
        ``channel_embedding_dim``, applied independently to every
        telemetry value.

    Examples
    --------
    >>> encoder = TelemetryChannelEncoder(channel_embedding_dim=16)
    >>> x = torch.randn(8, 32, 10)  # (batch, sequence_length, num_channels)
    >>> embeddings = encoder(x)
    >>> embeddings.shape
    torch.Size([8, 32, 10, 16])
    """

    def __init__(
        self,
        channel_embedding_dim: int,
        input_dim: int = 1,
    ) -> None:
        super().__init__()

        if not isinstance(input_dim, int) or isinstance(input_dim, bool):
            raise TypeError(
                f"input_dim must be an int, got {type(input_dim).__name__}"
            )
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

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

        self.input_dim = input_dim
        self.channel_embedding_dim = channel_embedding_dim

        self.projection = nn.Linear(input_dim, channel_embedding_dim)
        nn.init.xavier_uniform_(self.projection.weight)

        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed each telemetry value independently.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape
            ``(batch_size, sequence_length, num_channels)`` when
            ``input_dim == 1``, or
            ``(batch_size, sequence_length, num_channels, input_dim)``
            when ``input_dim > 1``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape
            ``(batch_size, sequence_length, num_channels,
            channel_embedding_dim)``.

        Raises
        ------
        TypeError
            If ``x`` is not a ``torch.Tensor``.
        ValueError
            If ``x`` does not have a supported number of dimensions, or
            if its trailing dimension does not match ``input_dim`` when
            ``input_dim > 1``.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")

        if self.input_dim == 1:
            if x.dim() != 3:
                raise ValueError(
                    "Expected input of shape (batch_size, sequence_length, "
                    f"num_channels) when input_dim=1, got tensor with shape "
                    f"{tuple(x.shape)}."
                )
            x = x.unsqueeze(-1)
        else:
            if x.dim() != 4:
                raise ValueError(
                    "Expected input of shape (batch_size, sequence_length, "
                    "num_channels, input_dim) when input_dim > 1, got tensor "
                    f"with shape {tuple(x.shape)}."
                )
            if x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected last dimension of input to be input_dim="
                    f"{self.input_dim}, got {x.shape[-1]}."
                )

        embeddings = self.projection(x)

        return embeddings