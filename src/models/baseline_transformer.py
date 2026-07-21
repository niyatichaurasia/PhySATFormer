"""BaselineTransformer: architecturally-matched baseline for PhySATFormer.

This module assembles the baseline architecture used as the scientific
control against PhySATFormer. It composes the same building blocks used
by PhySATFormer, with the sole exception that the channel-attention
stage is standard (non-physics-guided) rather than physics-guided:

    - ``TelemetryChannelEncoder``: scalar -> per-channel embedding
    - ``StandardChannelAttentionBlock``: vanilla attention across channels
    - ``ChannelPool``: channel-dimension pooling + projection to d_model
    - ``PositionalEncoding``: sinusoidal temporal positional encoding
    - ``TransformerEncoderBlock``: standard temporal self-attention

BaselineTransformer itself contains no attention, pooling, or encoding
logic of its own; it is strictly an orchestrator that wires these
components together into a two-stage (channel-axis, then temporal-axis)
Transformer, mirroring PhySATFormer exactly except for the absence of
any physics guidance. This makes it a fair, architecturally-matched
baseline for isolating the effect of physics-guided channel attention.
"""

import torch
import torch.nn as nn

from src.models.telemetry_channel_encoder import TelemetryChannelEncoder
from src.models.standard_channel_attention_block import StandardChannelAttentionBlock
from src.models.channel_pool import ChannelPool
from src.models.positional_encoding import PositionalEncoding
from src.models.encoder_block import TransformerEncoderBlock


class BaselineTransformer(nn.Module):
    """Architecturally-matched, non-physics-guided baseline for PhySATFormer.

    The model first embeds each scalar telemetry reading into a per-channel
    embedding, then applies a stack of standard (non-physics-guided)
    channel-attention blocks across the channel axis (independently per
    timestep), pools the channel dimension into a single per-timestep
    representation, injects temporal positional information, applies a
    stack of standard temporal Transformer encoder blocks, and finally
    produces per-timestep, per-channel anomaly logits via a linear
    prediction head applied independently to every timestep.

    This architecture is identical to :class:`~src.models.physatformer.
    PhySATFormer` in every respect except that its channel-attention stage
    uses :class:`~src.models.standard_channel_attention_block.
    StandardChannelAttentionBlock` instead of :class:`~src.models.
    channel_attention_block.ChannelAttentionBlock`, and therefore accepts
    no ``physics_matrix``. This makes it a scientifically fair baseline
    for isolating the effect of physics-guided channel attention.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each individual telemetry reading before
        channel embedding (typically 1, a single scalar value per
        channel per timestep). Must be positive.
    channel_embedding_dim : int
        Dimensionality of the per-channel embedding produced by
        ``TelemetryChannelEncoder`` and consumed by the channel-attention
        stage. Must be positive and typically must evenly divide
        ``num_heads`` of the channel-attention stage.
    d_model : int
        Dimensionality of the model's internal (hidden) representation
        used by the temporal Transformer stage. Must be positive.
    num_heads : int
        Number of attention heads used in both the channel-attention
        blocks and the temporal Transformer encoder blocks. Must be
        positive.
    num_channel_layers : int
        Number of stacked ``StandardChannelAttentionBlock`` layers. Must
        be positive.
    num_temporal_layers : int
        Number of stacked ``TransformerEncoderBlock`` layers. Must be
        positive.
    ff_dim : int
        Dimensionality of the feed-forward hidden layer within both the
        channel-attention blocks and the temporal Transformer encoder
        blocks. Must be positive.
    num_channels : int
        Number of telemetry channels for which the model predicts
        per-timestep anomaly logits. Must be positive.
    dropout : float, optional
        Dropout probability applied throughout the model, by default
        0.1.

    Attributes
    ----------
    telemetry_channel_encoder : TelemetryChannelEncoder
        Module that embeds raw scalar telemetry readings into
        per-channel embeddings.
    channel_attention_layers : nn.ModuleList
        Stack of ``StandardChannelAttentionBlock`` modules applied across
        the channel axis.
    channel_pool : ChannelPool
        Module that mean-pools the channel dimension and projects into
        ``d_model``-dimensional space.
    positional_encoding : PositionalEncoding
        Module that injects temporal positional information into the
        pooled per-timestep embeddings.
    temporal_layers : nn.ModuleList
        Stack of ``TransformerEncoderBlock`` modules applied across the
        temporal axis.
    classifier : nn.Linear
        Linear layer mapping the ``d_model`` representation at every
        timestep to ``num_channels`` raw anomaly logits.

    Examples
    --------
    >>> import torch
    >>> num_channels = 5
    >>> model = BaselineTransformer(
    ...     input_dim=1,
    ...     channel_embedding_dim=16,
    ...     d_model=64,
    ...     num_heads=4,
    ...     num_channel_layers=2,
    ...     num_temporal_layers=2,
    ...     ff_dim=128,
    ...     num_channels=num_channels,
    ... )
    >>> x = torch.randn(8, 32, num_channels)  # (batch, seq_len, channels)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([8, 32, 5])
    """

    def __init__(
        self,
        input_dim: int,
        channel_embedding_dim: int,
        d_model: int,
        num_heads: int,
        num_channel_layers: int,
        num_temporal_layers: int,
        ff_dim: int,
        num_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive int, got {input_dim!r}")
        if (
            not isinstance(channel_embedding_dim, int)
            or isinstance(channel_embedding_dim, bool)
            or channel_embedding_dim <= 0
        ):
            raise ValueError(
                f"channel_embedding_dim must be a positive int, got "
                f"{channel_embedding_dim!r}"
            )
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError(f"d_model must be a positive int, got {d_model!r}")

        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive int, got {num_heads!r}")

        if channel_embedding_dim % num_heads != 0:
            raise ValueError(
                "channel_embedding_dim must be divisible by num_heads."
            )

        if (
            not isinstance(num_channel_layers, int)
            or isinstance(num_channel_layers, bool)
            or num_channel_layers <= 0
        ):
            raise ValueError(
                f"num_channel_layers must be a positive int, got {num_channel_layers!r}"
            )

        if (
            not isinstance(num_temporal_layers, int)
            or isinstance(num_temporal_layers, bool)
            or num_temporal_layers <= 0
        ):
            raise ValueError(
                f"num_temporal_layers must be a positive int, got {num_temporal_layers!r}"
            )

        if not isinstance(ff_dim, int) or isinstance(ff_dim, bool) or ff_dim <= 0:
            raise ValueError(f"ff_dim must be a positive int, got {ff_dim!r}")

        if (
            not isinstance(num_channels, int)
            or isinstance(num_channels, bool)
            or num_channels <= 0
        ):
            raise ValueError(f"num_channels must be a positive int, got {num_channels!r}")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

        self.input_dim = input_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_channel_layers = num_channel_layers
        self.num_temporal_layers = num_temporal_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.num_channels = num_channels

        self.telemetry_channel_encoder = TelemetryChannelEncoder(
            channel_embedding_dim=channel_embedding_dim,
            input_dim=input_dim,
        )

        self.channel_attention_layers = nn.ModuleList(
            [
                StandardChannelAttentionBlock(
                    channel_embedding_dim=channel_embedding_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_channel_layers)
            ]
        )

        self.channel_pool = ChannelPool(
            channel_embedding_dim=channel_embedding_dim,
            d_model=d_model,
        )

        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.temporal_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_temporal_layers)
            ]
        )

        self.classifier = nn.Linear(d_model, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the full BaselineTransformer architecture.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, sequence_length,
            num_channels)`` when ``input_dim == 1``, or
            ``(batch_size, sequence_length, num_channels, input_dim)``
            when ``input_dim > 1``.

        Returns
        -------
        torch.Tensor
            Raw (pre-sigmoid) per-timestep, per-channel anomaly logits
            of shape ``(batch_size, sequence_length, num_channels)``.

        Raises
        ------
        TypeError
            If ``x`` is not a ``torch.Tensor``.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")

        # (B, T, C, input_dim) -> (B, T, C, d_c)
        channel_embeddings = self.telemetry_channel_encoder(x)

        batch_size, seq_len, num_channels, d_c = channel_embeddings.shape

        if num_channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {num_channels}."
            )

        # (B, T, C, d_c) -> (B*T, C, d_c) so channel attention operates
        # independently per (batch, timestep) position.
        channel_embeddings = channel_embeddings.reshape(
            batch_size * seq_len,
            num_channels,
            d_c,
        )

        for channel_attention_layer in self.channel_attention_layers:
            channel_embeddings = channel_attention_layer(channel_embeddings)

        # (B*T, C, d_c) -> (B, T, C, d_c)
        channel_embeddings = channel_embeddings.reshape(
            batch_size,
            seq_len,
            num_channels,
            d_c,
        )

        # (B, T, C, d_c) -> (B, T, d_model)
        hidden = self.channel_pool(channel_embeddings)

        # Temporal Transformer stage.
        hidden = self.positional_encoding(hidden)

        for temporal_layer in self.temporal_layers:
            hidden = temporal_layer(hidden)

        # (B, T, d_model) -> (B, T, num_channels), applied independently
        # to every timestep. No pooling, flattening, or aggregation over
        # time or channels, and no activation: raw logits are returned
        # for use with BCEWithLogitsLoss.
        logits = self.classifier(hidden)

        return logits
