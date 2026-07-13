"""PhySATFormer: Physics-Guided Spatio-Temporal Transformer.

This module assembles the complete PhySATFormer architecture by
composing the previously implemented building blocks of the project:

    - ``TelemetryChannelEncoder``: scalar -> per-channel embedding
    - ``ChannelAttentionBlock``: physics-guided attention across channels
    - ``ChannelPool``: channel-dimension pooling + projection to d_model
    - ``PositionalEncoding``: sinusoidal temporal positional encoding
    - ``TransformerEncoderBlock``: standard temporal self-attention

PhySATFormer itself contains no attention, pooling, or encoding logic of
its own; it is strictly an orchestrator that wires these components
together into a two-stage (channel-axis, then temporal-axis) Transformer
classifier.
"""

import torch
import torch.nn as nn

from src.models.telemetry_channel_encoder import TelemetryChannelEncoder
from src.models.channel_attention_block import ChannelAttentionBlock
from src.models.channel_pool import ChannelPool
from src.models.positional_encoding import PositionalEncoding
from src.models.encoder_block import TransformerEncoderBlock


class PhySATFormer(nn.Module):
    """Physics-Guided Spatio-Temporal Transformer for telemetry classification.

    The model first embeds each scalar telemetry reading into a per-channel
    embedding, then applies a stack of physics-guided channel-attention
    blocks across the channel axis (independently per timestep), pools the
    channel dimension into a single per-timestep representation, injects
    temporal positional information, applies a stack of standard temporal
    Transformer encoder blocks, mean-pools over the time axis, and finally
    produces class logits via a linear classification head.

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
        Number of stacked ``ChannelAttentionBlock`` layers. Must be
        positive.
    num_temporal_layers : int
        Number of stacked ``TransformerEncoderBlock`` layers. Must be
        positive.
    ff_dim : int
        Dimensionality of the feed-forward hidden layer within both the
        channel-attention blocks and the temporal Transformer encoder
        blocks. Must be positive.
    num_classes : int
        Number of output classes for classification. Must be positive.
    physics_matrix : torch.Tensor
        Static, non-trainable channel-relationship matrix of shape
        ``(num_channels, num_channels)``, as produced by
        :class:`~src.models.physics_matrix.PhysicsRelationshipMatrix`.
        Forwarded unchanged to every ``ChannelAttentionBlock``.
    dropout : float, optional
        Dropout probability applied throughout the model, by default
        0.1.

    Attributes
    ----------
    telemetry_channel_encoder : TelemetryChannelEncoder
        Module that embeds raw scalar telemetry readings into
        per-channel embeddings.
    channel_attention_layers : nn.ModuleList
        Stack of ``ChannelAttentionBlock`` modules applied across the
        channel axis.
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
        Linear layer mapping the pooled ``d_model`` representation to
        ``num_classes`` logits.

    Examples
    --------
    >>> import torch
    >>> num_channels = 5
    >>> physics_matrix = torch.eye(num_channels)
    >>> model = PhySATFormer(
    ...     input_dim=1,
    ...     channel_embedding_dim=16,
    ...     d_model=64,
    ...     num_heads=4,
    ...     num_channel_layers=2,
    ...     num_temporal_layers=2,
    ...     ff_dim=128,
    ...     num_classes=5,
    ...     physics_matrix=physics_matrix,
    ... )
    >>> x = torch.randn(8, 32, num_channels)  # (batch, seq_len, channels)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([8, 5])
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
        num_classes: int,
        physics_matrix: torch.Tensor,
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
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise ValueError(f"num_classes must be a positive int, got {num_classes!r}")
        
        if not isinstance(physics_matrix, torch.Tensor):
            raise TypeError(
                f"physics_matrix must be a torch.Tensor, got "
                f"{type(physics_matrix).__name__}"
            )
        
        if physics_matrix.dim() != 2 or physics_matrix.shape[0] != physics_matrix.shape[1]:
            raise ValueError(
                "physics_matrix must be a square 2-D tensor of shape "
                f"(num_channels, num_channels), got shape "
                f"{tuple(physics_matrix.shape)}."
            )
        
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

        self.input_dim = input_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_channel_layers = num_channel_layers
        self.num_temporal_layers = num_temporal_layers
        self.ff_dim = ff_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.num_channels = physics_matrix.shape[0]

        self.telemetry_channel_encoder = TelemetryChannelEncoder(
            channel_embedding_dim=channel_embedding_dim,
            input_dim=input_dim,
        )

        self.channel_attention_layers = nn.ModuleList(
            [
                ChannelAttentionBlock(
                    channel_embedding_dim=channel_embedding_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    physics_matrix=physics_matrix,
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

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the full PhySATFormer architecture.

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
            Class logits of shape ``(batch_size, num_classes)``.

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

        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)

        return logits