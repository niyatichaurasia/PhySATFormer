"""Baseline Transformer model for PhySATFormer.

This module implements a standard Transformer encoder classifier that
serves as the baseline architecture against which PhySATFormer variants
are compared. It reuses the shared building blocks already implemented
elsewhere in the codebase (``PositionalEncoding`` and
``TransformerEncoderBlock``) rather than duplicating their logic.
"""

import torch
import torch.nn as nn

from src.models.positional_encoding import PositionalEncoding
from src.models.encoder_block import TransformerEncoderBlock


class BaselineTransformer(nn.Module):
    """Standard Transformer encoder for sequence classification.

    The model projects raw input features into the model dimension,
    injects positional information, applies a stack of Transformer
    encoder blocks, mean-pools over the sequence dimension, and
    produces class logits via a linear classification head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features at each sequence position.
        Must be positive.
    d_model : int
        Dimensionality of the model's internal (hidden) representations.
        Must be positive.
    num_heads : int
        Number of attention heads used in each Transformer encoder
        block. Must be positive and typically must evenly divide
        ``d_model``.
    num_layers : int
        Number of stacked ``TransformerEncoderBlock`` layers. Must be
        positive.
    ff_dim : int
        Dimensionality of the feed-forward hidden layer within each
        Transformer encoder block. Must be positive.
    num_classes : int
        Number of output classes for classification. Must be positive.
    dropout : float, optional
        Dropout probability applied throughout the model, by default
        0.1.

    Attributes
    ----------
    input_projection : nn.Linear
        Linear layer mapping ``input_dim`` to ``d_model``.
    positional_encoding : PositionalEncoding
        Module that injects positional information into the
        input embeddings.
    encoder_layers : nn.ModuleList
        Stack of ``TransformerEncoderBlock`` modules.
    classifier : nn.Linear
        Linear layer mapping the pooled ``d_model`` representation to
        ``num_classes`` logits.

    Examples
    --------
    >>> model = BaselineTransformer(
    ...     input_dim=16,
    ...     d_model=64,
    ...     num_heads=4,
    ...     num_layers=2,
    ...     ff_dim=128,
    ...     num_classes=5,
    ... )
    >>> x = torch.randn(8, 32, 16)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([8, 5])
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if ff_dim <= 0:
            raise ValueError(f"ff_dim must be positive, got {ff_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass through the baseline Transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, sequence_length,
            input_dim)``.
        mask : torch.Tensor or None, optional
            Optional attention mask forwarded to each
            ``TransformerEncoderBlock``, by default None.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch_size, num_classes)``.
        """
        hidden = self.input_projection(x)
        hidden = self.positional_encoding(hidden)

        for encoder_layer in self.encoder_layers:
            hidden = encoder_layer(hidden)

        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)

        return logits