"""Sinusoidal positional encoding module for transformer embeddings."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional encodings to input embeddings.

    Implements the sinusoidal positional encoding described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        Dimensionality of the embeddings.
    dropout : float, optional
        Dropout probability applied after adding positional encodings,
        by default 0.1.
    max_length : int, optional
        Maximum supported sequence length, by default 5000.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_length: int = 5000,
    ) -> None:
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}.")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}.")

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        positional_encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, d_model).
        """
        sequence_length = x.size(1)
        max_length = self.positional_encoding.size(1)
        if sequence_length > max_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds the "
                f"maximum supported length {max_length}."
            )
        x = x + self.positional_encoding[:, :sequence_length, :].to(dtype=x.dtype)
        return self.dropout(x)