"""Multi-head self-attention mechanism for the PhySATFormer encoder."""

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module.

    Implements scaled dot-product multi-head self-attention as described
    in "Attention Is All You Need" (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        Dimensionality of the input and output embeddings.
    num_heads : int
        Number of attention heads. Must evenly divide ``d_model``.
    dropout : float, optional
        Dropout probability applied to the attention weights, by default 0.1.

    Attributes
    ----------
    d_model : int
        Dimensionality of the input and output embeddings.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimensionality of each attention head.
    query_projection : nn.Linear
        Linear layer used to project inputs into queries.
    key_projection : nn.Linear
        Linear layer used to project inputs into keys.
    value_projection : nn.Linear
        Linear layer used to project inputs into values.
    output_projection : nn.Linear
        Linear layer used to project concatenated head outputs back to
        ``d_model``.
    dropout : nn.Dropout
        Dropout layer applied to the attention weights.

    Raises
    ------
    ValueError
        If ``d_model`` is not positive, ``num_heads`` is not positive, or
        ``d_model`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape a projected tensor into separate attention heads.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of shape ``(batch_size, sequence_length, d_model)``.
        batch_size : int
            Number of sequences in the batch.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, num_heads, sequence_length,
            head_dim)``.
        """
        sequence_length = tensor.size(1)
        tensor = tensor.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Merge separate attention heads back into a single tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of shape ``(batch_size, num_heads, sequence_length,
            head_dim)``.
        batch_size : int
            Number of sequences in the batch.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, sequence_length, d_model)``.
        """
        sequence_length = tensor.size(2)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(batch_size, sequence_length, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention over the input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, sequence_length, d_model)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, sequence_length, d_model)``.
        """
        batch_size = x.size(0)

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, value)
        context = self._merge_heads(context, batch_size)

        return self.output_projection(context)