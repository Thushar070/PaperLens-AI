"""Padding-aware, project-implemented Transformer components."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions, stored as a non-trainable buffer."""

    def __init__(self, model_dim: int, max_length: int = 290) -> None:
        super().__init__()
        if model_dim < 1 or max_length < 1:
            raise ValueError("model_dim and max_length must be positive")

        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dim)
        )
        encoding = torch.zeros(max_length, model_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)
        self.model_dim = model_dim
        self.max_length = max_length

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.model_dim:
            raise ValueError("Expected embeddings with shape (batch, length, model_dim)")
        if embeddings.shape[1] > self.max_length:
            raise ValueError("Input sequence exceeds the configured maximum length")
        return embeddings + self.encoding[:, : embeddings.shape[1]].to(embeddings.dtype)


class ScaledDotProductAttention(nn.Module):
    """Scaled attention with boolean masks where True denotes a valid position."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        query_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Query, key, and value must have shape (batch, heads, length, dim)")
        if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
            raise ValueError("Query, key, and value batch/head dimensions must match")
        if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
            raise ValueError("Query/key dimensions and key/value lengths must match")
        if not (query.device == key.device == value.device):
            raise ValueError("Query, key, and value must be on the same device")

        batch, _, query_length, _ = query.shape
        key_length = key.shape[-2]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        valid = torch.ones_like(scores, dtype=torch.bool)

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != (batch, key_length):
                raise ValueError("key_padding_mask must be boolean with shape (batch, key_length)")
            valid &= key_padding_mask[:, None, None, :]

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise ValueError("attention_mask must be boolean; True means allowed")
            if attention_mask.ndim == 2 and attention_mask.shape == (query_length, key_length):
                mask = attention_mask[None, None, :, :]
            elif attention_mask.ndim == 3 and attention_mask.shape == (batch, query_length, key_length):
                mask = attention_mask[:, None, :, :]
            elif attention_mask.ndim == 4 and attention_mask.shape[-2:] == (query_length, key_length):
                mask = attention_mask
            else:
                raise ValueError("attention_mask must broadcast over batch/heads and query/key lengths")
            valid &= mask

        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).masked_fill(~valid, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )

        if query_padding_mask is not None:
            if query_padding_mask.dtype != torch.bool or query_padding_mask.shape != (batch, query_length):
                raise ValueError("query_padding_mask must be boolean with shape (batch, query_length)")
            weights = weights * query_padding_mask[:, None, :, None]

        output = torch.matmul(weights, value)
        return output, weights


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with explicit trainable Q, K, V, and output projections."""

    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        if model_dim < 1 or num_heads < 1 or model_dim % num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.attention = ScaledDotProductAttention()

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, inputs: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.model_dim:
            raise ValueError("Expected input with shape (batch, length, model_dim)")
        if padding_mask is not None and padding_mask.shape != inputs.shape[:2]:
            raise ValueError("padding_mask must match the input batch and sequence dimensions")

        query = self._split_heads(self.q_proj(inputs))
        key = self._split_heads(self.k_proj(inputs))
        value = self._split_heads(self.v_proj(inputs))
        attended, weights = self.attention(
            query,
            key,
            value,
            key_padding_mask=padding_mask,
            query_padding_mask=padding_mask,
        )
        batch, _, length, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().view(batch, length, self.model_dim)
        output = self.out_proj(merged)
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1)
        return output, weights


class PositionwiseFeedForward(nn.Module):
    """Two-layer ReLU feed-forward network applied independently per token."""

    def __init__(self, model_dim: int, feedforward_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if model_dim < 1 or feedforward_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("Dimensions must be positive and dropout must be in [0, 1)")
        self.layers = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TransformerEncoderBlock(nn.Module):
    """Post-normalized attention and feed-forward residual sublayers."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.self_attention = MultiHeadSelfAttention(model_dim, num_heads)
        self.feedforward = PositionwiseFeedForward(model_dim, feedforward_dim, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self, inputs: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.self_attention(inputs, padding_mask)
        hidden = self.attention_norm(inputs + self.attention_dropout(attended))
        if padding_mask is not None:
            hidden = hidden * padding_mask.unsqueeze(-1)
        fed = self.feedforward(hidden)
        output = self.output_norm(hidden + self.output_dropout(fed))
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1)
        return output, weights


class CustomTransformerClassifier(nn.Module):
    """Transformer encoder classifier using the saved Phase 2 token IDs."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.1,
        max_length: int = 290,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or num_layers < 1 or num_classes < 2:
            raise ValueError("vocab_size, num_layers, and num_classes are invalid")
        if not 0 <= pad_idx < vocab_size:
            raise ValueError("pad_idx must be a valid vocabulary index")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_length = max_length
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.position = PositionalEncoding(model_dim, max_length)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            TransformerEncoderBlock(model_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        )
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(
        self, token_ids: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2 or token_ids.shape[0] < 1 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have nonempty shape (batch, sequence_length)")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must use torch.long dtype")
        if token_ids.shape[1] > self.max_length:
            raise ValueError("Input sequence exceeds the configured maximum length")
        if token_ids.numel() and (token_ids.min() < 0 or token_ids.max() >= self.vocab_size):
            raise ValueError("token_ids contain an index outside the vocabulary")

        valid = token_ids.ne(self.pad_idx)
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Padding must be on the right")

        hidden = self.embedding(token_ids) * math.sqrt(self.model_dim)
        hidden = self.position(hidden) * valid.unsqueeze(-1)
        hidden = self.input_dropout(hidden)
        attention_weights = None
        for layer in self.encoder_layers:
            hidden, attention_weights = layer(hidden, valid)

        valid_float = valid.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        if return_attention:
            return logits, attention_weights
        return logits
