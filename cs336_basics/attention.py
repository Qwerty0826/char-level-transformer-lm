"""
Attention primitives: numerically-stable softmax, Rotary Positional
Embeddings (RoPE), and masked scaled dot-product attention.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics.nn_components import Linear


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically-stable softmax along dimension `dim`."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


class RotaryPositionalEmbedding(nn.Module):
    """
    Applies Rotary Position Embeddings to a query or key tensor.

    For dimension pair (2k, 2k+1) the vector is rotated by angle
    position * Theta^{-2k/d_k}.  Precomputes cos/sin tables stored
    as non-persistent buffers (not saved in checkpoints).
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)
        )
        positions = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(positions, inv_freq)   # (max_seq_len, d_k/2)
        self.register_buffer("cos_table", torch.cos(angles), persistent=False)
        self.register_buffer("sin_table", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:               (..., seq_len, d_k)
            token_positions: (..., seq_len) integer positions
        Returns:
            Rotated tensor of the same shape.
        """
        cos = self.cos_table[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_table[token_positions]

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q:    (batch, ..., seq_q, d_k)
        K:    (batch, ..., seq_k, d_k)
        V:    (batch, ..., seq_k, d_v)
        mask: Optional bool tensor broadcastable to (..., seq_q, seq_k).
              True = attend, False = block.
    Returns:
        (batch, ..., seq_q, d_v)
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... sq dk, ... sk dk -> ... sq sk") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return einsum(softmax(scores, dim=-1), V, "... sq sk, ... sk dv -> ... sq dv")


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE (Su et al., 2021).

    RoPE is applied to the projected Q and K tensors only.
    A lower-triangular causal mask prevents each position from
    attending to any future position.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float = 10_000.0,
        use_rope: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.num_heads

        Q = rearrange(self.W_Q(x), "b t (h dk) -> b h t dk", h=h)
        K = rearrange(self.W_K(x), "b t (h dk) -> b h t dk", h=h)
        V = rearrange(self.W_V(x), "b t (h dv) -> b h t dv", h=h)

        if self.use_rope:
            positions = torch.arange(T, device=x.device)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)

        causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        return self.W_O(rearrange(attn_out, "b h t dv -> b t (h dv)"))
