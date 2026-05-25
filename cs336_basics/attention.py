"""
Attention primitives.

Implements:
  - Numerically-stable softmax.
  - Rotary Positional Embeddings (RoPE, Su et al., 2021).
  - Masked scaled dot-product attention (causal + arbitrary mask).
  - Causal multi-head self-attention with optional:
      * Grouped Query Attention (GQA, Llama-2/3, Mistral)
      * KV caching for fast incremental decoding (O(T) instead of O(T^2))
      * Chunked computation for memory-efficient long-context training
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics.nn_components import Linear


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically-stable softmax along dimension `dim`."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021).

    Pairs of dimensions (2k, 2k+1) of the query/key vector are rotated by an
    angle that depends on the token's absolute position and the dimension
    pair index k.  Encodes relative position information through the dot
    product, generalises better to unseen sequence lengths than learned
    absolute embeddings, and is what Llama, PaLM, GPT-NeoX all use.
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
            Rotated tensor of the same shape and dtype as ``x``.
        """
        # cos/sin tables are stored in float32 for precision; cast to x's
        # dtype so we don't accidentally upcast x to float32 here (which
        # would create a dtype mismatch with V later in attention).
        cos = self.cos_table[token_positions].to(x.dtype)
        sin = self.sin_table[token_positions].to(x.dtype)

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q:    (..., seq_q, d_k)
        K:    (..., seq_k, d_k)
        V:    (..., seq_k, d_v)
        mask: Optional bool tensor broadcastable to (..., seq_q, seq_k).
              True = attend, False = block.
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... sq dk, ... sk dk -> ... sq sk") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return einsum(softmax(scores, dim=-1), V, "... sq sk, ... sk dv -> ... sq dv")


def chunked_causal_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Memory-efficient causal attention that processes the query sequence in
    chunks, never materialising the full (T, T) attention matrix.

    For seq_len T and chunk size C, peak memory is O(C * T) per chunk instead
    of O(T * T) for the standard implementation -- crucial when T is large.

    Mathematically equivalent to ``scaled_dot_product_attention`` with a
    lower-triangular causal mask; the output is identical up to fp rounding.
    """
    *batch, h, T, d_k = Q.shape
    out = torch.empty_like(Q)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        q_chunk = Q[..., start:end, :]                       # (..., h, c, d_k)
        k_chunk = K[..., :end, :]                            # causal: keys up to current
        v_chunk = V[..., :end, :]

        # Per-chunk causal mask: chunk's rows can attend to all keys up to
        # their absolute position, not beyond.
        c = end - start
        # rows: absolute positions [start, start+1, ..., end-1]
        # cols: absolute positions [0, 1, ..., end-1]
        row_idx = torch.arange(start, end, device=Q.device).view(c, 1)
        col_idx = torch.arange(0, end,   device=Q.device).view(1, end)
        mask = col_idx <= row_idx                             # (c, end), bool

        out[..., start:end, :] = scaled_dot_product_attention(
            q_chunk, k_chunk, v_chunk, mask=mask,
        )
    return out


# ---------------------------------------------------------------------------
# Causal multi-head / grouped-query self-attention
# ---------------------------------------------------------------------------

class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal self-attention with three modern extensions:

    1. Rotary Position Embedding (RoPE) on Q and K.
    2. Grouped Query Attention (GQA, Ainslie et al. 2023):
       fewer K/V heads than Q heads, reducing KV-cache size and bandwidth.
       Set ``num_kv_heads = num_heads`` for standard multi-head attention,
       ``num_kv_heads = 1`` for multi-query attention.
    3. KV cache for fast incremental decoding -- pass ``kv_cache=...`` to
       ``forward()`` and the layer will append to it and return the new cache.

    During training the layer behaves exactly like vanilla MHA; the cache
    machinery is only used at inference time.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float = 10_000.0,
        num_kv_heads: Optional[int] = None,
        use_rope: bool = True,
        chunk_size: Optional[int] = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) for GQA"
        )

        self.num_q_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_groups = num_heads // num_kv_heads      # repeats per kv head
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.chunk_size = chunk_size                       # None = standard attn

        # Projection dims: Q is full d_model, K/V are smaller under GQA.
        self.W_Q = Linear(d_model, num_heads    * self.d_k, device=device, dtype=dtype)
        self.W_K = Linear(d_model, num_kv_heads * self.d_k, device=device, dtype=dtype)
        self.W_V = Linear(d_model, num_kv_heads * self.d_k, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model,                 device=device, dtype=dtype)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand K/V from num_kv_heads to num_q_heads via repeat_interleave."""
        if self.head_groups == 1:
            return kv
        return kv.repeat_interleave(self.head_groups, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        return_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x:               (B, T_new, d_model) input tokens to attend to.
            kv_cache:        Optional (K_cache, V_cache) from a previous step.
                             Shapes: (B, num_kv_heads, T_cached, d_k).
            position_offset: First absolute position of ``x`` (i.e. T_cached
                             when extending a cache).
            return_cache:    If True, return ``(out, new_cache)``. If False
                             (default), return only ``out`` — preserves the
                             standard training signature.
            attention_mask:  Optional (B, T_new) bool tensor. True = real token,
                             False = padding. Combined with the causal mask in
                             the training/prefill path. Padded positions cannot
                             be attended to (key side) and contribute no signal.
        """
        B, T_new, _ = x.shape

        Q = rearrange(self.W_Q(x), "b t (h d) -> b h t d", h=self.num_q_heads)
        K_new = rearrange(self.W_K(x), "b t (h d) -> b h t d", h=self.num_kv_heads)
        V_new = rearrange(self.W_V(x), "b t (h d) -> b h t d", h=self.num_kv_heads)

        # Rotary embeddings: positions are offset by the cache length so that
        # the first new token sits at the correct absolute index.
        if self.use_rope:
            q_positions = torch.arange(
                position_offset, position_offset + T_new, device=x.device,
            )
            Q     = self.rope(Q,     q_positions)
            K_new = self.rope(K_new, q_positions)

        # Splice cache (already-rotated) onto the new K/V.
        if kv_cache is not None:
            K_cached, V_cached = kv_cache
            K_full = torch.cat([K_cached, K_new], dim=-2)
            V_full = torch.cat([V_cached, V_new], dim=-2)
        else:
            K_full, V_full = K_new, V_new

        # Expand for GQA. The cache stores the compact num_kv_heads version
        # (that's the entire point of GQA: smaller cache).
        K_attn = self._expand_kv(K_full)
        V_attn = self._expand_kv(V_full)

        T_total = K_attn.shape[-2]
        if kv_cache is None and position_offset == 0:
            # Training / prefill: standard (T, T) lower-triangular mask.
            if self.chunk_size is not None and T_new > self.chunk_size:
                # Chunked path doesn't support padding masks; intended for long
                # contiguous training, not batched SFT with variable lengths.
                attn_out = chunked_causal_attention(Q, K_attn, V_attn, self.chunk_size)
            else:
                causal = torch.ones(T_new, T_new, device=x.device, dtype=torch.bool).tril()
                if attention_mask is not None:
                    # (B, T) keys — block attention TO padding positions.
                    mask = causal[None, None, :, :] & attention_mask[:, None, None, :]
                else:
                    mask = causal
                attn_out = scaled_dot_product_attention(Q, K_attn, V_attn, mask=mask)
        else:
            # Decoding: new queries can attend to all cached + earlier new keys.
            row = torch.arange(position_offset, position_offset + T_new,
                               device=x.device).view(T_new, 1)
            col = torch.arange(T_total, device=x.device).view(1, T_total)
            mask = col <= row     # (T_new, T_total)
            attn_out = scaled_dot_product_attention(Q, K_attn, V_attn, mask=mask)

        out = self.W_O(rearrange(attn_out, "b h t d -> b t (h d)"))

        if return_cache:
            return out, (K_full, V_full)
        return out
