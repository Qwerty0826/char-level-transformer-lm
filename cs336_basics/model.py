"""
Transformer language model assembled from scratch-built primitives.

Architecture overview (pre-norm, decoder-only):
  token_embedding  →  [TransformerBlock] × num_layers  →  RMSNorm  →  lm_head

Each TransformerBlock:
  z = x + MHA(RMSNorm(x))
  y = z + FFN(RMSNorm(z))

Feed-forward uses SwiGLU (Shazeer, 2020):
  FFN(x) = W_2 * (SiLU(W_1 x) ⊙ W_3 x)
  d_ff ≈ 8/3 * d_model, rounded up to the nearest multiple of 64.

Optional weight tying between the input embedding and LM head (reduces
parameter count and often improves perplexity on small models).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cs336_basics.attention import CausalMultiHeadSelfAttention
from cs336_basics.nn_components import Embedding, Linear, RMSNorm


def _round_to_multiple(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# SwiGLU feed-forward network
# ---------------------------------------------------------------------------

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU feed-forward network (Shazeer, 2020; used in Llama, PaLM, etc.).

        FFN(x) = W_2 * (SiLU(W_1 x) ⊙ W_3 x)

    where SiLU(x) = x * sigmoid(x).

    d_ff is set to round(8/3 * d_model) aligned to the nearest 64 for
    hardware efficiency.  Pass an explicit d_ff to override.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        use_gate: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = _round_to_multiple(int(8 / 3 * d_model), 64)
        self.d_ff = d_ff
        self.use_gate = use_gate

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff,   d_model, device=device, dtype=dtype)
        if use_gate:
            self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.W1(x)
        activated = gate * torch.sigmoid(gate)   # SiLU
        if self.use_gate:
            activated = activated * self.W3(x)   # SwiGLU gating
        return self.W2(activated)


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Single Transformer block, supporting ablation variants.

    Default (pre_norm=True, use_norm=True):
        z = x + MHA(RMSNorm(x))
        y = z + FFN(RMSNorm(z))

    Ablations:
        use_norm=False  → remove RMSNorm (identity in its place)
        pre_norm=False  → post-norm: z = RMSNorm(x + MHA(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        max_seq_len: int = 2048,
        theta: float = 10_000.0,
        use_norm: bool = True,
        pre_norm: bool = True,
        use_rope: bool = True,
        use_gate: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.pre_norm = pre_norm

        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype) if use_norm else nn.Identity()
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, use_rope=use_rope,
            device=device, dtype=dtype,
        )
        self.ff_norm = RMSNorm(d_model, device=device, dtype=dtype) if use_norm else nn.Identity()
        self.ff = SwiGLUFeedForward(d_model, d_ff, use_gate=use_gate, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ff(self.ff_norm(x))
        else:
            x = self.attn_norm(x + self.attn(x))
            x = self.ff_norm(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Full Transformer Language Model
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """
    Decoder-only Transformer Language Model.

    Takes batched integer token sequences and returns logits over the
    vocabulary for every position (next-token prediction).

    Args:
        vocab_size:     Vocabulary size.
        context_length: Maximum context (sequence) length.
        d_model:        Hidden dimension.
        num_layers:     Number of Transformer blocks.
        num_heads:      Number of attention heads per block.
        d_ff:           Inner FFN dimension.  Defaults to ~8/3 * d_model.
        theta:          RoPE base frequency.
        tie_weights:    If True, share the token-embedding matrix with the
                        LM-head projection (weight tying).
        device:         Parameter device.
        dtype:          Parameter dtype.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int | None = None,
        theta: float = 10_000.0,
        tie_weights: bool = True,
        use_norm: bool = True,
        pre_norm: bool = True,
        use_rope: bool = True,
        use_gate: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, theta,
                use_norm=use_norm, pre_norm=pre_norm,
                use_rope=use_rope, use_gate=use_gate,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        if tie_weights:
            # Share the embedding weight with the LM-head projection.
            # lm_head.W has shape (vocab_size, d_model) == embedding.weight.
            self.lm_head.W = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, sequence_length) LongTensor

        Returns:
            logits: (batch_size, sequence_length, vocab_size)
        """
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count trainable parameters (optionally excluding embeddings)."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            emb_params = self.token_embedding.weight.numel()
            total -= emb_params
        return total

    def estimate_flops_per_token(self) -> int:
        """
        Rough FLOPs per forward token.  Rule: matmul A(m,n) @ B(n,p) = 2mnp.
        Counts QKV projections, attention (O(T^2*d)), FFN, and LM head.
        """
        d = self.d_model
        T = self.context_length
        num_layers = len(self.blocks)
        d_ff = self.blocks[0].ff.d_ff

        # Per-layer: 4 matmuls of d×d (Q,K,V,O) + QK attn + AV + 3 FFN matmuls
        attn_proj = 4 * 2 * T * d * d
        attn_scores = 2 * T * T * d           # QK^T
        attn_values = 2 * T * T * d           # A*V
        ffn = 3 * 2 * T * d * d_ff            # W1, W2, W3

        per_layer = attn_proj + attn_scores + attn_values + ffn
        return num_layers * per_layer + 2 * T * d * self.vocab_size  # LM head
