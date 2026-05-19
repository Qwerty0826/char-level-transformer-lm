"""
Decoder-only Transformer language model.

Architecture (pre-norm, modern LLM style):
    token_embedding  →  [TransformerBlock] × num_layers  →  RMSNorm  →  lm_head

Each TransformerBlock:
    z = x + MHA(RMSNorm(x))
    y = z + FFN(RMSNorm(z))

Components:
  - Multi-head self-attention with RoPE (Su 2021), optional Grouped Query
    Attention (Ainslie 2023), KV caching for fast incremental decoding.
  - Feed-forward: SwiGLU (Shazeer 2020):
        FFN(x) = W_2 (SiLU(W_1 x) ⊙ W_3 x)
        d_ff = ceil(8/3 * d_model) rounded up to nearest multiple of 64.
  - RMSNorm (Zhang & Sennrich 2019), upcasts to fp32 for stability.
  - Optional weight tying between token embedding and LM head.

Ablations exposed (Section 7.3): use_norm, pre_norm, use_rope, use_gate.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from cs336_basics.attention import CausalMultiHeadSelfAttention
from cs336_basics.nn_components import Embedding, Linear, RMSNorm


KVCache = Tuple[torch.Tensor, torch.Tensor]            # (K, V) for one layer
ModelKVCache = List[KVCache]                           # one per block


def _round_to_multiple(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# SwiGLU feed-forward network
# ---------------------------------------------------------------------------

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU feed-forward (Shazeer 2020): FFN(x) = W2(SiLU(W1 x) ⊙ W3 x)

    d_ff defaults to round(8/3 * d_model) aligned to the nearest 64
    (hardware-friendly tensor cores).  ``use_gate=False`` removes the W3
    branch — equivalent to a plain SiLU FFN for ablation.
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

        self.W1 = Linear(d_model, d_ff,    device=device, dtype=dtype)
        self.W2 = Linear(d_ff,    d_model, device=device, dtype=dtype)
        if use_gate:
            self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.W1(x)
        activated = gate * torch.sigmoid(gate)       # SiLU
        if self.use_gate:
            activated = activated * self.W3(x)       # SwiGLU gating
        return self.W2(activated)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single Transformer block.

    Default (pre_norm=True, use_norm=True, use_rope=True, use_gate=True):
        z = x + MHA(RMSNorm(x))
        y = z + FFN(RMSNorm(z))

    Ablations (PDF §7.3):
        use_norm=False   → swap RMSNorm for identity
        pre_norm=False   → post-norm placement
        use_rope=False   → no positional encoding (NoPE)
        use_gate=False   → SiLU-only FFN (no SwiGLU gate)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        max_seq_len: int = 2048,
        theta: float = 10_000.0,
        num_kv_heads: Optional[int] = None,
        use_norm: bool = True,
        pre_norm: bool = True,
        use_rope: bool = True,
        use_gate: bool = True,
        chunk_size: Optional[int] = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.pre_norm = pre_norm

        self.attn_norm = (
            RMSNorm(d_model, device=device, dtype=dtype) if use_norm else nn.Identity()
        )
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta,
            num_kv_heads=num_kv_heads, use_rope=use_rope,
            chunk_size=chunk_size,
            device=device, dtype=dtype,
        )
        self.ff_norm = (
            RMSNorm(d_model, device=device, dtype=dtype) if use_norm else nn.Identity()
        )
        self.ff = SwiGLUFeedForward(d_model, d_ff, use_gate=use_gate, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
        return_cache: bool = False,
    ):
        if return_cache or kv_cache is not None:
            # KV-cached path (always pre-norm; caching during training is unsupported).
            attn_out, new_cache = self.attn(
                self.attn_norm(x), kv_cache=kv_cache,
                position_offset=position_offset, return_cache=True,
            )
            x = x + attn_out
            x = x + self.ff(self.ff_norm(x))
            return x, new_cache

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
    Decoder-only Transformer language model.

    Args:
        vocab_size:     Vocabulary size.
        context_length: Maximum context (sequence) length supported.
        d_model:        Hidden dimension.
        num_layers:     Number of Transformer blocks.
        num_heads:      Query heads per block.
        d_ff:           Inner FFN dimension.  Defaults to ~8/3 × d_model.
        theta:          RoPE base frequency.
        num_kv_heads:   K/V heads per block (GQA).  None ⇒ same as num_heads
                        (vanilla MHA).  1 ⇒ multi-query attention.
        tie_weights:    Share token embedding ↔ LM head matrix.
        use_norm / pre_norm / use_rope / use_gate:  Section 7.3 ablation
                        switches; True/True/True/True is the canonical model.
        chunk_size:     If set, use chunked memory-efficient attention for
                        training sequences longer than this.
        device / dtype: Standard.

    Methods:
        forward(token_ids)               — training / one-shot inference.
        forward_with_cache(...)          — KV-cached step (decoding).
        generate(prompt_ids, ...)        — KV-cached autoregressive sampling.
        num_parameters / estimate_flops_per_token — diagnostics.
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
        num_kv_heads: Optional[int] = None,
        tie_weights: bool = True,
        use_norm: bool = True,
        pre_norm: bool = True,
        use_rope: bool = True,
        use_gate: bool = True,
        chunk_size: Optional[int] = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, theta,
                num_kv_heads=num_kv_heads,
                use_norm=use_norm, pre_norm=pre_norm,
                use_rope=use_rope, use_gate=use_gate,
                chunk_size=chunk_size,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        if tie_weights:
            self.lm_head.W = self.token_embedding.weight

    # ---- Standard forward (training / non-cached inference) ----------------

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    # ---- KV-cached forward (incremental decoding) --------------------------

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        kv_caches: Optional[ModelKVCache] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, ModelKVCache]:
        """
        Run a forward pass that accepts and returns a KV cache.

        Args:
            token_ids:       (B, T_new) tokens to process at this step.
            kv_caches:       Per-layer (K, V) cache from the previous call,
                             or None on the first step.
            position_offset: First absolute position of the new tokens.

        Returns:
            (logits, new_caches) where logits has shape (B, T_new, vocab_size).
        """
        x = self.token_embedding(token_ids)
        new_caches: ModelKVCache = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache_i = block(
                x, kv_cache=cache_i, position_offset=position_offset,
                return_cache=True,
            )
            new_caches.append(new_cache_i)
        x = self.final_norm(x)
        return self.lm_head(x), new_caches

    # ---- Sampling helpers --------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        KV-cached autoregressive generation.

        Args:
            prompt_ids:        (1, T_prompt) LongTensor on the model's device.
            max_new_tokens:    Maximum number of tokens to generate.
            temperature:       Softmax temperature (>0, <1 sharper, >1 flatter).
            top_p:             Nucleus threshold (0–1). None → no nucleus.
            top_k:             Keep only the top-k logits. None → no top-k.
            min_p:             Min-probability filter (Min-P sampling).
            repetition_penalty: 1.0 = no penalty.  > 1 discourages repetition
                               (Keskar et al. 2019).
            eos_id:            Optional EOS token to stop at.
            use_cache:         Enable KV cache (O(T) vs O(T²)).

        Returns:
            (1, T_prompt + N) LongTensor of full sequence.
        """
        device = prompt_ids.device
        self.eval()
        tokens = prompt_ids

        kv_caches: Optional[ModelKVCache] = None
        if use_cache:
            # Prefill: process the whole prompt and cache the K/V's.
            logits, kv_caches = self.forward_with_cache(tokens, None, 0)
            position_offset = tokens.shape[1]
        else:
            logits = self.forward(tokens)
            position_offset = tokens.shape[1]

        for _ in range(max_new_tokens):
            # Take logits for the last position.
            next_logits = logits[0, -1, :].clone()

            # Repetition penalty.
            if repetition_penalty != 1.0:
                generated = tokens[0]
                for tok in set(generated.tolist()):
                    if next_logits[tok] > 0:
                        next_logits[tok] /= repetition_penalty
                    else:
                        next_logits[tok] *= repetition_penalty

            # Temperature.
            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-8)

            # Top-k filter.
            if top_k is not None and top_k > 0 and top_k < next_logits.numel():
                kth = torch.topk(next_logits, top_k).values[-1]
                next_logits = torch.where(
                    next_logits < kth,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            probs = torch.softmax(next_logits, dim=-1)

            # Min-p filter.
            if min_p is not None and min_p > 0:
                threshold = probs.max() * min_p
                probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)
                probs = probs / probs.sum().clamp_min(1e-12)

            # Top-p (nucleus) filter.
            if top_p is not None and 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                keep_mask = cumulative - sorted_probs <= top_p   # include token that crosses
                # Always keep the top-1.
                keep_mask[0] = True
                filtered = torch.zeros_like(sorted_probs)
                filtered[keep_mask] = sorted_probs[keep_mask]
                # Scatter back to vocab order.
                probs = torch.zeros_like(probs)
                probs[sorted_idx] = filtered
                probs = probs / probs.sum().clamp_min(1e-12)

            # Sample one token.
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

            # Truncate to context_length if it overflows.
            if tokens.shape[1] > self.context_length:
                # Drop the oldest token from cache; for simplicity rebuild instead.
                kv_caches = None
                position_offset = 0
                tokens = tokens[:, -self.context_length:]
                logits, kv_caches = self.forward_with_cache(tokens, None, 0) if use_cache \
                    else (self.forward(tokens), None)
                position_offset = tokens.shape[1]
                continue

            # Feed only the newest token, reusing the cache.
            if use_cache:
                logits, kv_caches = self.forward_with_cache(
                    next_token.unsqueeze(0), kv_caches, position_offset,
                )
                position_offset += 1
            else:
                logits = self.forward(tokens)
                position_offset = tokens.shape[1]

        return tokens

    # ---- Diagnostics -------------------------------------------------------

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count trainable parameters (optionally excluding token embeddings)."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.token_embedding.weight.numel()
        return total

    def estimate_flops_per_token(self) -> int:
        """
        Forward FLOPs for a full T-length sequence (matmul A(m,n)@B(n,p) = 2mnp).
        Returns per-sequence cost; divide by T for true per-token cost.
        Backward is ≈ 2× forward; total training FLOPs ≈ 3× this number.
        """
        d = self.d_model
        T = self.context_length
        num_layers = len(self.blocks)
        d_ff = self.blocks[0].ff.d_ff
        n_q = self.blocks[0].attn.num_q_heads
        n_kv = self.num_kv_heads
        d_k = d // n_q
        ffn_mats = 3 if self.blocks[0].ff.use_gate else 2

        attn_qproj  = 2 * T * d * d                       # Q: (T,d)·(d, n_q*d_k = d)
        attn_kvproj = 2 * 2 * T * d * (n_kv * d_k)        # K + V projections (GQA-sized)
        attn_oproj  = 2 * T * d * d                       # O: (T,d)·(d,d)
        attn_qk     = 2 * T * T * d                       # QK^T (expanded heads)
        attn_av     = 2 * T * T * d                       # softmax(QK^T)V
        ffn         = ffn_mats * 2 * T * d * d_ff

        per_layer = attn_qproj + attn_kvproj + attn_oproj + attn_qk + attn_av + ffn
        return num_layers * per_layer + 2 * T * d * self.vocab_size       # LM head
