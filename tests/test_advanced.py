"""
Tests for advanced features: KV cache equivalence, Grouped Query Attention,
chunked attention, and the autoregressive generate() helper.
"""

import pytest
import torch

from cs336_basics.attention import (
    CausalMultiHeadSelfAttention,
    chunked_causal_attention,
    scaled_dot_product_attention,
)
from cs336_basics.model import TransformerLM


# ---- KV cache equivalence --------------------------------------------------

def test_kv_cache_matches_full_forward():
    """A model run incrementally with a KV cache must produce identical
    next-token logits to the same model fed the full sequence in one shot."""
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=64, context_length=32,
        d_model=64, num_layers=2, num_heads=4, d_ff=128,
        tie_weights=False,
    )
    model.eval()

    ids = torch.randint(0, 64, (1, 10))

    # Reference: full forward, take logits for the last token.
    with torch.no_grad():
        ref_logits = model(ids)[0, -1, :]

    # Incremental: prefill with ids[:-1] then step ids[-1].
    with torch.no_grad():
        _, caches = model.forward_with_cache(ids[:, :-1], None, 0)
        out, _    = model.forward_with_cache(ids[:, -1:], caches, ids.shape[1] - 1)
    cached_logits = out[0, -1, :]

    assert torch.allclose(ref_logits, cached_logits, atol=1e-4), \
        f"max diff = {(ref_logits - cached_logits).abs().max()}"


def test_kv_cache_token_by_token():
    """Reconstruct the full sequence one token at a time and match a single-pass run."""
    torch.manual_seed(1)
    model = TransformerLM(
        vocab_size=32, context_length=16,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
        tie_weights=False,
    )
    model.eval()

    ids = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        full = model(ids)

    caches = None
    offset = 0
    out_logits = []
    with torch.no_grad():
        for t in range(ids.shape[1]):
            logits, caches = model.forward_with_cache(ids[:, t:t+1], caches, offset)
            out_logits.append(logits)
            offset += 1
    incremental = torch.cat(out_logits, dim=1)

    assert torch.allclose(full, incremental, atol=1e-4)


# ---- Grouped Query Attention -----------------------------------------------

def test_gqa_shape_and_kv_cache_size():
    """GQA must produce the same output shape as MHA but a smaller KV cache."""
    d_model, num_heads, num_kv_heads = 32, 8, 2
    mha = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len=16,
                                       num_kv_heads=num_kv_heads)
    x = torch.randn(2, 10, d_model)
    out, (K, V) = mha(x, return_cache=True)
    assert out.shape == x.shape
    # K, V are stored with num_kv_heads, not num_heads
    assert K.shape == (2, num_kv_heads, 10, d_model // num_heads)
    assert V.shape == (2, num_kv_heads, 10, d_model // num_heads)


def test_gqa_equals_mha_when_kv_equals_heads():
    """num_kv_heads == num_heads must reduce to vanilla MHA."""
    torch.manual_seed(2)
    x = torch.randn(1, 5, 16)

    mha = CausalMultiHeadSelfAttention(16, 4, max_seq_len=8, num_kv_heads=4)
    out_mha = mha(x)
    assert out_mha.shape == x.shape


# ---- Chunked attention -----------------------------------------------------

def test_chunked_attention_matches_full():
    """Chunked causal attention must match a full (T,T) implementation."""
    torch.manual_seed(3)
    B, h, T, d = 1, 2, 16, 8
    Q = torch.randn(B, h, T, d)
    K = torch.randn(B, h, T, d)
    V = torch.randn(B, h, T, d)

    mask = torch.ones(T, T, dtype=torch.bool).tril()
    ref = scaled_dot_product_attention(Q, K, V, mask=mask)

    chunked = chunked_causal_attention(Q, K, V, chunk_size=4)
    assert torch.allclose(ref, chunked, atol=1e-5)


# ---- generate() helper -----------------------------------------------------

def test_generate_returns_correct_shape():
    torch.manual_seed(4)
    model = TransformerLM(
        vocab_size=32, context_length=16,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
        tie_weights=True,
    )
    prompt = torch.randint(0, 32, (1, 5))
    out = model.generate(prompt, max_new_tokens=7, temperature=1.0)
    assert out.shape == (1, 12)


def test_generate_with_top_p_and_top_k():
    """All sampling filters should produce a valid sequence."""
    torch.manual_seed(5)
    model = TransformerLM(
        vocab_size=32, context_length=16,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
    )
    prompt = torch.randint(0, 32, (1, 3))
    out = model.generate(
        prompt, max_new_tokens=5,
        temperature=0.8, top_p=0.9, top_k=10, min_p=0.05,
        repetition_penalty=1.1,
    )
    assert out.shape == (1, 8)
    assert (out >= 0).all() and (out < 32).all()


def test_generate_stream_matches_generate():
    """generate_stream() must yield the same tokens that generate() returns."""
    torch.manual_seed(6)
    model = TransformerLM(
        vocab_size=32, context_length=16,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
    )
    prompt = torch.randint(0, 32, (1, 4))

    torch.manual_seed(42)
    full = model.generate(prompt, max_new_tokens=8, temperature=0.7, top_p=0.9)

    torch.manual_seed(42)
    streamed = list(model.generate_stream(prompt, max_new_tokens=8, temperature=0.7, top_p=0.9))

    assert full.shape == (1, 12)
    assert streamed == full[0, prompt.shape[1]:].tolist()


def test_generate_stream_yields_incrementally():
    """generate_stream() must be a true generator (yield as it goes)."""
    torch.manual_seed(7)
    model = TransformerLM(
        vocab_size=32, context_length=8,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
    )
    prompt = torch.randint(0, 32, (1, 2))
    n_yielded = 0
    for _tok in model.generate_stream(prompt, max_new_tokens=4, temperature=1.0):
        n_yielded += 1
        if n_yielded == 2:
            break
    assert n_yielded == 2
