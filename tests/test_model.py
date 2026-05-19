"""
Unit tests for neural network components and the Transformer LM.
"""

import math
import pytest
import torch

from cs336_basics.attention import (
    CausalMultiHeadSelfAttention,
    RotaryPositionalEmbedding,
    scaled_dot_product_attention,
    softmax,
)
from cs336_basics.model import SwiGLUFeedForward, TransformerBlock, TransformerLM
from cs336_basics.nn_components import Embedding, Linear, RMSNorm


# ---- Linear ----------------------------------------------------------------

def test_linear_shape():
    lin = Linear(8, 4)
    assert lin(torch.randn(2, 3, 8)).shape == (2, 3, 4)


def test_linear_no_bias():
    lin = Linear(8, 4)
    assert not any("bias" in n for n, _ in lin.named_parameters())


def test_linear_gradient_flows():
    lin = Linear(8, 4)
    x = torch.randn(2, 8, requires_grad=True)
    lin(x).sum().backward()
    assert x.grad is not None


# ---- Embedding -------------------------------------------------------------

def test_embedding_shape():
    emb = Embedding(100, 16)
    ids = torch.randint(0, 100, (2, 10))
    assert emb(ids).shape == (2, 10, 16)


# ---- RMSNorm ---------------------------------------------------------------

def test_rmsnorm_unit_rms():
    norm = RMSNorm(16)
    x = torch.randn(50, 16) * 5.0
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert (rms - 1.0).abs().max() < 1e-4


def test_rmsnorm_dtype_preserved():
    norm = RMSNorm(8)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    assert norm(x).dtype == torch.bfloat16


# ---- Softmax ---------------------------------------------------------------

def test_softmax_sums_to_one():
    x = torch.randn(5, 10)
    s = softmax(x, dim=-1)
    assert (s.sum(-1) - 1.0).abs().max() < 1e-6


def test_softmax_matches_reference():
    x = torch.randn(3, 7)
    assert (softmax(x, -1) - torch.softmax(x, -1)).abs().max() < 1e-6


# ---- RoPE ------------------------------------------------------------------

def test_rope_shape_preserved():
    rope = RotaryPositionalEmbedding(theta=10000, d_k=16, max_seq_len=64)
    x = torch.randn(2, 4, 8, 16)
    positions = torch.arange(8)
    assert rope(x, positions).shape == x.shape


def test_rope_preserves_norm():
    rope = RotaryPositionalEmbedding(theta=10000, d_k=16, max_seq_len=64)
    x = torch.randn(10, 16)
    positions = torch.arange(10)
    x_rot = rope(x, positions)
    norms_diff = (x.norm(dim=-1) - x_rot.norm(dim=-1)).abs().max()
    assert norms_diff < 1e-5


# ---- Attention -------------------------------------------------------------

def test_sdpa_shape():
    Q = torch.randn(2, 8, 16)
    K = torch.randn(2, 8, 16)
    V = torch.randn(2, 8, 32)
    assert scaled_dot_product_attention(Q, K, V).shape == (2, 8, 32)


def test_sdpa_causal_mask():
    """Future tokens must not affect past outputs."""
    mha = CausalMultiHeadSelfAttention(d_model=32, num_heads=4, max_seq_len=16)
    x  = torch.randn(1, 10, 32)
    x2 = x.clone()
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])
    assert torch.allclose(mha(x)[:, :5, :], mha(x2)[:, :5, :], atol=1e-5)


# ---- SwiGLU ----------------------------------------------------------------

def test_swiglu_shape():
    ffn = SwiGLUFeedForward(64)
    x = torch.randn(2, 10, 64)
    assert ffn(x).shape == x.shape


def test_swiglu_d_ff_multiple_of_64():
    ffn = SwiGLUFeedForward(512)
    assert ffn.d_ff % 64 == 0


# ---- TransformerLM ---------------------------------------------------------

@pytest.fixture
def small_model():
    return TransformerLM(
        vocab_size=200, context_length=32,
        d_model=64, num_layers=2, num_heads=4, d_ff=128,
        tie_weights=True,
    )


def test_lm_output_shape(small_model):
    ids = torch.randint(0, 200, (2, 20))
    assert small_model(ids).shape == (2, 20, 200)


def test_weight_tying(small_model):
    assert small_model.lm_head.W is small_model.token_embedding.weight


def test_lm_gradient_flows(small_model):
    ids = torch.randint(0, 200, (2, 16))
    loss = small_model(ids).mean()
    loss.backward()
    for n, p in small_model.named_parameters():
        assert p.grad is not None, f"No gradient for {n}"


def test_lm_overfit_single_batch():
    """A small LM should drive loss near zero on a single repeated batch."""
    model = TransformerLM(
        vocab_size=20, context_length=8,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
    )
    from cs336_basics.optimizer import AdamW
    from cs336_basics.training import cross_entropy_loss

    opt = AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    ids = torch.randint(0, 20, (4, 8))

    first_loss = None
    for _ in range(100):
        opt.zero_grad()
        logits = model(ids)
        loss = cross_entropy_loss(logits[:, :-1].reshape(-1, 20), ids[:, 1:].reshape(-1))
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    assert loss.item() < first_loss * 0.2, \
        f"Should overfit: {first_loss:.3f} -> {loss.item():.3f}"
