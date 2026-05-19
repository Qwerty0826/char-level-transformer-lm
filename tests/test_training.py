"""
Unit tests for training utilities.
"""

import io
import math
import numpy as np
import pytest
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import (
    clip_gradient_norm,
    cross_entropy_loss,
    get_batch,
    get_lr_cosine_schedule,
    load_checkpoint,
    save_checkpoint,
)


# ---- Cross-entropy ---------------------------------------------------------

def test_cross_entropy_matches_pytorch():
    logits = torch.randn(4, 10, 50)
    targets = torch.randint(0, 50, (4, 10))
    ours = cross_entropy_loss(logits.view(-1, 50), targets.view(-1))
    ref  = torch.nn.functional.cross_entropy(logits.view(-1, 50), targets.view(-1))
    assert abs(ours.item() - ref.item()) < 1e-5


def test_cross_entropy_numerically_stable():
    logits = torch.tensor([[1000.0, 1001.0, 1002.0]])
    loss = cross_entropy_loss(logits, torch.tensor([2]))
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


# ---- LR schedule -----------------------------------------------------------

def test_lr_warmup_endpoints():
    assert get_lr_cosine_schedule(0, 1e-3, 1e-4, 100, 1000) == 0.0
    assert abs(get_lr_cosine_schedule(100, 1e-3, 1e-4, 100, 1000) - 1e-3) < 1e-12


def test_lr_postanneal():
    assert get_lr_cosine_schedule(5000, 1e-3, 1e-4, 100, 1000) == 1e-4


def test_lr_cosine_midpoint():
    mid = get_lr_cosine_schedule(550, 1e-3, 1e-4, 100, 1000)
    expected = 1e-4 + 0.5 * (1e-3 - 1e-4)   # cos(pi/2) = 0 → midpoint
    assert abs(mid - expected) < 1e-9


def test_lr_monotone_decay():
    lrs = [get_lr_cosine_schedule(t, 1e-3, 1e-4, 50, 500) for t in range(50, 500)]
    assert all(lrs[i] >= lrs[i+1] for i in range(len(lrs)-1))


# ---- Gradient clipping -----------------------------------------------------

def test_clip_reduces_norm():
    p = torch.nn.Parameter(torch.ones(100))
    p.grad = torch.ones(100)
    norm_before = clip_gradient_norm([p], max_norm=1.0)
    norm_after  = p.grad.norm().item()
    assert abs(norm_after - 1.0) < 1e-5


def test_clip_noop_when_below():
    p = torch.nn.Parameter(torch.ones(4))
    p.grad = torch.ones(4) * 0.1  # norm = 0.2
    norm_before = clip_gradient_norm([p], max_norm=1.0)
    assert (p.grad - 0.1).abs().max() < 1e-6  # unchanged


# ---- Data loading ----------------------------------------------------------

def test_get_batch_shapes():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, batch_size=4, context_length=16, device="cpu")
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


def test_get_batch_targets_shifted():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, batch_size=8, context_length=32, device="cpu")
    assert (y - x == 1).all()


# ---- Checkpointing ---------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    model = TransformerLM(
        vocab_size=50, context_length=8,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
    )
    opt = AdamW(model.parameters(), lr=1e-3)

    # Run a few steps to populate optimizer state
    ids = torch.randint(0, 50, (2, 8))
    for _ in range(3):
        opt.zero_grad()
        cross_entropy_loss(model(ids), ids).backward()
        opt.step()

    ck_path = str(tmp_path / "ckpt.pt")
    save_checkpoint(model, opt, iteration=42, out=ck_path)

    # Restore into a fresh model
    model2 = TransformerLM(
        vocab_size=50, context_length=8,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
    )
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    it = load_checkpoint(ck_path, model2, opt2)

    assert it == 42
    # Weights must match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
    # Optimizer step counter
    first_p = list(model2.parameters())[0]
    assert opt2.state[first_p]["t"] == 3
