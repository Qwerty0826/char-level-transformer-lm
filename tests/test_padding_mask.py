"""
Smoke tests for the attention padding-mask plumbing added for SFT.

Two invariants:

1. An all-ones padding mask must produce identical logits to passing no mask
   at all. Catches the most common off-by-one / broadcast-shape bug.

2. Padded positions on the key side must not influence the logits at real
   query positions. Done by comparing a short sequence's logits to the same
   sequence right-padded with garbage tokens that are masked out.
"""

import pytest
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import masked_cross_entropy_loss


def _tiny_model() -> TransformerLM:
    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=32, context_length=16,
        d_model=32, num_layers=2, num_heads=4,
        num_kv_heads=2, d_ff=64,
        tie_weights=False,
    ).eval()


def test_all_ones_mask_matches_no_mask():
    model = _tiny_model()
    ids = torch.randint(0, 32, (2, 8))
    mask_all_ones = torch.ones_like(ids, dtype=torch.bool)

    with torch.no_grad():
        logits_no_mask  = model(ids)
        logits_all_ones = model(ids, attention_mask=mask_all_ones)

    assert torch.allclose(logits_no_mask, logits_all_ones, atol=1e-6), \
        f"max diff = {(logits_no_mask - logits_all_ones).abs().max().item()}"


def test_padded_positions_do_not_affect_real_tokens():
    """
    Concatenate garbage tokens onto a real sequence, mask them out, and check
    that logits at the real positions are unchanged (up to fp tolerance).
    """
    model = _tiny_model()
    real = torch.randint(0, 32, (1, 5))                       # the "real" 5 tokens

    # Right-pad with deliberately-different garbage.
    garbage = torch.randint(0, 32, (1, 3))
    padded  = torch.cat([real, garbage], dim=1)               # (1, 8)
    mask    = torch.cat([
        torch.ones_like(real, dtype=torch.bool),
        torch.zeros_like(garbage, dtype=torch.bool),
    ], dim=1)

    with torch.no_grad():
        ref     = model(real)                                 # (1, 5, V)
        padded_logits = model(padded, attention_mask=mask)    # (1, 8, V)

    # Logits at the real positions (0..4) must match the no-padding reference.
    assert torch.allclose(ref, padded_logits[:, :5, :], atol=1e-5), \
        f"max diff at real positions = {(ref - padded_logits[:, :5, :]).abs().max().item()}"


def test_masked_loss_zero_mask_is_safe():
    """A degenerate all-zero mask must not produce NaN — it should return 0."""
    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    mask = torch.zeros(2, 4)
    loss = masked_cross_entropy_loss(logits, targets, mask)
    assert torch.isfinite(loss), "masked CE blew up on an all-zero mask"
    assert loss.item() == 0.0, f"all-zero mask should give 0, got {loss.item()}"


def test_masked_loss_all_ones_matches_mean():
    """An all-ones mask must match plain mean cross-entropy."""
    from cs336_basics.training import cross_entropy_loss
    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    mask = torch.ones(2, 4)
    masked = masked_cross_entropy_loss(logits, targets, mask)
    plain  = cross_entropy_loss(logits.view(-1, 8), targets.view(-1))
    assert torch.allclose(masked, plain, atol=1e-6), \
        f"masked={masked.item()} plain={plain.item()}"
