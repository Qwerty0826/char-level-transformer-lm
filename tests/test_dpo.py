"""
Smoke tests for the DPO loss.

We don't test correctness against a black-box reference (that would require
either F.logsigmoid or a known-good DPO implementation). Instead we check
invariants:

  1. When policy == ref, the loss is exactly log(2) and reward_margin is 0.
     (This is the "no learning yet" state: log σ(0) = -log(2).)

  2. When the policy strictly prefers chosen over ref by some Δ and matches
     ref on rejected, the loss is smaller than log(2) and the margin is
     positive.

  3. Log-sigmoid implementation matches the reference torch.sigmoid path
     up to fp tolerance, including in the |x| > 30 regime where naive
     log(sigmoid(x)) would underflow.

  4. Loss is finite under reasonable inputs; gradients flow through the
     policy logits and NOT through the (no_grad) ref logits.
"""

import math

import pytest
import torch

from cs336_basics.dpo import _log_sigmoid, dpo_loss, sequence_log_probs


def _make_inputs(B=4, T=8, V=16, seed=0):
    torch.manual_seed(seed)
    chosen_targets   = torch.randint(0, V, (B, T))
    rejected_targets = torch.randint(0, V, (B, T))
    # mask: response only on second half
    mask = torch.zeros(B, T)
    mask[:, T // 2:] = 1.0
    return chosen_targets, rejected_targets, mask, mask.clone()


def test_log_sigmoid_matches_reference():
    """Manual log σ must equal log(torch.sigmoid(x)) for all reasonable x."""
    x = torch.linspace(-40.0, 40.0, 401)
    expected = torch.log(torch.sigmoid(x).clamp_min(1e-300))
    got = _log_sigmoid(x)
    # Stable formula handles |x| > 30 where the naive version underflows;
    # compare only where the naive value is finite.
    finite = torch.isfinite(expected)
    assert torch.allclose(got[finite], expected[finite], atol=1e-5), \
        f"max diff = {(got[finite] - expected[finite]).abs().max().item()}"


def test_loss_is_log2_when_policy_equals_ref():
    B, T, V = 4, 8, 16
    chosen_targets, rejected_targets, cmask, rmask = _make_inputs(B, T, V)

    torch.manual_seed(0)
    policy_chosen   = torch.randn(B, T, V)
    policy_rejected = torch.randn(B, T, V)

    # Reference is identical to policy → margin = 0 → loss = log(2)
    loss, diag = dpo_loss(
        policy_chosen, policy_rejected,
        policy_chosen.clone(), policy_rejected.clone(),
        chosen_targets, cmask,
        rejected_targets, rmask,
        beta=0.1,
    )

    assert torch.isclose(loss, torch.tensor(math.log(2.0)), atol=1e-5), \
        f"loss should be log(2)={math.log(2):.6f}, got {loss.item():.6f}"
    assert torch.isclose(diag.reward_margin, torch.tensor(0.0), atol=1e-5)


def test_margin_positive_drops_loss_below_log2():
    """If the policy prefers chosen over ref, the loss must drop below log(2)."""
    B, T, V = 4, 8, 16
    chosen_targets, rejected_targets, cmask, rmask = _make_inputs(B, T, V)

    torch.manual_seed(1)
    policy_rejected = torch.randn(B, T, V)
    ref_chosen      = torch.randn(B, T, V)
    ref_rejected    = policy_rejected.clone()

    # Boost the chosen target logits in the policy by a strong margin.
    policy_chosen = ref_chosen.clone()
    for b in range(B):
        for t in range(T // 2, T):
            policy_chosen[b, t, chosen_targets[b, t]] += 5.0

    loss, diag = dpo_loss(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected,
        chosen_targets, cmask,
        rejected_targets, rmask,
        beta=0.1,
    )
    assert loss.item() < math.log(2.0), \
        f"loss {loss.item():.4f} should be < log(2)={math.log(2):.4f}"
    assert diag.reward_margin.item() > 0
    assert diag.accuracy.item() > 0.5


def test_gradients_flow_to_policy_not_ref():
    B, T, V = 2, 4, 8
    chosen_targets, rejected_targets, cmask, rmask = _make_inputs(B, T, V)

    torch.manual_seed(2)
    policy_chosen   = torch.randn(B, T, V, requires_grad=True)
    policy_rejected = torch.randn(B, T, V, requires_grad=True)
    ref_chosen      = torch.randn(B, T, V)    # no grad
    ref_rejected    = torch.randn(B, T, V)

    loss, _ = dpo_loss(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected,
        chosen_targets, cmask,
        rejected_targets, rmask,
        beta=0.1,
    )
    loss.backward()
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None
    assert ref_chosen.grad is None
    assert ref_rejected.grad is None
    assert torch.isfinite(policy_chosen.grad).all()


def test_sequence_log_probs_respects_mask():
    """Tokens outside the mask must contribute 0 to the summed log-prob."""
    B, T, V = 2, 6, 8
    torch.manual_seed(3)
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    full_mask = torch.ones(B, T)
    half_mask = torch.zeros(B, T)
    half_mask[:, T // 2:] = 1.0

    lp_full = sequence_log_probs(logits, targets, full_mask)
    lp_half = sequence_log_probs(logits, targets, half_mask)
    # lp_full = lp_half + (contribution from first half)
    assert (lp_full > lp_half).all() or (lp_full < lp_half).all() or torch.allclose(lp_full, lp_half), \
        "masking should monotonically remove a fixed contribution"
    # An all-zero mask gives zero.
    zero = sequence_log_probs(logits, targets, torch.zeros(B, T))
    assert torch.allclose(zero, torch.zeros(B))
