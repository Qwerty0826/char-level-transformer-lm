"""
Direct Preference Optimization (Rafailov et al., 2023), from scratch.

Given a frozen reference model π_ref and a trainable policy π_θ, plus
preference pairs (prompt x, chosen response y_w, rejected response y_l):

    L_DPO = -E[ log σ( β · ( log π_θ(y_w|x)/π_ref(y_w|x)
                           - log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]

In words: the policy is rewarded for raising the log-ratio of chosen
relative to reference, and lowering the log-ratio of rejected. β controls
how aggressively to chase the preference (smaller = more conservative,
closer to SFT behaviour).

This module supplies two primitives:

  sequence_log_probs(logits, targets, mask)
    Sum of token-level log-probs over the response (mask=1) positions only.

  dpo_loss(policy_..., ref_..., beta)
    Computes the DPO loss above plus the standard diagnostics
    (chosen/rejected reward, reward margin, preference accuracy).

We keep the from-scratch ethos: no torch.nn.functional. log-softmax is
computed via the same numerically stable shifted - log_sum_exp pattern
used in cross_entropy_loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def sequence_log_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Sum of log p(target | context) at positions where ``response_mask`` is 1.

    Args:
        logits:        (B, T, V) policy or reference output.
        targets:       (B, T)    next-token labels (already shifted).
        response_mask: (B, T)    1.0 on response tokens, 0.0 elsewhere.

    Returns:
        (B,) — per-sequence sum of response log-probabilities.
    """
    # Numerically stable log-softmax (same pattern as training.cross_entropy_loss).
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - logits_max
    log_sum_exp = shifted.exp().sum(dim=-1, keepdim=True).log()
    log_probs = shifted - log_sum_exp                        # (B, T, V)

    target_lp = log_probs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)                                            # (B, T)

    mask = response_mask.to(target_lp.dtype)
    return (target_lp * mask).sum(dim=-1)                    # (B,)


def _log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log σ(x).

        log σ(x) = -log(1 + exp(-x))     when x >= 0
                 =  x - log(1 + exp(x))  when x <  0

    Avoids overflow in exp() for large-magnitude x.
    """
    return torch.where(
        x >= 0,
        -torch.log1p(torch.exp(-x)),
         x - torch.log1p(torch.exp(x)),
    )


@dataclass
class DPODiagnostics:
    """Per-batch diagnostics. All scalars are detached."""
    chosen_reward:   torch.Tensor   # β · (policy - ref) log-ratio on chosen
    rejected_reward: torch.Tensor   # β · (policy - ref) log-ratio on rejected
    reward_margin:   torch.Tensor   # chosen_reward - rejected_reward  (must trend up)
    accuracy:        torch.Tensor   # fraction of pairs where margin > 0
    policy_chosen_logp:   torch.Tensor
    policy_rejected_logp: torch.Tensor


def dpo_loss(
    policy_chosen_logits:   torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits:      torch.Tensor,
    ref_rejected_logits:    torch.Tensor,
    chosen_targets:   torch.Tensor,
    chosen_mask:      torch.Tensor,
    rejected_targets: torch.Tensor,
    rejected_mask:    torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, DPODiagnostics]:
    """
    Direct Preference Optimization loss.

    Both policy and reference logits must be passed in (the caller decides
    whether to use one forward pass on a concatenated batch or two passes).
    For numerical correctness, the reference logits should be computed
    with ``torch.no_grad()`` and ideally in the SAME dtype as the policy
    at log-prob computation time — dtype mismatch between policy and ref
    at this step can swamp the DPO signal.

    Returns:
        loss:        scalar tensor (mean over the batch)
        diagnostics: DPODiagnostics with detached per-batch summary stats.
    """
    policy_chosen_lp   = sequence_log_probs(policy_chosen_logits,   chosen_targets,   chosen_mask)
    policy_rejected_lp = sequence_log_probs(policy_rejected_logits, rejected_targets, rejected_mask)
    ref_chosen_lp      = sequence_log_probs(ref_chosen_logits,      chosen_targets,   chosen_mask)
    ref_rejected_lp    = sequence_log_probs(ref_rejected_logits,    rejected_targets, rejected_mask)

    log_ratio_chosen   = policy_chosen_lp   - ref_chosen_lp     # (B,)
    log_ratio_rejected = policy_rejected_lp - ref_rejected_lp   # (B,)

    margin = beta * (log_ratio_chosen - log_ratio_rejected)     # (B,)
    loss = -_log_sigmoid(margin).mean()

    diagnostics = DPODiagnostics(
        chosen_reward   = (beta * log_ratio_chosen).detach().mean(),
        rejected_reward = (beta * log_ratio_rejected).detach().mean(),
        reward_margin   = margin.detach().mean(),
        accuracy        = (margin > 0).float().mean().detach(),
        policy_chosen_logp   = policy_chosen_lp.detach().mean(),
        policy_rejected_logp = policy_rejected_lp.detach().mean(),
    )
    return loss, diagnostics
