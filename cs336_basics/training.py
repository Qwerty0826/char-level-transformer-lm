"""
Training utilities: loss, learning-rate schedule, gradient clipping,
data loading (memory-mapped), and checkpoint save/load.
"""

from __future__ import annotations

import io
import math
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cross-entropy loss (numerically stable log-softmax)
# ---------------------------------------------------------------------------

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Numerically-stable cross-entropy loss.

        loss_i = -log softmax(logits_i)[target_i]
               = -(logits_i[target_i] - max_i) + log sum_j exp(logits_j - max_j)

    Args:
        logits:  (..., vocab_size)  — raw unnormalised scores.
        targets: (...)              — integer token indices.

    Returns:
        Scalar mean cross-entropy across all positions.
    """
    # Subtract max for numerical stability (doesn't change result)
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - logits_max
    log_sum_exp = shifted.exp().sum(dim=-1, keepdim=True).log()

    # log_softmax = logits - logits_max - log_sum_exp
    log_probs = shifted - log_sum_exp          # (..., vocab_size)

    # Gather the log-probability of the target token at each position
    target_log_probs = log_probs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)                              # (...)

    return -target_log_probs.mean()


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy averaged over positions selected by ``loss_mask``.

    For SFT we want loss only on assistant-response tokens; for DPO we
    compute per-sequence log-probs over response tokens only. Both reduce
    to the same primitive: sum(per-position-loss * mask) / sum(mask).

    Args:
        logits:    (..., vocab_size) — raw unnormalised scores.
        targets:   (...)             — integer token indices.
        loss_mask: (...)             — 0/1 float or bool mask. Positions
                                       with 0 contribute nothing to the loss.

    Returns:
        Scalar masked cross-entropy. Safe when ``loss_mask.sum() == 0``
        (returns 0 in that pathological case rather than NaN).
    """
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - logits_max
    log_sum_exp = shifted.exp().sum(dim=-1, keepdim=True).log()
    log_probs = shifted - log_sum_exp                          # (..., V)

    target_log_probs = log_probs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)                                              # (...)

    nll = -target_log_probs                                    # (...)
    mask = loss_mask.to(nll.dtype)
    denom = mask.sum().clamp_min(1.0)                          # guard /0
    return (nll * mask).sum() / denom


# ---------------------------------------------------------------------------
# Cosine learning rate schedule with linear warmup
# ---------------------------------------------------------------------------

def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """
    Cosine annealing schedule with linear warmup (LLaMA / Touvron et al. 2023).

    Three phases:
      1. Warmup   (t <  T_w):           alpha_t = (t / T_w) * alpha_max
      2. Annealing (T_w <= t <= T_c):   cosine decay from alpha_max to alpha_min
      3. Post-anneal (t >  T_c):        alpha_t = alpha_min

    Args:
        t:         Current training step (1-indexed).
        alpha_max: Peak learning rate.
        alpha_min: Final (minimum) learning rate.
        T_w:       Number of warmup steps.
        T_c:       Total cosine-decay steps (annealing ends here).

    Returns:
        Learning rate for step t.
    """
    if t < T_w:
        return (t / T_w) * alpha_max
    if t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1.0 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)
    return alpha_min


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def clip_gradient_norm(
    parameters,
    max_norm: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Clip the global L2-norm of all parameter gradients in-place.

    If the global norm exceeds max_norm the gradients are rescaled so that
    the resulting norm equals max_norm (approximately; eps avoids division
    by zero).

    Args:
        parameters: Iterable of torch.nn.Parameters.
        max_norm:   Maximum allowed global L2-norm.
        eps:        Small constant for numerical stability.

    Returns:
        The global gradient norm before clipping (as a scalar Tensor).
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return torch.tensor(0.0)

    total_norm = torch.sqrt(
        sum(p.grad.data.pow(2).sum() for p in params_with_grad)
    )

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)

    return total_norm


# ---------------------------------------------------------------------------
# Data loading (memory-mapped)
# ---------------------------------------------------------------------------

def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch from a token-ID array.

    Args:
        data:           1-D numpy array of uint16 token IDs (memory-mapped ok).
        batch_size:     Number of sequences per batch.
        context_length: Length of each sequence.
        device:         PyTorch device string ('cpu', 'mps', 'cuda:0', …).

    Returns:
        (inputs, targets) — both LongTensors of shape (batch_size, context_length).
        inputs[i] = data[s:s+context_length]
        targets[i] = data[s+1:s+context_length+1]
    """
    n = len(data)
    starts = np.random.randint(0, n - context_length, size=(batch_size,))
    x = np.stack([data[s: s + context_length] for s in starts])
    y = np.stack([data[s + 1: s + context_length + 1] for s in starts])
    x_t = torch.from_numpy(x.astype(np.int64)).to(device)
    y_t = torch.from_numpy(y.astype(np.int64)).to(device)
    return x_t, y_t


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, io.IOBase],
) -> None:
    """
    Save model + optimizer state along with the current iteration index.

    Args:
        model:     The nn.Module to checkpoint.
        optimizer: The Optimizer to checkpoint.
        iteration: Current training step.
        out:       File path or file-like object to write to.
    """
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, io.IOBase],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load a checkpoint into model and optimizer, returning the saved iteration.

    Args:
        src:       File path or file-like object to read from.
        model:     Model to restore weights into.
        optimizer: Optimizer to restore state into.

    Returns:
        The iteration number stored in the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
