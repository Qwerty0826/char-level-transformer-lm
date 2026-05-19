"""
AdamW optimizer implemented from scratch as a torch.optim.Optimizer subclass.

Follows Algorithm 1 from Loshchilov & Hutter (2019):
  "Decoupled Weight Decay Regularization."

Note: weight decay is applied *after* the gradient step (not coupled with
the gradient update as in the original Adam) — this is the key difference
that makes AdamW superior to L2-regularised Adam for transformers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimiser (Loshchilov & Hutter, 2019).

    Args:
        params:       Iterable of parameters or parameter groups.
        lr:           Learning rate (alpha).
        betas:        (beta1, beta2) — moment decay coefficients.
        eps:          Small constant for numerical stability (default 1e-8).
        weight_decay: Decoupled weight-decay coefficient (lambda).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]

                # Lazy state initialisation (first step)
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state["t"] + 1
                m: torch.Tensor = state["m"]
                v: torch.Tensor = state["v"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Bias-corrected learning rate
                alpha_t = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

                # Parameter update: theta -= alpha_t * m / (sqrt(v) + eps)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Decoupled weight decay: theta -= lr * lambda * theta
                p.data.mul_(1.0 - lr * weight_decay)

                state["t"] = t

        return loss
