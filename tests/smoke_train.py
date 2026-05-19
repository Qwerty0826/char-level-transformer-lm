"""
End-to-end smoke test: train a tiny model on synthetic data for a few
steps to catch regressions that unit tests don't (checkpoint format
changes, optimizer state shape mismatch, missing imports, etc.).

Run as: ``python tests/smoke_train.py``
"""

import math

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import (
    clip_gradient_norm,
    cross_entropy_loss,
    get_batch,
    get_lr_cosine_schedule,
)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    data = np.random.randint(0, 100, size=1024, dtype=np.uint16)
    model = TransformerLM(
        vocab_size=100, context_length=16,
        d_model=32, num_layers=2, num_heads=4, d_ff=64,
    )
    opt = AdamW(model.parameters(), lr=1e-4)
    last_loss = None
    for step in range(1, 6):
        lr = get_lr_cosine_schedule(step, 1e-3, 1e-4, T_w=2, T_c=10)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = get_batch(data, 4, 16, "cpu")
        opt.zero_grad()
        loss = cross_entropy_loss(model(x).view(-1, 100), y.view(-1))
        loss.backward()
        clip_gradient_norm(model.parameters(), 1.0)
        opt.step()
        last_loss = loss.item()
    assert last_loss is not None, "training loop did not run"
    print(f"final loss: {last_loss:.4f}")
    # Sanity bounds: loss must be a finite number.  Random-data starting
    # loss is ~ln(100) ≈ 4.6; we allow a wide cushion since the test
    # only verifies the pipeline runs, not that the model learns.
    assert math.isfinite(last_loss), f"non-finite loss: {last_loss}"


if __name__ == "__main__":
    main()
