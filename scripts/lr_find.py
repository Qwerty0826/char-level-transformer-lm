#!/usr/bin/env python3
"""
Learning-rate finder (Smith 2015 / fast.ai).

Sweeps the learning rate exponentially from `--lr_min` to `--lr_max` over
`--num_iters` steps and records the loss at each rate.  A good choice for
the peak LR is roughly one decade below the point where loss starts to
diverge — this script prints that automatically.

The output is a small CSV (step, lr, loss) and a Markdown summary,
including the suggested peak LR for the cosine schedule.

Example:
    python scripts/lr_find.py \\
        --config configs/tinystories.yaml \\
        --num_iters 100 \\
        --out results/lr_find.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import (
    clip_gradient_norm,
    cross_entropy_loss,
    get_batch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exponential learning-rate range test")
    p.add_argument("--config",    required=True, help="Base YAML config")
    p.add_argument("--num_iters", type=int,   default=100)
    p.add_argument("--lr_start",  type=float, default=1e-7)
    p.add_argument("--lr_end",    type=float, default=1.0)
    p.add_argument("--out",       default="results/lr_find.csv")
    p.add_argument("--device",    default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Device: {device}")

    data = np.load(cfg["train_data"], mmap_mode="r")
    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg.get("d_ff"),
        theta=cfg.get("theta", 10_000.0),
        tie_weights=cfg.get("tie_weights", True),
        device=device,
    )

    opt = AdamW(
        model.parameters(),
        lr=args.lr_start,
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        weight_decay=cfg.get("weight_decay", 0.1),
    )

    # Geometric ramp from lr_start to lr_end over num_iters
    ratio = (args.lr_end / args.lr_start) ** (1.0 / args.num_iters)

    history = []
    best_loss = float("inf")

    model.train()
    print(f"\nSweeping LR {args.lr_start:.0e} → {args.lr_end:.0e} over {args.num_iters} iters")
    print("-" * 50)
    for it in range(args.num_iters):
        lr = args.lr_start * (ratio ** it)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        x, y = get_batch(data, cfg.get("batch_size", 32),
                         cfg["context_length"], device)
        logits = model(x)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        clip_gradient_norm(model.parameters(), cfg.get("grad_clip", 1.0))
        opt.step()

        l = loss.item()
        history.append((it, lr, l))
        if l < best_loss:
            best_loss = l

        if it % max(1, args.num_iters // 20) == 0 or it == args.num_iters - 1:
            print(f"  iter {it:4d}  lr {lr:.2e}  loss {l:.4f}")

        # Stop if loss diverges (4× best)
        if l > 4 * best_loss and it > 10:
            print(f"  (diverged at iter {it}, lr {lr:.2e})")
            break

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "lr", "loss"])
        for row in history:
            w.writerow(row)

    # Suggested peak LR: one decade below the steepest descent (loss minimum)
    losses = [l for _, _, l in history]
    if len(losses) < 5:
        print("\nNot enough samples to suggest a peak LR.")
        return
    min_idx = int(np.argmin(losses))
    min_lr = history[min_idx][1]
    suggested = min_lr / 10.0

    print("\n" + "=" * 50)
    print(f"  Min loss: {losses[min_idx]:.4f}  @ lr={min_lr:.2e}")
    print(f"  Suggested peak LR for cosine schedule: {suggested:.2e}")
    print(f"  CSV written to: {args.out}")


if __name__ == "__main__":
    main()
