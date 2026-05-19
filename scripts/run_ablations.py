#!/usr/bin/env python3
"""
Run the four architecture ablation studies from CS336 Section 7.3.

Each ablation trains for --steps steps (default 2000) and reports
final validation loss.  Results are written to a Markdown table.

Ablations:
  baseline  — full model (RMSNorm, pre-norm, RoPE, SwiGLU)
  no_norm   — remove RMSNorm (identity in its place)
  post_norm — post-norm instead of pre-norm
  no_rope   — no positional encoding (NoPE)
  no_gate   — SiLU-only FFN (remove SwiGLU gate)

Example:
    python scripts/run_ablations.py \
        --config configs/tinystories.yaml \
        --steps  2000 \
        --out    results/ablations.md
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

# Make sure the package is importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import (
    clip_gradient_norm,
    cross_entropy_loss,
    get_batch,
    get_lr_cosine_schedule,
)


ABLATIONS = {
    "baseline":  {},
    "no_norm":   {"use_norm":  False},
    "post_norm": {"pre_norm":  False},
    "no_rope":   {"use_rope":  False},
    "no_gate":   {"use_gate":  False},
}


def run_one(name: str, cfg: dict, ablation_flags: dict, device: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Ablation: {name}")
    print(f"  Flags: {ablation_flags or 'none (baseline)'}")
    print(f"{'='*60}")

    train_data = np.load(cfg["train_data"], mmap_mode="r")
    val_data   = np.load(cfg["val_data"],   mmap_mode="r")

    dtype = torch.bfloat16 if cfg.get("dtype") == "bfloat16" else torch.float32

    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg.get("d_ff"),
        theta=cfg.get("theta", 10_000.0),
        tie_weights=cfg.get("tie_weights", True),
        use_norm=ablation_flags.get("use_norm", True),
        pre_norm=ablation_flags.get("pre_norm", True),
        use_rope=ablation_flags.get("use_rope", True),
        use_gate=ablation_flags.get("use_gate", True),
        device=device,
        dtype=dtype,
    )
    print(f"  Parameters: {model.num_parameters():,}")

    opt = AdamW(
        model.parameters(),
        lr=cfg.get("lr_max", 1e-3),
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        weight_decay=cfg.get("weight_decay", 0.1),
    )

    steps       = cfg["_steps"]
    batch_size  = cfg.get("batch_size", 32)
    ctx_len     = cfg["context_length"]
    warmup      = min(cfg.get("warmup_steps", 200), steps // 10)
    lr_max      = cfg.get("lr_max", 1e-3)
    lr_min      = cfg.get("lr_min", 1e-4)
    grad_clip   = cfg.get("grad_clip", 1.0)
    val_batches = 20

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        lr = get_lr_cosine_schedule(step, lr_max, lr_min, warmup, steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        x, y = get_batch(train_data, batch_size, ctx_len, device)
        loss = cross_entropy_loss(model(x).view(-1, model.vocab_size), y.view(-1))
        loss.backward()
        clip_gradient_norm(model.parameters(), grad_clip)
        opt.step()

        if step % 200 == 0 or step == steps:
            print(f"  step {step:5d}/{steps} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {time.time()-t0:.0f}s")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for _ in range(val_batches):
            x, y = get_batch(val_data, batch_size, ctx_len, device)
            val_loss += cross_entropy_loss(
                model(x).view(-1, model.vocab_size), y.view(-1)
            ).item()
    val_loss /= val_batches
    val_ppl   = math.exp(val_loss)

    print(f"  VAL loss: {val_loss:.4f}  ppl: {val_ppl:.2f}")
    return {"name": name, "val_loss": val_loss, "val_ppl": val_ppl,
            "params": model.num_parameters(), "elapsed": time.time() - t0}


def main():
    p = argparse.ArgumentParser(description="Run CS336 ablation studies")
    p.add_argument("--config", required=True, help="YAML config (tinystories.yaml)")
    p.add_argument("--steps",  type=int, default=2000,
                   help="Training steps per ablation (default: 2000)")
    p.add_argument("--ablations", nargs="*", default=list(ABLATIONS.keys()),
                   choices=list(ABLATIONS.keys()),
                   help="Which ablations to run (default: all)")
    p.add_argument("--out", default="results/ablations.md",
                   help="Markdown output path")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["_steps"] = args.steps

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}  |  Steps per ablation: {args.steps}")

    results = []
    for name in args.ablations:
        r = run_one(name, cfg, ABLATIONS[name], device)
        results.append(r)

    # Sort by val loss
    results.sort(key=lambda r: r["val_loss"])
    baseline = next((r for r in results if r["name"] == "baseline"), results[0])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# Ablation Study Results\n\n")
        f.write(f"Training steps per run: {args.steps}  \n")
        f.write(f"Config: `{args.config}`\n\n")
        f.write("| Ablation | Val Loss | Perplexity | Δ vs Baseline | Params |\n")
        f.write("|----------|----------|------------|---------------|--------|\n")
        for r in results:
            delta = r["val_loss"] - baseline["val_loss"]
            sign  = "+" if delta > 0 else ""
            f.write(
                f"| {r['name']:12s} | {r['val_loss']:.4f} | "
                f"{r['val_ppl']:8.2f} | {sign}{delta:+.4f} | "
                f"{r['params']:,} |\n"
            )
        f.write(f"\n*Generated by `scripts/run_ablations.py`*\n")

    print(f"\nResults written to: {args.out}")
    print("\n| Ablation     | Val Loss | Perplexity | Δ vs Baseline |")
    print("|--------------|----------|------------|---------------|")
    for r in results:
        delta = r["val_loss"] - baseline["val_loss"]
        print(f"| {r['name']:12s} | {r['val_loss']:.4f}   | {r['val_ppl']:8.2f}   | {delta:+.4f}        |")


if __name__ == "__main__":
    main()
