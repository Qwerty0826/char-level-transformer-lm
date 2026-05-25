#!/usr/bin/env python3
"""
Main training script for the Transformer LM.

Supports full CLI configuration, memory-mapped data loading, cosine LR
scheduling, gradient clipping, periodic validation, checkpointing, and
optional Weights & Biases or CSV experiment logging.

Example (TinyStories on Apple Silicon MPS):

    python scripts/train.py \\
        --train_data data/tinystories_tokens_train.npy \\
        --val_data   data/tinystories_tokens_val.npy   \\
        --vocab_size 10000                             \\
        --context_length 256                           \\
        --d_model 512 --num_layers 4 --num_heads 16    \\
        --d_ff 1344                                    \\
        --batch_size 32 --total_steps 5000             \\
        --lr_max 1e-3 --lr_min 1e-4 --warmup_steps 200 \\
        --checkpoint_dir checkpoints/tinystories       \\
        --compile
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    # Pre-parse to detect --config so YAML values become argparse defaults.
    # Explicit CLI args always win over the YAML file.
    import sys
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    yaml_defaults: dict = {}
    if pre_args.config:
        import yaml
        with open(pre_args.config) as _f:
            yaml_defaults = yaml.safe_load(_f) or {}

    p = argparse.ArgumentParser(description="Train a Transformer LM from scratch")
    p.add_argument("--config", default=None, help="YAML config file (CLI flags override)")

    # Data
    p.add_argument("--train_data", help="Path to tokenised train .npy file")
    p.add_argument("--val_data",   help="Path to tokenised val .npy file")

    # Tokenizer / vocab
    p.add_argument("--vocab_size", type=int, default=10_000)

    # Model
    p.add_argument("--context_length", type=int,   default=256)
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--num_layers",     type=int,   default=4)
    p.add_argument("--num_heads",      type=int,   default=16)
    p.add_argument("--num_kv_heads",   type=int,   default=None,
                   help="K/V heads for Grouped Query Attention. None = same as num_heads.")
    p.add_argument("--d_ff",           type=int,   default=1344)
    p.add_argument("--theta",          type=float, default=10_000.0, help="RoPE base frequency")
    p.add_argument("--no_tie_weights", action="store_true", help="Disable weight tying")

    # Optimiser
    p.add_argument("--lr_max",          type=float, default=1e-3)
    p.add_argument("--lr_min",          type=float, default=1e-4)
    p.add_argument("--warmup_steps",    type=int,   default=200)
    p.add_argument("--total_steps",     type=int,   default=5_000)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--weight_decay",    type=float, default=0.1)
    p.add_argument("--beta1",           type=float, default=0.9)
    p.add_argument("--beta2",           type=float, default=0.95)
    p.add_argument("--grad_accum_steps",type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)")

    # Logging / checkpointing
    p.add_argument("--log_interval",       type=int, default=100)
    p.add_argument("--val_interval",       type=int, default=500)
    p.add_argument("--val_batches",        type=int, default=20,
                   help="Number of batches to estimate validation loss")
    p.add_argument("--checkpoint_dir",     default="checkpoints")
    p.add_argument("--checkpoint_interval",type=int, default=1000)
    p.add_argument("--resume",             default=None, help="Path to checkpoint to resume from")
    p.add_argument("--log_csv",            default=None, help="Path to CSV log file")

    # Hardware
    p.add_argument("--device", default=None,
                   help="Device string (default: auto-detect mps/cuda/cpu)")
    p.add_argument("--dtype",  default="float32",
                   choices=["float32", "bfloat16"],
                   help="Parameter dtype (bfloat16 recommended for CUDA, float32 for MPS)")
    p.add_argument("--compile", action="store_true",
                   help="Compile the model with torch.compile")
    p.add_argument("--compile_backend", default=None,
                   help="torch.compile backend (auto-selected based on device if unset)")

    # Ablations (Section 7.3)
    p.add_argument("--no_norm",    action="store_true", help="Ablation: remove RMSNorm layers")
    p.add_argument("--post_norm",  action="store_true", help="Ablation: use post-norm instead of pre-norm")
    p.add_argument("--no_rope",    action="store_true", help="Ablation: remove RoPE positional encoding")
    p.add_argument("--no_gate",    action="store_true", help="Ablation: use SiLU-only FFN (remove SwiGLU gate)")

    # W&B
    p.add_argument("--wandb",         action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb_project", default="transformer-lm")
    p.add_argument("--wandb_run",     default=None)

    if yaml_defaults:
        p.set_defaults(**yaml_defaults)

    args = p.parse_args()
    if args.train_data is None or args.val_data is None:
        p.error("--train_data and --val_data are required (or set via --config)")
    return args


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_val_loss(model, data, batch_size, context_length, device, n_batches):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        total += cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return total / n_batches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # -- Data ----------------------------------------------------------------
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data   = np.load(args.val_data,   mmap_mode="r")
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

    # -- Model ---------------------------------------------------------------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        tie_weights=not args.no_tie_weights,
        use_norm=not args.no_norm,
        pre_norm=not args.post_norm,
        use_rope=not args.no_rope,
        use_gate=not args.no_gate,
        device=device,
        dtype=dtype,
    )

    total_params = model.num_parameters()
    non_emb      = model.num_parameters(non_embedding=True)
    print(f"Parameters: {total_params:,} total, {non_emb:,} non-embedding")
    print(f"Estimated FLOPs/token: {model.estimate_flops_per_token():,}")

    # Optionally compile (use aot_eager on MPS to avoid broken inductor)
    if args.compile:
        backend = args.compile_backend
        if backend is None:
            backend = "aot_eager" if device == "mps" else "inductor"
        print(f"Compiling model with backend={backend!r} ...")
        model = torch.compile(model, backend=backend)

    # -- Optimizer -----------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at step {start_step}")

    # -- W&B -----------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config=vars(args),
                resume="allow" if args.resume else "never",
            )
        except ImportError:
            print("wandb not installed; skipping W&B logging.")

    # -- CSV logger ----------------------------------------------------------
    csv_file = None
    csv_writer = None
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.log_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if os.path.getsize(args.log_csv) == 0:
            csv_writer.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "elapsed_s"])

    # -- Checkpoint dir ------------------------------------------------------
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # -- Peak hardware FLOPs/s (for MFU calculation) -------------------------
    # MFU = achieved_flops / peak_flops.  The "right" peak depends on the
    # numerical precision in use: tensor cores are fp16/bf16 only on most
    # consumer cards, so fp32 training has a much lower theoretical peak.
    # Picking the wrong table makes MFU look ~8× lower than reality.
    PEAK_FLOPS_TC = {     # fp16 / bf16 tensor-core peaks
        "A100":    312e12,
        "H100":    989e12,
        "T4":       65e12,
        "V100":    125e12,
        "L4":      121e12,
        "RTX4090": 330e12,
        "RTX3090": 142e12,
    }
    PEAK_FLOPS_FP32 = {   # plain fp32 (no tensor cores) peaks
        "A100":   19.5e12,
        "H100":   67.0e12,
        "T4":      8.1e12,
        "V100":   15.7e12,
        "L4":     30.3e12,
        "RTX4090": 82.6e12,
        "RTX3090": 35.6e12,
    }
    peak_table = PEAK_FLOPS_FP32 if args.dtype == "float32" else PEAK_FLOPS_TC
    peak_flops = None
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "").upper()
        for k, v in peak_table.items():
            if k.upper() in gpu_name:
                peak_flops = v
                break

    # Per-step FLOPs ≈ 3 × forward (forward 1× + backward 2×).
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    flops_per_token_fwd = base_model.estimate_flops_per_token()
    flops_per_step = 3 * flops_per_token_fwd * args.batch_size * args.grad_accum_steps

    # -- Training loop -------------------------------------------------------
    model.train()
    t0 = time.time()
    running_loss = 0.0
    tokens_processed = start_step * args.batch_size * args.context_length * args.grad_accum_steps

    for step in range(start_step + 1, args.total_steps + 1):
        # Learning rate
        lr = get_lr_cosine_schedule(
            t=step,
            alpha_max=args.lr_max,
            alpha_min=args.lr_min,
            T_w=args.warmup_steps,
            T_c=args.total_steps,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
            logits = model(x)
            loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / args.grad_accum_steps).backward()
            accum_loss += loss.item() / args.grad_accum_steps

        grad_norm = clip_gradient_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += accum_loss
        tokens_processed += args.batch_size * args.context_length * args.grad_accum_steps

        # -- Logging ---------------------------------------------------------
        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed  = time.time() - t0
            tokens_per_sec = tokens_processed / elapsed

            # MFU: model FLOPs utilisation (achieved / hardware-peak FLOPs)
            mfu_str = ""
            if peak_flops is not None and elapsed > 0:
                step_time = elapsed / max(1, step - start_step)
                achieved_flops_per_sec = flops_per_step / step_time
                mfu = achieved_flops_per_sec / peak_flops
                mfu_str = f" | MFU {mfu*100:.1f}%"

            print(
                f"step {step:6d} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"{tokens_per_sec:,.0f} tok/s{mfu_str}"
            )
            if wandb_run:
                log_dict = {"train/loss": avg_loss, "train/lr": lr,
                            "train/grad_norm": grad_norm.item(),
                            "train/tokens_per_sec": tokens_per_sec}
                if peak_flops is not None:
                    log_dict["train/mfu"] = achieved_flops_per_sec / peak_flops
                wandb_run.log(log_dict, step=step)
            running_loss = 0.0

        # -- Validation ------------------------------------------------------
        if step % args.val_interval == 0:
            val_loss = estimate_val_loss(
                model, val_data, args.batch_size,
                args.context_length, device, args.val_batches,
            )
            val_ppl  = math.exp(val_loss)
            elapsed  = time.time() - t0
            print(f"  [val] step {step} | loss {val_loss:.4f} | ppl {val_ppl:.2f} | {elapsed:.1f}s")
            if wandb_run:
                wandb_run.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)
            if csv_writer:
                avg_loss_log = running_loss / max(1, args.log_interval)
                csv_writer.writerow([step, avg_loss_log, val_loss, lr, grad_norm.item(), elapsed])
                csv_file.flush()

        # -- Checkpoint ------------------------------------------------------
        if step % args.checkpoint_interval == 0:
            ck_path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, ck_path)
            print(f"  Saved checkpoint: {ck_path}")

    # Final checkpoint
    ck_path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, args.total_steps, ck_path)
    print(f"Training complete. Final checkpoint: {ck_path}")

    if csv_file:
        csv_file.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
