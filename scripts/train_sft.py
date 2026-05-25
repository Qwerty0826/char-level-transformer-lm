#!/usr/bin/env python3
"""
Supervised Fine-Tuning script.

Three diffs from scripts/train.py:
  (a) Load weights from a base pretrain checkpoint with a *fresh* AdamW
      (not the pretraining optimizer state).
  (b) Loss is computed only on assistant-response tokens via
      ``masked_cross_entropy_loss`` + the ``loss_mask`` from the packed batch.
  (c) Lower default learning rate (~3e-5). SFT learning rate is typically
      10–50× lower than pretraining.

Input: a .pt file produced by scripts/build_sft_dataset.py containing
``train`` and ``val`` dicts with input_ids, target_ids, loss_mask,
attention_mask tensors.

Example:
    python scripts/train_sft.py \\
        --base_checkpoint checkpoints/base_60m/final.pt \\
        --sft_data        data/tinystories_v2_sft.pt \\
        --checkpoint_dir  checkpoints/sft \\
        --total_steps 2000 --lr_max 3e-5 \\
        --batch_size 32 --device cuda --dtype bfloat16 --compile
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import (
    clip_gradient_norm,
    get_lr_cosine_schedule,
    masked_cross_entropy_loss,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    import sys
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    yaml_defaults: dict = {}
    if pre_args.config:
        import yaml
        with open(pre_args.config) as _f:
            yaml_defaults = yaml.safe_load(_f) or {}

    p = argparse.ArgumentParser(description="Supervised fine-tuning")
    p.add_argument("--config", default=None, help="YAML config (model + LR defaults)")
    p.add_argument("--base_checkpoint", required=True, help="Path to pretrain .pt")
    p.add_argument("--sft_data",        required=True, help="Path to packed SFT .pt")

    # Model (must match the base checkpoint's architecture)
    p.add_argument("--vocab_size",     type=int, default=16_000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--d_model",        type=int, default=640)
    p.add_argument("--num_layers",     type=int, default=10)
    p.add_argument("--num_heads",      type=int, default=10)
    p.add_argument("--num_kv_heads",   type=int, default=2)
    p.add_argument("--d_ff",           type=int, default=1728)
    p.add_argument("--theta",          type=float, default=10_000.0)
    p.add_argument("--no_tie_weights", action="store_true")

    # Optimizer (SFT defaults are lower than pretrain)
    p.add_argument("--lr_max",       type=float, default=3.0e-5)
    p.add_argument("--lr_min",       type=float, default=3.0e-6)
    p.add_argument("--warmup_steps", type=int,   default=100)
    p.add_argument("--total_steps",  type=int,   default=2_000)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.95)

    p.add_argument("--log_interval",        type=int, default=20)
    p.add_argument("--val_interval",        type=int, default=200)
    p.add_argument("--val_batches",         type=int, default=20)
    p.add_argument("--checkpoint_dir",      default="checkpoints/sft")
    p.add_argument("--checkpoint_interval", type=int, default=500)

    p.add_argument("--device", default=None)
    p.add_argument("--dtype",  default="float32",
                   choices=["float32", "bfloat16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_backend", default=None)
    p.add_argument("--seed", type=int, default=0)

    if yaml_defaults:
        p.set_defaults(**yaml_defaults)
    return p.parse_args()


def get_device(req: str | None) -> str:
    if req is not None:
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_weights(ckpt_path: str, model: torch.nn.Module) -> int:
    """
    Load model weights from a base pretrain checkpoint, ignoring optimizer
    state. Handles the ``_orig_mod.`` prefix that torch.compile leaves behind.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning: missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  Warning: unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    return ckpt.get("iteration", 0)


def iter_batches(data: dict, batch_size: int, device: str, seed: int):
    """Infinite shuffled batch iterator over packed SFT tensors."""
    n = data["input_ids"].shape[0]
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            yield {k: v[idx].to(device, non_blocking=True) for k, v in data.items()}


@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, device, n_batches) -> float:
    model.eval()
    total, count = 0.0, 0
    it = iter_batches(val_data, batch_size, device, seed=1234)
    for _ in range(n_batches):
        batch = next(it)
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = masked_cross_entropy_loss(
            logits, batch["target_ids"], batch["loss_mask"],
        )
        total += loss.item()
        count += 1
    model.train()
    return total / max(1, count)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # -- Data ----------------------------------------------------------------
    print(f"Loading SFT data: {args.sft_data}")
    pkg = torch.load(args.sft_data, map_location="cpu", weights_only=False)
    train_data, val_data = pkg["train"], pkg["val"]
    print(f"  Train examples: {train_data['input_ids'].shape[0]:,}")
    print(f"  Val examples:   {val_data['input_ids'].shape[0]:,}")
    print(f"  Seq length:     {train_data['input_ids'].shape[1]}")

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
        device=device,
        dtype=dtype,
    )
    print(f"Parameters: {model.num_parameters():,}")

    print(f"Loading base weights: {args.base_checkpoint}")
    load_base_weights(args.base_checkpoint, model)

    if args.compile:
        backend = args.compile_backend or ("inductor" if device == "cuda" else "aot_eager")
        print(f"Compiling model with backend={backend!r}")
        model = torch.compile(model, backend=backend)

    # -- Optimizer (FRESH — do not load pretrain optimizer state) -----------
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # -- Training loop ------------------------------------------------------
    model.train()
    train_iter = iter_batches(train_data, args.batch_size, device, args.seed)
    t0 = time.time()
    running_loss = 0.0

    for step in range(1, args.total_steps + 1):
        lr = get_lr_cosine_schedule(
            t=step,
            alpha_max=args.lr_max,
            alpha_min=args.lr_min,
            T_w=args.warmup_steps,
            T_c=args.total_steps,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = next(train_iter)
        optimizer.zero_grad()
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = masked_cross_entropy_loss(
            logits, batch["target_ids"], batch["loss_mask"],
        )
        loss.backward()
        grad_norm = clip_gradient_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()

        if step % args.log_interval == 0:
            avg = running_loss / args.log_interval
            elapsed = time.time() - t0
            ppl = math.exp(min(avg, 20.0))  # cap to avoid overflow on noisy early steps
            print(
                f"step {step:5d} | loss {avg:.4f} | ppl {ppl:.2f} | "
                f"lr {lr:.2e} | gn {grad_norm:.2f} | {elapsed:.1f}s"
            )
            running_loss = 0.0

        if step % args.val_interval == 0:
            val_loss = estimate_val_loss(
                model, val_data, args.batch_size, device, args.val_batches,
            )
            val_ppl = math.exp(min(val_loss, 20.0))
            print(f"  [val] step {step} | loss {val_loss:.4f} | ppl {val_ppl:.2f}")

        if step % args.checkpoint_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, path)
            print(f"  Saved {path}")

    path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, args.total_steps, path)
    print(f"\nDone. Final SFT checkpoint: {path}")


if __name__ == "__main__":
    main()
