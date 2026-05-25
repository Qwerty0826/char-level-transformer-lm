#!/usr/bin/env python3
"""
Direct Preference Optimization training loop, from scratch.

Loads the SFT checkpoint twice:
  - policy: trainable
  - ref:    frozen (eval mode, requires_grad_(False)), SAME dtype as policy

For each batch of preference pairs (prompt, chosen, rejected):
  1. Format both as chat-template SFT sequences (loss mask on response only).
  2. Forward policy and ref on chosen + rejected (ref under no_grad).
  3. Compute DPO loss + diagnostics (reward_margin must trend up).
  4. Backward through policy only.

Critical correctness notes (see plan's risk callouts):
  - Policy and ref MUST share dtype at log-prob computation time. Use
    bf16 on both (or fp32 on both). Mismatch swamps the DPO signal.
  - Reference weights are loaded once at startup and never updated.
  - Beta=0.05-0.1 is the safe range for small models. Larger β = more
    aggressive preference chasing, but unstable.

Example:
    python scripts/train_dpo.py \\
        --sft_checkpoint   checkpoints/sft/final.pt \\
        --preferences      data/pref_labeled.jsonl  \\
        --vocab            data/tinystories_v2_vocab.json \\
        --merges           data/tinystories_v2_merges.txt \\
        --checkpoint_dir   checkpoints/dpo \\
        --total_steps 1500 --batch_size 4 --lr_max 5e-6 \\
        --beta 0.1 --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from cs336_basics.data_sft import (
    EOT,
    Message,
    format_sft_example,
    pad_and_collate,
)
from cs336_basics.dpo import dpo_loss
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import (
    clip_gradient_norm,
    get_lr_cosine_schedule,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO training")
    p.add_argument("--sft_checkpoint", required=True)
    p.add_argument("--preferences",    required=True, help="Labeled preferences JSONL")
    p.add_argument("--vocab",          required=True)
    p.add_argument("--merges",         required=True)

    # Model (must match SFT)
    p.add_argument("--vocab_size",     type=int, default=16_000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--d_model",        type=int, default=640)
    p.add_argument("--num_layers",     type=int, default=10)
    p.add_argument("--num_heads",      type=int, default=10)
    p.add_argument("--num_kv_heads",   type=int, default=2)
    p.add_argument("--d_ff",           type=int, default=1728)

    # DPO
    p.add_argument("--beta",           type=float, default=0.1)
    p.add_argument("--max_length",     type=int, default=512)

    # Optimizer (DPO LR is typically even lower than SFT — small δ is the goal)
    p.add_argument("--lr_max",       type=float, default=5.0e-6)
    p.add_argument("--lr_min",       type=float, default=5.0e-7)
    p.add_argument("--warmup_steps", type=int,   default=50)
    p.add_argument("--total_steps",  type=int,   default=1_500)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="DPO commonly disables weight decay (anchor is ref, not 0).")
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.95)

    p.add_argument("--log_interval",        type=int, default=10)
    p.add_argument("--checkpoint_dir",      default="checkpoints/dpo")
    p.add_argument("--checkpoint_interval", type=int, default=500)

    p.add_argument("--device", default=None)
    p.add_argument("--dtype",  default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed",   type=int, default=0)
    return p.parse_args()


def get_device(req: str | None) -> str:
    if req:
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_model(args, device, dtype) -> TransformerLM:
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        tie_weights=True,
        device=device,
        dtype=dtype,
    )


def load_sft_weights(ckpt_path: str, model: torch.nn.Module) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def pack_preferences(
    jsonl_path: str,
    tokenizer: Tokenizer,
    max_length: int,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Read labeled preferences JSONL → packed chosen/rejected tensors."""
    chosen_examples:   list[tuple[list[int], list[int]]] = []
    rejected_examples: list[tuple[list[int], list[int]]] = []

    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for tag, text in [("chosen", row["chosen"]), ("rejected", row["rejected"])]:
                messages = [
                    Message(role="user",      content=row["prompt"]),
                    Message(role="assistant", content=text),
                ]
                ids, mask = format_sft_example(messages, tokenizer, max_length)
                (chosen_examples if tag == "chosen" else rejected_examples).append((ids, mask))

    chosen_batch   = pad_and_collate(chosen_examples,   max_length=max_length, pad_id=pad_id)
    rejected_batch = pad_and_collate(rejected_examples, max_length=max_length, pad_id=pad_id)

    return {
        "chosen_input_ids":      chosen_batch["input_ids"],
        "chosen_target_ids":     chosen_batch["target_ids"],
        "chosen_loss_mask":      chosen_batch["loss_mask"],
        "chosen_attention_mask": chosen_batch["attention_mask"],
        "rejected_input_ids":      rejected_batch["input_ids"],
        "rejected_target_ids":     rejected_batch["target_ids"],
        "rejected_loss_mask":      rejected_batch["loss_mask"],
        "rejected_attention_mask": rejected_batch["attention_mask"],
    }


def iter_pref_batches(data: dict, batch_size: int, device: str, seed: int):
    n = data["chosen_input_ids"].shape[0]
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            yield {k: v[idx].to(device, non_blocking=True) for k, v in data.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Device: {device}, dtype: {dtype}, beta: {args.beta}")

    tokenizer = Tokenizer.from_files(
        args.vocab, args.merges,
        special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"],
    )
    pad_id = tokenizer.encode(EOT)[0]
    print(f"Tokenizer vocab_size={tokenizer.vocab_size}, pad_id={pad_id}")

    print(f"\nPacking preferences from {args.preferences} ...")
    data = pack_preferences(args.preferences, tokenizer, args.max_length, pad_id)
    n = data["chosen_input_ids"].shape[0]
    print(f"  {n:,} preference pairs packed (max_length={args.max_length})")
    if n < args.batch_size:
        raise SystemExit(f"Not enough preference pairs ({n}) for batch_size={args.batch_size}")

    print("\nLoading policy and reference models ...")
    policy = make_model(args, device, dtype)
    ref    = make_model(args, device, dtype)
    load_sft_weights(args.sft_checkpoint, policy)
    load_sft_weights(args.sft_checkpoint, ref)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # Critical: assert dtype parity between policy and ref params.
    p_dtypes = {p.dtype for p in policy.parameters()}
    r_dtypes = {p.dtype for p in ref.parameters()}
    assert p_dtypes == r_dtypes, f"dtype mismatch: policy={p_dtypes}, ref={r_dtypes}"
    print(f"  policy/ref dtype: {p_dtypes}")

    optimizer = AdamW(
        policy.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train_iter = iter_pref_batches(data, args.batch_size, device, args.seed)

    policy.train()
    running_loss = 0.0
    running_margin = 0.0
    running_acc = 0.0
    t0 = time.time()

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

        # Policy forward (both chosen and rejected).
        p_chosen_logits = policy(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        p_rejected_logits = policy(
            batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        # Reference forward (frozen, no grad).
        with torch.no_grad():
            r_chosen_logits = ref(
                batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            r_rejected_logits = ref(
                batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )

        loss, diag = dpo_loss(
            p_chosen_logits,  p_rejected_logits,
            r_chosen_logits,  r_rejected_logits,
            batch["chosen_target_ids"],   batch["chosen_loss_mask"],
            batch["rejected_target_ids"], batch["rejected_loss_mask"],
            beta=args.beta,
        )
        loss.backward()
        grad_norm = clip_gradient_norm(policy.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        running_margin += diag.reward_margin.item()
        running_acc += diag.accuracy.item()

        if step % args.log_interval == 0:
            n_log = args.log_interval
            elapsed = time.time() - t0
            print(
                f"step {step:4d} | loss {running_loss/n_log:.4f} | "
                f"margin {running_margin/n_log:+.4f} | "
                f"acc {running_acc/n_log:.2%} | "
                f"chosen_r {diag.chosen_reward.item():+.3f} | "
                f"rej_r {diag.rejected_reward.item():+.3f} | "
                f"lr {lr:.2e} | gn {grad_norm:.2f} | {elapsed:.1f}s"
            )
            running_loss = 0.0
            running_margin = 0.0
            running_acc = 0.0

        if step % args.checkpoint_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
            save_checkpoint(policy, optimizer, step, path)
            print(f"  Saved {path}")

    path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(policy, optimizer, args.total_steps, path)
    print(f"\nDone. Final DPO checkpoint: {path}")


if __name__ == "__main__":
    main()
