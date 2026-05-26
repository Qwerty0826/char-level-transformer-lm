#!/usr/bin/env python3
"""
Build a packed SFT dataset from roneneldan/TinyStoriesInstruct.

The HF dataset's `text` field looks like:

    Summary: A young girl finds a lost puppy and returns it.
    Features: Dialogue, MoralValue
    Words: puppy, return, kind
    Random sentence: She smiled warmly at the dog.
    Story: Once upon a time, ...

We split each example on the literal token "Story:" — everything before
becomes the user message (the constraints), everything after becomes the
assistant response (the story). This preserves the natural "instruction →
completion" structure the dataset was built for.

Output: a single .pt file containing
    input_ids:       (N, L-1) long
    target_ids:      (N, L-1) long
    loss_mask:       (N, L-1) float
    attention_mask:  (N, L-1) bool

where L = --max_length and N = number of examples that survived filtering.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cs336_basics.data_sft import (
    EOT,
    Message,
    format_sft_example,
    pad_and_collate,
)
from cs336_basics.tokenizer import Tokenizer


HF_DATASET = "roneneldan/TinyStoriesInstruct"
STORY_MARKER = "Story:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build packed SFT dataset from TinyStoriesInstruct")
    p.add_argument("--vocab",  required=True, help="Path to tokenizer vocab.json")
    p.add_argument("--merges", required=True, help="Path to tokenizer merges.txt")
    p.add_argument("--output", required=True, help="Output .pt path")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_examples", type=int, default=50_000,
                   help="Cap the number of examples to keep training time bounded.")
    p.add_argument("--min_response_tokens", type=int, default=20,
                   help="Drop examples whose response (loss-mask) is shorter than this.")
    p.add_argument("--val_fraction", type=float, default=0.02,
                   help="Hold out this fraction as a SFT validation set.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_example(text: str) -> tuple[str, str] | None:
    """Return (user_prompt, story) or None if the example can't be split."""
    if STORY_MARKER not in text:
        return None
    head, story = text.split(STORY_MARKER, 1)
    user = head.strip()
    story = story.strip()
    if not user or not story:
        return None
    return user, story


def main():
    args = parse_args()
    tokenizer = Tokenizer.from_files(
        args.vocab, args.merges,
        special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"],
    )
    pad_id = tokenizer.encode(EOT)[0]
    print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}, pad_id={pad_id}")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e

    print(f"\nLoading {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET, split="train")
    print(f"  {len(ds):,} raw examples")

    print(f"\nFormatting (max_length={args.max_length}, max_examples={args.max_examples}) ...")
    t0 = time.time()
    packed: list[tuple[list[int], list[int]]] = []
    skipped_parse = 0
    skipped_short = 0
    for ex in ds:
        text = ex.get("text", "") or ""
        parsed = parse_example(text)
        if parsed is None:
            skipped_parse += 1
            continue
        user, story = parsed
        messages = [
            Message(role="user",      content=user),
            Message(role="assistant", content=story),
        ]
        ids, mask = format_sft_example(messages, tokenizer, args.max_length)
        # After truncation, drop if the response is too short to be useful.
        if sum(mask) < args.min_response_tokens:
            skipped_short += 1
            continue
        packed.append((ids, mask))
        if len(packed) >= args.max_examples:
            break

    print(f"  Kept {len(packed):,} examples in {time.time() - t0:.1f}s")
    print(f"  Skipped: parse_fail={skipped_parse:,}, too_short={skipped_short:,}")

    print(f"\nCollating + padding to {args.max_length} ...")
    batch = pad_and_collate(packed, max_length=args.max_length, pad_id=pad_id)

    # Train/val split (random shuffle).
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(batch["input_ids"].shape[0], generator=g)
    for k in batch:
        batch[k] = batch[k][perm]

    n_val = int(len(packed) * args.val_fraction)
    n_train = len(packed) - n_val
    train_batch = {k: v[:n_train] for k, v in batch.items()}
    val_batch   = {k: v[n_train:] for k, v in batch.items()}

    print(f"  Train: {n_train:,} examples")
    print(f"  Val:   {n_val:,} examples")

    # Quick diagnostic: average response length after truncation.
    mean_resp = train_batch["loss_mask"].sum(dim=1).mean().item()
    print(f"  Avg response tokens per example (train): {mean_resp:.1f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train": train_batch,
            "val":   val_batch,
            "pad_id": pad_id,
            "max_length": args.max_length,
            "vocab_size": tokenizer.vocab_size,
        },
        out_path,
    )
    print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
