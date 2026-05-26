#!/usr/bin/env python3
"""
Build a packed SFT dataset from roneneldan/TinyStoriesInstruct.

The HF dataset stores the original .txt file line-by-line: each row is one
line, and stories are separated by a row whose text is `<|endoftext|>`.
A single story block looks like:

    Features: Dialogue, MoralValue
    Words: puppy, return, kind
    Summary: A young girl finds a lost puppy and returns it.
    Story:
    Once upon a time, ...
    ...
    <|endoftext|>

We accumulate lines until the separator, then split each block on the
"Story:" marker — lines before become the user message (constraints),
lines after become the assistant response (the story body).

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
STORY_SEPARATOR = "<|endoftext|>"


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


def parse_block(lines: list[str]) -> tuple[str, str] | None:
    """Parse one story block (a list of lines between <|endoftext|> separators).

    Returns (user_prompt, story) or None if no Story: marker is found or either
    side is empty.
    """
    story_idx = None
    inline_story = ""
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(STORY_MARKER):
            story_idx = i
            inline_story = stripped[len(STORY_MARKER):].strip()
            break
    if story_idx is None:
        return None

    prompt_lines = [l for l in lines[:story_idx] if l.strip()]
    body_lines = [l for l in lines[story_idx + 1:] if l.strip()]
    if inline_story:
        body_lines.insert(0, inline_story)

    user = "\n".join(prompt_lines).strip()
    story = "\n".join(body_lines).strip()
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
    blocks_seen = 0
    current_lines: list[str] = []

    def consume_block() -> bool:
        """Process the accumulated block; return True if we should stop."""
        nonlocal skipped_parse, skipped_short, blocks_seen
        blocks_seen += 1
        parsed = parse_block(current_lines)
        if parsed is None:
            skipped_parse += 1
            return False
        user, story = parsed
        messages = [
            Message(role="user",      content=user),
            Message(role="assistant", content=story),
        ]
        ids, mask = format_sft_example(messages, tokenizer, args.max_length)
        if sum(mask) < args.min_response_tokens:
            skipped_short += 1
            return False
        packed.append((ids, mask))
        return len(packed) >= args.max_examples

    for ex in ds:
        line = ex.get("text", "") or ""
        if line.strip() == STORY_SEPARATOR:
            if current_lines:
                stop = consume_block()
                current_lines = []
                if stop:
                    break
        else:
            current_lines.append(line)
    # Trailing block without final <|endoftext|>.
    if current_lines and len(packed) < args.max_examples:
        consume_block()

    print(f"  Saw {blocks_seen:,} story blocks; kept {len(packed):,} examples in {time.time() - t0:.1f}s")
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
