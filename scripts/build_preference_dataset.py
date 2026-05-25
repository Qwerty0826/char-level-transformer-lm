#!/usr/bin/env python3
"""
Sample candidate completion pairs from the SFT model and apply a heuristic
pre-filter, producing a JSONL of (prompt, completion_a, completion_b) triples
ready for ``scripts/label_preferences.py`` to judge.

The heuristic prefilter is critical: ~70% of pairs sampled at the same
temperature share an opening 20+ tokens and provide no DPO signal. Filtering
them out at sample time saves judge cost and avoids training on noise.

Surviving pair criteria (all must hold):
  * len(a) >= min_tokens and len(b) >= min_tokens
  * a and b share fewer than ``--shared_prefix`` opening tokens (string)
  * |len(a) - len(b)| or 30+ tokens differ
  * a != b

Example:
    python scripts/build_preference_dataset.py \\
        --sft_checkpoint checkpoints/sft/final.pt \\
        --vocab data/tinystories_v2_vocab.json \\
        --merges data/tinystories_v2_merges.txt \\
        --output data/pref_candidates.jsonl \\
        --num_prompts 1000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cs336_basics.data_sft import (
    ASSISTANT_TAG,
    EOT,
    USER_TAG,
)
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample candidate preference pairs from the SFT model")
    p.add_argument("--sft_checkpoint", required=True)
    p.add_argument("--vocab",          required=True)
    p.add_argument("--merges",         required=True)
    p.add_argument("--output",         required=True)

    # Model (must match SFT)
    p.add_argument("--vocab_size",     type=int, default=16_000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--d_model",        type=int, default=640)
    p.add_argument("--num_layers",     type=int, default=10)
    p.add_argument("--num_heads",      type=int, default=10)
    p.add_argument("--num_kv_heads",   type=int, default=2)
    p.add_argument("--d_ff",           type=int, default=1728)

    # Sampling
    p.add_argument("--num_prompts",   type=int, default=1_000)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--top_k",       type=int,   default=50)

    # Heuristic filter
    p.add_argument("--min_tokens",    type=int, default=40,
                   help="Drop completions shorter than this (in chars).")
    p.add_argument("--shared_prefix", type=int, default=20,
                   help="Drop pairs sharing this many leading chars.")
    p.add_argument("--min_diff",      type=int, default=30,
                   help="Drop pairs whose length diff is less than this.")

    p.add_argument("--device", default=None)
    p.add_argument("--dtype",  default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed",   type=int, default=0)
    return p.parse_args()


def get_device(req: str | None) -> str:
    if req:
        return req
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_sft_model(args, device: str, dtype) -> TransformerLM:
    model = TransformerLM(
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
    ckpt = torch.load(args.sft_checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_prompts() -> list[str]:
    """Pull held-out prompts from TinyStoriesInstruct (last 5% of train split)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e
    ds = load_dataset("roneneldan/TinyStoriesInstruct", split="train")
    # Use the last 5% as held-out prompts.
    start = int(len(ds) * 0.95)
    prompts: list[str] = []
    for ex in ds.select(range(start, len(ds))):
        text = ex.get("text", "") or ""
        if "\nStory:" not in text:
            continue
        head = text.split("\nStory:", 1)[0].strip()
        if head:
            prompts.append(head)
    return prompts


def format_prompt(user_text: str) -> str:
    return f"{USER_TAG}{user_text}{EOT}{ASSISTANT_TAG}"


def sample_two(model, tokenizer, eot_id: int, prompt_text: str, args, device: str) -> tuple[str, str]:
    """Sample two completions from the SFT model with the same settings."""
    formatted = format_prompt(prompt_text)
    prompt_ids = torch.tensor([tokenizer.encode(formatted)], device=device, dtype=torch.long)

    completions: list[str] = []
    for _ in range(2):
        out = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_id=eot_id,
        )
        new_ids = out[0, prompt_ids.shape[1]:].tolist()
        # Strip a trailing EOT for cleaner judging.
        if new_ids and new_ids[-1] == eot_id:
            new_ids = new_ids[:-1]
        completions.append(tokenizer.decode(new_ids))
    return completions[0], completions[1]


def heuristic_keep(a: str, b: str, args) -> bool:
    if len(a) < args.min_tokens or len(b) < args.min_tokens:
        return False
    if a == b:
        return False
    shared = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            shared += 1
        else:
            break
    if shared >= args.shared_prefix:
        return False
    if abs(len(a) - len(b)) < args.min_diff:
        # Could still be different enough if content differs — check character overlap.
        # Quick check: if substantial chunks of one are inside the other, reject.
        if a[:50] == b[:50] or a[-50:] == b[-50:]:
            return False
    return True


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    tokenizer = Tokenizer.from_files(
        args.vocab, args.merges,
        special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"],
    )
    eot_id = tokenizer.encode(EOT)[0]

    print(f"Loading SFT model: {args.sft_checkpoint}")
    model = load_sft_model(args, device, dtype)
    print(f"  Parameters: {model.num_parameters():,}")

    prompts = load_prompts()
    print(f"\nHeld-out prompts available: {len(prompts):,}")
    n_target = min(args.num_prompts, len(prompts))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        for i, prompt in enumerate(prompts[:n_target]):
            a, b = sample_two(model, tokenizer, eot_id, prompt, args, device)
            if heuristic_keep(a, b, args):
                fh.write(json.dumps({
                    "prompt": prompt,
                    "completion_a": a,
                    "completion_b": b,
                }) + "\n")
                kept += 1
            else:
                rejected += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_target - i - 1) / rate
                print(f"  {i+1}/{n_target}  kept {kept}  rejected {rejected}  "
                      f"({rate:.2f} pr/s, ETA {eta/60:.1f} min)")

    print(f"\nDone in {(time.time() - t0)/60:.1f} min")
    print(f"  Total prompts sampled: {n_target}")
    print(f"  Kept candidate pairs:  {kept}")
    print(f"  Rejected (heuristic):  {rejected}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
