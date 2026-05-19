#!/usr/bin/env python3
"""
Evaluate a trained Transformer LM checkpoint.

Reports perplexity, bits-per-character, and optionally generated samples.

Example:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/tinystories/final.pt \\
        --data data/tinystories_tokens_val.npy \\
        --vocab  data/tinystories_vocab.json \\
        --merges data/tinystories_merges.txt \\
        --n_batches 100 \\
        --generate_samples 3
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import cross_entropy_loss, get_batch
from scripts.generate import compute_bpc, generate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Transformer LM")

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data",       required=True, help=".npy token file")
    p.add_argument("--vocab",      required=True)
    p.add_argument("--merges",     required=True)
    p.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])

    # Model shape
    p.add_argument("--vocab_size",     type=int,   default=10_000)
    p.add_argument("--context_length", type=int,   default=256)
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--num_layers",     type=int,   default=4)
    p.add_argument("--num_heads",      type=int,   default=16)
    p.add_argument("--d_ff",           type=int,   default=1344)
    p.add_argument("--theta",          type=float, default=10_000.0)
    p.add_argument("--no_tie_weights", action="store_true")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_batches",  type=int, default=50, help="Batches for perplexity estimation")

    p.add_argument("--generate_samples", type=int,   default=0)
    p.add_argument("--prompt",           default="Once upon a time")
    p.add_argument("--max_tokens",       type=int,   default=200)
    p.add_argument("--temperature",      type=float, default=0.8)
    p.add_argument("--top_p",            type=float, default=0.95)

    p.add_argument("--device", default=None)

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Tokenizer
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    eos_id = tokenizer._special_to_id.get("<|endoftext|>")

    # Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        tie_weights=not args.no_tie_weights,
        device=device,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (step {ckpt.get('iteration', '?')})")
    print(f"Parameters: {model.num_parameters():,} total")

    # Perplexity
    data = np.load(args.data, mmap_mode="r")
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(args.n_batches):
            x, y = get_batch(data, args.batch_size, args.context_length, device)
            logits = model(x)
            loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / args.n_batches
    ppl = math.exp(avg_loss)
    bpc = avg_loss / math.log(2)

    print(f"\n=== Evaluation Results ===")
    print(f"  Avg loss (nats): {avg_loss:.4f}")
    print(f"  Perplexity:      {ppl:.2f}")
    print(f"  Bits-per-token:  {bpc:.4f}")

    # Text generation
    if args.generate_samples > 0:
        print(f"\n=== Generated Samples (T={args.temperature}, top_p={args.top_p}) ===")
        for i in range(args.generate_samples):
            print(f"\n--- Sample {i+1} ---")
            text = generate(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_id=eos_id,
                device=device,
            )
            print(text)


if __name__ == "__main__":
    main()
