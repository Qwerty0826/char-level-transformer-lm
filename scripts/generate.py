#!/usr/bin/env python3
"""
Text generation from a trained Transformer LM checkpoint.

Uses the model's built-in KV-cached ``generate()`` (O(T) decoding rather
than O(T^2)) with support for:

  - Temperature scaling
  - Top-p (nucleus) sampling
  - Top-k filtering
  - Min-p filtering
  - Repetition penalty (Keskar et al., 2019)

Example:
    python scripts/generate.py \\
        --checkpoint checkpoints/tinystories/final.pt \\
        --vocab  data/tinystories_vocab.json \\
        --merges data/tinystories_merges.txt \\
        --prompt "Once upon a time" \\
        --max_tokens 256 \\
        --temperature 0.8 --top_p 0.95
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import cross_entropy_loss


# ---------------------------------------------------------------------------
# Bits-per-character (held-out evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_bpc(
    model: TransformerLM,
    tokenizer: Tokenizer,
    text: str,
    context_length: int,
    device: str,
) -> float:
    """Bits-per-character on `text` using non-overlapping windows."""
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return float("nan")

    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(ids) - 1, context_length):
        end = min(start + context_length, len(ids) - 1)
        chunk_x = torch.tensor([ids[start:end]],   dtype=torch.long, device=device)
        chunk_y = torch.tensor([ids[start+1:end+1]], dtype=torch.long, device=device)
        logits = model(chunk_x)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), chunk_y.view(-1))
        total_loss   += loss.item() * (end - start)
        total_tokens += (end - start)

    avg_loss_nats = total_loss / total_tokens
    # Convert nats per token -> bits per character using the corpus length ratio.
    nats_per_char = avg_loss_nats * total_tokens / len(text)
    return nats_per_char / math.log(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a trained LM")

    # Checkpoint + tokenizer
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--vocab",   required=True, help="Path to vocab JSON")
    p.add_argument("--merges",  required=True, help="Path to merges TXT")
    p.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])

    # Model shape (must match checkpoint)
    p.add_argument("--vocab_size",     type=int,   default=10_000)
    p.add_argument("--context_length", type=int,   default=256)
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--num_layers",     type=int,   default=4)
    p.add_argument("--num_heads",      type=int,   default=16)
    p.add_argument("--num_kv_heads",   type=int,   default=None)
    p.add_argument("--d_ff",           type=int,   default=1344)
    p.add_argument("--theta",          type=float, default=10_000.0)
    p.add_argument("--no_tie_weights", action="store_true")

    # Generation
    p.add_argument("--prompt",     default="Once upon a time")
    p.add_argument("--max_tokens", type=int,   default=256)
    p.add_argument("--temperature",type=float, default=0.8)
    p.add_argument("--top_p",      type=float, default=0.95)
    p.add_argument("--top_k",      type=int,   default=None)
    p.add_argument("--min_p",      type=float, default=None)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--n_samples",  type=int,   default=1)
    p.add_argument("--no_cache",   action="store_true", help="Disable KV cache (slower)")

    p.add_argument("--eval_bpc",  action="store_true", help="Compute bits-per-character on prompt")

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
    print(f"Device: {device}")

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
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        tie_weights=not args.no_tie_weights,
        device=device,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    # Allow loading a torch.compile'd state dict by stripping its prefix.
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint (step {ckpt.get('iteration', '?')})")

    if args.eval_bpc and len(args.prompt) > 1:
        bpc = compute_bpc(model, tokenizer, args.prompt, args.context_length, device)
        print(f"Bits-per-character on prompt: {bpc:.4f}")

    # Generate
    for i in range(args.n_samples):
        print(f"\n--- Sample {i+1} (T={args.temperature}, top_p={args.top_p}, "
              f"top_k={args.top_k}, min_p={args.min_p}, rep_pen={args.repetition_penalty}) ---")
        ids = tokenizer.encode(args.prompt)
        prompt_t = torch.tensor([ids], dtype=torch.long, device=device)

        t0 = time.time()
        out_ids = model.generate(
            prompt_t,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            eos_id=eos_id,
            use_cache=not args.no_cache,
        )
        elapsed = time.time() - t0
        n_new = out_ids.shape[1] - len(ids)
        tok_s = n_new / max(elapsed, 1e-6)
        text = tokenizer.decode(out_ids[0].tolist())
        print(text)
        print(f"\n  Generated {n_new} tokens in {elapsed:.2f}s ({tok_s:.1f} tok/s)")


if __name__ == "__main__":
    main()
