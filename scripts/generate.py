#!/usr/bin/env python3
"""
Text generation from a trained Transformer LM checkpoint.

Implements temperature scaling and top-p (nucleus) sampling.

Example:
    python scripts/generate.py \\
        --checkpoint checkpoints/tinystories/final.pt \\
        --vocab  data/tinystories_vocab.json \\
        --merges data/tinystories_merges.txt \\
        --prompt "Once upon a time" \\
        --max_tokens 256 \\
        --temperature 0.8 \\
        --top_p 0.95
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import load_checkpoint


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def sample_top_p(probs: torch.Tensor, p: float) -> int:
    """
    Nucleus (top-p) sampling: sample from the smallest set of tokens whose
    cumulative probability mass exceeds p.

    Args:
        probs: 1-D probability distribution over the vocabulary.
        p:     Cumulative probability threshold (0 < p <= 1).

    Returns:
        Sampled token index.
    """
    # Sort descending
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    # Keep tokens until the cumulative prob exceeds p
    # We include the token that pushes us over the threshold
    mask = cumulative - sorted_probs > p   # exclude first token that crosses
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()  # renormalise

    token_pos = torch.multinomial(sorted_probs, num_samples=1).item()
    return sorted_idx[token_pos].item()


def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
    device: str = "cpu",
) -> str:
    """
    Autoregressively generate text from a prompt.

    Args:
        model:          Trained TransformerLM.
        tokenizer:      Tokenizer matching the model's vocabulary.
        prompt:         Seed text for generation.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature:    Softmax temperature (< 1 → sharper, > 1 → flatter).
        top_p:          Nucleus sampling probability mass threshold.
                        If None, uses greedy (temperature-only) sampling.
        eos_id:         Token ID that signals end-of-sequence.
        device:         Device string.

    Returns:
        Full generated string (prompt + new tokens).
    """
    model.eval()
    ctx_len = model.context_length

    # Encode prompt
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context window
            input_ids = tokens[:, -ctx_len:]

            logits = model(input_ids)          # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]     # (vocab_size,)

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Convert to probabilities
            probs = torch.softmax(next_logits, dim=-1)

            # Sample next token
            if top_p is not None and top_p < 1.0:
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated_ids.append(next_token)
            tokens = torch.cat(
                [tokens, torch.tensor([[next_token]], device=device)], dim=1
            )

            if eos_id is not None and next_token == eos_id:
                break

    return tokenizer.decode(ids + generated_ids)


# ---------------------------------------------------------------------------
# Bits-per-character evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_bpc(
    model: TransformerLM,
    tokenizer: Tokenizer,
    text: str,
    context_length: int,
    device: str,
) -> float:
    """
    Compute bits-per-character on a text string.

    bpc = cross_entropy_in_nats / log(2)
    """
    from cs336_basics.training import cross_entropy_loss

    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return float("nan")

    # Chunk into non-overlapping windows
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(ids) - 1, context_length):
        end = min(start + context_length, len(ids) - 1)
        chunk_x = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
        chunk_y = torch.tensor([ids[start+1:end+1]], dtype=torch.long, device=device)

        logits = model(chunk_x)
        loss   = cross_entropy_loss(logits.view(-1, logits.size(-1)), chunk_y.view(-1))
        total_loss   += loss.item() * (end - start)
        total_tokens += (end - start)

    avg_loss_nats = total_loss / total_tokens
    return avg_loss_nats / math.log(2)


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
    p.add_argument("--d_ff",           type=int,   default=1344)
    p.add_argument("--theta",          type=float, default=10_000.0)
    p.add_argument("--no_tie_weights", action="store_true")

    # Generation
    p.add_argument("--prompt",     default="Once upon a time")
    p.add_argument("--max_tokens", type=int,   default=256)
    p.add_argument("--temperature",type=float, default=0.8)
    p.add_argument("--top_p",      type=float, default=0.95)
    p.add_argument("--n_samples",  type=int,   default=1)

    # Eval
    p.add_argument("--eval_bpc",  action="store_true", help="Compute bits-per-character on prompt")

    # Hardware
    p.add_argument("--device", default=None)

    return p.parse_args()


def main():
    args = parse_args()

    # Device
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
        d_ff=args.d_ff,
        theta=args.theta,
        tie_weights=not args.no_tie_weights,
        device=device,
    )

    # Load weights only (no optimizer needed for inference)
    import torch as _torch
    ckpt = _torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (step {ckpt.get('iteration', '?')})")

    # BPC evaluation
    if args.eval_bpc and len(args.prompt) > 1:
        bpc = compute_bpc(model, tokenizer, args.prompt, args.context_length, device)
        print(f"Bits-per-character on prompt: {bpc:.4f}")

    # Generate
    for i in range(args.n_samples):
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
