#!/usr/bin/env python3
"""
Interactive REPL for chatting with a trained Transformer LM.

Type a prompt; the model continues it.  Slash-commands control sampling:

    /temp 0.8         set temperature
    /top_p 0.9        set nucleus probability mass
    /top_k 40         set top-k filter (0 = off)
    /min_p 0.05       set min-p filter (0 = off)
    /rep 1.2          repetition penalty
    /max 200          max tokens per generation
    /reset            clear conversation context (start fresh)
    /show             print current settings
    /help             list commands
    /quit             exit

Each generation continues from your previous context, so the model can
build a coherent dialogue (within its context window).

Example:
    python scripts/chat.py \
        --checkpoint checkpoints/tinystories/final.pt \
        --vocab data/tinystories_vocab.json \
        --merges data/tinystories_merges.txt
"""

from __future__ import annotations

import argparse
import sys

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive chat with a trained LM")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab",      required=True)
    p.add_argument("--merges",     required=True)
    p.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])

    # Model shape
    p.add_argument("--vocab_size",     type=int, default=10_000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model",        type=int, default=512)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--num_heads",      type=int, default=16)
    p.add_argument("--num_kv_heads",   type=int, default=None)
    p.add_argument("--d_ff",           type=int, default=1344)
    p.add_argument("--theta",          type=float, default=10_000.0)
    p.add_argument("--no_tie_weights", action="store_true")

    # Sampling defaults
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--top_k",       type=int,   default=None)
    p.add_argument("--min_p",       type=float, default=None)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--max_tokens",  type=int, default=200)

    p.add_argument("--device", default=None)
    return p.parse_args()


HELP = """\
Commands:
  /temp <float>     set temperature
  /top_p <float>    set nucleus probability mass (0..1)
  /top_k <int>      set top-k filter (0 = off)
  /min_p <float>    set min-p filter (0 = off)
  /rep <float>      set repetition penalty (1.0 = none)
  /max <int>        max tokens per generation
  /reset            clear context
  /show             show current settings
  /help             this message
  /quit             exit
"""


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
    print(f"Loading model on {device}...")

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    eos_id = tokenizer._special_to_id.get("<|endoftext|>")

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
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    cfg = {
        "temperature": args.temperature,
        "top_p":       args.top_p,
        "top_k":       args.top_k,
        "min_p":       args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens":  args.max_tokens,
    }

    print(f"Model ready ({model.num_parameters():,} params).")
    print("Type a prompt, or /help for commands.\n")

    context_ids: list[int] = []     # rolling conversation history

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        if user.startswith("/"):
            parts = user.split()
            cmd = parts[0]
            if cmd == "/quit":
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/show":
                for k, v in cfg.items():
                    print(f"  {k}: {v}")
                print(f"  context_tokens: {len(context_ids)}")
            elif cmd == "/reset":
                context_ids = []
                print("(context cleared)")
            elif cmd == "/temp" and len(parts) == 2:
                cfg["temperature"] = float(parts[1])
            elif cmd == "/top_p" and len(parts) == 2:
                cfg["top_p"] = float(parts[1])
            elif cmd == "/top_k" and len(parts) == 2:
                v = int(parts[1])
                cfg["top_k"] = v if v > 0 else None
            elif cmd == "/min_p" and len(parts) == 2:
                v = float(parts[1])
                cfg["min_p"] = v if v > 0 else None
            elif cmd == "/rep" and len(parts) == 2:
                cfg["repetition_penalty"] = float(parts[1])
            elif cmd == "/max" and len(parts) == 2:
                cfg["max_tokens"] = int(parts[1])
            else:
                print(f"Unknown command: {user}. Type /help.")
            continue

        # Append the new user input to context.
        new_ids = tokenizer.encode(user)
        context_ids = context_ids + new_ids
        # Truncate from the left if we exceed the context window.
        if len(context_ids) > model.context_length - cfg["max_tokens"]:
            context_ids = context_ids[-(model.context_length - cfg["max_tokens"]):]

        prompt_t = torch.tensor([context_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out_ids = model.generate(
                prompt_t,
                max_new_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                top_k=cfg["top_k"],
                min_p=cfg["min_p"],
                repetition_penalty=cfg["repetition_penalty"],
                eos_id=eos_id,
                use_cache=True,
            )
        full_ids = out_ids[0].tolist()
        new_part = full_ids[len(context_ids):]
        completion = tokenizer.decode(new_part)
        print(completion)
        context_ids = full_ids


if __name__ == "__main__":
    main()
