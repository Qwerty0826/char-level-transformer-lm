#!/usr/bin/env python3
"""
CLI launcher for the Gradio playground.

The UI itself lives in ``cs336_basics.playground`` so it ships inside the
installable package (the HuggingFace Space imports it from there); this
script only parses arguments and launches the app locally.

Example (base only):
    python scripts/playground.py \\
        --checkpoint checkpoints/base_60m/final.pt \\
        --vocab data/tinystories_v2_vocab.json \\
        --merges data/tinystories_v2_merges.txt

Example (3-way base / SFT / DPO comparison):
    python scripts/playground.py \\
        --checkpoint     checkpoints/base_60m/final.pt \\
        --checkpoint_sft checkpoints/sft/final.pt \\
        --checkpoint_dpo checkpoints/dpo/final.pt \\
        --vocab data/tinystories_v2_vocab.json \\
        --merges data/tinystories_v2_merges.txt \\
        --share        # exposes a public ngrok-style URL
"""

from __future__ import annotations

import argparse

from cs336_basics.playground import build_ui, load_models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio playground for a trained LM")
    p.add_argument("--checkpoint", required=True,
                   help="Base model checkpoint (pretrain output)")
    p.add_argument("--checkpoint_sft", default=None,
                   help="Optional SFT checkpoint; enables the Aligned tab when set")
    p.add_argument("--checkpoint_dpo", default=None,
                   help="Optional DPO checkpoint; enables the Aligned tab when set")
    p.add_argument("--vocab",  required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--special_tokens", nargs="*",
                   default=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"])

    # Model shape (must match every checkpoint passed in). Defaults are the 60M
    # post-training config. For the legacy 17M TinyStories run pass:
    #   --vocab_size 10000 --context_length 256 --d_model 512 --num_layers 4
    #   --num_heads 16 --d_ff 1344
    p.add_argument("--vocab_size",     type=int, default=16_000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--d_model",        type=int, default=640)
    p.add_argument("--num_layers",     type=int, default=10)
    p.add_argument("--num_heads",      type=int, default=10)
    p.add_argument("--num_kv_heads",   type=int, default=2)
    p.add_argument("--d_ff",           type=int, default=1728)
    p.add_argument("--theta",          type=float, default=10_000.0)
    p.add_argument("--no_tie_weights", action="store_true")

    p.add_argument("--device", default=None)
    p.add_argument("--dtype",  default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--host",   default="127.0.0.1")
    p.add_argument("--port",   type=int, default=7860)
    p.add_argument("--share",  action="store_true",
                   help="Expose a temporary public URL via Gradio's tunnel")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models, tokenizer, device, eos_id = load_models(args)
    base_params = models["base"]["model"].num_parameters()
    print(f"[playground] {base_params:,}-param model on {device}  "
          f"({len(models)} checkpoint(s) loaded)")
    app = build_ui(models, tokenizer, device, eos_id)
    app.queue().launch(
        server_name=args.host, server_port=args.port,
        share=args.share, show_error=True,
    )


if __name__ == "__main__":
    main()
