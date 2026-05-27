#!/usr/bin/env python3
"""
Upload the trained checkpoints + tokenizer files to a HuggingFace model
repo so the HuggingFace Spaces app (``spaces/app.py``) can pull them on
cold start.

Run once (or each time you re-train) after authenticating with:

    huggingface-cli login

Example:

    python scripts/upload_checkpoints_to_hf.py \\
        --repo_id pragadeeshsk/transformer-lm-60m-tinystories \\
        --base_checkpoint checkpoints/base_60m/final.pt \\
        --sft_checkpoint  checkpoints/sft_v2/final.pt \\
        --dpo_checkpoint  checkpoints/dpo_v2/final.pt \\
        --vocab  data/tinystories_v2_vocab.json \\
        --merges data/tinystories_v2_merges.txt

The resulting repo layout matches what ``spaces/app.py`` expects:

    <repo>/
        base_60m/final.pt
        sft_v2/final.pt
        dpo_v2/final.pt
        tinystories_v2_vocab.json
        tinystories_v2_merges.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload checkpoints + tokenizer to a HF model repo")
    p.add_argument("--repo_id", required=True,
                   help="Target HF model repo id, e.g. 'username/transformer-lm-60m-tinystories'")
    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--sft_checkpoint",  required=True)
    p.add_argument("--dpo_checkpoint",  required=True)
    p.add_argument("--vocab",           required=True)
    p.add_argument("--merges",          required=True)
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private (default: public).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    api = HfApi()
    create_repo(args.repo_id, exist_ok=True, private=args.private)
    print(f"Repo ready: https://huggingface.co/{args.repo_id}")

    uploads = [
        (Path(args.base_checkpoint), "base_60m/final.pt"),
        (Path(args.sft_checkpoint),  "sft_v2/final.pt"),
        (Path(args.dpo_checkpoint),  "dpo_v2/final.pt"),
        (Path(args.vocab),           "tinystories_v2_vocab.json"),
        (Path(args.merges),          "tinystories_v2_merges.txt"),
    ]

    for src, dest in uploads:
        if not src.exists():
            raise SystemExit(f"Missing file: {src}")
        size_mb = src.stat().st_size / 1e6
        print(f"  uploading {dest}  ({size_mb:,.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(f"\nDone. Set the Space's MODEL_REPO variable to: {args.repo_id}")


if __name__ == "__main__":
    main()
