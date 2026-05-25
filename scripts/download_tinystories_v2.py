#!/usr/bin/env python3
"""
One-shot setup for the 60M TinyStoriesV2-GPT4 pretrain.

Steps (all idempotent — re-running skips work that's already done):
  1. Download `roneneldan/TinyStoriesV2-GPT4-train` from Hugging Face Hub.
  2. Write a flat text file with `<|endoftext|>` separating stories.
  3. Train a byte-level BPE tokenizer (vocab_size=16000) with chat-template
     specials baked in: <|endoftext|>, <|user|>, <|assistant|>, <|system|>.
     Baking specials in at training time means SFT never needs an embedding
     resize — the model is pretrained with these IDs present in the vocab.
  4. Encode the corpus to a uint16 .npy file.
  5. Save 90/10 train/val splits as data/tinystories_v2_tokens_{train,val}.npy.

Example (Colab):
    python scripts/download_tinystories_v2.py --output_dir data/

Output files (default --output_dir=data/):
    data/tinystories_v2.txt
    data/tinystories_v2_vocab.json
    data/tinystories_v2_merges.txt
    data/tinystories_v2_tokens.npy           (full corpus, uint16)
    data/tinystories_v2_tokens_train.npy     (90%)
    data/tinystories_v2_tokens_val.npy       (10%)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer, train_bpe


CHAT_SPECIALS = [
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
]
# The GPT-4-regenerated TinyStories live as a data file inside the original
# roneneldan/TinyStories repo, not as a standalone HF dataset. We grab the
# raw .txt directly with hf_hub_download — the file already uses <|endoftext|>
# between stories, so no reformatting is needed.
HF_DATASET = "roneneldan/TinyStories"
HF_DATA_FILE = "TinyStoriesV2-GPT4-train.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download + tokenize TinyStoriesV2-GPT4")
    p.add_argument("--output_dir", default="data/")
    p.add_argument("--prefix",     default="tinystories_v2")
    p.add_argument("--vocab_size", type=int, default=16_000)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--encode_chunk_bytes", type=int, default=20 * 1024 * 1024,
                   help="Chunk size for memory-efficient encoding (default 20MB)")
    p.add_argument("--skip_download",   action="store_true",
                   help="Skip the HF download step if the .txt file already exists.")
    p.add_argument("--skip_tokenizer",  action="store_true",
                   help="Skip BPE training if the vocab/merges files already exist.")
    p.add_argument("--skip_encode",     action="store_true",
                   help="Skip encoding if the tokens .npy already exists.")
    return p.parse_args()


def download_corpus(out_txt: Path) -> None:
    """
    Pull TinyStoriesV2-GPT4-train.txt from the HF Hub and copy it to out_txt.
    The source file already separates stories with <|endoftext|>, so no
    transformation is needed — we just stream the bytes.
    """
    print(f"Downloading {HF_DATA_FILE} from {HF_DATASET} ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        ) from e

    t0 = time.time()
    local_path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=HF_DATA_FILE,
        repo_type="dataset",
    )
    print(f"  Cached at {local_path}")

    # Stream-copy so we don't load the whole file into memory.
    chunk_bytes = 16 * 1024 * 1024
    with open(local_path, "r", encoding="utf-8") as src, open(out_txt, "w", encoding="utf-8") as dst:
        while True:
            chunk = src.read(chunk_bytes)
            if not chunk:
                break
            dst.write(chunk)

    size_gb = out_txt.stat().st_size / 1e9
    print(f"  Done in {time.time() - t0:.1f}s  ({size_gb:.2f} GB)")


def train_tokenizer(
    input_txt: Path,
    vocab_size: int,
    vocab_out: Path,
    merges_out: Path,
) -> Tokenizer:
    print(f"\nTraining BPE on {input_txt} ...")
    print(f"  vocab_size={vocab_size}")
    print(f"  special_tokens={CHAT_SPECIALS}")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=str(input_txt),
        vocab_size=vocab_size,
        special_tokens=CHAT_SPECIALS,
    )
    print(f"  Done in {time.time() - t0:.1f}s ({len(vocab)} tokens, {len(merges)} merges)")

    tokenizer = Tokenizer(vocab, merges, CHAT_SPECIALS)
    tokenizer.save(str(vocab_out), str(merges_out))
    print(f"  Saved {vocab_out}")
    print(f"  Saved {merges_out}")
    return tokenizer


def encode_corpus(
    tokenizer: Tokenizer,
    input_txt: Path,
    out_npy: Path,
    chunk_bytes: int,
) -> None:
    """Streaming encode to keep memory bounded."""
    print(f"\nEncoding {input_txt} -> {out_npy} ...")
    total_bytes = input_txt.stat().st_size
    t0 = time.time()
    all_ids: list[int] = []

    with open(input_txt, "r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            all_ids.extend(tokenizer.encode(chunk))
            pct = fh.tell() / total_bytes * 100
            print(f"  {pct:5.1f}%  {len(all_ids):,} tokens", end="\r", flush=True)

    print()
    # uint16 fits vocab_size up to 65535 — comfortable for 16K vocab.
    arr = np.array(all_ids, dtype=np.uint16)
    np.save(out_npy, arr)
    elapsed = time.time() - t0
    print(f"  Encoded {len(arr):,} tokens in {elapsed:.1f}s "
          f"({total_bytes / elapsed / 1e6:.1f} MB/s)")
    print(f"  Compression: {total_bytes / len(arr):.2f} bytes/token")


def split_train_val(
    tokens_npy: Path,
    train_out: Path,
    val_out: Path,
    val_fraction: float,
) -> None:
    print(f"\nSplitting {tokens_npy} into train/val ...")
    data = np.load(tokens_npy, mmap_mode="r")
    n = len(data)
    split = int(n * (1.0 - val_fraction))
    np.save(train_out, data[:split])
    np.save(val_out,   data[split:])
    print(f"  Train: {split:,} tokens -> {train_out}")
    print(f"  Val:   {n - split:,} tokens -> {val_out}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path     = out_dir / f"{args.prefix}.txt"
    vocab_path   = out_dir / f"{args.prefix}_vocab.json"
    merges_path  = out_dir / f"{args.prefix}_merges.txt"
    tokens_path  = out_dir / f"{args.prefix}_tokens.npy"
    train_path   = out_dir / f"{args.prefix}_tokens_train.npy"
    val_path     = out_dir / f"{args.prefix}_tokens_val.npy"

    # 1-2. Download.
    if args.skip_download or txt_path.exists():
        print(f"Skipping download — using existing {txt_path}")
    else:
        download_corpus(txt_path)

    # 3. Train BPE.
    if args.skip_tokenizer or (vocab_path.exists() and merges_path.exists()):
        print(f"Skipping BPE training — loading {vocab_path}")
        tokenizer = Tokenizer.from_files(
            str(vocab_path), str(merges_path), CHAT_SPECIALS
        )
    else:
        tokenizer = train_tokenizer(
            txt_path, args.vocab_size, vocab_path, merges_path
        )

    # 4. Encode.
    if args.skip_encode or tokens_path.exists():
        print(f"Skipping encode — using existing {tokens_path}")
    else:
        encode_corpus(tokenizer, txt_path, tokens_path, args.encode_chunk_bytes)

    # 5. Train/val split.
    split_train_val(tokens_path, train_path, val_path, args.val_fraction)

    print("\nDone. Special token IDs:")
    for tok in CHAT_SPECIALS:
        ids = tokenizer.encode(tok)
        print(f"  {tok!r:18s} -> {ids}")

    print(f"\nReady to train. Example:")
    print(f"  python scripts/train.py --config configs/tinystories_60m.yaml")


if __name__ == "__main__":
    main()
