#!/usr/bin/env python3
"""
Train and serialise a byte-level BPE tokenizer, then encode a dataset
into a flat NumPy array of token IDs.

Example (TinyStories):

    python scripts/train_tokenizer.py \\
        --input data/tinystories.txt \\
        --vocab_size 10000 \\
        --output_dir data/ \\
        --prefix tinystories \\
        --encode data/tinystories.txt \\
        --encode_out data/tinystories_tokens.npy
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer, train_bpe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a BPE tokenizer and optionally encode a corpus")

    # Training
    p.add_argument("--input",      required=True, help="Training corpus text file")
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])

    # Serialisation
    p.add_argument("--output_dir", default="data/")
    p.add_argument("--prefix",     default="bpe",
                   help="Filename prefix for vocab.json and merges.txt")

    # Encoding (optional)
    p.add_argument("--encode",     default=None,
                   help="Text file to encode after training")
    p.add_argument("--encode_out", default=None,
                   help="Output .npy path for encoded tokens (uint16)")
    p.add_argument("--encode_chunk_bytes", type=int, default=10 * 1024 * 1024,
                   help="Chunk size in bytes for memory-efficient encoding")

    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # -- Train BPE -----------------------------------------------------------
    print(f"Training BPE tokenizer on {args.input!r} ...")
    print(f"  vocab_size={args.vocab_size}, special_tokens={args.special_tokens}")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Merges performed: {len(merges)}")

    # Longest token
    longest = max(vocab.values(), key=len)
    print(f"  Longest token ({len(longest)} bytes): {longest!r}")

    # -- Serialise -----------------------------------------------------------
    vocab_path  = os.path.join(args.output_dir, f"{args.prefix}_vocab.json")
    merges_path = os.path.join(args.output_dir, f"{args.prefix}_merges.txt")
    tokenizer = Tokenizer(vocab, merges, args.special_tokens)
    tokenizer.save(vocab_path, merges_path)
    print(f"  Saved vocab  -> {vocab_path}")
    print(f"  Saved merges -> {merges_path}")

    # -- Encode (optional) ---------------------------------------------------
    if args.encode:
        out_path = args.encode_out or (args.encode + ".npy")
        print(f"\nEncoding {args.encode!r} -> {out_path!r} ...")

        file_size = os.path.getsize(args.encode)
        chunk_bytes = args.encode_chunk_bytes

        all_ids = []
        n_tokens = 0
        t1 = time.time()

        with open(args.encode, "r", encoding="utf-8") as fh:
            while True:
                chunk = fh.read(chunk_bytes)
                if not chunk:
                    break
                ids = tokenizer.encode(chunk)
                all_ids.extend(ids)
                n_tokens += len(ids)
                pct = fh.tell() / file_size * 100
                print(f"  {pct:5.1f}%  {n_tokens:,} tokens", end="\r", flush=True)

        print()
        arr = np.array(all_ids, dtype=np.uint16)
        np.save(out_path, arr)
        elapsed2 = time.time() - t1
        print(f"  Encoded {n_tokens:,} tokens in {elapsed2:.1f}s")
        print(f"  Throughput: {file_size / elapsed2 / 1e6:.2f} MB/s")
        print(f"  Saved: {out_path}")

        # Compression ratio
        with open(args.encode, "rb") as fh:
            raw_bytes = fh.read(100_000)
        sample_ids = tokenizer.encode(raw_bytes.decode("utf-8", errors="replace"))
        ratio = len(raw_bytes) / max(1, len(sample_ids))
        print(f"  Compression ratio (bytes/token): {ratio:.2f}")


if __name__ == "__main__":
    main()
