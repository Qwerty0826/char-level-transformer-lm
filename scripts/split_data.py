#!/usr/bin/env python3
"""
Split a tokenised .npy file into train and validation sets.

Example:
    python scripts/split_data.py \
        --input  data/tinystories_tokens.npy \
        --val_fraction 0.1 \
        --train_out data/tinystories_tokens_train.npy \
        --val_out   data/tinystories_tokens_val.npy
"""

from __future__ import annotations

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser(description="Split a tokenised .npy into train/val")
    p.add_argument("--input",        required=True, help="Source .npy token file")
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Fraction of tokens reserved for validation (default: 0.1)")
    p.add_argument("--train_out", required=True, help="Output path for train split")
    p.add_argument("--val_out",   required=True, help="Output path for val split")
    args = p.parse_args()

    data = np.load(args.input, mmap_mode="r")
    n = len(data)
    split = int(n * (1.0 - args.val_fraction))

    print(f"Total tokens : {n:,}")
    print(f"Train tokens : {split:,}  ({100*(1-args.val_fraction):.0f}%)")
    print(f"Val tokens   : {n - split:,}  ({100*args.val_fraction:.0f}%)")

    np.save(args.train_out, data[:split])
    np.save(args.val_out,   data[split:])
    print(f"Saved: {args.train_out}")
    print(f"Saved: {args.val_out}")


if __name__ == "__main__":
    main()
