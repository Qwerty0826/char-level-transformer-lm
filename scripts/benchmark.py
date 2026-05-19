#!/usr/bin/env python3
"""
Measure forward/backward throughput, peak memory, and Model FLOPs Utilisation.

Useful to verify that:
  - KV-cache decoding actually beats non-cached decoding (≥10× expected).
  - Mixed precision (bfloat16) gives the expected throughput speedup.
  - The achieved MFU is reasonable (10–40% on most hardware).

Example:
    python scripts/benchmark.py \
        --config configs/tinystories.yaml \
        --warmup 5 --iters 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.model import TransformerLM
from cs336_basics.training import cross_entropy_loss, get_batch


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark training + inference")
    p.add_argument("--config", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters",  type=int, default=30)
    p.add_argument("--device", default=None)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--gen_tokens", type=int, default=64, help="Tokens to generate in inference bench")
    return p.parse_args()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench_train(model, data, cfg, device, warmup, iters):
    """Forward + backward + step throughput in tokens/sec."""
    B = cfg.get("batch_size", 32)
    T = cfg["context_length"]
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    model.train()

    for _ in range(warmup):
        x, y = get_batch(data, B, T, device)
        opt.zero_grad()
        loss = cross_entropy_loss(model(x).view(-1, model.vocab_size), y.view(-1))
        loss.backward()
        opt.step()
    sync(device)

    t0 = time.time()
    for _ in range(iters):
        x, y = get_batch(data, B, T, device)
        opt.zero_grad()
        loss = cross_entropy_loss(model(x).view(-1, model.vocab_size), y.view(-1))
        loss.backward()
        opt.step()
    sync(device)
    elapsed = time.time() - t0

    tokens = iters * B * T
    return tokens / elapsed, elapsed / iters


def bench_inference(model, device, prompt_len, gen_tokens, use_cache):
    """Pure decoding tokens/sec (single-batch)."""
    prompt = torch.randint(0, model.vocab_size, (1, prompt_len), device=device)
    sync(device)
    t0 = time.time()
    with torch.no_grad():
        model.generate(
            prompt, max_new_tokens=gen_tokens,
            temperature=1.0, top_p=None, top_k=None,
            use_cache=use_cache,
        )
    sync(device)
    elapsed = time.time() - t0
    return gen_tokens / elapsed, elapsed


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Device:          {device}")
    if device == "cuda":
        print(f"GPU:             {torch.cuda.get_device_name(0)}")

    dtype = torch.bfloat16 if cfg.get("dtype") == "bfloat16" else torch.float32

    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg.get("d_ff"),
        theta=cfg.get("theta", 10_000.0),
        tie_weights=cfg.get("tie_weights", True),
        device=device,
        dtype=dtype,
    )
    print(f"Parameters:      {model.num_parameters():,}")
    print(f"Context length:  {cfg['context_length']}")
    print(f"FLOPs/fwd tok:   {model.estimate_flops_per_token():,}")

    if not args.no_compile and cfg.get("compile", False):
        backend = cfg.get("compile_backend") or ("aot_eager" if device == "mps" else "inductor")
        print(f"Compiling with backend={backend!r} ...")
        model = torch.compile(model, backend=backend)

    train_data = np.load(cfg["train_data"], mmap_mode="r")

    print("\n=== Training throughput ===")
    tok_per_s, step_time = bench_train(model, train_data, cfg, device, args.warmup, args.iters)
    print(f"Tokens/sec:      {tok_per_s:,.0f}")
    print(f"Step time:       {step_time*1000:.1f} ms")

    # Memory (CUDA only)
    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU mem:    {peak_gb:.2f} GB")

    print("\n=== Inference throughput (KV cache vs no cache) ===")
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    prompt_len = 32
    fast_tps, fast_t = bench_inference(base_model, device, prompt_len, args.gen_tokens, use_cache=True)
    slow_tps, slow_t = bench_inference(base_model, device, prompt_len, args.gen_tokens, use_cache=False)
    print(f"With cache:      {fast_tps:.1f} tok/s ({fast_t:.2f}s for {args.gen_tokens} tokens)")
    print(f"No cache:        {slow_tps:.1f} tok/s ({slow_t:.2f}s for {args.gen_tokens} tokens)")
    print(f"Speedup:         {fast_tps / max(slow_tps, 1e-6):.1f}×")


if __name__ == "__main__":
    main()
