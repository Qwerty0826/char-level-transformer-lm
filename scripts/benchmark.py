#!/usr/bin/env python3
"""
Benchmark training and inference throughput for a TransformerLM.

Reports:
  - Training tokens/sec, step time, peak GPU memory
  - MFU (Model FLOPs Utilisation), with the correct peak for fp32 vs
    fp16/bf16 (tensor cores).  Picking the wrong peak makes MFU look
    8x lower than it really is.
  - Inference throughput at multiple context lengths, comparing the
    KV-cached path against the full-recomputation path.  KV cache
    only wins when context >> kernel-launch overhead — this benchmark
    surfaces the crossover point.

Example:
    python scripts/benchmark.py --config configs/tinystories.yaml --iters 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.model import TransformerLM
from cs336_basics.training import cross_entropy_loss, get_batch


# ---------------------------------------------------------------------------
# Peak FLOPs lookup (kept in sync with scripts/train.py)
# ---------------------------------------------------------------------------

PEAK_FLOPS_TC = {     # fp16 / bf16 tensor-core peaks (FLOPS)
    "A100":    312e12, "H100":    989e12, "T4":       65e12,
    "V100":    125e12, "L4":      121e12, "RTX4090": 330e12, "RTX3090": 142e12,
}
PEAK_FLOPS_FP32 = {   # plain fp32 (no tensor cores) peaks
    "A100":    19.5e12, "H100":    67.0e12, "T4":       8.1e12,
    "V100":    15.7e12, "L4":      30.3e12, "RTX4090":  82.6e12, "RTX3090":  35.6e12,
}


def lookup_peak(device: str, dtype_label: str) -> float | None:
    if device != "cuda":
        return None
    table = PEAK_FLOPS_FP32 if dtype_label == "float32" else PEAK_FLOPS_TC
    gpu = torch.cuda.get_device_name(0).replace(" ", "").upper()
    for k, v in table.items():
        if k.upper() in gpu:
            return v
    return None


# ---------------------------------------------------------------------------
# Benchmark primitives
# ---------------------------------------------------------------------------

def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench_train(model, data, cfg, device, warmup, iters) -> Tuple[float, float]:
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

    return iters * B * T / elapsed, elapsed / iters


def bench_inference(model, device, prompt_len, gen_tokens, use_cache) -> Tuple[float, float]:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark training + inference")
    p.add_argument("--config", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters",  type=int, default=30)
    p.add_argument("--device", default=None)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument(
        "--ctx_sweep", type=int, nargs="+",
        default=[32, 64, 128, 224],
        help="Generation lengths to sweep for the KV-cache vs no-cache comparison",
    )
    return p.parse_args()


def main() -> None:
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

    dtype_label = cfg.get("dtype", "float32")
    dtype = torch.bfloat16 if dtype_label == "bfloat16" else torch.float32

    print(f"Device:          {device}")
    if device == "cuda":
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Precision:       {dtype_label}")

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
    flops_per_seq = model.estimate_flops_per_token()
    print(f"Parameters:      {model.num_parameters():,}")
    print(f"Context length:  {cfg['context_length']}")
    print(f"FLOPs/fwd seq:   {flops_per_seq:,}")

    if not args.no_compile and cfg.get("compile", False):
        backend = cfg.get("compile_backend") or ("aot_eager" if device == "mps" else "inductor")
        print(f"Compiling with backend={backend!r} ...")
        model = torch.compile(model, backend=backend)

    train_data = np.load(cfg["train_data"], mmap_mode="r")

    # ---- Training throughput ----------------------------------------------
    print("\n=== Training throughput ===")
    tok_per_s, step_time = bench_train(model, train_data, cfg, device, args.warmup, args.iters)
    print(f"Tokens/sec:      {tok_per_s:>10,.0f}")
    print(f"Step time:       {step_time * 1000:>10.1f} ms")

    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU mem:    {peak_gb:>10.2f} GB")

    # MFU using the right peak for the dtype actually being used.
    peak = lookup_peak(device, dtype_label)
    if peak is not None:
        achieved = 3 * flops_per_seq * cfg.get("batch_size", 32) / step_time
        mfu = achieved / peak
        print(f"MFU ({dtype_label:>8s}):  {mfu * 100:>10.1f}%  "
              f"({achieved/1e12:.2f} / {peak/1e12:.1f} TFLOPS peak)")

    # ---- Inference throughput: KV cache sweep -----------------------------
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    print("\n=== Inference throughput (KV cache vs full recompute) ===")
    print(f"Sweeping generation length: {args.ctx_sweep}")
    print(f"{'gen_tokens':>10}  {'cache tok/s':>12}  {'no-cache tok/s':>16}  {'speedup':>8}")
    print("-" * 56)
    rows: List[Tuple[int, float, float, float]] = []
    for gen_tok in args.ctx_sweep:
        # Single shared prompt length keeps prefill cost constant across rows.
        cache_tps,    _ = bench_inference(base_model, device, prompt_len=16, gen_tokens=gen_tok, use_cache=True)
        no_cache_tps, _ = bench_inference(base_model, device, prompt_len=16, gen_tokens=gen_tok, use_cache=False)
        speedup = cache_tps / max(no_cache_tps, 1e-6)
        rows.append((gen_tok, cache_tps, no_cache_tps, speedup))
        print(f"{gen_tok:>10}  {cache_tps:>12.1f}  {no_cache_tps:>16.1f}  {speedup:>7.2f}×")

    # Quick interpretation hint.
    crossover = next((r[0] for r in rows if r[3] > 1.0), None)
    if crossover is None:
        print(
            "\nKV cache loses across all tested lengths.  Expected for tiny\n"
            "models (≤100M params) on GPU: per-step kernel launch overhead\n"
            "dominates the saved attention/projection FLOPs.  KV cache\n"
            "speedups appear once the model is large or the context is\n"
            "much longer than tested here."
        )
    else:
        print(f"\nKV cache crossover at gen_tokens ≥ {crossover}.")


if __name__ == "__main__":
    main()
