#!/usr/bin/env python3
"""
End-to-end evaluation across base / SFT / DPO checkpoints.

Produces ``results.md`` containing:
  1. Held-out perplexity per checkpoint.
  2. Pairwise judge win-rate matrix between {base, SFT, DPO}.
  3. Swap-consistency rate (position-bias control; must be >=70% to trust).
  4. A handful of sample outputs per checkpoint for qualitative inspection.

The judge is a local open-weight LLM (Qwen2.5-3B-Instruct by default), so
no external API is needed. Every pair is judged in BOTH orders and pairs
where the judge flips are discarded from the win count but reported in
the swap-consistency metric.

Example:
    python scripts/eval_all.py \\
        --base_checkpoint  checkpoints/base_60m/final.pt \\
        --sft_checkpoint   checkpoints/sft/final.pt      \\
        --dpo_checkpoint   checkpoints/dpo/final.pt      \\
        --val_data         data/tinystories_v2_tokens_val.npy \\
        --vocab            data/tinystories_v2_vocab.json \\
        --merges           data/tinystories_v2_merges.txt \\
        --output           results.md \\
        --num_eval_prompts 150 --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data_sft import ASSISTANT_TAG, EOT, USER_TAG
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import cross_entropy_loss, get_batch


JUDGE_SYSTEM = (
    "You are an expert evaluator of short stories for children. "
    "Read two completions to the same prompt and decide which is better."
)
JUDGE_RUBRIC = """Compare the two completions. Pick the one that is more coherent, has clearer narrative structure, and ends more satisfyingly.

Prompt:
{prompt}

Completion A:
{a}

Completion B:
{b}

Answer with a single letter, A or B. Do not explain."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end eval matrix")

    p.add_argument("--base_checkpoint", required=True)
    p.add_argument("--sft_checkpoint",  required=True)
    p.add_argument("--dpo_checkpoint",  required=True)
    p.add_argument("--val_data",        required=True, help=".npy token file for PPL")
    p.add_argument("--vocab",           required=True)
    p.add_argument("--merges",          required=True)
    p.add_argument("--output",          default="results.md")

    # Model (must match the three checkpoints — assumed identical architecture)
    p.add_argument("--vocab_size",     type=int, default=16_000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--d_model",        type=int, default=640)
    p.add_argument("--num_layers",     type=int, default=10)
    p.add_argument("--num_heads",      type=int, default=10)
    p.add_argument("--num_kv_heads",   type=int, default=2)
    p.add_argument("--d_ff",           type=int, default=1728)

    # PPL eval
    p.add_argument("--ppl_batches",    type=int, default=50)
    p.add_argument("--ppl_batch_size", type=int, default=32)

    # Win-rate eval
    p.add_argument("--num_eval_prompts", type=int, default=150)
    p.add_argument("--max_new_tokens",   type=int, default=200)
    p.add_argument("--temperature",      type=float, default=0.8)
    p.add_argument("--top_p",            type=float, default=0.95)

    # Judge
    p.add_argument("--judge_model",   default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--load_in_4bit",  action="store_true")

    p.add_argument("--device", default=None)
    p.add_argument("--dtype",  default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed",   type=int, default=0)
    return p.parse_args()


def get_device(req: str | None) -> str:
    if req:
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lm(ckpt_path: str, args, device, dtype) -> TransformerLM:
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        tie_weights=True,
        device=device,
        dtype=dtype,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def eval_ppl(model, val_data, args, device) -> tuple[float, float]:
    total = 0.0
    for _ in range(args.ppl_batches):
        x, y = get_batch(val_data, args.ppl_batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        total += loss.item()
    avg = total / args.ppl_batches
    return avg, math.exp(avg)


def load_held_out_prompts(n: int, seed: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e
    ds = load_dataset("roneneldan/TinyStoriesInstruct", split="train")
    # Last 5% reserved as held-out (same convention as build_preference_dataset).
    start = int(len(ds) * 0.95)
    prompts: list[str] = []
    for ex in ds.select(range(start, len(ds))):
        text = ex.get("text", "") or ""
        if "\nStory:" not in text:
            continue
        head = text.split("\nStory:", 1)[0].strip()
        if head:
            prompts.append(head)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


@torch.no_grad()
def sample_completion(model, tokenizer, eot_id, prompt_text: str, args, device) -> str:
    formatted = f"{USER_TAG}{prompt_text}{EOT}{ASSISTANT_TAG}"
    ids = torch.tensor([tokenizer.encode(formatted)], device=device, dtype=torch.long)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eot_id,
    )
    new = out[0, ids.shape[1]:].tolist()
    if new and new[-1] == eot_id:
        new = new[:-1]
    return tokenizer.decode(new)


def load_judge(name: str, device: str, load_in_4bit: bool):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=device)
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("  bitsandbytes not installed — loading at bf16.")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return tok, model


@torch.no_grad()
def judge_one(tok, model, prompt: str, a: str, b: str) -> str | None:
    user_text = JUDGE_RUBRIC.format(prompt=prompt, a=a, b=b)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user_text},
    ]
    inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)
    out = model.generate(inputs, max_new_tokens=4, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    answer = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True).strip().upper()
    for ch in answer:
        if ch in ("A", "B"):
            return ch
    return None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = get_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    tokenizer = Tokenizer.from_files(
        args.vocab, args.merges,
        special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"],
    )
    eot_id = tokenizer.encode(EOT)[0]

    val_data = np.load(args.val_data, mmap_mode="r")

    names = ["base", "sft", "dpo"]
    ckpts = {
        "base": args.base_checkpoint,
        "sft":  args.sft_checkpoint,
        "dpo":  args.dpo_checkpoint,
    }

    # ============ 1. PERPLEXITY ============
    print("\n=== 1. Held-out perplexity ===")
    ppl_results = {}
    for name in names:
        print(f"Loading {name} ...")
        model = load_lm(ckpts[name], args, device, dtype)
        loss, ppl = eval_ppl(model, val_data, args, device)
        ppl_results[name] = (loss, ppl)
        print(f"  {name}: loss {loss:.4f}  ppl {ppl:.2f}")
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # ============ 2. SAMPLE COMPLETIONS ============
    print(f"\n=== 2. Sampling {args.num_eval_prompts} completions per model ===")
    prompts = load_held_out_prompts(args.num_eval_prompts, args.seed)
    print(f"  Held-out prompts loaded: {len(prompts)}")

    completions: dict[str, list[str]] = {n: [] for n in names}
    for name in names:
        print(f"  Generating from {name} ...")
        model = load_lm(ckpts[name], args, device, dtype)
        t0 = time.time()
        for i, p in enumerate(prompts):
            completions[name].append(sample_completion(model, tokenizer, eot_id, p, args, device))
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(prompts)}  ({(time.time() - t0)/60:.1f} min)")
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # ============ 3. JUDGE WIN-RATE ============
    print(f"\n=== 3. Judging pairs with {args.judge_model} ===")
    tok, judge_model = load_judge(args.judge_model, device, args.load_in_4bit)

    # win_counts[(x, y)] = number of times x beat y (consistent across orderings).
    pairs = list(itertools.combinations(names, 2))
    win_counts = {(x, y): 0 for x, y in pairs}
    win_counts.update({(y, x): 0 for x, y in pairs})
    flipped = {(x, y): 0 for x, y in pairs}
    invalid = {(x, y): 0 for x, y in pairs}

    t0 = time.time()
    for i, prompt in enumerate(prompts):
        for x, y in pairs:
            v1 = judge_one(tok, judge_model, prompt, completions[x][i], completions[y][i])
            v2 = judge_one(tok, judge_model, prompt, completions[y][i], completions[x][i])
            if v1 is None or v2 is None:
                invalid[(x, y)] += 1
                continue
            if v1 == "A" and v2 == "B":
                win_counts[(x, y)] += 1
            elif v1 == "B" and v2 == "A":
                win_counts[(y, x)] += 1
            else:
                flipped[(x, y)] += 1
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate
            print(f"  {i+1}/{len(prompts)}  ({rate:.2f} pr/s, ETA {eta/60:.1f} min)")

    # Compute win-rate matrix and swap-consistency.
    matrix = {n1: {n2: None for n2 in names} for n1 in names}
    swap_consistency: dict[tuple, float] = {}
    for x, y in pairs:
        total = win_counts[(x, y)] + win_counts[(y, x)] + flipped[(x, y)]
        if total == 0:
            continue
        wr_x = win_counts[(x, y)] / total
        wr_y = win_counts[(y, x)] / total
        matrix[x][y] = wr_x
        matrix[y][x] = wr_y
        swap_consistency[(x, y)] = (win_counts[(x, y)] + win_counts[(y, x)]) / total

    # ============ 4. WRITE RESULTS.md ============
    out_path = Path(args.output)
    lines: list[str] = []
    lines.append("# Post-Training Eval Results\n")
    lines.append("## Held-out perplexity\n")
    lines.append("| Checkpoint | Loss (nats) | Perplexity |")
    lines.append("|---|---|---|")
    for n in names:
        loss, ppl = ppl_results[n]
        lines.append(f"| {n} | {loss:.4f} | {ppl:.2f} |")

    lines.append("\n## Pairwise judge win-rate matrix\n")
    lines.append(f"Judge: `{args.judge_model}`  ·  "
                 f"Prompts: {args.num_eval_prompts}  ·  Both orders run.\n")
    lines.append("Each cell = P(row beats column), consistent (non-flipped) judgements only.\n")
    lines.append("| ↓ vs → | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    for n1 in names:
        row = [n1]
        for n2 in names:
            if n1 == n2:
                row.append("—")
            elif matrix[n1][n2] is None:
                row.append("n/a")
            else:
                row.append(f"{matrix[n1][n2]:.1%}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Swap-consistency (position-bias control)\n")
    lines.append("| Pair | Consistent | Flipped | Consistency rate |")
    lines.append("|---|---|---|---|")
    for x, y in pairs:
        cons = win_counts[(x, y)] + win_counts[(y, x)]
        flip = flipped[(x, y)]
        total = cons + flip
        if total == 0:
            continue
        rate = cons / total
        flag = "" if rate >= 0.70 else " ⚠️ (low signal)"
        lines.append(f"| {x} vs {y} | {cons} | {flip} | {rate:.1%}{flag} |")

    lines.append("\n## Headline\n")
    sft_vs_base = matrix.get("sft", {}).get("base")
    dpo_vs_sft  = matrix.get("dpo", {}).get("sft")
    dpo_vs_base = matrix.get("dpo", {}).get("base")
    if sft_vs_base is not None:
        lines.append(f"- SFT beats base in **{sft_vs_base:.1%}** of pairs.")
    if dpo_vs_sft is not None:
        lines.append(f"- DPO beats SFT in **{dpo_vs_sft:.1%}** of pairs.")
    if dpo_vs_base is not None:
        lines.append(f"- DPO beats base in **{dpo_vs_base:.1%}** of pairs.")

    lines.append("\n## Sample completions (first 3 prompts)\n")
    for i in range(min(3, len(prompts))):
        lines.append(f"### Prompt {i+1}\n")
        lines.append(f"> {prompts[i]}\n")
        for n in names:
            lines.append(f"**{n}:**\n")
            lines.append(f"> {completions[n][i]}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    # Also save raw data for post-hoc analysis.
    raw_path = out_path.with_suffix(".raw.json")
    raw_path.write_text(json.dumps({
        "ppl": {n: {"loss": l, "ppl": p} for n, (l, p) in ppl_results.items()},
        "completions": completions,
        "win_counts":  {f"{x}>{y}": v for (x, y), v in win_counts.items()},
        "flipped":     {f"{x}vs{y}": v for (x, y), v in flipped.items()},
        "invalid":     {f"{x}vs{y}": v for (x, y), v in invalid.items()},
        "prompts": prompts,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {out_path} and {raw_path}")


if __name__ == "__main__":
    main()
