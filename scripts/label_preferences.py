#!/usr/bin/env python3
"""
Label candidate preference pairs with a local open-weight LLM judge.

For each (prompt, completion_a, completion_b) triple:
  * Run the judge in BOTH orders (A vs B, B vs A) — this is the standard
    position-bias control. A judge that gives different verdicts depending
    on order is providing noise, not signal, so we drop those pairs.
  * If both orderings agree on a winner, emit a labeled pair with
    ``chosen = winner_text`` and ``rejected = loser_text``.

Default judge: ``Qwen/Qwen2.5-3B-Instruct`` (~3GB in 4-bit, fits on a T4
during the same session as the eval). The 7B variant gives slightly higher
agreement but isn't necessary for story-quality preferences at this scale.

Example:
    python scripts/label_preferences.py \\
        --input  data/pref_candidates.jsonl \\
        --output data/pref_labeled.jsonl    \\
        --judge_model Qwen/Qwen2.5-3B-Instruct --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


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
    p = argparse.ArgumentParser(description="Label preference pairs with a local LLM judge")
    p.add_argument("--input",  required=True, help="Candidate pairs JSONL (from build_preference_dataset.py)")
    p.add_argument("--output", required=True, help="Labeled pairs JSONL")
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load the judge in 4-bit via bitsandbytes (saves VRAM).")
    p.add_argument("--max_pairs", type=int, default=10_000,
                   help="Cap on number of input pairs to process.")
    p.add_argument("--max_judge_tokens", type=int, default=4,
                   help="Tokens to generate for the A/B answer (usually 1).")
    return p.parse_args()


def get_device(req: str | None) -> str:
    if req:
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_judge(name: str, device: str, load_in_4bit: bool):
    """Load Qwen-style instruction-tuned judge. Returns (tokenizer, model)."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise RuntimeError("pip install transformers") from e

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
            print("  Warning: bitsandbytes not installed; loading at bf16.")
    print(f"Loading judge: {name} (4bit={load_in_4bit})")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return tok, model


@torch.no_grad()
def judge_pair(tok, model, prompt: str, a: str, b: str, max_new_tokens: int) -> str | None:
    """Return 'A' or 'B' or None if the judge produced neither."""
    user_text = JUDGE_RUBRIC.format(prompt=prompt, a=a, b=b)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user_text},
    ]
    inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)
    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    answer = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True).strip().upper()
    # The judge sometimes adds extra punctuation/words — look for the first A or B.
    for ch in answer:
        if ch in ("A", "B"):
            return ch
    return None


def main():
    args = parse_args()
    device = get_device(args.device)
    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    with in_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    candidates = candidates[:args.max_pairs]
    print(f"Loaded {len(candidates):,} candidate pairs from {in_path}")

    tok, model = load_judge(args.judge_model, device, args.load_in_4bit)

    n_consistent = 0
    n_flipped = 0
    n_invalid = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        for i, c in enumerate(candidates):
            v1 = judge_pair(tok, model, c["prompt"], c["completion_a"], c["completion_b"],
                            args.max_judge_tokens)
            v2 = judge_pair(tok, model, c["prompt"], c["completion_b"], c["completion_a"],
                            args.max_judge_tokens)
            if v1 is None or v2 is None:
                n_invalid += 1
                continue

            # In the swapped order, the judge should now prefer the OTHER letter.
            # v1='A' AND v2='B' → judge consistently prefers the original 'completion_a'.
            # v1='B' AND v2='A' → judge consistently prefers the original 'completion_b'.
            # Anything else → judge flipped (position bias) → drop.
            if v1 == "A" and v2 == "B":
                chosen, rejected = c["completion_a"], c["completion_b"]
                n_consistent += 1
            elif v1 == "B" and v2 == "A":
                chosen, rejected = c["completion_b"], c["completion_a"]
                n_consistent += 1
            else:
                n_flipped += 1
                continue

            fh.write(json.dumps({
                "prompt":   c["prompt"],
                "chosen":   chosen,
                "rejected": rejected,
            }) + "\n")

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(candidates) - i - 1) / rate
                print(f"  {i+1}/{len(candidates)}  consistent {n_consistent}  "
                      f"flipped {n_flipped}  invalid {n_invalid}  "
                      f"({rate:.2f} pr/s, ETA {eta/60:.1f} min)")

    elapsed = time.time() - t0
    swap_consistency = n_consistent / max(1, n_consistent + n_flipped)
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Consistent pairs: {n_consistent} ({swap_consistency:.1%})  -> {out_path}")
    print(f"  Flipped (dropped):  {n_flipped}")
    print(f"  Invalid responses:  {n_invalid}")
    if swap_consistency < 0.70:
        print("  WARNING: judge swap-consistency < 70%. Preferences may be unreliable.")


if __name__ == "__main__":
    main()
