#!/usr/bin/env python3
"""
Interactive Gradio playground for a trained Transformer LM.

Up to four tabs:

  1. Generate         single prompt + all five samplers as sliders +
                      live token-by-token streaming + tokens/sec metric.
  2. Compare          two side-by-side configs to visualise how
                      different sampling strategies change the output.
  3. Aligned          (shown when --checkpoint_sft / --checkpoint_dpo
                      are passed) same prompt fed to base, SFT, and DPO
                      side by side — visualises what post-training did.
  4. Model card       architecture stats (parameters, layers, FLOPs,
                      checkpoint step, KV-cache info).

The generation streams tokens as they are sampled so the UI feels
responsive even for long outputs.

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
import time
from typing import Callable, Generator, Optional, Tuple

import gradio as gr
import torch

from cs336_basics.data_sft import ASSISTANT_TAG, EOT, USER_TAG
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from scripts.serve import StreamingDecoder       # re-use the API server's decoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_device(req: Optional[str]) -> str:
    if req:
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(args: argparse.Namespace, device: str, dtype: torch.dtype) -> TransformerLM:
    return TransformerLM(
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
        dtype=dtype,
    )


def _load_into(model: TransformerLM, checkpoint_path: str, device: str) -> int:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return int(ckpt.get("iteration", 0))


def load_models(args: argparse.Namespace) -> Tuple[dict, Tokenizer, str, Optional[int]]:
    """Load base + optional SFT/DPO checkpoints sharing one tokenizer / arch.

    Returns ``(models, tokenizer, device, eos_id)`` where ``models`` is a
    dict keyed by ``"base"`` / ``"sft"`` / ``"dpo"`` with each value
    ``{"model": TransformerLM, "step": int, "path": str}``.
    """
    device = _resolve_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    eos_id = tokenizer._special_to_id.get("<|endoftext|>")

    models: dict = {}
    for tag, path in [
        ("base", args.checkpoint),
        ("sft",  args.checkpoint_sft),
        ("dpo",  args.checkpoint_dpo),
    ]:
        if not path:
            continue
        model = _build_model(args, device, dtype)
        step = _load_into(model, path, device)
        models[tag] = {"model": model, "step": step, "path": path}
        print(f"[playground] Loaded {tag.upper()} from {path}  (step {step:,})")

    return models, tokenizer, device, eos_id


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def make_generator(model: TransformerLM, tokenizer: Tokenizer, device: str, eos_id: Optional[int]):
    """Closure that yields (full_text, metrics) updates for the UI."""

    def _sample_kwargs(temperature, top_p, top_k, min_p, rep_penalty):
        return dict(
            temperature=float(temperature),
            top_p=float(top_p) if top_p > 0 else None,
            top_k=int(top_k) if top_k > 0 else None,
            min_p=float(min_p) if min_p > 0 else None,
            repetition_penalty=float(rep_penalty),
            eos_id=eos_id,
        )

    def generate_streaming(
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        rep_penalty: float,
        max_tokens: int,
    ) -> Generator[Tuple[str, str], None, None]:
        if not prompt.strip():
            yield "", "Enter a prompt to begin."
            return

        ids = tokenizer.encode(prompt)
        prompt_t = torch.tensor([ids], dtype=torch.long, device=device)

        decoder = StreamingDecoder(tokenizer)
        new_text = ""
        n_tokens = 0
        t0 = time.time()

        for tok in model.generate_stream(
            prompt_t, int(max_tokens),
            **_sample_kwargs(temperature, top_p, top_k, min_p, rep_penalty),
        ):
            piece = decoder.feed(tok)
            if piece:
                new_text += piece
            n_tokens += 1
            elapsed = time.time() - t0
            tok_s = n_tokens / max(elapsed, 1e-6)
            metrics = (
                f"{n_tokens} tokens  |  {tok_s:5.1f} tok/s  |  "
                f"{elapsed:.1f}s elapsed"
            )
            yield prompt + new_text, metrics

        # Flush any UTF-8 tail bytes.
        tail = decoder.flush()
        if tail:
            new_text += tail
            elapsed = time.time() - t0
            tok_s = n_tokens / max(elapsed, 1e-6)
            yield (
                prompt + new_text,
                f"{n_tokens} tokens  |  {tok_s:5.1f} tok/s  |  {elapsed:.1f}s elapsed  |  done",
            )

    def generate_one_shot(
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        rep_penalty: float,
        max_tokens: int,
    ) -> Tuple[str, str]:
        """Used by the compare tab — returns the final output only."""
        last = ("", "")
        for last in generate_streaming(
            prompt, temperature, top_p, top_k, min_p, rep_penalty, max_tokens,
        ):
            pass
        return last

    return generate_streaming, generate_one_shot


def _strip_chat_artifacts(text: str, prompt_prefix: str) -> str:
    """Trim the wrapped chat prompt back off the model's continuation."""
    if text.startswith(prompt_prefix):
        text = text[len(prompt_prefix):]
    eot_at = text.find(EOT)
    if eot_at != -1:
        text = text[:eot_at]
    return text.strip()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui(
    models: dict,
    tokenizer: Tokenizer,
    device: str,
    eos_id: Optional[int],
) -> gr.Blocks:
    base = models["base"]["model"]
    base_step = models["base"]["step"]
    stream_fn, oneshot_fn = make_generator(base, tokenizer, device, eos_id)
    # Per-checkpoint one-shot generators for the Aligned tab.
    oneshot_by_tag: dict[str, Callable] = {
        tag: make_generator(spec["model"], tokenizer, device, eos_id)[1]
        for tag, spec in models.items()
    }
    extra_tags = [t for t in ("sft", "dpo") if t in models]

    flops_per_seq = base.estimate_flops_per_token()
    blocks = base.blocks
    num_q_heads = blocks[0].attn.num_q_heads
    num_kv_heads = blocks[0].attn.num_kv_heads
    d_k = blocks[0].attn.d_k
    d_ff = blocks[0].ff.d_ff

    loaded_md = "  ·  ".join(
        f"**{tag.upper()}** @ step {spec['step']:,}"
        for tag, spec in models.items()
    )

    model_card_md = f"""
### Architecture (shared by every loaded checkpoint)

| Property | Value |
|---|---|
| Total parameters | **{base.num_parameters():,}** |
| Non-embedding parameters | {base.num_parameters(non_embedding=True):,} |
| Vocab size | {base.vocab_size:,} |
| Context length | {base.context_length} |
| Layers | {len(blocks)} |
| Hidden dim (`d_model`) | {base.d_model} |
| FFN inner dim (`d_ff`) | {d_ff} |
| Query heads | {num_q_heads} |
| KV heads (GQA) | {num_kv_heads}{'  *(GQA enabled)*' if num_kv_heads < num_q_heads else ''} |
| Head dim | {d_k} |
| FLOPs / forward seq | {flops_per_seq:,} |
| Device | `{device}` |

### Loaded checkpoints

{chr(10).join(f"- **{tag.upper()}** — step {spec['step']:,}  ·  `{spec['path']}`" for tag, spec in models.items())}

### Sampling pipeline

`logits → repetition_penalty → temperature → top_k → softmax → min_p → top_p → multinomial`
"""

    theme = gr.themes.Soft() if hasattr(gr, "themes") else None
    with gr.Blocks(title="Transformer LM Playground", theme=theme) as app:
        gr.Markdown(
            "# Transformer LM Playground\n"
            f"Decoder-only Transformer (RoPE + SwiGLU + RMSNorm + GQA),  "
            f"**{base.num_parameters() / 1e6:.1f}M params**, "
            f"context **{base.context_length}**.  \n"
            f"Loaded: {loaded_md}."
        )

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Prompt", value="Once upon a time",
                        lines=3, max_lines=10, placeholder="Start typing a story...",
                    )
                    output = gr.Textbox(
                        label="Output (streamed)",
                        lines=18, max_lines=40,
                        interactive=False,
                    )
                    metrics = gr.Textbox(label="Metrics", lines=1, interactive=False)
                    with gr.Row():
                        gen_btn = gr.Button("Generate", variant="primary", scale=2)
                        stop_btn = gr.Button("Stop", scale=1)
                        clear_btn = gr.Button("Clear", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("#### Sampling controls")
                    temperature = gr.Slider(
                        0.1, 2.0, value=0.8, step=0.05, label="Temperature",
                        info="<1 sharpens, >1 flattens.",
                    )
                    top_p = gr.Slider(
                        0.0, 1.0, value=0.95, step=0.01, label="Top-p (nucleus)",
                        info="Keep smallest set whose prob ≥ this.",
                    )
                    top_k = gr.Slider(
                        0, 200, value=50, step=1, label="Top-k",
                        info="0 disables.",
                    )
                    min_p = gr.Slider(
                        0.0, 0.5, value=0.0, step=0.01, label="Min-p",
                        info="0 disables.  Keep tokens with prob ≥ min_p × max_prob.",
                    )
                    rep_penalty = gr.Slider(
                        1.0, 2.0, value=1.0, step=0.05, label="Repetition penalty",
                        info="1.0 = no penalty.",
                    )
                    max_tokens = gr.Slider(
                        16, 1024, value=200, step=16, label="Max new tokens",
                    )

            gen_event = gen_btn.click(
                fn=stream_fn,
                inputs=[prompt, temperature, top_p, top_k, min_p, rep_penalty, max_tokens],
                outputs=[output, metrics],
            )
            stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])
            clear_btn.click(
                fn=lambda: ("", ""), inputs=None, outputs=[output, metrics],
            )

            gr.Examples(
                examples=[
                    ["Once upon a time there was a little girl who", 0.8, 0.95, 50, 0.0, 1.0, 200],
                    ["The old wizard looked at the dragon and said,", 1.0, 0.9, 40, 0.0, 1.1, 200],
                    ["In a small village by the sea,",                0.6, 0.95, 50, 0.05, 1.0, 200],
                ],
                inputs=[prompt, temperature, top_p, top_k, min_p, rep_penalty, max_tokens],
            )

        with gr.Tab("Compare"):
            gr.Markdown(
                "Generate two completions from the same prompt with **different sampling settings** "
                "to see how each parameter changes the model's behaviour."
            )
            cmp_prompt = gr.Textbox(label="Prompt", value="Once upon a time", lines=3)
            cmp_max = gr.Slider(16, 512, value=150, step=16, label="Max new tokens")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Config A — conservative**")
                    t_a = gr.Slider(0.1, 2.0, value=0.5, step=0.05, label="Temperature A")
                    p_a = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top-p A")
                    k_a = gr.Slider(0, 200, value=40, step=1, label="Top-k A")
                    out_a = gr.Textbox(label="Output A", lines=14, interactive=False)
                    met_a = gr.Textbox(label="Metrics A", lines=1, interactive=False)
                with gr.Column():
                    gr.Markdown("**Config B — exploratory**")
                    t_b = gr.Slider(0.1, 2.0, value=1.1, step=0.05, label="Temperature B")
                    p_b = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p B")
                    k_b = gr.Slider(0, 200, value=80, step=1, label="Top-k B")
                    out_b = gr.Textbox(label="Output B", lines=14, interactive=False)
                    met_b = gr.Textbox(label="Metrics B", lines=1, interactive=False)
            cmp_btn = gr.Button("Generate both", variant="primary")

            def _compare(prompt, ta, pa, ka, tb, pb, kb, mx):
                ta_out = oneshot_fn(prompt, ta, pa, ka, 0.0, 1.0, mx)
                tb_out = oneshot_fn(prompt, tb, pb, kb, 0.0, 1.0, mx)
                return ta_out[0], ta_out[1], tb_out[0], tb_out[1]

            cmp_btn.click(
                _compare,
                inputs=[cmp_prompt, t_a, p_a, k_a, t_b, p_b, k_b, cmp_max],
                outputs=[out_a, met_a, out_b, met_b],
            )

        if extra_tags:
            with gr.Tab("Aligned (Base → SFT → DPO)"):
                gr.Markdown(
                    "Same prompt → every loaded checkpoint. The base model only "
                    "saw raw stories during pretraining; SFT taught it the chat "
                    "template; DPO sharpened it on preference pairs. Use a prompt "
                    "phrased as a *user instruction* to see the gap.\n\n"
                    "Base receives the raw instruction; SFT and DPO receive the "
                    "same instruction wrapped as `<|user|>…<|endoftext|><|assistant|>` "
                    "(the chat template they were trained on)."
                )
                al_prompt = gr.Textbox(
                    label="User instruction",
                    value="Write a short story about a girl named Lily who finds a magic stone in the forest.",
                    lines=3,
                )
                with gr.Row():
                    al_temp = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
                    al_top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
                    al_top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
                    al_max = gr.Slider(16, 512, value=220, step=16, label="Max new tokens")
                al_btn = gr.Button("Generate from every checkpoint", variant="primary")

                column_labels = {
                    "base": "Base (pretrain only)",
                    "sft":  "+ SFT (chat-template instruction tuning)",
                    "dpo":  "+ SFT + DPO (preference-aligned)",
                }
                output_boxes: dict[str, gr.Textbox] = {}
                metric_boxes: dict[str, gr.Textbox] = {}
                with gr.Row():
                    for tag in ("base", *extra_tags):
                        with gr.Column():
                            gr.Markdown(f"#### {column_labels[tag]}")
                            output_boxes[tag] = gr.Textbox(
                                label="", lines=16, interactive=False,
                                show_copy_button=True,
                            )
                            metric_boxes[tag] = gr.Textbox(
                                label="metrics", lines=1, interactive=False,
                            )

                ordered_tags = ["base", *extra_tags]

                def _aligned(prompt, t, p, k, mx):
                    chat_prompt = f"{USER_TAG}{prompt.strip()}{EOT}{ASSISTANT_TAG}"
                    outs: list[str] = []
                    for tag in ordered_tags:
                        raw_prompt = prompt if tag == "base" else chat_prompt
                        text, metric = oneshot_by_tag[tag](
                            raw_prompt, t, p, k, 0.0, 1.0, mx,
                        )
                        if tag != "base":
                            text = _strip_chat_artifacts(text, chat_prompt)
                        outs.extend([text, metric])
                    return tuple(outs)

                al_outputs: list = []
                for tag in ordered_tags:
                    al_outputs.append(output_boxes[tag])
                    al_outputs.append(metric_boxes[tag])

                al_btn.click(
                    _aligned,
                    inputs=[al_prompt, al_temp, al_top_p, al_top_k, al_max],
                    outputs=al_outputs,
                )

                gr.Examples(
                    examples=[
                        ["Write a short story about a girl named Lily who finds a magic stone in the forest.", 0.8, 0.95, 50, 220],
                        ["Write a story where a small dragon learns to share its treasure.", 0.7, 0.9, 50, 220],
                        ["Tell me a bedtime story about two friends who get lost and find their way home.", 0.8, 0.95, 50, 220],
                    ],
                    inputs=[al_prompt, al_temp, al_top_p, al_top_k, al_max],
                )

        with gr.Tab("Model card"):
            gr.Markdown(model_card_md)

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

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
