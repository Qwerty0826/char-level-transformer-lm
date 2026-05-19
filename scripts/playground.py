#!/usr/bin/env python3
"""
Interactive Gradio playground for a trained Transformer LM.

Three tabs:

  1. Generate         single prompt + all five samplers as sliders +
                      live token-by-token streaming + tokens/sec metric.
  2. Compare          two side-by-side configs to visualise how
                      different sampling strategies change the output.
  3. Model card       architecture stats (parameters, layers, FLOPs,
                      checkpoint step, KV-cache info).

The generation streams tokens as they are sampled so the UI feels
responsive even for long outputs.

Example:
    python scripts/playground.py \\
        --checkpoint checkpoints/tinystories/final.pt \\
        --vocab data/tinystories_vocab.json \\
        --merges data/tinystories_merges.txt \\
        --share        # exposes a public ngrok-style URL
"""

from __future__ import annotations

import argparse
import time
from typing import Generator, Optional, Tuple

import gradio as gr
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from scripts.serve import StreamingDecoder       # re-use the API server's decoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio playground for a trained LM")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab",      required=True)
    p.add_argument("--merges",     required=True)
    p.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])

    # Model shape (must match checkpoint)
    p.add_argument("--vocab_size",     type=int, default=10_000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model",        type=int, default=512)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--num_heads",      type=int, default=16)
    p.add_argument("--num_kv_heads",   type=int, default=None)
    p.add_argument("--d_ff",           type=int, default=1344)
    p.add_argument("--theta",          type=float, default=10_000.0)
    p.add_argument("--no_tie_weights", action="store_true")

    p.add_argument("--device", default=None)
    p.add_argument("--host",   default="127.0.0.1")
    p.add_argument("--port",   type=int, default=7860)
    p.add_argument("--share",  action="store_true",
                   help="Expose a temporary public URL via Gradio's tunnel")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args: argparse.Namespace) -> Tuple[TransformerLM, Tokenizer, str, Optional[int], int]:
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    eos_id = tokenizer._special_to_id.get("<|endoftext|>")

    model = TransformerLM(
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
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    return model, tokenizer, device, eos_id, int(ckpt.get("iteration", 0))


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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui(
    model: TransformerLM,
    tokenizer: Tokenizer,
    device: str,
    eos_id: Optional[int],
    ckpt_step: int,
) -> gr.Blocks:
    stream_fn, oneshot_fn = make_generator(model, tokenizer, device, eos_id)

    flops_per_seq = model.estimate_flops_per_token()
    blocks = model.blocks
    num_q_heads = blocks[0].attn.num_q_heads
    num_kv_heads = blocks[0].attn.num_kv_heads
    d_k = blocks[0].attn.d_k
    d_ff = blocks[0].ff.d_ff

    model_card_md = f"""
### Architecture

| Property | Value |
|---|---|
| Total parameters | **{model.num_parameters():,}** |
| Non-embedding parameters | {model.num_parameters(non_embedding=True):,} |
| Vocab size | {model.vocab_size:,} |
| Context length | {model.context_length} |
| Layers | {len(blocks)} |
| Hidden dim (`d_model`) | {model.d_model} |
| FFN inner dim (`d_ff`) | {d_ff} |
| Query heads | {num_q_heads} |
| KV heads (GQA) | {num_kv_heads}{'  *(GQA enabled)*' if num_kv_heads < num_q_heads else ''} |
| Head dim | {d_k} |
| FLOPs / forward seq | {flops_per_seq:,} |
| Checkpoint step | {ckpt_step:,} |
| Device | `{device}` |

### Sampling pipeline

`logits → repetition_penalty → temperature → top_k → softmax → min_p → top_p → multinomial`
"""

    with gr.Blocks(title="Transformer LM Playground") as app:
        gr.Markdown(
            "# Transformer LM Playground\n"
            f"Decoder-only Transformer (RoPE + SwiGLU + RMSNorm + GQA),  "
            f"**{model.num_parameters() / 1e6:.1f}M params**, "
            f"context **{model.context_length}**, checkpoint at step **{ckpt_step:,}**."
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

        with gr.Tab("Model card"):
            gr.Markdown(model_card_md)

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    model, tokenizer, device, eos_id, ckpt_step = load_model(args)
    print(f"[playground] Loaded {model.num_parameters():,}-param model on {device} "
          f"(step {ckpt_step:,})")
    app = build_ui(model, tokenizer, device, eos_id, ckpt_step)
    # In Gradio 6.x, ``theme`` is passed to launch() rather than Blocks().
    launch_kwargs = dict(
        server_name=args.host, server_port=args.port,
        share=args.share, show_error=True,
    )
    if hasattr(gr, "themes"):
        launch_kwargs["theme"] = gr.themes.Soft()
    app.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
