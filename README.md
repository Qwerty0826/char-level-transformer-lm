# Transformer Language Model — From Scratch

A decoder-only Transformer language model implemented from scratch in PyTorch, following the Stanford CS336 (Spring 2025) curriculum on building LLMs. Every component — tokenizer, attention, normalisation, optimizer, training loop, sampling — is written by hand without `torch.nn.functional`, `nn.Linear`, or `nn.Embedding`.

The architecture follows modern LLM design choices used in Llama, Mistral, and GPT-NeoX:

- **Byte-level BPE** tokenizer with GPT-2 regex pre-tokenization and incremental pair-count updates
- **Pre-norm** Transformer with **RMSNorm**
- **Rotary Position Embeddings (RoPE)**
- **SwiGLU** feed-forward network
- **Grouped Query Attention (GQA)** support — reduces KV-cache by 4–8× vs MHA
- **KV-cached decoding** — O(T) generation instead of O(T²), ≥10× faster
- **AdamW** with decoupled weight decay, cosine LR schedule with warmup, gradient clipping
- **Mixed precision** (bfloat16) and `torch.compile` support
- **MFU tracking** (Model FLOPs Utilisation) during training
- **Multiple sampling strategies**: temperature, top-p, top-k, min-p, repetition penalty
- **OpenAI-compatible REST API** with SSE streaming (drop-in for Open WebUI, SillyTavern, Jan, LangChain)
- **Gradio playground** with side-by-side sampling comparison, live token streaming, model card

55 unit tests, all passing, covering primitives, attention equivalence under KV caching, sampling, API endpoints, and end-to-end overfit on a single batch.

---

## Four ways to interact with the model

| Interface | Command | Use case |
|---|---|---|
| **Web playground** (Gradio) | `python scripts/playground.py …` | Visual demo with sampling sliders + side-by-side compare |
| **OpenAI-compatible API** (FastAPI) | `python scripts/serve.py … --port 8000` | Drop-in for Open WebUI / SillyTavern / Jan / openai-python / LangChain |
| **Terminal REPL** | `python scripts/chat.py …` | Quick interactive testing, `/temp 0.5`, `/top_p 0.9`, etc. |
| **One-shot CLI** | `python scripts/generate.py …` | Scripts, automation, batch generation |

---

## Results (TinyStories, 5K-step run on Tesla T4)

Trained the default 17M-param configuration on TinyStories for 5,000 steps with `batch_size=32`, fp32, `lr_max=1e-3`, cosine schedule, 200 warmup steps.

| Metric | Value |
|---|---|
| Final training loss | **1.64** |
| **Validation loss** | **2.26** |
| **Validation perplexity** | **9.59** |
| Throughput | 22,861 tokens/sec |
| MFU (fp32, vs T4 fp32 peak 8.1 TFLOPS) | **~28%** |
| Wall-clock training time | ~30 min on T4 |
| Peak GPU memory | 3.74 GB |

### Sample outputs

Prompt: *"Once upon a time there was a little girl who"* (T=0.8, top-p=0.95, top-k=50):

> Once upon a time there was a little girl who loved to play. One day, she decided to invite her friends to a party. They played games and had fun. The party was a happy day.
>
> At the party, the little girl met a big dog. The dog was sad because he lost his toy. The dog gave the girl a big hug. The girl said, "I am sorry, dog. Please don't be angry." The dog felt sad and said, "I am sorry, dog."
>
> The girl went to the dog and gave him the toy. The dog was happy and let the dog play together. The girl and the toy became friends and they played together. The moral of the story is to be kind and share with others.

> Once upon a time there was a little girl who liked to watch the things. One day, the girl met a boy named Tom who was playing with her. The sky was full of pretty colors.
>
> The little girl said to the bird, "I like to watch the sunset too!" The bird said, "Yes, I like to whistle. It has a nest that can feel so special." Tom liked the idea. They talked and laughed as they drank their water.

The model demonstrates fluent narrative structure — characters, dialogue, simple plot arcs, and the "moral of the story" pattern characteristic of TinyStories.

### Why KV cache "lost" on this model

The benchmark on the trained 17M model shows the KV-cached generation path is roughly tied with full-recompute (≈0.9× speedup at 64-token generation). This is an instructive finding rather than a bug:

- A 17M-param model is **bandwidth-bound** on a T4 — the time is dominated by loading the weight matrices from HBM, not by the matmul itself.
- Per-step kernel launch overhead (~10 μs × dozens of kernels per layer × 64 tokens) is comparable to the actual compute.
- The full-recompute path does ~30× more FLOPs but amortises weight loads across many tokens per launch.

KV cache wins decisively when the model is large enough to be compute-bound (≥1B params) or the context is long enough that O(T²) attention starts to dominate. This is the same reason production LLM serving stacks (vLLM, TGI) use fused kernels rather than naive Python loops over a cache.

The cache implementation is verified mathematically equivalent to the non-cached path (`test_kv_cache_matches_full_forward`, error < 3 × 10⁻⁶). It's the right architectural choice; the speedup is just hardware-dependent.

---

## Architecture at a glance

```
                  token_ids (B, T)
                       │
                       ▼
              ┌────────────────┐
              │   Embedding    │  (V × d_model)
              └────────┬───────┘
                       │
        ┌──────────────┴───────────────┐
        │                              │
        │     Transformer Block × N    │
        │  ┌─────────────────────┐     │
        │  │      RMSNorm        │     │
        │  │         │           │     │
        │  │    GQA + RoPE       │     │
        │  │         │           │     │
        │  │     ⊕ residual      │     │
        │  │         │           │     │
        │  │      RMSNorm        │     │
        │  │         │           │     │
        │  │   SwiGLU FFN        │     │
        │  │         │           │     │
        │  │     ⊕ residual      │     │
        │  └─────────┬───────────┘     │
        └────────────┼─────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   RMSNorm    │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   LM head    │  (d_model × V, tied to embedding)
              └──────┬───────┘
                     │
                     ▼
              logits (B, T, V)
```

| Component       | Choice                                                  |
|-----------------|---------------------------------------------------------|
| Tokenizer       | Byte-level BPE, GPT-2 regex, multiprocessing            |
| Position enc.   | RoPE (Su et al. 2021)                                   |
| Normalisation   | RMSNorm (Zhang & Sennrich 2019), pre-norm placement    |
| Attention       | Causal MHA, optional GQA, optional chunked-memory      |
| Feed-forward    | SwiGLU (Shazeer 2020), `d_ff = round_64(8/3 · d_model)`|
| Optimiser       | AdamW (Loshchilov & Hutter 2019), decoupled WD          |
| LR schedule     | Cosine annealing + linear warmup (LLaMA style)         |
| Weight tying    | Input embedding ↔ output LM head (optional)            |
| Decoding        | KV cache + temperature/top-p/top-k/min-p/rep-penalty   |

**Default TinyStories model**: 4 layers, 16 heads, d_model=512, d_ff=1344 → **17M parameters** (12M non-embedding).

---

## Repository layout

```
cs336_basics/                 Importable Python package
├── tokenizer.py              Byte-level BPE: train_bpe(), Tokenizer
├── nn_components.py          Linear, Embedding, RMSNorm (from scratch)
├── attention.py              Softmax, RoPE, SDPA, chunked attention,
│                             CausalMultiHeadSelfAttention with GQA + KV cache
├── model.py                  SwiGLU FFN, TransformerBlock, TransformerLM
│                             (training, forward_with_cache, generate)
├── optimizer.py              AdamW from scratch
└── training.py               Cross-entropy, LR schedule, gradient clipping,
                              memory-mapped data loader, checkpoint save/load

scripts/
├── train_tokenizer.py        Train BPE + encode a corpus to .npy
├── split_data.py             90/10 train/val split for tokenised .npy files
├── train.py                  Full training loop (CLI / YAML config, MFU,
│                             W&B + CSV, gradient accumulation, resume)
├── generate.py               Text generation (KV-cached, all samplers)
├── chat.py                   Interactive REPL with slash-commands
├── playground.py             Gradio web UI: sampling sliders, side-by-side
│                             compare, live streaming, model card
├── serve.py                  OpenAI-compatible FastAPI server with SSE
│                             streaming (/v1/chat/completions, /v1/completions,
│                             /v1/models, /health) — Open WebUI / SillyTavern / Jan
├── evaluate.py               Perplexity / BPC / sample generation
├── lr_find.py                Learning-rate range test (Smith 2015)
├── benchmark.py              Throughput / memory / MFU benchmarks
└── run_ablations.py          Section 7.3 ablations (RMSNorm, post-norm,
                              NoPE, SwiGLU vs SiLU) — produces Markdown report

configs/
├── tinystories.yaml          Apple Silicon MPS / low-resource
└── owt.yaml                  OpenWebText for CUDA

tests/                        55 unit tests (pytest), all passing
└── ...                       KV-cache equivalence, GQA, chunked attention,
                              streaming generator, OpenAI API endpoints
                              (chat/completion, streaming, stop sequences),
                              UTF-8 streaming decoder, end-to-end overfit

.github/workflows/tests.yml   CI: tests on Python 3.10 / 3.11 / 3.12
```

---

## Quick start

### 1. Install

```bash
pip install -e .
# Optional for W&B logging:
pip install wandb
```

### 2. Train the BPE tokenizer and encode a corpus

```bash
python scripts/train_tokenizer.py \
    --input data/tinystories.txt \
    --vocab_size 10000 \
    --output_dir data/ --prefix tinystories \
    --encode data/tinystories.txt \
    --encode_out data/tinystories_tokens.npy

python scripts/split_data.py \
    --input data/tinystories_tokens.npy --val_fraction 0.1 \
    --train_out data/tinystories_tokens_train.npy \
    --val_out   data/tinystories_tokens_val.npy
```

### 3. (Optional) Find a good learning rate

```bash
python scripts/lr_find.py --config configs/tinystories.yaml --num_iters 80
```

### 4. Train

```bash
# Apple Silicon MPS
python scripts/train.py --config configs/tinystories.yaml

# CUDA / Colab T4 (faster: bigger batch, bfloat16, inductor)
python scripts/train.py --config configs/tinystories.yaml \
    --dtype bfloat16 --batch_size 64 --compile_backend inductor --wandb
```

The training loop reports loss, learning rate, gradient norm, tokens/sec, and **MFU** (model FLOPs utilisation) at every `log_interval`.

### 5. Generate text

```bash
python scripts/generate.py \
    --checkpoint checkpoints/tinystories/final.pt \
    --vocab data/tinystories_vocab.json --merges data/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --max_tokens 256 --temperature 0.8 --top_p 0.95 --top_k 50
```

### 6. Chat interactively

```bash
python scripts/chat.py \
    --checkpoint checkpoints/tinystories/final.pt \
    --vocab data/tinystories_vocab.json --merges data/tinystories_merges.txt
```

```
> Once upon a time there was a
[model continues...]
> /temp 0.5
> /top_p 0.9
> Tell me about the bunny.
[model continues with new settings]
```

### 7. Run ablations (PDF §7.3)

```bash
python scripts/run_ablations.py \
    --config configs/tinystories.yaml --steps 1500 \
    --out results/ablations.md
```

This trains the baseline plus four ablations (no-norm, post-norm, no-RoPE, no-gate) and writes a comparison table to Markdown.

### 8. Benchmark

```bash
python scripts/benchmark.py --config configs/tinystories.yaml --iters 30
```

Reports training tokens/sec, step time, peak GPU memory, and KV-cache speedup at inference (typically ≥10×).

### 9. Serve as an OpenAI-compatible API

```bash
pip install -e ".[serve]"

python scripts/serve.py \
    --checkpoint checkpoints/tinystories/final.pt \
    --vocab data/tinystories_vocab.json \
    --merges data/tinystories_merges.txt \
    --port 8000
```

Exposes:

- `POST /v1/chat/completions` (chat messages, multi-turn)
- `POST /v1/completions`      (legacy text completion)
- `GET  /v1/models`           (model discovery)
- `GET  /health`

Both POST endpoints support `stream=true` for token-by-token Server-Sent Events.

**Drop-in compatibility:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="transformer-lm",
    messages=[{"role": "user", "content": "Once upon a time"}],
    max_tokens=100, temperature=0.8, top_p=0.95, stream=True,
)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Works the same way with **Open WebUI** (point its OpenAI provider at `http://localhost:8000/v1`), **SillyTavern**, **Jan**, or any LangChain `ChatOpenAI` client.

### 10. Serve as a Gradio web playground

```bash
pip install -e ".[ui]"

python scripts/playground.py \
    --checkpoint checkpoints/tinystories/final.pt \
    --vocab data/tinystories_vocab.json \
    --merges data/tinystories_merges.txt \
    --port 7860 \
    --share        # optional: public ngrok-style tunnel
```

Three tabs:
- **Generate** — prompt box + sliders for all five samplers + live token streaming + tok/s metric
- **Compare** — same prompt, two sampling configs side-by-side (visualises what each knob does)
- **Model card** — parameters, layers, FLOPs, GQA ratio, checkpoint step

Deploys cleanly to **HuggingFace Spaces** — drop the script into a Space and you have a public live demo.

---

## Key design decisions

### KV cache for incremental decoding

Vanilla generation re-runs the full forward pass over the entire context every step (O(T²) total work to generate T tokens). With a KV cache, the keys and values from each layer are saved across steps; each new step only computes attention for the **new** query against all cached keys/values — O(T) total.

`TransformerLM.generate()` uses this automatically; `model.forward_with_cache()` is the lower-level API. The included test `test_kv_cache_matches_full_forward` proves the cached path produces logits identical (within fp tolerance) to the non-cached path.

### Grouped Query Attention (GQA)

Llama-2/3, Mistral, and Gemma all use GQA: fewer K/V heads than Q heads (the K/V projections are smaller, and at inference time several Q heads share the same K/V). This reduces the KV cache by `num_q_heads / num_kv_heads` — the dominant memory cost at long contexts.

Set `num_kv_heads=num_heads` (default) for vanilla MHA; `num_kv_heads=1` for multi-query attention; anything in between for GQA.

### Memory-efficient (chunked) attention

`scaled_dot_product_attention` materialises an `(T, T)` score matrix. For long contexts this becomes the dominant memory cost. The `chunked_causal_attention` helper processes Q in chunks of `chunk_size` rows, never holding more than `(chunk, T)` scores at once — same arithmetic, lower peak memory.

Pass `chunk_size=...` to `TransformerLM` to enable it automatically for sequences longer than the chunk size.

### MFU (Model FLOPs Utilisation)

MFU is the ratio of FLOPs the model actually crunched per second to the hardware's theoretical peak. It's the metric LLM training teams care about: 20–50% on a tuned A100 setup, 10–25% on consumer GPUs is typical.

`scripts/train.py` computes per-step FLOPs from `model.estimate_flops_per_token()`, divides by wall-clock time, and divides by a hard-coded peak from a table of common GPUs. Logged to console and W&B as `train/mfu`.

### Byte-level BPE tokenizer

Trains on raw UTF-8 bytes — every byte sequence is representable, no OOV tokens. The GPT-2 regex (`'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`) prevents merges from spanning word/punctuation boundaries.

Pair counts are updated **incrementally** after each merge — only the pairs adjacent to the merged tokens are recomputed, not the whole corpus. On a 22 MB TinyStories file: ~14 seconds to train a 10K vocabulary.

### RMSNorm with fp32 upcast

The activation is upcast to float32 for the variance computation (the dominant source of numerical instability in low-precision training), then cast back to the input dtype. This is the same pattern used by Meta's official Llama code.

### Pre-norm placement

`x = x + sublayer(norm(x))` instead of `x = norm(x + sublayer(x))`. Keeps the residual path "clean" (no normalisation between residual additions), improves gradient flow, and is what every modern LLM uses. Ablation switch available (`--post_norm`).

### Weight tying

The token embedding matrix `(V × d)` is shared with the LM head projection. Saves `V × d` parameters (5M on a 10K-vocab d=512 model) and typically improves perplexity slightly on small models.

---

## Sampling strategies

`model.generate()` and `scripts/generate.py` support combining several samplers:

- **Temperature** — `logits / T`. T<1 sharpens, T>1 flattens.
- **Top-k** — keep only the k highest-prob tokens, renormalise.
- **Top-p (nucleus)** — keep the smallest set whose cumulative prob ≥ p.
- **Min-p** — keep tokens with prob ≥ `min_p × max_prob` (Nguyen 2023). Adapts the kept set to the distribution's sharpness automatically.
- **Repetition penalty** — divide logits of already-generated tokens by a factor >1 to discourage loops (Keskar et al. 2019).

You can combine them; they're applied in this order: rep-penalty → temperature → top-k → softmax → min-p → top-p → sample.

---

## Ablation studies (PDF §7.3)

`scripts/run_ablations.py` runs five short training runs back-to-back and produces a Markdown comparison table:

| Ablation   | What changes                                  | Expected effect             |
|------------|-----------------------------------------------|-----------------------------|
| baseline   | Full model                                    | best                        |
| no_norm    | Identity in place of RMSNorm                  | training unstable / worse   |
| post_norm  | Norm placed after residual add                | slightly worse, less stable |
| no_rope    | No positional encoding (NoPE)                 | catastrophic                |
| no_gate    | SiLU FFN without W3 gate                      | slightly worse              |

Run with `--steps 1500` for quick signal or `--steps 5000` for the full picture.

---

## Tests

```bash
pytest tests/ -v
# 55 passed in ~2 s
```

The advanced test suite includes:

- **`test_kv_cache_matches_full_forward`** — single-shot vs cache: identical logits
- **`test_kv_cache_token_by_token`** — fully incremental decoding still matches
- **`test_generate_stream_matches_generate`** — streaming generator yields the same tokens as `.generate()`
- **`test_gqa_shape_and_kv_cache_size`** — GQA produces smaller K/V cache
- **`test_chunked_attention_matches_full`** — memory-efficient = math-identical
- **`test_streaming_decoder_handles_partial_utf8`** — multi-byte UTF-8 is buffered correctly
- **`test_chat_streaming_sse_format`** — API streams in OpenAI SSE format with `[DONE]` terminator
- **`test_stop_sequence_honored`** — `stop=["fox"]` truncates output before the trigger
- **`test_lm_overfit_single_batch`** — full model learns a single batch
- **`test_sdpa_causal_mask`** — future tokens can't leak into past

---

## Hardware notes

| Setup                | Throughput      | TinyStories 5K steps |
|----------------------|-----------------|----------------------|
| Apple M1 MPS         | ~4,000 tok/s    | 2.5–3 h              |
| Colab T4 (bfloat16)  | ~20,000 tok/s   | 30–50 min            |
| Colab A100 (bfloat16)| ~80,000 tok/s   | 10–15 min            |

On Apple Silicon MPS:
- Use `compile_backend: aot_eager` (Inductor has broken MPS kernels).
- Use `dtype: float32` (bfloat16 has broken kernels too).
- Do **not** set `torch.set_float32_matmul_precision('high')` on MPS.

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.*
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
- Zhang & Sennrich (2019). *Root Mean Square Layer Normalization.*
- Shazeer (2020). *GLU Variants Improve Transformer.*
- Ainslie et al. (2023). *GQA: Training Generalised Multi-Query Transformer Models.*
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization.*
- Smith (2015). *Cyclical Learning Rates for Training Neural Networks.*
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.*
- Holtzman et al. (2019). *The Curious Case of Neural Text Degeneration* (top-p).
- Keskar et al. (2019). *CTRL: A Conditional Transformer LM* (repetition penalty).
- Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units* (BPE).
- Eldan & Li (2023). *TinyStories: How Small Can Language Models Be?*
