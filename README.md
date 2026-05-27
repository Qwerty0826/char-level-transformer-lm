# End-to-End From-Scratch Implementation of a Transformer LM

[![CI](https://github.com/Qwerty0826/char-level-transformer-lm/actions/workflows/tests.yml/badge.svg)](https://github.com/Qwerty0826/char-level-transformer-lm/actions/workflows/tests.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Spaces-FF9D00)](https://huggingface.co/spaces/Grux11/transformer-lm-60m)
[![Weights](https://img.shields.io/badge/Weights-Hub-FFD21E)](https://huggingface.co/Grux11/transformer-lm-60m-tinystories)
[![Python](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A 60M-parameter decoder-only Transformer built end to end in PyTorch.
Pretraining, supervised fine-tuning, and Direct Preference Optimization
are all written using only `torch.nn.Parameter`. No `nn.Linear`, no
`nn.functional`, no HuggingFace TRL.

## Features

- **Modern decoder-only 60M Transformer.** Pre-norm RMSNorm, RoPE,
  SwiGLU, grouped-query attention at 5:1, weight tying, KV-cached
  decoding.
- **Pretraining on TinyStoriesV2-GPT4.** Validation perplexity **17.19**
  at **228 K tokens/sec** and **26.2 % MFU** on a single Tesla L4
  in bfloat16 with `torch.compile`.
- **Post-training, from scratch.** Masked-loss SFT on TinyStoriesInstruct,
  then DPO against a frozen reference. The DPO implementation is ~140
  lines with a numerically stable `log σ`. Training diagnostics over
  600 steps: reward margin grew from +0.01 to **+0.36**, preference
  accuracy from **45 %** to **87.5 %**.
- **Judge harness with bias controls.** Both-order judging plus
  swap-consistency filtering. 3 × 3 pairwise win-rate matrix,
  cross-validated across Qwen2.5-7B and 14B.
- **BPE tokenizer trained in-house.** GPT-2 regex pre-tokenisation,
  multiprocessing, incremental pair-count updates. Chat-template
  specials are baked in at training time so the embedding table never
  needs to be resized for post-training.
- **Five samplers, one pipeline.** Temperature, top-p, top-k, min-p,
  repetition penalty. The KV cache is verified mathematically equivalent
  to full recomputation (max error < 3 × 10⁻⁶).
- **Four ways to run it.** Gradio playground on HuggingFace Spaces
  (Generate, Compare, **Aligned** base / SFT / DPO comparison),
  OpenAI-compatible FastAPI server with SSE streaming, terminal REPL,
  one-shot CLI.
- **67 tests on every push.** GitHub Actions runs the full suite on
  Python 3.10, 3.11, and 3.12, plus a five-step smoke train to catch
  training-loop regressions.

**Stack** — PyTorch · FastAPI · Gradio · HuggingFace Hub · pytest ·
GitHub Actions

## Demo

**Try it live →** <https://huggingface.co/spaces/Grux11/transformer-lm-60m>

The Space sleeps after 48 hours of inactivity. First load wakes it;
expect a ~30 second cold start on the free CPU tier.

The **Aligned** tab feeds the same prompt to all three checkpoints at
once. Base shows what 60M of pretraining produces on its own. SFT adds
chat-template instruction-following. DPO is the preference-aligned
variant.

![Aligned tab — base, SFT, and DPO completing the same instruction prompt](docs/playground-aligned.png)

The **Generate** tab streams a single model token by token, with all
five sampling controls as live sliders and a tokens-per-second meter.
Output ends cleanly at `<|endoftext|>`.

![Generate tab — streaming 60M base completion of "Once upon a time there was a little girl who"](docs/playground-generate.png)

## Quick start

The three trained checkpoints (base, SFT, DPO) and tokenizer files live
on the HuggingFace Hub at
[Grux11/transformer-lm-60m-tinystories](https://huggingface.co/Grux11/transformer-lm-60m-tinystories).

**Run the playground locally**

```bash
pip install -e ".[ui]"

python -c "from huggingface_hub import snapshot_download; \
           snapshot_download('Grux11/transformer-lm-60m-tinystories', \
                             local_dir='hf_cache')"

python scripts/playground.py \
    --checkpoint     hf_cache/base_60m/final.pt \
    --checkpoint_sft hf_cache/sft_v2/final.pt \
    --checkpoint_dpo hf_cache/dpo_v2/final.pt \
    --vocab  hf_cache/tinystories_v2_vocab.json \
    --merges hf_cache/tinystories_v2_merges.txt
```

Then open <http://127.0.0.1:7860>.

**Or generate from a single checkpoint**

```bash
python scripts/generate.py \
    --checkpoint hf_cache/sft_v2/final.pt \
    --vocab hf_cache/tinystories_v2_vocab.json \
    --merges hf_cache/tinystories_v2_merges.txt \
    --vocab_size 16000 --d_model 640 --num_layers 10 \
    --num_heads 10 --num_kv_heads 2 --d_ff 1728 --context_length 512 \
    --prompt "Once upon a time" --max_tokens 200 \
    --temperature 0.8 --top_p 0.95 --top_k 50
```

Two more entry points use the same arguments. `scripts/chat.py` opens
an interactive REPL with `/temp`, `/top_p`, `/top_k`, `/reset`
slash-commands. `scripts/serve.py` launches an OpenAI-compatible
HTTP server (`/v1/chat/completions`, `/v1/completions`, SSE streaming)
that works as a drop-in for the OpenAI Python SDK, LangChain, Open
WebUI, Jan, and SillyTavern.

## Results

Pretraining run, single Tesla L4, bfloat16, `torch.compile`, 20K steps.

| Metric | Value |
|---|---|
| Validation perplexity | **17.19** |
| Training throughput | 228,109 tokens/sec |
| MFU (bf16, against L4 ~120 TFLOPS peak) | **26.2 %** |
| Wall-clock | ~48 min |

DPO training over 600 steps at β=0.1, lr 5 × 10⁻⁶, on 178
swap-consistent preference pairs.

| Diagnostic | Step ~20 | Step 600 |
|---|---|---|
| Reward margin  β · (log πθ/πref difference) | +0.01 | **+0.36** |
| Preference accuracy on training batches | 45 % | **87.5 %** |

Pairwise win-rate matrix from `scripts/eval_all.py`. Qwen2.5-7B judge,
150 held-out prompts, both orders judged, swap-consistent pairs only.

|              | beats Base | beats SFT | beats DPO |
|--------------|:----------:|:---------:|:---------:|
| **Base**     | —          | 10.0 %    | 10.0 %    |
| **SFT**      | **38.0 %** | —         | 34.7 %    |
| **DPO**      | **32.7 %** | 31.3 %    | —         |

SFT beats Base 38 % to 10 %. That's the headline result. DPO vs SFT
sits inside the judge's noise band at this scale, even though the
training-side numbers say DPO is doing exactly what it should. Running
the same eval with the larger Qwen2.5-14B judge gave the same picture.

→ Held-out PPL, SFT trainer hyperparameters, two-judge
cross-validation, KV-cache crossover sweep, hardware reference, and
extended sample completions are in
[`docs/results.md`](docs/results.md).

## Sample outputs

Prompt is a chat-template instruction for SFT and DPO, and raw text for
base:

> *Write a short story about a girl named Lily who finds a magic stone
> in the forest.*

Settings: T=0.7, top-p=0.95, top-k=50, max 220 tokens.

**Base.** Continues the instruction as if it were the first sentence of
the story.

> Write a short story about a girl named Lily who finds a magic stone in
> the forest. Lily loved the shopping, and they would grant jigchy. She
> was always curious and happy to see the magic stone. One day, Lily and
> her mom went to the forest to find the magic stone…

**SFT.** Recognises the chat template and produces a structured story
with dialogue.

> One day, a little girl named Lily was playing in her room. She saw a
> small bird sitting on a branch… "Mom, can you please help me get the
> bird down?" Lily asked. Her mom said, "Of course, Lily. Let's go help
> the bird." … And they lived happily ever after.

**DPO.** Same instruction-following plus the moral-lesson closing that
preference-aligned models tend to drift toward.

> Once upon a time, there was a little girl named Lily… her mom said,
> "Lily, you need to be careful with theitty. It can be dangerous." Lily
> nodded and promised to be more careful next time. From that day on,
> Lily learned to be more careful with the things she found.

Word-level artefacts like `jigchy` and `theitty` are characteristic of
byte-level BPE on a 60M model and don't go away with more training.

## Architecture

```
        token_ids (B, T)
              │
              ▼
       Embedding (V × d_model)
              │
       ┌──────┴──────┐
       │  N × Block  │
       │  ┌────────┐ │
       │  │ RMSNorm│ │
       │  │   │    │ │
       │  │  GQA   │ │
       │  │ + RoPE │ │
       │  │   │    │ │
       │  │ ⊕ res  │ │
       │  │   │    │ │
       │  │ RMSNorm│ │
       │  │   │    │ │
       │  │ SwiGLU │ │
       │  │   │    │ │
       │  │ ⊕ res  │ │
       │  └────────┘ │
       └──────┬──────┘
              │
          RMSNorm
              │
        LM head (tied)
              │
              ▼
       logits (B, T, V)
```

| Component | Choice |
|---|---|
| Position encoding | Rotary position embeddings (RoPE) |
| Normalisation | RMSNorm, pre-norm placement, fp32 upcast for variance |
| Attention | Causal multi-head with grouped-query and KV cache; chunked-memory variant for long context |
| Feed-forward | SwiGLU, `d_ff = round_64(8/3 · d_model)` |
| Tokenizer | Byte-level BPE, GPT-2 regex pre-tokenisation, chat-template specials baked in |
| Optimiser | AdamW with decoupled weight decay |
| LR schedule | Cosine annealing + linear warmup |
| Weight tying | Token embedding shared with LM head |
| Sampling | Temperature, top-k, top-p, min-p, repetition penalty, composed in one pipeline |

Full 60M config in
[`configs/tinystories_60m.yaml`](configs/tinystories_60m.yaml):
10 layers, 10 query heads, 2 KV heads (GQA 5:1), `d_model=640`,
`d_ff=1728`, context 512, vocab 16K.

## Implementation details

**Chat-template loss masking.** Conversations format as
`<|user|>{prompt}<|endoftext|><|assistant|>{response}<|endoftext|>`.
The loss mask is 1 on response tokens and 0 on the prompt and the
leading `<|assistant|>` marker. The model is trained to start
generating *after* the marker, not to predict it. The mask is shifted
left by one to align with `target_ids[t] = sequence[t+1]`. See
[`cs336_basics/data_sft.py`](cs336_basics/data_sft.py).

**DPO loss.** Given preference pairs `(x, y_w, y_l)` and a frozen
reference `π_ref`:

```
L_DPO = -E[ log σ( β · ( log π_θ(y_w|x)/π_ref(y_w|x)
                       - log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]
```

[`cs336_basics/dpo.py`](cs336_basics/dpo.py) implements this in ~140
lines. The `log σ` is computed with the branch
`-log(1 + exp(-x))` for `x ≥ 0` and `x - log(1 + exp(x))` otherwise,
which is numerically stable. Log-probabilities use the same shifted
log-softmax pattern as the cross-entropy loss. Policy and reference
are loaded into two separate `TransformerLM` instances and held in
matched bfloat16 at log-prob computation. Dtype mismatch at this step
silently swamps the DPO signal, so `train_dpo.py` asserts dtype
equality at startup.

**Judge harness.** Each preference pair is judged in both orders.
Pairs where the verdict flips are dropped as position bias. The
remaining swap-consistent pairs are the eval signal, and the
swap-consistency rate itself is the reliability metric. Rates below
70 % are flagged as low signal. Cross-validating with a stronger judge
tests the judge-limited hypothesis directly.

**KV cache and GQA.** `model.generate()` uses an incremental KV cache.
`test_kv_cache_matches_full_forward` verifies it produces identical
logits to full recomputation (max error < 3 × 10⁻⁶). Grouped-query
attention shares K/V projections across query heads in groups of
`num_q_heads / num_kv_heads` (5:1 in the 60M config), cutting KV-cache
memory at long context.

## Reproduce the pipeline

End-to-end on a single Tesla L4 in roughly six hours. An A100 helps
steps 6 and 8 if you substitute a 14B-or-larger judge.

```bash
pip install -e .

# 1. Download TinyStoriesV2-GPT4, train BPE (chat specials baked in), encode, split.
python scripts/download_tinystories_v2.py --vocab_size 16000 --output_dir data/

# 2. Pretrain the 60M base.
python scripts/train.py --config configs/tinystories_60m.yaml --device cuda

# 3. Pack SFT examples from TinyStoriesInstruct.
python scripts/build_sft_dataset.py \
    --vocab data/tinystories_v2_vocab.json \
    --merges data/tinystories_v2_merges.txt \
    --output data/tinystories_v2_sft.pt

# 4. Supervised fine-tune from the pretrain checkpoint.
python scripts/train_sft.py \
    --base_checkpoint checkpoints/base_60m/final.pt \
    --sft_data        data/tinystories_v2_sft.pt \
    --checkpoint_dir  checkpoints/sft \
    --total_steps 6000 --batch_size 32 --lr_max 3e-5 \
    --device cuda --dtype bfloat16 --compile

# 5. Sample candidate preference pairs from the SFT model.
python scripts/build_preference_dataset.py \
    --sft_checkpoint checkpoints/sft/final.pt \
    --vocab  data/tinystories_v2_vocab.json \
    --merges data/tinystories_v2_merges.txt \
    --output data/pref_candidates.jsonl \
    --num_prompts 1000 --device cuda

# 6. Label preferences with a local LLM judge in both orders.
python scripts/label_preferences.py \
    --input        data/pref_candidates.jsonl \
    --output       data/pref_labeled.jsonl \
    --judge_model  Qwen/Qwen2.5-7B-Instruct --load_in_4bit --device cuda

# 7. Direct Preference Optimization from a frozen reference.
python scripts/train_dpo.py \
    --sft_checkpoint checkpoints/sft/final.pt \
    --preferences    data/pref_labeled.jsonl \
    --vocab  data/tinystories_v2_vocab.json \
    --merges data/tinystories_v2_merges.txt \
    --checkpoint_dir checkpoints/dpo \
    --total_steps 600 --batch_size 4 --beta 0.1 --lr_max 5e-6 \
    --device cuda --dtype bfloat16

# 8. PPL + 3×3 pairwise win-rate matrix + swap-consistency → results.md.
python scripts/eval_all.py \
    --base_checkpoint checkpoints/base_60m/final.pt \
    --sft_checkpoint  checkpoints/sft/final.pt \
    --dpo_checkpoint  checkpoints/dpo/final.pt \
    --val_data data/tinystories_v2_tokens_val.npy \
    --vocab    data/tinystories_v2_vocab.json \
    --merges   data/tinystories_v2_merges.txt \
    --judge_model Qwen/Qwen2.5-7B-Instruct --load_in_4bit \
    --output results.md --device cuda
```

The notebooks in [`notebooks/`](notebooks/) wrap these commands with
Colab setup (Drive mount, repo clone, dependency install) and include
captured stdout from the runs that produced the numbers above.

To deploy the playground to HuggingFace Spaces, upload the checkpoints
with [`scripts/upload_checkpoints_to_hf.py`](scripts/upload_checkpoints_to_hf.py)
and copy [`spaces/`](spaces/) into a new Gradio Space.

## Repository layout

```
cs336_basics/             importable Python package — all model code
├── tokenizer.py          byte-level BPE: train_bpe(), Tokenizer
├── nn_components.py      Linear, Embedding, RMSNorm
├── attention.py          softmax, RoPE, SDPA, chunked attention,
│                         CausalMultiHeadSelfAttention with GQA + KV cache
├── model.py              SwiGLU FFN, TransformerBlock, TransformerLM
├── optimizer.py          AdamW with decoupled weight decay
├── data_sft.py           chat-template formatter, masked-loss packer
├── dpo.py                DPO loss + per-batch diagnostics
├── streaming.py          incremental UTF-8 decoder for streaming output
└── training.py           cross-entropy + masked variant, LR schedule,
                          gradient clipping, data loader, checkpoint I/O

scripts/                  one CLI per pipeline stage
├── download_tinystories_v2.py
├── train.py
├── build_sft_dataset.py · train_sft.py
├── build_preference_dataset.py · label_preferences.py
├── train_dpo.py
├── eval_all.py · evaluate.py
├── playground.py · serve.py · chat.py · generate.py
├── benchmark.py · lr_find.py · run_ablations.py
├── train_tokenizer.py · split_data.py
└── upload_checkpoints_to_hf.py

spaces/                   HuggingFace Spaces deployment of the playground
notebooks/                Colab notebooks (pretraining and post-training)
configs/                  YAML configs (17M reference, 60M target, OWT)
docs/                     deep-dive evaluation results
tests/                    pytest suite (67 unit + integration tests)
.github/workflows/        tests.yml — CI on Python 3.10, 3.11, 3.12
```

## Tests

```bash
pytest tests/ -q
```

67 tests across model internals, training loop, post-training,
tokenizer, KV-cache equivalence, GQA shapes, chunked attention,
streaming generator, OpenAI-compatible API endpoints, UTF-8 decoder
edge cases, and a single-batch overfit check. GitHub Actions runs the
suite on every push and PR across Python 3.10, 3.11, and 3.12, plus a
five-step smoke train
([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

## License

MIT. See [`LICENSE`](LICENSE).

## References

Ideas the model and pipeline build on.

| Topic | Reference |
|---|---|
| Transformer | Vaswani et al., *Attention Is All You Need* (2017) |
| RoPE | Su et al., *RoFormer* (2021) |
| RMSNorm | Zhang & Sennrich (2019) |
| SwiGLU | Shazeer, *GLU Variants Improve Transformer* (2020) |
| GQA | Ainslie et al. (2023) |
| AdamW | Loshchilov & Hutter (2019) |
| Llama-3 stack | Touvron et al. (2023) |
| Top-p sampling | Holtzman et al. (2019) |
| Min-p sampling | Nguyen (2023) |
| BPE | Sennrich et al. (2016) |
| TinyStories | Eldan & Li (2023) |
| DPO | Rafailov et al. (2023) |
| LLM-as-judge methodology | Zheng et al., *MT-Bench / Chatbot Arena* (2023) |
| Curriculum | Stanford CS336 — *Language Models from Scratch* (Spring 2025) |
