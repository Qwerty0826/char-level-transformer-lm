# Evaluation results (detail)

This file holds the deep-dive numbers and methodology the main README
summarises in three tables. Everything here is reproducible from the
commands in [README §Reproduce the pipeline](../README.md#reproduce-the-pipeline).

## Pretraining

20,000 steps on TinyStoriesV2-GPT4. Batch size 64, context 512,
bfloat16, `torch.compile`. AdamW with cosine annealing (peak LR
3 × 10⁻⁴, min LR 3 × 10⁻⁵, warmup 500 steps), gradient clipping at
1.0, decoupled weight decay 0.1. Single Tesla L4.

| Metric | Value |
|---|---|
| Validation perplexity | **17.19** |
| Validation loss | 2.84 |
| Final training loss (smoothed) | 2.87 |
| Throughput | 228,109 tokens/sec |
| MFU (bf16, L4 ~120 TFLOPS peak) | **26.2 %** |
| Wall-clock | ~48 min |

17.19 PPL in token space on a 16K BPE vocab works out to ~1 bit per
character on TinyStories prose, close to the Shannon entropy floor for
that distribution. The model is bandwidth-bound on L4 at this size,
which is why throughput holds at ~228K tok/s.

### 17M correctness reference

A 17M-parameter configuration (`configs/tinystories.yaml`) was trained
earlier on the original TinyStories on a Tesla T4 in fp32. Val PPL
9.59, 29.1 % MFU, ~30 min wall-clock. The 17M run validated every
primitive (attention, RoPE, RMSNorm, AdamW, samplers, KV cache) before
the 60M scale-up, and is retained as a regression target.

## Supervised fine-tuning

6,000 steps of masked-loss SFT on TinyStoriesInstruct.

| Hyperparameter | Value |
|---|---|
| Steps | 6,000 |
| Batch size | 32 |
| Peak LR | 3 × 10⁻⁵ |
| Warmup | 200 steps |
| Schedule | Cosine |
| Optimiser | AdamW (same as pretrain) |
| Compute | bfloat16, `torch.compile` |

The loss is computed only on assistant-response tokens. The prompt and
the leading `<|assistant|>` marker are conditioning context (mask = 0);
the response body and the closing `<|endoftext|>` are in the loss
(mask = 1). The mask shifts left by one to align with the
`target_ids[t] = sequence[t+1]` convention used by `get_batch` in
`scripts/train.py`. Implementation in
[`cs336_basics/data_sft.py`](../cs336_basics/data_sft.py).

## Direct Preference Optimization

178 swap-consistent preference pairs, filtered down from 516 candidates
sampled at T=0.9 from the SFT model and labeled by
Qwen2.5-7B-Instruct in 4-bit. The labeler runs each pair in both
orders and drops the cases where the verdict flips
([`scripts/label_preferences.py`](../scripts/label_preferences.py)).

| Hyperparameter | Value |
|---|---|
| Steps | 600 |
| Batch size | 4 |
| β | 0.1 |
| Peak LR | 5 × 10⁻⁶ |
| Reference | frozen SFT, matched bfloat16 |
| Optimiser | AdamW |

Training-side diagnostics from the run:

| Diagnostic | Step ~20 | Step 600 |
|---|---|---|
| Reward margin  β · (log πθ/πref difference) | +0.01 | **+0.36** |
| Preference accuracy on training batches | 45 % | **87.5 %** |

The margin grew about 30×, accuracy nearly doubled. That's DPO doing
what it's supposed to: pushing the chosen response above the reference
and pulling the rejected response below.

## Pairwise judge win-rate (Qwen-7B)

3 × 3 matrix from [`scripts/eval_all.py`](../scripts/eval_all.py).
150 held-out prompts. Judge: Qwen2.5-7B-Instruct in 4-bit. Both
orderings of every pair judged. Win-rate counts only swap-consistent
(non-flipping) judgements.

|              | beats Base | beats SFT | beats DPO |
|--------------|:----------:|:---------:|:---------:|
| **Base**     | —          | 10.0 %    | 10.0 %    |
| **SFT**      | **38.0 %** | —         | 34.7 %    |
| **DPO**      | **32.7 %** | 31.3 %    | —         |

SFT beats Base 38 % to 10 %. That's the headline qualitative gain.
DPO vs SFT is closer to a coin flip than to a clean win, even though
the training-side numbers above say DPO is learning. Two-judge
cross-validation below tests whether the eval was judge-limited.

## Held-out perplexity

Validation set: a held-out slice of TinyStoriesV2-GPT4 (raw stories,
not chat-formatted).

| Checkpoint | Val loss | Val PPL |
|---|---|---|
| Base       | 2.89     | **18.06** |
| + SFT      | 3.06     | 21.33 |
| + SFT + DPO | 3.06    | 21.25 |

PPL rises after SFT because the SFT model is fitted to the
chat-template distribution, not raw stories. This is expected and is
why PPL is not a useful headline metric for preference-tuned models.
The SFT-vs-DPO PPLs are basically identical, which is consistent with
DPO being a preference objective rather than a likelihood objective.

## Two-judge cross-validation

To check whether the eval was bottlenecked by judge quality, the v2
checkpoints were re-judged with Qwen2.5-14B-Instruct on a 75-prompt
subset.

| Pair | Qwen-7B (150 prompts) | Qwen-14B (75 prompts) |
|---|:---:|:---:|
| Base vs SFT swap-consistency | 48.0 % | 38.7 % |
| Base vs DPO swap-consistency | 42.7 % | 48.6 % |
| SFT vs DPO swap-consistency | **66.0 %** | 36.0 % |
| SFT beats Base | 38.0 % | 24.0 % |
| DPO beats SFT | 31.3 % | 24.0 % |
| DPO beats Base | 32.7 % | 31.1 % |

The larger judge does not increase swap-consistency. On the SFT-vs-DPO
comparison it drops sharply, with the 14B judge finding the two models
harder to distinguish than the 7B did. Both judges put DPO-vs-SFT
inside the noise band, so the result is not judge-bound. The
output-level difference between SFT and DPO at 60M parameters falls
inside both judges' reliability margins, while the DPO objective
itself continues to optimise correctly (margin growth, accuracy
growth).

## Extended sample completions

Prompt: *"Write a short story about a girl named Lily who finds a
magic stone in the forest."*  ·  T=0.7, top-p=0.95, top-k=50,
max 220 tokens.

### Base

> Write a short story about a girl named Lily who finds a magic stone in
> the forest. Lily loved the shopping, and they would grant jigchy. She
> was always curious and happy to see the magic stone. One day, Lily and
> her mom went to the forest to find the magic stone. They walked and
> walked until they found a big tree with lots of yummy bugs and coach.
> Lily was so excited to eat the bugs that she forgot about the magic
> stone. She climbed down the tree and sat down to read it. The forest
> was full of creatures, and Lily had the power to should go home.
> `<|endoftext|>`

Base reads the instruction as the first sentence of a story and
continues from there. Word-level artefacts like `jigchy` and "yummy
bugs and coach" are characteristic of byte-level BPE on a small model.

### SFT

> One day, a little girl named Lily was playing in her room. She saw a
> small bird sitting on a branch. Lily looked sad. She wanted to help
> the bird, so she tried to talk to it. She asked her mom for help.
>
> "Mom, can you please help me get the bird down?" Lily asked. Her mom
> said, "Of course, Lily. Let's go help the bird."
>
> As they went on the carousel, something unexpected happened. A little
> bird flew down and landed on the ground. The bird picked up the bird
> with its beak and dropped it on the ground. The bird landed on the
> ground, and Lily was surprised. She picked up the bird and said,
> "Thank you, little bird, for helping me."
>
> Lily and the bird became good friends. They played together every day
> and always helped each other. And they lived happily ever after.

SFT recognises the chat template and produces a structured story.
Lily is preserved as the protagonist; the magic stone drifts into a
bird-helping plot, which is a common limit at this parameter count.

### DPO

> Once upon a time, there was a little girl named Lily. She loved to
> read books about cy around her house. One day, Lily's mom said,
> "Lily, it's time to go to the forest today." Lily was so excited to
> go to the forest.
>
> At the forest, Lily saw a big, round, and cheap toy. She was so happy
> and wanted to read it. But then, she remembered what her mom said and
> said, "Lily, you should not get too excited if you don't get lost."
>
> Lily didn't listen and kept reading. When she got tired, she decided
> to go home. She told her mom about the woof and how she loved playing
> with theitty. Her mom said, "Lily, you need to be careful with
> theitty. It can be dangerous." Lily nodded and promised to be more
> careful next time.
>
> From that day on, Lily learned to be more careful with the things she
> found. She also learned to listen to her mom and to be careful when
> playing and reading.

DPO writes a story shaped around a moral lesson. The phrasing is more
cautious and cozier than SFT's, which matches the preference data the
labeler emitted (the 7B judge tended to pick the more polished /
safer-sounding completion). Word-level artefacts like `theitty` and
"big, round, and cheap toy" remain visible.

## Inference throughput: KV cache crossover

`scripts/benchmark.py` sweeps generation lengths to locate the point
where the KV cache starts paying off. On the 17M model on T4 in fp32:

| gen tokens | KV cache (tok/s) | No cache (tok/s) | Speedup |
|---:|---:|---:|---:|
| 32  | 81.3  | 116.3 | 0.70× |
| **64**  | **118.6** | **97.9**  | **1.21×** |
| 128 | 95.2  | 90.8  | 1.05× |
| 224 | 105.5 | 115.0 | 0.92× |

Crossover sits at `gen_tokens ≥ 64`. At shorter contexts the cache
loses on a model this small because the 17M weights are bandwidth-bound
on T4: time is dominated by reading parameter matrices out of HBM, and
the per-step kernel launch overhead is comparable to the actual
compute. The full-recompute path does many more FLOPs but amortises
weight loads across many tokens per launch. The cache wins decisively
once the model is compute-bound (≥1B params) or the context is long
enough for O(T²) attention to dominate, which is why production serving
stacks (vLLM, TGI) use fused kernels on cached paths rather than naive
Python loops.

Cache correctness is verified by
`test_kv_cache_matches_full_forward` (max logit error < 3 × 10⁻⁶) and
`test_kv_cache_token_by_token` (max logit error < 1 × 10⁻⁴).

## Hardware reference

Throughput numbers across the platforms the project has been run on.

| Setup | Throughput | Wall-clock |
|---|---|---|
| 17M · Apple M1 MPS (fp32) | ~4,000 tok/s | 2.5–3 h (5K steps) |
| 17M · Colab T4 (fp32) | 21,088 tok/s | ~30 min (5K steps) |
| **60M · Colab L4 (bf16)** | **228,109 tok/s** | **~48 min (20K steps), 26.2 % MFU** |
| 60M · Colab A100 (bf16) | ~400,000 tok/s | ~30 min (estimated) |

Tesla T4 (Turing) has no bf16 tensor cores, so bf16 falls back to
non-tensor-core compute and runs slower than fp32 on that card. On T4
use `--dtype float32`. On Ampere and newer (A100, L4, RTX 3xxx/4xxx),
bf16 is the default. On Apple Silicon MPS, `dtype: float32` and
`compile_backend: aot_eager` are the only stable combination at the
time of this run; Inductor and bf16 kernels have known issues on MPS.

## MFU calculation

MFU is the ratio of FLOPs the model actually performs per second to the
hardware's theoretical peak. `scripts/train.py` and `scripts/benchmark.py`
derive per-step FLOPs from `model.estimate_flops_per_token()`, divide
by wall-clock time, and divide by the appropriate peak from a per-GPU
table — selected by the **configured dtype**, so a bf16 run is compared
to the bf16 tensor-core peak and an fp32 run to the plain fp32 peak.
Reporting bf16 throughput against the fp32 peak (or vice versa) is the
easy way to publish numbers that look impressive without being correct.
The 60M run measures **26.2 % MFU** against the L4 bf16 peak
(~120 TFLOPS).

## Methodology notes

**Position-bias filtering.** Every preference pair is judged twice,
once as `(A, B)` and once as `(B, A)`. If the verdict flips between
the two orderings, the pair is dropped from training data and from
eval counts. The remaining swap-consistent pairs are the only ones
used. The swap-consistency rate itself is reported as a reliability
metric. Rates below 70 % are flagged as low signal in the eval output.

**Why not use PPL as the headline metric.** PPL measures likelihood of
the validation distribution, which the SFT and DPO objectives
deliberately move the model away from (toward the chat-template
distribution and toward preferred outputs respectively). Reporting PPL
as the headline would penalise the two checkpoints for doing what
they're meant to do.

**Reproducing this file.** Every number above comes from the commands
in [README §Reproduce the pipeline](../README.md#reproduce-the-pipeline).
The win-rate matrix and swap-consistency rates are written by
`scripts/eval_all.py` to `results.md` and `results.raw.json` on each
run; the numbers here are from the most recent run with `sft_v2` and
`dpo_v2` checkpoints.
