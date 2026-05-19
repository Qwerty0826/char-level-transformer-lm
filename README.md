# Transformer Language Model — Built from Scratch

A complete decoder-only Transformer language model implemented from the ground up in PyTorch, following the CS336 (Stanford, Spring 2025) curriculum on building LLMs from scratch.

Every component — byte-level BPE tokenizer, multi-head self-attention with Rotary Position Embeddings, SwiGLU feed-forward layers, RMSNorm, AdamW optimizer, cosine LR schedule, and the full training loop — is written without relying on `torch.nn.functional` or pre-built layer implementations.

---

## Architecture

| Component | Design choice |
|-----------|---------------|
| Tokenizer | Byte-level BPE (GPT-2 regex pre-tokenizer, multiprocessing) |
| Position encoding | Rotary Position Embedding (RoPE, Su et al. 2021) |
| Normalisation | RMSNorm (Zhang & Sennrich 2019), pre-norm placement |
| Feed-forward | SwiGLU (Shazeer 2020): `W₂(SiLU(W₁x) ⊙ W₃x)` |
| Optimiser | AdamW (Loshchilov & Hutter 2019) from scratch |
| LR schedule | Cosine annealing with linear warmup (LLaMA style) |
| Weight tying | Input embedding ↔ output LM head (optional) |

**Default model (TinyStories):** 4 layers, 16 heads, d_model=512, d_ff=1344 → ~17M non-embedding parameters.

---

## Repository Layout

```
cs336_basics/          Core library (importable package)
  tokenizer.py         BPE training + Tokenizer class
  nn_components.py     Linear, Embedding, RMSNorm
  attention.py         softmax, RoPE, scaled dot-product attention, CausalMHA
  model.py             SwiGLU FFN, TransformerBlock, TransformerLM
  optimizer.py         AdamW
  training.py          cross_entropy, LR schedule, gradient clipping,
                       get_batch, save/load checkpoint

scripts/
  train_tokenizer.py   Train BPE tokenizer and encode a corpus to .npy
  train.py             Full training loop (CLI, W&B, CSV logging)
  generate.py          Autoregressive generation with top-p sampling
  evaluate.py          Perplexity + bits-per-character evaluation

configs/
  tinystories.yaml     Apple Silicon / low-resource config
  owt.yaml             OpenWebText config for CUDA

tests/                 35 unit tests (pytest), all passing
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy regex einops pyyaml tqdm
# Optional for W&B logging:
pip install wandb
```

### 2. Get the data

**TinyStories** (≈2.1M short children's stories):

The `data/tinystories.txt` file contains a small validation subset (~22K stories, 22 MB). For the full dataset:
```bash
# Download from HuggingFace or use the provided small validation set
```

### 3. Train a BPE tokenizer

```bash
python scripts/train_tokenizer.py \
    --input data/tinystories.txt \
    --vocab_size 10000 \
    --output_dir data/ \
    --prefix tinystories \
    --encode data/tinystories.txt \
    --encode_out data/tinystories_tokens.npy
```

### 4. Train the model

```bash
python scripts/train.py \
    --train_data data/tinystories_tokens.npy \
    --val_data   data/tinystories_tokens.npy \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
    --batch_size 32 --total_steps 5000 \
    --lr_max 1e-3 --lr_min 1e-4 --warmup_steps 200 \
    --checkpoint_dir checkpoints/tinystories \
    --compile
```

On Apple Silicon MPS the script auto-selects `aot_eager` as the `torch.compile` backend (the default Inductor backend has broken kernels on MPS).

### 5. Generate text

```bash
python scripts/generate.py \
    --checkpoint checkpoints/tinystories/final.pt \
    --vocab  data/tinystories_vocab.json \
    --merges data/tinystories_merges.txt \
    --prompt "Once upon a time there was" \
    --max_tokens 256 --temperature 0.8 --top_p 0.95
```

### 6. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/tinystories/final.pt \
    --data   data/tinystories_tokens.npy \
    --vocab  data/tinystories_vocab.json \
    --merges data/tinystories_merges.txt \
    --n_batches 100 --generate_samples 2
```

---

## Running Tests

```bash
pytest tests/ -v
# Expected: 35 passed
```

---

## Key Design Decisions

### Byte-level BPE tokenizer

The tokenizer is trained on raw UTF-8 bytes — every possible byte sequence can be represented without an out-of-vocabulary token. The GPT-2 pre-tokenization regex prevents merges from crossing word/punctuation boundaries, giving the same quality as word-level tokenizers while retaining byte-level completeness.

Pair counts are updated incrementally after each merge (only affected pairs are recomputed), making training significantly faster than a naïve O(vocab × corpus) implementation.

### RoPE positional embeddings

Absolute sinusoidal positions are replaced by Rotary Position Embeddings, which encode relative position information through 2D rotations applied to query and key vectors. This generalises better to sequence lengths longer than those seen during training compared to learned absolute positions.

The cos/sin tables are precomputed at init and stored as non-persistent buffers (excluded from checkpoints since they are deterministic).

### Pre-norm Transformer

Layer normalisation is applied *before* the sub-layer input (pre-norm), not after (post-norm). This provides a "clean residual stream" from input to output with no normalisation on the residual path, improving gradient flow and training stability. All modern large LMs (GPT-3, LLaMA, Mistral) use this variant.

### SwiGLU activation

The feed-forward network uses a gated linear unit with SiLU (Swish) activation. The inner dimension is set to ⌈8/3 × d_model⌉ rounded to the nearest multiple of 64 — this approximates the parameter count of a standard 4× FFN while the three-matrix structure provides better performance empirically.

### Weight tying

The input token embedding matrix (vocab_size × d_model) is shared with the LM head projection. This reduces parameter count by ~5M on a 10K vocabulary model and typically improves perplexity by ~0.1–0.2 nats on small models.

---

## Ablation Studies

The architecture supports easy ablation of individual components:

| Ablation | How |
|----------|-----|
| Remove RMSNorm | Set `attn_norm` / `ff_norm` to identity in `TransformerBlock` |
| Post-norm vs pre-norm | Swap norm placement in `TransformerBlock.forward` |
| RoPE vs NoPE | Skip the `self.rope(Q, positions)` / `self.rope(K, positions)` calls |
| SwiGLU vs SiLU | Remove `* self.W3(x)` from `SwiGLUFeedForward.forward` |

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.*
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
- Zhang & Sennrich (2019). *Root Mean Square Layer Normalization.*
- Shazeer (2020). *GLU Variants Improve Transformer.*
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization.*
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.*
- Eldan & Li (2023). *TinyStories: How Small Can Language Models Be?*
- Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units.*
