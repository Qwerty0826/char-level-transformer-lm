# Character-Level Transformer Language Model (From Scratch)

This project implements a decoder-only Transformer Language Model from scratch, including tokenization, model architecture, and training pipeline.

## 🔧 Features

- Byte-level BPE tokenizer (UTF-8 based)
- Decoder-only Transformer architecture
- Multi-head self-attention + feed-forward layers
- Cross-entropy loss with AdamW optimizer
- Perplexity-based evaluation
- Text generation pipeline

## 📚 Architecture

The model follows a standard Transformer LM pipeline:

Input Text → BPE Tokenization → Token Embeddings → Transformer Blocks → Linear Head → Next-token Prediction

Key components implemented:

- Token Embeddings
- Multi-head Self-Attention
- Feed-forward layers
- Training loop with optimizer state

## ⚙️ Training

- Dataset: TinyStories (subset)
- Loss: Cross-Entropy
- Optimizer: AdamW
- Evaluation: Perplexity, Bits-per-character

## 📊 Results (WIP)

- Initial training shows decreasing loss trends
- Planned ablations:
  - Model depth
  - Context length
  - Tokenization strategy

## 🚧 Status

Work in progress:
- Efficient BPE merging
- Full attention implementation
- Scaled training

## 📌 References

- Vaswani et al. (2017) — Attention is All You Need
- Stanford CS336 Assignment 1
