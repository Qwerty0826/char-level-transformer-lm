---
title: Transformer LM (60M, From Scratch)
emoji: 🌀
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# 60M Transformer LM — From-Scratch Playground

Interactive playground for a decoder-only Transformer language model
built from scratch in PyTorch — no `nn.Linear`, no `nn.functional`,
no HuggingFace TRL.

Three checkpoints are loaded side-by-side:

- **Base** — 60M-param pretrain on TinyStoriesV2-GPT4
- **+ SFT** — masked-loss supervised fine-tuning on TinyStoriesInstruct
- **+ SFT + DPO** — Direct Preference Optimization (Rafailov et al. 2023)
  against a frozen reference of the SFT model, trained on preference
  pairs labeled by a local Qwen2.5-7B-Instruct judge

Open the **Aligned (Base → SFT → DPO)** tab and type an instruction-style
prompt — the same prompt is fed to all three checkpoints in parallel and
the outputs render in three columns.

Source code, results, and reproduction instructions:
[github.com/Qwerty0826/char-level-transformer-lm](https://github.com/Qwerty0826/char-level-transformer-lm)
