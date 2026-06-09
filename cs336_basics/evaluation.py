"""
Evaluation helpers shared by the CLI scripts.

Lives in the library (not ``scripts/``) so that anything pip-installing
this project gets it without depending on script modules.
"""

from __future__ import annotations

import math

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import cross_entropy_loss


@torch.no_grad()
def compute_bpc(
    model: TransformerLM,
    tokenizer: Tokenizer,
    text: str,
    context_length: int,
    device: str,
) -> float:
    """Bits-per-character on `text` using non-overlapping windows."""
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return float("nan")

    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(ids) - 1, context_length):
        end = min(start + context_length, len(ids) - 1)
        chunk_x = torch.tensor([ids[start:end]],   dtype=torch.long, device=device)
        chunk_y = torch.tensor([ids[start+1:end+1]], dtype=torch.long, device=device)
        logits = model(chunk_x)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), chunk_y.view(-1))
        total_loss   += loss.item() * (end - start)
        total_tokens += (end - start)

    avg_loss_nats = total_loss / total_tokens
    # Convert nats per token -> bits per character using the corpus length ratio.
    nats_per_char = avg_loss_nats * total_tokens / len(text)
    return nats_per_char / math.log(2)
