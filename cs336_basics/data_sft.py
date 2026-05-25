"""
SFT data: chat-template formatting + padded batching.

The chat template is the simplest format that maps cleanly to single-turn
instruction following — what TinyStoriesInstruct provides:

    <|user|>{prompt}<|endoftext|><|assistant|>{response}<|endoftext|>

The loss mask is 1 only on the *response* tokens (the assistant's output
and its closing <|endoftext|>), 0 on the prompt and on the leading
<|assistant|> marker. The marker is part of the conditioning, not the
target — the model is trained to start generating after it.

Autoregressive shift convention:
    input_ids[t]   = sequence[t]
    target_ids[t]  = sequence[t+1]
    loss_mask[t]   = 1 iff position t+1 is a response token

(Matches scripts/train.py's get_batch convention.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from cs336_basics.tokenizer import Tokenizer


USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
SYSTEM_TAG = "<|system|>"
EOT = "<|endoftext|>"


@dataclass
class Message:
    role: str       # "user" | "assistant" | "system"
    content: str


def format_sft_example(
    messages: Sequence[Message],
    tokenizer: Tokenizer,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """
    Encode a conversation into (token_ids, token_mask) where token_mask[i]
    is 1 iff token i is a response (assistant content) token.

    The leading <|assistant|> marker is NOT in the mask — the model conditions
    on it. The trailing <|endoftext|> closing the response IS in the mask —
    we want the model to learn to stop.
    """
    ids: list[int] = []
    mask: list[int] = []

    def append(text: str, is_response: bool) -> None:
        chunk_ids = tokenizer.encode(text)
        ids.extend(chunk_ids)
        mask.extend([1 if is_response else 0] * len(chunk_ids))

    for msg in messages:
        if msg.role == "system":
            append(SYSTEM_TAG + msg.content + EOT, is_response=False)
        elif msg.role == "user":
            append(USER_TAG + msg.content + EOT, is_response=False)
        elif msg.role == "assistant":
            # The marker is conditioning context, not a target.
            append(ASSISTANT_TAG, is_response=False)
            # Response body + closing EOT are in the loss.
            append(msg.content + EOT, is_response=True)
        else:
            raise ValueError(f"Unknown role: {msg.role!r}")

    return ids[:max_length], mask[:max_length]


def pad_and_collate(
    examples: list[tuple[list[int], list[int]]],
    max_length: int,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """
    Pad a batch of (token_ids, token_mask) pairs to ``max_length`` and apply
    the autoregressive shift.

    Returns a dict with:
      input_ids:      (B, L-1)  long
      target_ids:     (B, L-1)  long
      loss_mask:      (B, L-1)  float  — 1 on positions whose target is a response token
      attention_mask: (B, L-1)  bool   — 1 on real tokens (not padding)
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2 to shift inputs/targets.")

    B = len(examples)
    L = max_length
    input_ids     = torch.full((B, L - 1), pad_id, dtype=torch.long)
    target_ids    = torch.full((B, L - 1), pad_id, dtype=torch.long)
    loss_mask     = torch.zeros((B, L - 1), dtype=torch.float)
    attention_mask = torch.zeros((B, L - 1), dtype=torch.bool)

    for i, (tok_ids, tok_mask) in enumerate(examples):
        # Truncate to max_length BEFORE the shift so we get exactly L tokens.
        tok_ids  = tok_ids[:L]
        tok_mask = tok_mask[:L]
        n = len(tok_ids)
        if n < 2:
            continue

        # input[t] = seq[t], target[t] = seq[t+1] for t in [0, n-2].
        ids_t  = torch.tensor(tok_ids,  dtype=torch.long)
        mask_t = torch.tensor(tok_mask, dtype=torch.float)
        input_ids[i,  :n-1] = ids_t[:-1]
        target_ids[i, :n-1] = ids_t[1:]
        # Loss at position t is on target[t] = seq[t+1]; include it iff seq[t+1]
        # is a response token.
        loss_mask[i,  :n-1] = mask_t[1:]
        # Attention mask covers all real input positions [0, n-2].
        attention_mask[i, :n-1] = True

    return dict(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
    )
