"""
Smoke tests for the SFT data pipeline.

Critical invariants:
1. The loss mask covers ONLY response tokens — not the assistant marker,
   not the user prompt, not padding.
2. After the autoregressive shift, loss_mask[t] = 1 iff target_ids[t] is
   a response token.
3. The model can be called on (input_ids, attention_mask) and the masked
   loss is a finite scalar.
"""

import pytest
import torch

from cs336_basics.data_sft import (
    Message,
    USER_TAG,
    ASSISTANT_TAG,
    EOT,
    format_sft_example,
    pad_and_collate,
)
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer, train_bpe
from cs336_basics.training import masked_cross_entropy_loss


# ---- A tiny tokenizer trained on a synthetic corpus, with chat specials ----


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    """Train a tiny BPE tokenizer with chat specials so tests are self-contained."""
    corpus_dir = tmp_path_factory.mktemp("sft")
    corpus = corpus_dir / "tiny.txt"
    # Use repeated text so BPE produces interesting merges; include the EOT.
    text = ("Hello world. The fox jumps over the lazy dog. " * 20 + EOT + "\n") * 5
    corpus.write_text(text, encoding="utf-8")

    vocab, merges = train_bpe(
        input_path=str(corpus),
        vocab_size=400,
        special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"],
    )
    return Tokenizer(vocab, merges, ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"])


def test_response_only_in_loss(tokenizer):
    messages = [
        Message(role="user",      content="Tell me a story."),
        Message(role="assistant", content="Once upon a time."),
    ]
    ids, mask = format_sft_example(messages, tokenizer, max_length=256)

    assert len(ids) == len(mask)
    # The mask must contain only 0s and 1s.
    assert set(mask) <= {0, 1}

    # The first token must be the <|user|> marker — fully masked.
    user_id = tokenizer.encode(USER_TAG)[0]
    assert ids[0] == user_id
    assert mask[0] == 0

    # The <|assistant|> marker must NOT be in the loss.
    asst_id = tokenizer.encode(ASSISTANT_TAG)[0]
    asst_pos = ids.index(asst_id)
    assert mask[asst_pos] == 0, "assistant marker should not be in loss"

    # All positions after the marker (the response body + closing EOT) ARE in the loss.
    for j in range(asst_pos + 1, len(ids)):
        assert mask[j] == 1, f"position {j} ({ids[j]}) should be in loss"


def test_collator_shifts_correctly(tokenizer):
    messages = [
        Message(role="user",      content="Hi."),
        Message(role="assistant", content="Hello."),
    ]
    example = format_sft_example(messages, tokenizer, max_length=64)
    pad_id = tokenizer.encode(EOT)[0]

    batch = pad_and_collate([example, example], max_length=64, pad_id=pad_id)
    assert batch["input_ids"].shape == (2, 63)
    assert batch["target_ids"].shape == (2, 63)
    assert batch["loss_mask"].shape == (2, 63)
    assert batch["attention_mask"].shape == (2, 63)

    # input[t] = ids[t], target[t] = ids[t+1] — verify on the first row.
    ids, _ = example
    n = len(ids)
    assert torch.equal(batch["input_ids"][0, :n-1], torch.tensor(ids[:-1]))
    assert torch.equal(batch["target_ids"][0, :n-1], torch.tensor(ids[1:]))

    # loss_mask[t] = 1 iff target[t] (= ids[t+1]) was tagged as response.
    _, src_mask = example
    expected = torch.tensor(src_mask[1:], dtype=torch.float)
    assert torch.equal(batch["loss_mask"][0, :n-1], expected)

    # Padding positions have attention_mask = False.
    assert batch["attention_mask"][0, n-1:].sum() == 0


def test_end_to_end_loss_finite(tokenizer):
    """Wire the SFT batch through a tiny model with masked CE — must produce a finite scalar."""
    torch.manual_seed(0)
    vocab_size = tokenizer.vocab_size
    model = TransformerLM(
        vocab_size=vocab_size, context_length=64,
        d_model=32, num_layers=2, num_heads=4,
        num_kv_heads=2, d_ff=64,
        tie_weights=False,
    )

    messages = [
        Message(role="user",      content="Tell me a story."),
        Message(role="assistant", content="The fox jumps over the lazy dog."),
    ]
    example = format_sft_example(messages, tokenizer, max_length=64)
    pad_id = tokenizer.encode(EOT)[0]
    batch = pad_and_collate([example], max_length=64, pad_id=pad_id)

    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    loss = masked_cross_entropy_loss(logits, batch["target_ids"], batch["loss_mask"])
    assert torch.isfinite(loss), f"loss is non-finite: {loss}"
    loss.backward()  # gradient flows cleanly through masked positions
    grad_norm = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(grad_norm), "gradients are non-finite"
