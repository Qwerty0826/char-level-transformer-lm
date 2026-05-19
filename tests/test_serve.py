"""
End-to-end tests for the OpenAI-compatible API server.

Loads a tiny untrained model into the server's global state, then drives
both endpoints through FastAPI's in-process TestClient — no network, no
external process.  Covers:

  - /health and /v1/models
  - /v1/chat/completions  (streaming + non-streaming)
  - /v1/completions       (streaming + non-streaming)
  - StreamingDecoder UTF-8 buffering
  - stop sequence honoured
  - OpenAI SSE format and [DONE] terminator
"""

import json

import pytest
import torch

# Skip the whole file if FastAPI deps aren't installed.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer, train_bpe
from scripts import serve
from scripts.serve import StreamingDecoder, _format_chat, _stop_hit


# ---------------------------------------------------------------------------
# Tokenizer + tiny model fixtures
# ---------------------------------------------------------------------------

CORPUS = "the quick brown fox jumps over the lazy dog. " * 20 + "<|endoftext|>"


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serve")
    p = tmp / "corpus.txt"
    p.write_text(CORPUS, encoding="utf-8")
    vocab, merges = train_bpe(str(p), vocab_size=300, special_tokens=["<|endoftext|>"])
    return Tokenizer(vocab, merges, ["<|endoftext|>"])


@pytest.fixture(scope="module")
def loaded_app(tokenizer):
    """Populate serve.STATE with a tiny model and yield the FastAPI app."""
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        d_model=32, num_layers=1, num_heads=2, d_ff=64,
    )
    model.eval()

    serve.STATE.model = model
    serve.STATE.tokenizer = tokenizer
    serve.STATE.device = "cpu"
    serve.STATE.eos_id = tokenizer._special_to_id.get("<|endoftext|>")
    serve.STATE.model_name = "test-lm"

    yield serve.app

    serve.STATE.model = None
    serve.STATE.tokenizer = None


@pytest.fixture
def client(loaded_app):
    return TestClient(loaded_app)


# ---------------------------------------------------------------------------
# StreamingDecoder unit tests (no server needed)
# ---------------------------------------------------------------------------

def test_streaming_decoder_handles_partial_utf8(tokenizer):
    """An incomplete multi-byte sequence must be buffered until completable."""
    dec = StreamingDecoder(tokenizer)
    # Feed a token whose bytes do not form valid UTF-8 in isolation by
    # constructing a fake one-byte token (the byte 0xC3 — start of a 2-byte char).
    # First 256 token IDs are the raw bytes, so id 0xC3 = the byte b"\xc3".
    out_a = dec.feed(0xC3)
    assert out_a == ""           # buffered, not yet decodable
    out_b = dec.feed(0xA9)       # 0xA9 completes "é"
    assert out_b == "é"


def test_streaming_decoder_flush(tokenizer):
    dec = StreamingDecoder(tokenizer)
    dec.feed(0xC3)               # buffer holds an incomplete byte
    tail = dec.flush()           # should emit U+FFFD replacement
    assert tail == "�"


# ---------------------------------------------------------------------------
# Chat / completion formatting helpers
# ---------------------------------------------------------------------------

def test_format_chat_multi_turn():
    from scripts.serve import ChatMessage
    msgs = [
        ChatMessage(role="system",    content="You are concise."),
        ChatMessage(role="user",      content="Hi."),
        ChatMessage(role="assistant", content="Hello!"),
        ChatMessage(role="user",      content="Tell me a story."),
    ]
    prompt = _format_chat(msgs)
    assert prompt.startswith("You are concise.")
    assert "User: Hi." in prompt
    assert "Assistant: Hello!" in prompt
    assert prompt.endswith("Assistant:")


def test_stop_hit_finds_earliest():
    # "and" appears at index 17 ("once upon a time "), well before "end" at 27.
    assert _stop_hit("once upon a time and then end here", ["end", "and"]) == 17
    assert _stop_hit("hello world", ["xyz"]) is None
    assert _stop_hit("anything", []) is None


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models_listing(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "test-lm"


def test_chat_non_streaming(client):
    r = client.post("/v1/chat/completions", json={
        "model": "test-lm",
        "messages": [{"role": "user", "content": "Tell me a story."}],
        "max_tokens": 8,
        "temperature": 1.0,
        "stream": False,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["usage"]["completion_tokens"] >= 1
    assert body["choices"][0]["finish_reason"] in ("stop", "length")


def test_completion_non_streaming(client):
    r = client.post("/v1/completions", json={
        "model": "test-lm",
        "prompt": "the quick brown",
        "max_tokens": 6,
        "stream": False,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert isinstance(body["choices"][0]["text"], str)


def _parse_sse(stream_text: str) -> list:
    chunks = []
    for line in stream_text.split("\n"):
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                chunks.append("DONE")
            else:
                chunks.append(json.loads(payload))
    return chunks


def test_chat_streaming_sse_format(client):
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "test-lm",
        "messages": [{"role": "user", "content": "Tell me a story."}],
        "max_tokens": 8,
        "stream": True,
    }) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in r.iter_text())

    events = _parse_sse(body)
    assert events[-1] == "DONE"
    # First content chunk must carry role=assistant.
    assert events[0]["choices"][0]["delta"].get("role") == "assistant"
    # Some content chunks should arrive (small model, but at least empty deltas).
    finish_reasons = [e["choices"][0].get("finish_reason") for e in events if e != "DONE"]
    assert any(fr in ("stop", "length") for fr in finish_reasons)


def test_completion_streaming(client):
    with client.stream("POST", "/v1/completions", json={
        "model": "test-lm",
        "prompt": "the quick brown",
        "max_tokens": 6,
        "stream": True,
    }) as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())

    events = _parse_sse(body)
    assert events[-1] == "DONE"
    # Completion chunks must carry "text" in choices (not "delta").
    text_chunks = [e for e in events if e != "DONE" and "text" in e["choices"][0]]
    assert len(text_chunks) >= 1


def test_stop_sequence_honored(client):
    r = client.post("/v1/completions", json={
        "model": "test-lm",
        "prompt": "the quick brown fox",
        "max_tokens": 64,
        "stop": ["fox"],          # very common token in our toy corpus
        "stream": False,
    })
    assert r.status_code == 200
    body = r.json()
    # If "fox" appears, output must be truncated before it.
    assert "fox" not in body["choices"][0]["text"]
