#!/usr/bin/env python3
"""
OpenAI-compatible REST API server for a trained Transformer LM.

Exposes a FastAPI service speaking the same JSON dialect as
``api.openai.com``, so any OpenAI-SDK client, LangChain, Open WebUI,
SillyTavern, Jan, or curl works as a drop-in client.

Endpoints
---------
GET  /v1/models                  list available models
POST /v1/chat/completions        chat-style request (multi-turn messages)
POST /v1/completions             legacy text-completion request
GET  /health                     liveness probe

Both POST endpoints support ``stream=true`` for token-by-token Server-Sent
Events (SSE) streaming.  KV cache is enabled by default for O(T) decoding.

Sampling parameters honoured (OpenAI names map onto our samplers):
    temperature, top_p, top_k, min_p, presence_penalty (≡ rep penalty),
    max_tokens, stop.

Example
-------
    python scripts/serve.py \\
        --checkpoint checkpoints/tinystories/final.pt \\
        --vocab data/tinystories_vocab.json \\
        --merges data/tinystories_merges.txt \\
        --port 8000

    curl http://localhost:8000/v1/chat/completions \\
         -H 'Content-Type: application/json' \\
         -d '{"model":"transformer-lm",
              "messages":[{"role":"user","content":"Once upon a time"}],
              "stream":false}'
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cs336_basics.model import TransformerLM
from cs336_basics.streaming import StreamingDecoder
from cs336_basics.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# OpenAI request / response schemas (subset we care about)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "transformer-lm"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = 0.0   # mapped to repetition_penalty
    frequency_penalty: Optional[float] = 0.0  # alias for presence_penalty here
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    n: Optional[int] = 1                      # honoured: must be 1


class CompletionRequest(BaseModel):
    model: str = "transformer-lm"
    prompt: str
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    n: Optional[int] = 1


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class _State:
    model: Optional[TransformerLM] = None
    tokenizer: Optional[Tokenizer] = None
    device: str = "cpu"
    eos_id: Optional[int] = None
    model_name: str = "transformer-lm"

STATE = _State()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Transformer LM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_chat(messages: List[ChatMessage]) -> str:
    """
    Flatten a multi-turn message list into a single prompt for a base LM.

    Uses a plain ``Role: text`` format with an open ``Assistant:`` tail so
    the model knows where to continue.  System messages are prepended
    verbatim.  Works reasonably with any base LM regardless of training.
    """
    parts: List[str] = []
    for m in messages:
        if m.role == "system":
            parts.append(m.content.strip())
        elif m.role == "user":
            parts.append(f"User: {m.content.strip()}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content.strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _rep_penalty(req: ChatCompletionRequest | CompletionRequest) -> float:
    """Map OpenAI's presence/frequency penalty (added) onto our multiplicative penalty.

    OpenAI uses additive logit penalties; our model uses Keskar-style
    multiplicative penalty (divide positive logits, multiply negative).
    Translate by ``rep = 1 + max(presence, frequency)`` so 0 ⇒ 1.0 (no
    penalty) and larger ⇒ stronger discouragement.
    """
    add = max(req.presence_penalty or 0.0, req.frequency_penalty or 0.0)
    return 1.0 + max(add, 0.0)


def _sampling_kwargs(req: ChatCompletionRequest | CompletionRequest) -> Dict[str, Any]:
    return dict(
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p,
        top_k=req.top_k if (req.top_k and req.top_k > 0) else None,
        min_p=req.min_p if (req.min_p and req.min_p > 0) else None,
        repetition_penalty=_rep_penalty(req),
        eos_id=STATE.eos_id,
    )


def _stop_hit(text: str, stop: Optional[List[str]]) -> Optional[int]:
    """Return index where any stop sequence first appears, else None."""
    if not stop:
        return None
    earliest = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx >= 0 and (earliest is None or idx < earliest):
            earliest = idx
    return earliest


def _encode_prompt(text: str) -> torch.Tensor:
    assert STATE.tokenizer is not None and STATE.model is not None
    ids = STATE.tokenizer.encode(text)
    if not ids:
        # An empty prompt would crash the model; insert a single space.
        ids = STATE.tokenizer.encode(" ")
    # Truncate to leave room for max_tokens within the context window.
    max_prompt = STATE.model.context_length - 1
    if len(ids) > max_prompt:
        ids = ids[-max_prompt:]
    return torch.tensor([ids], dtype=torch.long, device=STATE.device)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [{
            "id": STATE.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if STATE.model is None:
        raise HTTPException(503, "Model not loaded")
    prompt = _format_chat(req.messages)
    prompt_ids = _encode_prompt(prompt)
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _stream_chat(prompt_ids, req, cmpl_id, created),
            media_type="text/event-stream",
        )

    text, finish_reason, n_completion = _run_generation(prompt_ids, req)
    return {
        "id": cmpl_id,
        "object": "chat.completion",
        "created": created,
        "model": STATE.model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_ids.shape[1],
            "completion_tokens": n_completion,
            "total_tokens": prompt_ids.shape[1] + n_completion,
        },
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    if STATE.model is None:
        raise HTTPException(503, "Model not loaded")
    prompt_ids = _encode_prompt(req.prompt)
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _stream_completion(prompt_ids, req, cmpl_id, created),
            media_type="text/event-stream",
        )

    text, finish_reason, n_completion = _run_generation(prompt_ids, req)
    return {
        "id": cmpl_id,
        "object": "text_completion",
        "created": created,
        "model": STATE.model_name,
        "choices": [{
            "index": 0,
            "text": text,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_ids.shape[1],
            "completion_tokens": n_completion,
            "total_tokens": prompt_ids.shape[1] + n_completion,
        },
    }


# ---------------------------------------------------------------------------
# Non-streaming generation
# ---------------------------------------------------------------------------

def _run_generation(
    prompt_ids: torch.Tensor,
    req: ChatCompletionRequest | CompletionRequest,
) -> tuple[str, str, int]:
    assert STATE.model is not None
    kwargs = _sampling_kwargs(req)
    decoder = StreamingDecoder(STATE.tokenizer)  # type: ignore[arg-type]
    accumulated = ""
    n_completion = 0
    finish_reason = "length"
    for tok in STATE.model.generate_stream(prompt_ids, req.max_tokens or 256, **kwargs):
        n_completion += 1
        accumulated += decoder.feed(tok)
        idx = _stop_hit(accumulated, req.stop)
        if idx is not None:
            accumulated = accumulated[:idx]
            finish_reason = "stop"
            break
        if STATE.eos_id is not None and tok == STATE.eos_id:
            finish_reason = "stop"
            break
    else:
        # Loop completed without break → max_tokens was the cause.
        finish_reason = "length"
    accumulated += decoder.flush()
    return accumulated, finish_reason, n_completion


# ---------------------------------------------------------------------------
# Streaming generation (SSE)
# ---------------------------------------------------------------------------

def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_chat(
    prompt_ids: torch.Tensor,
    req: ChatCompletionRequest,
    cmpl_id: str,
    created: int,
) -> Iterator[str]:
    base = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": STATE.model_name,
    }
    # 1) Initial chunk announces the assistant role.
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    decoder = StreamingDecoder(STATE.tokenizer)  # type: ignore[arg-type]
    kwargs = _sampling_kwargs(req)
    accumulated = ""
    finish_reason = "length"

    for tok in STATE.model.generate_stream(prompt_ids, req.max_tokens or 256, **kwargs):
        piece = decoder.feed(tok)
        if not piece:
            continue
        new_accumulated = accumulated + piece
        idx = _stop_hit(new_accumulated, req.stop)
        if idx is not None:
            keep = new_accumulated[len(accumulated): idx]
            if keep:
                yield _sse({**base, "choices": [{"index": 0, "delta": {"content": keep}, "finish_reason": None}]})
            finish_reason = "stop"
            break
        if STATE.eos_id is not None and tok == STATE.eos_id:
            if piece:
                yield _sse({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
            finish_reason = "stop"
            break

        yield _sse({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
        accumulated = new_accumulated

    tail = decoder.flush()
    if tail:
        yield _sse({**base, "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}]})

    # 2) Terminal chunk with finish_reason.
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
    yield "data: [DONE]\n\n"


def _stream_completion(
    prompt_ids: torch.Tensor,
    req: CompletionRequest,
    cmpl_id: str,
    created: int,
) -> Iterator[str]:
    base = {
        "id": cmpl_id,
        "object": "text_completion",
        "created": created,
        "model": STATE.model_name,
    }
    decoder = StreamingDecoder(STATE.tokenizer)  # type: ignore[arg-type]
    kwargs = _sampling_kwargs(req)
    accumulated = ""
    finish_reason = "length"

    for tok in STATE.model.generate_stream(prompt_ids, req.max_tokens or 256, **kwargs):
        piece = decoder.feed(tok)
        if not piece:
            continue
        new_accumulated = accumulated + piece
        idx = _stop_hit(new_accumulated, req.stop)
        if idx is not None:
            keep = new_accumulated[len(accumulated): idx]
            if keep:
                yield _sse({**base, "choices": [{"index": 0, "text": keep, "logprobs": None, "finish_reason": None}]})
            finish_reason = "stop"
            break
        if STATE.eos_id is not None and tok == STATE.eos_id:
            if piece:
                yield _sse({**base, "choices": [{"index": 0, "text": piece, "logprobs": None, "finish_reason": None}]})
            finish_reason = "stop"
            break
        yield _sse({**base, "choices": [{"index": 0, "text": piece, "logprobs": None, "finish_reason": None}]})
        accumulated = new_accumulated

    tail = decoder.flush()
    if tail:
        yield _sse({**base, "choices": [{"index": 0, "text": tail, "logprobs": None, "finish_reason": None}]})

    yield _sse({**base, "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": finish_reason}]})
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible API server")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab",      required=True)
    p.add_argument("--merges",     required=True)
    p.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])

    # Model shape (must match the checkpoint)
    p.add_argument("--vocab_size",     type=int, default=10_000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model",        type=int, default=512)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--num_heads",      type=int, default=16)
    p.add_argument("--num_kv_heads",   type=int, default=None)
    p.add_argument("--d_ff",           type=int, default=1344)
    p.add_argument("--theta",          type=float, default=10_000.0)
    p.add_argument("--no_tie_weights", action="store_true")

    p.add_argument("--model_name", default="transformer-lm",
                   help="ID returned by /v1/models and echoed in responses")
    p.add_argument("--device", default=None)
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--port",   type=int, default=8000)
    return p.parse_args()


def load_model(args: argparse.Namespace) -> None:
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[serve] Loading model on {device} ...")

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    eos_id = tokenizer._special_to_id.get("<|endoftext|>")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        tie_weights=not args.no_tie_weights,
        device=device,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.device = device
    STATE.eos_id = eos_id
    STATE.model_name = args.model_name

    n_params = model.num_parameters()
    print(f"[serve] Loaded checkpoint (step {ckpt.get('iteration', '?')}, {n_params:,} params)")
    print(f"[serve] Ready at  /v1/chat/completions  /v1/completions  /v1/models  /health")


def main() -> None:
    import uvicorn
    args = parse_args()
    load_model(args)
    print(f"[serve] Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
