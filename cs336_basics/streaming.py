"""
Incremental byte-level decoder for streaming generation.

BPE tokens are raw byte sequences; a single token can carry the first
byte of a 4-byte UTF-8 codepoint with the rest delivered later. A naive
per-token decode would produce U+FFFD replacement characters. This
buffer holds incomplete bytes until they're decodable.

Used by both the FastAPI server (`scripts/serve.py`) and the Gradio
playground (`scripts/playground.py`). Lives here so neither of those
modules has to import the other.
"""

from __future__ import annotations

from cs336_basics.tokenizer import Tokenizer


class StreamingDecoder:
    """Incremental UTF-8 decoder over a stream of BPE token ids."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self._buffer: bytes = b""

    def feed(self, token_id: int) -> str:
        """Append one token's bytes; return any newly decodable text."""
        self._buffer += self._tok._vocab[token_id]
        try:
            text = self._buffer.decode("utf-8")
            self._buffer = b""
            return text
        except UnicodeDecodeError as e:
            # Decode the prefix that is valid; keep the incomplete tail.
            text = self._buffer[: e.start].decode("utf-8")
            self._buffer = self._buffer[e.start :]
            return text

    def flush(self) -> str:
        """Emit any remaining bytes at end-of-stream (replace if invalid)."""
        if not self._buffer:
            return ""
        text = self._buffer.decode("utf-8", errors="replace")
        self._buffer = b""
        return text
