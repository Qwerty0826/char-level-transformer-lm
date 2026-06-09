"""
Incremental byte-level decoder for streaming generation.

BPE tokens are raw byte sequences; a single token can carry the first
byte of a 4-byte UTF-8 codepoint with the rest delivered later. A naive
per-token decode would produce U+FFFD replacement characters. This
decoder holds incomplete bytes until they're decodable.

Built on ``codecs.getincrementaldecoder`` rather than a hand-rolled
buffer: the stdlib decoder distinguishes a *truncated* multi-byte
sequence (buffer and wait) from an *invalid* byte such as a lone
continuation byte (replace immediately). A hand-rolled "decode the
valid prefix, keep the tail" loop conflates the two and stalls forever
once an invalid byte enters the buffer — and byte-level BPE vocabularies
contain all 256 single bytes, so the model can sample one at any time.

Used by both the FastAPI server (`scripts/serve.py`) and the Gradio
playground. Lives here so neither of those modules has to import the
other.
"""

from __future__ import annotations

import codecs

from cs336_basics.tokenizer import Tokenizer


class StreamingDecoder:
    """Incremental UTF-8 decoder over a stream of BPE token ids."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, token_id: int) -> str:
        """Append one token's bytes; return any newly decodable text."""
        return self._decoder.decode(self._tok._vocab[token_id], final=False)

    def flush(self) -> str:
        """Emit any remaining bytes at end-of-stream (replace if invalid)."""
        return self._decoder.decode(b"", final=True)
