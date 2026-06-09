"""
Unit tests for the incremental streaming decoder.

The hard requirement: byte-level BPE vocabularies contain all 256 single
bytes, so the model can emit bytes that are *invalid* UTF-8 (e.g. a lone
continuation byte), not just *incomplete* sequences. The decoder must
replace invalid bytes immediately and keep streaming — a decoder that
buffers them stalls the whole stream until flush.
"""

from cs336_basics.streaming import StreamingDecoder


class _FakeTokenizer:
    """Minimal stand-in exposing the _vocab mapping the decoder reads."""

    def __init__(self, vocab: dict[int, bytes]) -> None:
        self._vocab = vocab


def test_incomplete_multibyte_is_buffered_then_emitted():
    dec = StreamingDecoder(_FakeTokenizer({0: b"\xc3", 1: b"\xa9"}))
    assert dec.feed(0) == ""          # first byte of "é" — wait for the rest
    assert dec.feed(1) == "é"
    assert dec.flush() == ""


def test_invalid_byte_does_not_stall_the_stream():
    """A lone continuation byte must be replaced immediately, and every
    subsequent token must still stream out (regression: the old buffer
    held the invalid byte forever and withheld all later text)."""
    dec = StreamingDecoder(_FakeTokenizer({0: b"\x80", 1: b"hello", 2: b" world"}))
    assert dec.feed(0) == "�"
    assert dec.feed(1) == "hello"
    assert dec.feed(2) == " world"
    assert dec.flush() == ""


def test_invalid_byte_mid_stream():
    dec = StreamingDecoder(_FakeTokenizer({0: b"ab\x80cd"}))
    assert dec.feed(0) == "ab�cd"


def test_flush_replaces_trailing_incomplete_sequence():
    dec = StreamingDecoder(_FakeTokenizer({0: b"\xc3"}))
    assert dec.feed(0) == ""
    assert dec.flush() == "�"


def test_four_byte_codepoint_split_across_tokens():
    # U+1F600 GRINNING FACE = f0 9f 98 80, delivered one byte per token.
    emoji = "😀".encode("utf-8")
    vocab = {i: bytes([b]) for i, b in enumerate(emoji)}
    dec = StreamingDecoder(_FakeTokenizer(vocab))
    pieces = [dec.feed(i) for i in range(4)]
    assert pieces[:3] == ["", "", ""]
    assert pieces[3] == "😀"
    assert dec.flush() == ""
