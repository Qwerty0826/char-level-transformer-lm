"""
Unit tests for the BPE tokenizer.
"""

import pytest
from cs336_basics.tokenizer import Tokenizer, train_bpe


CORPUS = """\
low low low low low
lower lower widest widest
newest newest newest newest newest newest<|endoftext|>hello world
"""

SPECIAL = ["<|endoftext|>"]


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    """Tokenizer trained on a tiny in-memory corpus."""
    tmp = tmp_path_factory.mktemp("data")
    p = tmp / "corpus.txt"
    p.write_text(CORPUS, encoding="utf-8")
    vocab, merges = train_bpe(str(p), vocab_size=280, special_tokens=SPECIAL)
    return Tokenizer(vocab, merges, SPECIAL)


def test_roundtrip(tiny_tokenizer):
    tok = tiny_tokenizer
    for text in ["hello", "low lower", "newest widest", "Hello, world!"]:
        assert tok.decode(tok.encode(text)) == text


def test_special_token_preserved(tiny_tokenizer):
    tok = tiny_tokenizer
    ids = tok.encode("hello<|endoftext|>world")
    decoded = tok.decode(ids)
    assert decoded == "hello<|endoftext|>world"
    # Special token should map to a single ID
    eos_id = tok._special_to_id["<|endoftext|>"]
    assert eos_id in ids


def test_encode_iterable(tiny_tokenizer):
    tok = tiny_tokenizer
    text = "low lower newest"
    ids_encode = tok.encode(text)
    ids_iter = list(tok.encode_iterable(iter([text])))
    assert ids_encode == ids_iter


def test_from_files_roundtrip(tiny_tokenizer, tmp_path):
    tok = tiny_tokenizer
    vocab_path  = str(tmp_path / "vocab.json")
    merges_path = str(tmp_path / "merges.txt")
    tok.save(vocab_path, merges_path)

    tok2 = Tokenizer.from_files(vocab_path, merges_path, SPECIAL)
    text = "newest widest lower"
    assert tok2.encode(text) == tok.encode(text)


def test_unicode_roundtrip(tiny_tokenizer):
    tok = tiny_tokenizer
    text = "café résumé naïve"
    assert tok.decode(tok.encode(text)) == text


def test_decode_invalid_ids_no_crash(tiny_tokenizer):
    tok = tiny_tokenizer
    # Should replace undecodable bytes with U+FFFD, not crash
    result = tok.decode([0, 1, 2, 3])
    assert isinstance(result, str)


def test_encode_stream_matches_whole_encode(tiny_tokenizer):
    """Chunked streaming must produce the exact IDs of a whole-text encode,
    even when the raw chunk size would split words or the special token."""
    import io

    tok = tiny_tokenizer
    text = (
        "low lower newest<|endoftext|>widest stories here\n"
        "another line<|endoftext|>and a tail without separator"
    )
    whole = tok.encode(text)

    # Sweep pathological chunk sizes — several land mid-word and inside
    # <|endoftext|>, which a naive read/encode loop would shatter.
    for chunk_chars in (1, 3, 5, 7, 11, 64):
        streamed: list[int] = []
        for ids in tok.encode_stream(
            io.StringIO(text), chunk_chars=chunk_chars, boundary="<|endoftext|>",
        ):
            streamed.extend(ids)
        assert streamed == whole, f"mismatch at chunk_chars={chunk_chars}"


def test_encode_stream_newline_fallback(tiny_tokenizer):
    """Without the boundary token in the text, splitting falls back to
    whitespace-run boundaries and must still match the whole-text encode."""
    import io

    tok = tiny_tokenizer
    text = "low lower\nnewest widest\n  indented line\nno trailing newline"
    whole = tok.encode(text)
    for chunk_chars in (2, 6, 13):
        streamed: list[int] = []
        for ids in tok.encode_stream(
            io.StringIO(text), chunk_chars=chunk_chars, boundary="<|endoftext|>",
        ):
            streamed.extend(ids)
        assert streamed == whole, f"mismatch at chunk_chars={chunk_chars}"


def test_constructor_does_not_mutate_caller_vocab():
    vocab = {i: bytes([i]) for i in range(256)}
    before = dict(vocab)
    Tokenizer(vocab, [], special_tokens=["<|endoftext|>"])
    assert vocab == before, "Tokenizer.__init__ must not mutate the vocab passed in"
