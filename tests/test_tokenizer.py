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
