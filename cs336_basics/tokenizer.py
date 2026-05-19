"""
Byte-level BPE tokenizer: training and encode/decode.

Follows the GPT-2 pre-tokenization scheme and implements the full
BPE merge algorithm from scratch, including multiprocessing-accelerated
pre-tokenization and incremental pair-count updates during merging.
"""

from __future__ import annotations

import json
import os
import re as _re
from collections import defaultdict
from itertools import chain
from multiprocessing import Pool, cpu_count
from typing import Iterable, Iterator, Optional

import regex

# ---------------------------------------------------------------------------
# GPT-2 pre-tokenization pattern (Radford et al., 2019).
# Requires the `regex` package for Unicode category support (\p{L}, \p{N}).
# ---------------------------------------------------------------------------
_GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ---------------------------------------------------------------------------
# Pre-tokenization helpers
# ---------------------------------------------------------------------------

def _find_chunk_boundaries(text: str, special_token: str, n_chunks: int) -> list[int]:
    """Return byte-offset boundaries that always fall at the start of special_token."""
    positions = [0]
    idx = 0
    while True:
        pos = text.find(special_token, idx)
        if pos == -1:
            break
        positions.append(pos)
        idx = pos + 1
    positions.append(len(text))

    if len(positions) <= n_chunks + 1:
        return positions

    step = max(1, (len(positions) - 1) // n_chunks)
    boundaries = [positions[i * step] for i in range(n_chunks)]
    boundaries.append(len(text))
    return boundaries


def _pretokenize_chunk(args: tuple) -> dict[tuple[bytes, ...], int]:
    """
    Worker: pre-tokenize a text chunk and return pre-token frequencies.
    Special tokens are stripped before applying the GPT-2 regex so that
    no merge ever crosses a document boundary.
    """
    chunk: str
    special_tokens: list[str]
    chunk, special_tokens = args

    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)

    if special_tokens:
        split_pat = "|".join(_re.escape(t) for t in special_tokens)
        parts = _re.split(split_pat, chunk)
    else:
        parts = [chunk]

    for part in parts:
        for m in regex.finditer(_GPT2_PAT, part):
            word_bytes = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            word_freqs[word_bytes] += 1

    return dict(word_freqs)


def _merge_word_freqs(
    dicts: list[dict[tuple[bytes, ...], int]]
) -> dict[tuple[bytes, ...], int]:
    merged: dict[tuple[bytes, ...], int] = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return dict(merged)


# ---------------------------------------------------------------------------
# Core BPE training
# ---------------------------------------------------------------------------

def _get_pair_counts(
    word_freqs: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq
    return pair_counts


def _apply_merge_to_word(
    word: tuple[bytes, ...], a: bytes, b: bytes
) -> tuple[bytes, ...]:
    ab = a + b
    result: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            result.append(ab)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path:     Path to the plain-text training corpus.
        vocab_size:     Maximum final vocabulary size (including the 256 base
                        byte tokens and all special tokens).
        special_tokens: Strings that are always preserved as single tokens
                        and never split by BPE.

    Returns:
        vocab:   Mapping from integer token ID to the bytes it represents.
        merges:  Ordered list of (token_a, token_b) merge pairs.
    """
    # -- Vocabulary initialisation -------------------------------------------
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    n_merges = vocab_size - len(vocab)
    if n_merges <= 0:
        return vocab, []

    # -- Pre-tokenisation (parallel) -----------------------------------------
    with open(input_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    # Determine the delimiter used for chunking (use the first special token
    # if available, otherwise fall back to a newline/whitespace boundary).
    chunk_token = special_tokens[0] if special_tokens else "\n"
    n_workers = max(1, min(cpu_count(), 8))
    boundaries = _find_chunk_boundaries(text, chunk_token, n_workers)

    chunks = [
        text[boundaries[i]: boundaries[i + 1]]
        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i + 1]
    ]

    if len(chunks) > 1:
        with Pool(len(chunks)) as pool:
            results = pool.map(
                _pretokenize_chunk,
                [(c, special_tokens) for c in chunks],
            )
    else:
        results = [_pretokenize_chunk((text, special_tokens))]

    word_freqs = _merge_word_freqs(results)

    # -- Build initial pair counts -------------------------------------------
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq

    # Index: pair → set of words that contain it (for fast lookup)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word in word_freqs:
        for i in range(len(word) - 1):
            pair_to_words[(word[i], word[i + 1])].add(word)

    # -- Iterative merging ---------------------------------------------------
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(n_merges):
        if not pair_counts:
            break

        # Select the most frequent pair; break ties lexicographically (take max)
        best_pair = max(
            (p for p, c in pair_counts.items() if c > 0),
            key=lambda p: (pair_counts[p], p),
            default=None,
        )
        if best_pair is None or pair_counts[best_pair] <= 0:
            break

        a, b = best_pair
        ab = a + b
        merges.append(best_pair)
        vocab[len(vocab)] = ab

        # Update all words that contain this pair
        affected = list(pair_to_words.get(best_pair, set()))

        for word in affected:
            freq = word_freqs.get(word, 0)
            if freq == 0:
                continue

            new_word = _apply_merge_to_word(word, a, b)

            # Remove old pair contributions
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                pair_counts[p] -= freq
                pair_to_words[p].discard(word)

            # Add new pair contributions
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(new_word)

            del word_freqs[word]
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

        del pair_counts[best_pair]

    return vocab, merges


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _vocab_to_json(vocab: dict[int, bytes]) -> dict[str, str]:
    return {str(k): v.hex() for k, v in vocab.items()}


def _vocab_from_json(data: dict[str, str]) -> dict[int, bytes]:
    return {int(k): bytes.fromhex(v) for k, v in data.items()}


def _merges_to_lines(merges: list[tuple[bytes, bytes]]) -> list[str]:
    return [f"{a.hex()} {b.hex()}" for a, b in merges]


def _merges_from_lines(lines: list[str]) -> list[tuple[bytes, bytes]]:
    result = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        result.append((bytes.fromhex(parts[0]), bytes.fromhex(parts[1])))
    return result


# ---------------------------------------------------------------------------
# Tokenizer class
# ---------------------------------------------------------------------------

class Tokenizer:
    """
    Byte-level BPE tokenizer.

    Encodes arbitrary Unicode text into integer token IDs by:
      1. Splitting on special tokens (preserved verbatim).
      2. Applying the GPT-2 pre-tokenization regex.
      3. Greedily applying BPE merges in creation order.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ) -> None:
        self._vocab: dict[int, bytes] = vocab
        self._merges: list[tuple[bytes, bytes]] = merges

        # Reverse lookup: bytes → token ID
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # Merge rank: (a, b) → index in merge list (lower = applied first)
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # Special-token handling
        self._special_tokens: list[str] = list(special_tokens or [])
        self._special_to_id: dict[str, int] = {}
        self._special_re: Optional[_re.Pattern] = None

        if self._special_tokens:
            for tok in self._special_tokens:
                tok_bytes = tok.encode("utf-8")
                if tok_bytes not in self._bytes_to_id:
                    new_id = len(self._vocab)
                    self._vocab[new_id] = tok_bytes
                    self._bytes_to_id[tok_bytes] = new_id
                self._special_to_id[tok] = self._bytes_to_id[tok_bytes]

            # Build a regex that matches any special token (longest first)
            sorted_toks = sorted(self._special_tokens, key=len, reverse=True)
            pat = "(" + "|".join(_re.escape(t) for t in sorted_toks) + ")"
            self._special_re = _re.compile(pat)

    # ------------------------------------------------------------------
    # Class method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[list[str]] = None,
    ) -> "Tokenizer":
        """Construct a Tokenizer from serialised vocab and merges files."""
        with open(vocab_filepath, "r", encoding="utf-8") as fh:
            vocab = _vocab_from_json(json.load(fh))
        with open(merges_filepath, "r", encoding="utf-8") as fh:
            merges = _merges_from_lines(fh.readlines())
        return cls(vocab, merges, special_tokens)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, vocab_path: str, merges_path: str) -> None:
        """Serialise vocabulary and merge list to disk."""
        with open(vocab_path, "w", encoding="utf-8") as fh:
            json.dump(_vocab_to_json(self._vocab), fh, indent=2)
        with open(merges_path, "w", encoding="utf-8") as fh:
            fh.write("# BPE merges (format: hex_token_a hex_token_b)\n")
            fh.write("\n".join(_merges_to_lines(self._merges)))
            fh.write("\n")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _bpe_encode_word(self, word: tuple[bytes, ...]) -> list[bytes]:
        """
        Apply BPE merges to a single pre-token (sequence of single bytes).
        Uses the merge-rank approach: repeatedly apply the lowest-rank merge.
        """
        tokens: list[bytes] = list(word)
        while len(tokens) > 1:
            best_rank = float("inf")
            best_i = -1
            for i in range(len(tokens) - 1):
                rank = self._merge_rank.get((tokens[i], tokens[i + 1]), float("inf"))  # type: ignore[arg-type]
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i == -1:
                break
            tokens[best_i] = tokens[best_i] + tokens[best_i + 1]
            del tokens[best_i + 1]
        return tokens

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode text that contains no special tokens."""
        ids: list[int] = []
        for m in regex.finditer(_GPT2_PAT, text):
            word = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            for tok in self._bpe_encode_word(word):
                ids.append(self._bytes_to_id[tok])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of integer token IDs."""
        if self._special_re is None:
            return self._encode_chunk(text)

        ids: list[int] = []
        for part in self._special_re.split(text):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            elif part:
                ids.extend(self._encode_chunk(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Memory-efficient encoding of an iterable of strings."""
        for chunk in iterable:
            yield from self.encode(chunk)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join(self._vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)
