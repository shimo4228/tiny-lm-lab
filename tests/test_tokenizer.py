"""CharTokenizer: vocab construction, round-trip, unknown handling, persistence."""

from pathlib import Path

import pytest

from tiny_lm.tokenizer import UNK_CHAR, CharTokenizer


def test_vocab_built_from_unique_chars(tiny_text: str) -> None:
    tok = CharTokenizer.from_text(tiny_text)
    # vocab = unique chars + the <unk> slot
    assert tok.vocab_size == len(set(tiny_text)) + 1


def test_encode_decode_round_trip(tokenizer: CharTokenizer, tiny_text: str) -> None:
    ids = tokenizer.encode(tiny_text)
    assert tokenizer.decode(ids) == tiny_text


def test_encode_returns_ints_in_vocab_range(tokenizer: CharTokenizer, tiny_text: str) -> None:
    ids = tokenizer.encode(tiny_text)
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


def test_unknown_char_maps_to_unk(tokenizer: CharTokenizer) -> None:
    ids = tokenizer.encode("★")  # not in the tiny corpus
    assert ids == [tokenizer.unk_id]
    assert tokenizer.decode(ids) == UNK_CHAR


def test_empty_text() -> None:
    tok = CharTokenizer.from_text("ab")
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_from_text_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError):
        CharTokenizer.from_text("")


def test_save_load_round_trip(tokenizer: CharTokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.vocab_size == tokenizer.vocab_size
    text = "こんにちは world"
    assert loaded.encode(text) == tokenizer.encode(text)


def test_encoding_is_deterministic(tiny_text: str) -> None:
    a = CharTokenizer.from_text(tiny_text)
    b = CharTokenizer.from_text(tiny_text)
    assert a.encode(tiny_text) == b.encode(tiny_text)
