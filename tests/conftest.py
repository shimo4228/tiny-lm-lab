"""Shared fixtures: a tiny corpus and a tiny model config for fast tests."""

import pytest

from tiny_lm.model import GPT, GPTConfig
from tiny_lm.tokenizer import CharTokenizer

TINY_TEXT = "こんにちは世界。hello world.\n" * 20


@pytest.fixture()
def tiny_text() -> str:
    return TINY_TEXT


@pytest.fixture()
def tokenizer(tiny_text: str) -> CharTokenizer:
    return CharTokenizer.from_text(tiny_text)


@pytest.fixture()
def tiny_config(tokenizer: CharTokenizer) -> GPTConfig:
    return GPTConfig(
        vocab_size=tokenizer.vocab_size,
        ctx_len=16,
        d_model=32,
        n_heads=2,
        n_layers=2,
    )


@pytest.fixture()
def tiny_model(tiny_config: GPTConfig) -> GPT:
    return GPT(tiny_config)
