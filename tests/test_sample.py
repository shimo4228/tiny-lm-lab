"""Sampling: greedy determinism, temperature/top-k behavior, length control."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from tiny_lm.model import GPT, GPTConfig, cross_entropy_loss
from tiny_lm.sample import generate
from tiny_lm.tokenizer import CharTokenizer


@pytest.fixture()
def trained(tiny_text: str) -> tuple[GPT, CharTokenizer]:
    """A model quickly overfit on the tiny corpus so sampling is meaningful."""
    tokenizer = CharTokenizer.from_text(tiny_text)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, ctx_len=16, d_model=32, n_heads=2, n_layers=2
    )
    model = GPT(config)
    ids = mx.array([tokenizer.encode(tiny_text[:17])])
    x, y = ids[:, :-1], ids[:, 1:]

    def loss_fn(m: GPT) -> mx.array:
        return cross_entropy_loss(m(x), y)

    opt = optim.AdamW(learning_rate=1e-2)
    step = nn.value_and_grad(model, loss_fn)
    for _ in range(50):
        _loss, grads = step(model)
        opt.update(model, grads)
        mx.eval(model.parameters())
    return model, tokenizer


def test_greedy_is_deterministic(trained: tuple[GPT, CharTokenizer]) -> None:
    model, tokenizer = trained
    a = generate(model, tokenizer, prompt="こんにちは", max_new_tokens=10, temperature=0.0)
    b = generate(model, tokenizer, prompt="こんにちは", max_new_tokens=10, temperature=0.0)
    assert a == b


def test_generate_length(trained: tuple[GPT, CharTokenizer]) -> None:
    model, tokenizer = trained
    prompt = "こんにちは"
    out = generate(model, tokenizer, prompt=prompt, max_new_tokens=12, temperature=0.0)
    assert out.startswith(prompt)
    assert len(out) == len(prompt) + 12


def test_seeded_sampling_is_deterministic(trained: tuple[GPT, CharTokenizer]) -> None:
    model, tokenizer = trained
    a = generate(model, tokenizer, prompt="hello", max_new_tokens=10, temperature=1.0, seed=7)
    b = generate(model, tokenizer, prompt="hello", max_new_tokens=10, temperature=1.0, seed=7)
    assert a == b


def test_top_k_one_equals_greedy(trained: tuple[GPT, CharTokenizer]) -> None:
    model, tokenizer = trained
    greedy = generate(model, tokenizer, prompt="hello", max_new_tokens=10, temperature=0.0)
    topk1 = generate(
        model, tokenizer, prompt="hello", max_new_tokens=10, temperature=1.0, top_k=1, seed=0
    )
    assert greedy == topk1


def test_top_k_larger_than_vocab_is_noop(trained: tuple[GPT, CharTokenizer]) -> None:
    """Regression: MLX silently returns 0.0 for out-of-range negative indices,
    so an unclamped top_k > vocab_size corrupted the truncation threshold."""
    model, tokenizer = trained
    unfiltered = generate(
        model, tokenizer, prompt="hello", max_new_tokens=10, temperature=1.0, top_k=0, seed=3
    )
    oversized = generate(
        model,
        tokenizer,
        prompt="hello",
        max_new_tokens=10,
        temperature=1.0,
        top_k=tokenizer.vocab_size + 100,
        seed=3,
    )
    assert oversized == unfiltered


def test_empty_prompt_rejected(trained: tuple[GPT, CharTokenizer]) -> None:
    model, tokenizer = trained
    with pytest.raises(ValueError):
        generate(model, tokenizer, prompt="", max_new_tokens=5)


def test_negative_max_new_tokens_rejected(trained: tuple[GPT, CharTokenizer]) -> None:
    model, tokenizer = trained
    with pytest.raises(ValueError):
        generate(model, tokenizer, prompt="a", max_new_tokens=-1)
