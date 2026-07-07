"""GPT model: shapes, causality, attention weight properties, loss, training sanity."""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from tiny_lm.model import GPT, GPTConfig, cross_entropy_loss


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, ctx_len=8, d_model=30, n_heads=4, n_layers=1)  # 30 % 4 != 0


def test_logits_shape(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    x = mx.zeros((3, tiny_config.ctx_len), dtype=mx.int32)
    logits = tiny_model(x)
    assert logits.shape == (3, tiny_config.ctx_len, tiny_config.vocab_size)


def test_shorter_sequence_ok(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    x = mx.zeros((1, 4), dtype=mx.int32)
    logits = tiny_model(x)
    assert logits.shape == (1, 4, tiny_config.vocab_size)


def test_sequence_beyond_ctx_len_raises(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    x = mx.zeros((1, tiny_config.ctx_len + 1), dtype=mx.int32)
    with pytest.raises(ValueError):
        tiny_model(x)


def test_causality_future_tokens_do_not_affect_past(
    tiny_model: GPT, tiny_config: GPTConfig
) -> None:
    """The defining property of a causal LM: changing token t+1.. leaves logits at t unchanged."""
    T = tiny_config.ctx_len
    v = tiny_config.vocab_size
    base = mx.arange(T).reshape(1, T) % v
    modified = mx.concatenate([base[:, : T // 2], (base[:, T // 2 :] + 1) % v], axis=1)
    logits_base = tiny_model(base.astype(mx.int32))
    logits_mod = tiny_model(modified.astype(mx.int32))
    # positions strictly before the modification point must be identical
    cut = T // 2
    diff = mx.abs(logits_base[:, :cut, :] - logits_mod[:, :cut, :]).max().item()
    assert diff < 1e-5


def test_attention_maps_properties(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    T = 8
    x = mx.zeros((1, T), dtype=mx.int32)
    maps = tiny_model.attention_maps(x)
    assert len(maps) == tiny_config.n_layers
    for m in maps:
        assert m.shape == (tiny_config.n_heads, T, T)
        # rows sum to 1 (softmax)
        row_sums = m.sum(axis=-1)
        assert mx.abs(row_sums - 1.0).max().item() < 1e-5
        # strictly-upper triangle (future positions) is zero
        for t in range(T):
            future = m[:, t, t + 1 :]
            if future.size:
                assert mx.abs(future).max().item() < 1e-7


def test_initial_loss_near_uniform(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    """At init the model should be near-uniform: loss ~= ln(vocab_size)."""
    x = mx.zeros((4, tiny_config.ctx_len), dtype=mx.int32)
    y = mx.zeros((4, tiny_config.ctx_len), dtype=mx.int32)
    loss = cross_entropy_loss(tiny_model(x), y).item()
    assert abs(loss - math.log(tiny_config.vocab_size)) < 1.0


def test_cross_entropy_matches_manual() -> None:
    logits = mx.array([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])  # (1, 2, 3)
    targets = mx.array([[0, 1]])
    expected = -math.log(math.exp(2.0) / (math.exp(2.0) + 2.0))
    got = cross_entropy_loss(logits, targets).item()
    assert abs(got - expected) < 1e-5


def test_model_can_overfit_tiny_sequence(tiny_config: GPTConfig) -> None:
    """Training sanity: a few dozen steps on one repeated sequence must crush the loss."""
    mx.random.seed(0)  # deterministic init so the loss threshold is stable
    model = GPT(tiny_config)
    x = (mx.arange(tiny_config.ctx_len).reshape(1, -1) % tiny_config.vocab_size).astype(mx.int32)
    y = mx.concatenate([x[:, 1:], x[:, :1]], axis=1)

    def loss_fn(m: GPT) -> mx.array:
        return cross_entropy_loss(m(x), y)

    opt = optim.AdamW(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    initial = loss_fn(model).item()
    for _ in range(100):
        _loss, grads = loss_and_grad(model)
        opt.update(model, grads)
        mx.eval(model.parameters())
    final = loss_fn(model).item()
    assert final < initial * 0.3
    assert final < 1.0


def test_param_count_positive(tiny_model: GPT) -> None:
    assert tiny_model.num_params() > 0
