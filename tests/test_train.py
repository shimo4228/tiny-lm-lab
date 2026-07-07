"""Training loop: loss decreases on real-ish text, checkpoint save/load round-trip."""

from pathlib import Path

import mlx.core as mx
import pytest

from tiny_lm.model import GPTConfig
from tiny_lm.tokenizer import CharTokenizer
from tiny_lm.train import TrainResult, load_checkpoint, run_training


def _train_tiny(tmp_path: Path, tiny_text: str, steps: int = 30) -> TrainResult:
    tokenizer = CharTokenizer.from_text(tiny_text)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        ctx_len=16,
        d_model=32,
        n_heads=2,
        n_layers=2,
    )
    return run_training(
        text=tiny_text,
        tokenizer=tokenizer,
        config=config,
        steps=steps,
        batch_size=4,
        learning_rate=3e-3,
        val_fraction=0.1,
        eval_every=10,
        seed=0,
        out_dir=tmp_path / "ckpt",
    )


def test_loss_decreases(tmp_path: Path, tiny_text: str) -> None:
    result = _train_tiny(tmp_path, tiny_text, steps=40)
    assert len(result.train_losses) == 40
    first5 = sum(result.train_losses[:5]) / 5
    last5 = sum(result.train_losses[-5:]) / 5
    assert last5 < first5


def test_val_loss_recorded(tmp_path: Path, tiny_text: str) -> None:
    result = _train_tiny(tmp_path, tiny_text, steps=30)
    assert len(result.val_losses) == 3  # eval_every=10 -> steps 10, 20, 30
    assert all(v > 0 for _, v in result.val_losses)


def test_checkpoint_files_written(tmp_path: Path, tiny_text: str) -> None:
    result = _train_tiny(tmp_path, tiny_text)
    out = result.out_dir
    assert (out / "model.safetensors").exists()
    assert (out / "tokenizer.json").exists()
    assert (out / "config.json").exists()
    assert (out / "history.json").exists()


def test_checkpoint_round_trip_same_logits(tmp_path: Path, tiny_text: str) -> None:
    result = _train_tiny(tmp_path, tiny_text)
    model, tokenizer, config = load_checkpoint(result.out_dir)
    assert tokenizer.vocab_size == config.vocab_size
    x = mx.array([tokenizer.encode(tiny_text[: config.ctx_len])])
    logits_loaded = model(x)
    logits_trained = result.model(x)
    assert mx.abs(logits_loaded - logits_trained).max().item() < 1e-6


def test_too_short_val_split_fails_fast(tmp_path: Path, tiny_text: str) -> None:
    """Regression: a val split shorter than ctx_len+1 used to crash at the first
    eval step (after minutes of training), losing the run with no checkpoint.
    It must be rejected before the training loop starts."""
    tokenizer = CharTokenizer.from_text(tiny_text)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        ctx_len=128,  # tiny_text has ~560 chars -> val split of 10% is ~56 < 129
        d_model=32,
        n_heads=2,
        n_layers=2,
    )
    with pytest.raises(ValueError, match="validation"):
        run_training(
            text=tiny_text,
            tokenizer=tokenizer,
            config=config,
            steps=5,
            batch_size=2,
            learning_rate=1e-3,
            val_fraction=0.1,
            eval_every=100,
            seed=0,
            out_dir=tmp_path / "ckpt",
        )


def test_training_is_seed_reproducible(tmp_path: Path, tiny_text: str) -> None:
    """Same seed -> same batch order and near-identical losses.

    Bitwise equality does not hold on Metal GPU (non-deterministic reduction
    order), so compare with a tolerance instead.
    """
    r1 = _train_tiny(tmp_path / "a", tiny_text, steps=10)
    r2 = _train_tiny(tmp_path / "b", tiny_text, steps=10)
    assert r1.train_losses == pytest.approx(r2.train_losses, rel=1e-3)
