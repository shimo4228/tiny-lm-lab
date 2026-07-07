"""Training loop and checkpoint I/O, plus a CLI entry point.

Usage:
    uv run python -m tiny_lm.train --corpus <dir> --steps 3000 --out runs/zenn
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from tiny_lm.data import get_batch, load_corpus, train_val_split
from tiny_lm.model import GPT, GPTConfig, cross_entropy_loss
from tiny_lm.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    model: GPT
    train_losses: list[float]
    val_losses: list[tuple[int, float]]  # (step, loss)
    out_dir: Path


def _estimate_val_loss(
    model: GPT, val_data: mx.array, batch_size: int, ctx_len: int, seed: int, iters: int = 8
) -> float:
    losses = []
    for i in range(iters):
        x, y = get_batch(val_data, batch_size, ctx_len, seed=seed + i)
        losses.append(cross_entropy_loss(model(x), y).item())
    return sum(losses) / len(losses)


def run_training(
    text: str,
    tokenizer: CharTokenizer,
    config: GPTConfig,
    steps: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    eval_every: int,
    seed: int,
    out_dir: Path | str,
) -> TrainResult:
    mx.random.seed(seed)
    ids = tokenizer.encode(text)
    train_ids, val_ids = train_val_split(ids, val_fraction)
    # fail fast: a too-short split would otherwise crash at the first eval,
    # after the training time is already spent, with no checkpoint written
    needed = config.ctx_len + 1
    if len(train_ids) < needed:
        raise ValueError(f"train split has {len(train_ids)} tokens; needs >= {needed}")
    if len(val_ids) < needed:
        raise ValueError(
            f"validation split has {len(val_ids)} tokens; needs >= {needed} "
            f"(reduce ctx_len, grow the corpus, or raise val_fraction)"
        )
    train_data = mx.array(train_ids)
    val_data = mx.array(val_ids)

    model = GPT(config)
    optimizer = optim.AdamW(learning_rate=learning_rate)

    def loss_fn(m: GPT, x: mx.array, y: mx.array) -> mx.array:
        return cross_entropy_loss(m(x), y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    train_losses: list[float] = []
    val_losses: list[tuple[int, float]] = []
    started = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, batch_size, config.ctx_len, seed=seed * 1_000_003 + step)
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        train_losses.append(loss.item())
        if step % eval_every == 0:
            val_loss = _estimate_val_loss(model, val_data, batch_size, config.ctx_len, seed)
            val_losses.append((step, val_loss))
            elapsed = time.time() - started
            logger.info(
                "step %d/%d train %.4f val %.4f (%.1fs)",
                step,
                steps,
                train_losses[-1],
                val_loss,
                elapsed,
            )

    out = save_checkpoint(Path(out_dir), model, tokenizer, config, train_losses, val_losses)
    return TrainResult(model=model, train_losses=train_losses, val_losses=val_losses, out_dir=out)


def save_checkpoint(
    out_dir: Path,
    model: GPT,
    tokenizer: CharTokenizer,
    config: GPTConfig,
    train_losses: list[float],
    val_losses: list[tuple[int, float]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "model.safetensors"), dict(tree_flatten(model.parameters())))
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    history = {"train_losses": train_losses, "val_losses": val_losses}
    (out_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    return out_dir


def load_checkpoint(ckpt_dir: Path | str) -> tuple[GPT, CharTokenizer, GPTConfig]:
    ckpt_dir = Path(ckpt_dir)
    config = GPTConfig(**json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8")))
    tokenizer = CharTokenizer.load(ckpt_dir / "tokenizer.json")
    model = GPT(config)
    weights = mx.load(str(ckpt_dir / "model.safetensors"))
    if not isinstance(weights, dict):
        raise ValueError(f"unexpected checkpoint format in {ckpt_dir}")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model, tokenizer, config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a char-level GPT on a markdown corpus")
    parser.add_argument("--corpus", type=Path, required=True, help="directory of .md files")
    parser.add_argument("--out", type=Path, required=True, help="checkpoint output directory")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=0, help="cap corpus size (0 = no cap)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    text = load_corpus(args.corpus)
    if args.max_chars > 0:
        text = text[: args.max_chars]
    tokenizer = CharTokenizer.from_text(text)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        ctx_len=args.ctx_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    logger.info("corpus: %d chars, vocab: %d", len(text), tokenizer.vocab_size)
    result = run_training(
        text=text,
        tokenizer=tokenizer,
        config=config,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        eval_every=args.eval_every,
        seed=args.seed,
        out_dir=args.out,
    )
    model_params = result.model.num_params()
    logger.info("done: %d params, checkpoint at %s", model_params, result.out_dir)


if __name__ == "__main__":
    main()
