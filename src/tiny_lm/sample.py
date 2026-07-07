"""Autoregressive text generation from a trained checkpoint.

Usage:
    uv run python -m tiny_lm.sample --ckpt runs/zenn --prompt "エージェント" --tokens 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from tiny_lm.model import GPT
from tiny_lm.tokenizer import CharTokenizer
from tiny_lm.train import load_checkpoint


def generate(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 0,
    seed: int = 0,
) -> str:
    """Sample continuation of `prompt`. temperature=0 means greedy argmax."""
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    key = mx.random.key(seed)
    ids = list(tokenizer.encode(prompt))
    for _ in range(max_new_tokens):
        window = ids[-model.config.ctx_len :]
        logits = model(mx.array([window]))[0, -1, :]  # next-token logits
        if temperature <= 0.0:
            next_id = int(mx.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_k > 0:
                # keep only the k largest logits; mask the rest to -inf.
                # clamp k first: an out-of-range negative index does NOT raise in
                # MLX — it silently yields 0.0 and would corrupt the threshold
                k = min(top_k, int(logits.size))
                kth = mx.sort(logits)[-k]
                logits = mx.where(logits < kth, mx.array(-mx.inf), logits)
            key, subkey = mx.random.split(key)
            next_id = int(mx.random.categorical(logits, key=subkey).item())
        ids.append(next_id)
    return tokenizer.decode(ids)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained tiny-lm checkpoint")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    model, tokenizer, _config = load_checkpoint(args.ckpt)
    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
