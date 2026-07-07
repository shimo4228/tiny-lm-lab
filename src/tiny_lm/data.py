"""Corpus loading, markdown cleanup, split, and deterministic batching."""

from __future__ import annotations

import random
import re
from pathlib import Path

import mlx.core as mx

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```.*?^```[ \t]*$\n?", re.DOTALL | re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def clean_markdown(text: str) -> str:
    """Strip YAML frontmatter, fenced code blocks, and image links; keep prose."""
    text = _FRONTMATTER_RE.sub("", text)
    text = _CODE_FENCE_RE.sub("", text)
    text = _IMAGE_RE.sub("", text)
    return text


def load_corpus(root: Path | str, pattern: str = "**/*.md") -> str:
    """Concatenate cleaned text of all markdown files under `root`."""
    root = Path(root)
    files = sorted(root.glob(pattern))
    parts = [clean_markdown(f.read_text(encoding="utf-8")) for f in files]
    corpus = "\n\n".join(p.strip() for p in parts if p.strip())
    if not corpus:
        raise ValueError(f"no markdown content found under {root}")
    return corpus


def train_val_split(ids: list[int], val_fraction: float) -> tuple[list[int], list[int]]:
    """Contiguous (no shuffle) split so validation text stays coherent."""
    if not 0.0 < val_fraction <= 0.5:
        raise ValueError(f"val_fraction must be in (0, 0.5], got {val_fraction}")
    cut = int(len(ids) * (1.0 - val_fraction))
    return ids[:cut], ids[cut:]


def get_batch(
    data: mx.array, batch_size: int, ctx_len: int, seed: int
) -> tuple[mx.array, mx.array]:
    """Sample `batch_size` random windows; y is x shifted one position left."""
    n = data.shape[0]
    if n < ctx_len + 1:
        raise ValueError(f"data has {n} tokens; needs at least ctx_len+1 = {ctx_len + 1}")
    rng = random.Random(seed)
    starts = [rng.randrange(0, n - ctx_len) for _ in range(batch_size)]
    x = mx.stack([data[s : s + ctx_len] for s in starts])
    y = mx.stack([data[s + 1 : s + ctx_len + 1] for s in starts])
    return x, y
