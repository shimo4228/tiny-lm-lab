"""Corpus loading, markdown cleanup, split, and deterministic batching."""

from pathlib import Path

import mlx.core as mx
import pytest

from tiny_lm.data import clean_markdown, get_batch, load_corpus, train_val_split

MD_WITH_NOISE = """---
title: テスト記事
emoji: "🌊"
---

# 見出し

本文の段落です。attention の話。

```python
print("code should be dropped")
```

![図1](/images/fig1.png)

次の段落。
"""


def test_clean_markdown_strips_frontmatter() -> None:
    cleaned = clean_markdown(MD_WITH_NOISE)
    assert "title:" not in cleaned
    assert "emoji" not in cleaned


def test_clean_markdown_strips_code_fences() -> None:
    cleaned = clean_markdown(MD_WITH_NOISE)
    assert "print(" not in cleaned
    assert "code should be dropped" not in cleaned


def test_clean_markdown_strips_image_links() -> None:
    cleaned = clean_markdown(MD_WITH_NOISE)
    assert "/images/fig1.png" not in cleaned


def test_clean_markdown_keeps_prose() -> None:
    cleaned = clean_markdown(MD_WITH_NOISE)
    assert "本文の段落です" in cleaned
    assert "次の段落" in cleaned


def test_clean_markdown_no_frontmatter_passthrough() -> None:
    text = "ただの本文。\n二行目。\n"
    assert "ただの本文。" in clean_markdown(text)


def test_load_corpus_reads_md_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("---\ntitle: a\n---\n記事A本文", encoding="utf-8")
    (tmp_path / "b.md").write_text("記事B本文", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("not markdown", encoding="utf-8")
    corpus = load_corpus(tmp_path)
    assert "記事A本文" in corpus
    assert "記事B本文" in corpus
    assert "not markdown" not in corpus


def test_load_corpus_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_corpus(tmp_path)


def test_train_val_split_proportions() -> None:
    ids = list(range(100))
    train, val = train_val_split(ids, val_fraction=0.1)
    assert len(train) == 90
    assert len(val) == 10
    assert train + val == ids  # contiguous split, no shuffle


def test_train_val_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        train_val_split(list(range(10)), val_fraction=0.9)


def test_get_batch_shapes_and_shift() -> None:
    data = mx.array(list(range(100)))
    x, y = get_batch(data, batch_size=4, ctx_len=8, seed=0)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y is x shifted by one position
    assert mx.array_equal(x[:, 1:], y[:, :-1]).item()


def test_get_batch_deterministic_with_seed() -> None:
    data = mx.array(list(range(100)))
    x1, y1 = get_batch(data, batch_size=4, ctx_len=8, seed=42)
    x2, y2 = get_batch(data, batch_size=4, ctx_len=8, seed=42)
    assert mx.array_equal(x1, x2).item()
    assert mx.array_equal(y1, y2).item()


def test_get_batch_varies_with_seed() -> None:
    data = mx.array(list(range(1000)))
    x1, _ = get_batch(data, batch_size=4, ctx_len=8, seed=1)
    x2, _ = get_batch(data, batch_size=4, ctx_len=8, seed=2)
    assert not mx.array_equal(x1, x2).item()


def test_get_batch_rejects_short_data() -> None:
    data = mx.array(list(range(5)))
    with pytest.raises(ValueError):
        get_batch(data, batch_size=2, ctx_len=8, seed=0)
