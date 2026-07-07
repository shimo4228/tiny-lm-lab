"""End-to-end: train CLI -> sample CLI -> report CLI on a tiny corpus."""

from pathlib import Path

import pytest

from tiny_lm import sample, train, viz


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("corpus")
    body = "エージェントの知識は循環する。attention is all you need.\n" * 30
    (d / "article.md").write_text(f"---\ntitle: t\n---\n{body}", encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def ckpt_dir(corpus_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("runs") / "e2e"
    train.main(
        [
            "--corpus",
            str(corpus_dir),
            "--out",
            str(out),
            "--steps",
            "20",
            "--batch-size",
            "4",
            "--ctx-len",
            "16",
            "--d-model",
            "32",
            "--n-heads",
            "2",
            "--n-layers",
            "2",
            "--eval-every",
            "10",
            "--val-fraction",
            "0.1",
            "--max-chars",
            "500",
        ]
    )
    return out


def test_train_cli_writes_checkpoint(ckpt_dir: Path) -> None:
    assert (ckpt_dir / "model.safetensors").exists()
    assert (ckpt_dir / "history.json").exists()


def test_sample_cli_prints_text(ckpt_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sample.main(
        ["--ckpt", str(ckpt_dir), "--prompt", "エージェント", "--tokens", "10", "--seed", "1"]
    )
    out = capsys.readouterr().out
    assert out.startswith("エージェント")
    assert len(out.strip()) >= len("エージェント")


def test_report_cli_writes_html(ckpt_dir: Path, tmp_path: Path) -> None:
    report = tmp_path / "report.html"
    viz.main(
        [
            "--ckpt",
            str(ckpt_dir),
            "--prompt",
            "エージェントの知識",
            "--out",
            str(report),
            "--sample-tokens",
            "10",
        ]
    )
    html_text = report.read_text(encoding="utf-8")
    assert "<title>" in html_text
    assert "Attention patterns" in html_text
    assert html_text.count("<svg") == 2 * 2 + 1  # n_layers * n_heads + loss curve
