"""Zero-dependency HTML/SVG report: attention heatmaps and loss curve."""

from pathlib import Path

import mlx.core as mx
import pytest

from tiny_lm.viz import attention_heatmap_svg, loss_curve_svg, write_report


def _uniform_maps(n_layers: int, n_heads: int, t: int) -> list[mx.array]:
    """Causal uniform attention: row t attends equally to positions 0..t."""
    rows = []
    for t_i in range(t):
        row = [1.0 / (t_i + 1)] * (t_i + 1) + [0.0] * (t - t_i - 1)
        rows.append(row)
    single = mx.array([rows] * n_heads)
    return [single] * n_layers


def test_attention_heatmap_svg_structure() -> None:
    maps = _uniform_maps(n_layers=2, n_heads=2, t=4)
    svg = attention_heatmap_svg(maps[0], tokens=list("abcd"), layer_index=0)
    assert svg.count("<svg") == 2  # one per head
    # each head draws a 4x4 grid of cells
    assert svg.count("<rect") >= 2 * 16


def test_attention_heatmap_rejects_token_mismatch() -> None:
    maps = _uniform_maps(n_layers=1, n_heads=1, t=4)
    with pytest.raises(ValueError):
        attention_heatmap_svg(maps[0], tokens=list("ab"), layer_index=0)


def test_loss_curve_svg_contains_polyline() -> None:
    svg = loss_curve_svg([5.0, 4.0, 3.0, 2.5, 2.0])
    assert "<svg" in svg
    assert "polyline" in svg


def test_loss_curve_rejects_empty() -> None:
    with pytest.raises(ValueError):
        loss_curve_svg([])


def test_loss_curve_with_val_losses() -> None:
    svg = loss_curve_svg([5.0, 4.0, 3.0], val_losses=[(1, 4.5), (2, 3.5)])
    assert svg.count("polyline") == 2


def test_write_report(tmp_path: Path) -> None:
    maps = _uniform_maps(n_layers=2, n_heads=2, t=4)
    out = tmp_path / "report.html"
    write_report(
        out,
        title="test run",
        train_losses=[5.0, 4.0, 3.0],
        val_losses=[(1, 4.5)],
        attention_maps=maps,
        tokens=list("abcd"),
        samples=["生成テキスト例"],
    )
    html = out.read_text(encoding="utf-8")
    assert "<title>test run</title>" in html
    assert "生成テキスト例" in html
    assert html.count("<svg") == 5  # 2 layers x 2 heads + 1 loss curve
    # html-escapes special characters in samples
    write_report(
        out,
        title="t",
        train_losses=[1.0],
        val_losses=[],
        attention_maps=maps,
        tokens=list("abcd"),
        samples=["<script>alert(1)</script>"],
    )
    assert "<script>" not in out.read_text(encoding="utf-8")
