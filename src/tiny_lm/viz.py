"""Zero-dependency HTML/SVG rendering: attention heatmaps and loss curves.

Everything here is pure string generation — deterministic and unit-testable
without matplotlib or a browser.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import mlx.core as mx

CELL = 18  # px per attention cell
MARGIN = 4


def _heat_color(w: float) -> str:
    """0 -> near-white, 1 -> deep blue."""
    w = min(max(w, 0.0), 1.0)
    r = int(247 - 200 * w)
    g = int(251 - 170 * w)
    b = int(255 - 80 * w)
    return f"rgb({r},{g},{b})"


def attention_heatmap_svg(layer_map: mx.array, tokens: list[str], layer_index: int) -> str:
    """One SVG per head for a single layer's attention weights (H, T, T)."""
    n_heads, t, t2 = layer_map.shape
    if t != t2 or len(tokens) != t:
        raise ValueError(f"tokens ({len(tokens)}) must match attention size ({t}x{t2})")
    size = t * CELL + 2 * MARGIN
    parts: list[str] = []
    for h in range(n_heads):
        rects: list[str] = []
        for row in range(t):
            for col in range(t):
                w = float(layer_map[h, row, col].item())
                x = MARGIN + col * CELL
                y = MARGIN + row * CELL
                label = html.escape(f"{tokens[row]}→{tokens[col]}: {w:.3f}")
                rects.append(
                    f'<rect x="{x}" y="{y}" width="{CELL - 1}" height="{CELL - 1}" '
                    f'fill="{_heat_color(w)}"><title>{label}</title></rect>'
                )
        parts.append(
            f"<figure><figcaption>layer {layer_index} · head {h}</figcaption>"
            f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(rects)
            + "</svg></figure>"
        )
    return "\n".join(parts)


def loss_curve_svg(
    train_losses: list[float],
    val_losses: list[tuple[int, float]] | None = None,
    width: int = 640,
    height: int = 240,
) -> str:
    """Polyline chart of training loss, with optional validation loss overlay."""
    if not train_losses:
        raise ValueError("train_losses must not be empty")
    val_losses = val_losses or []
    all_vals = train_losses + [v for _, v in val_losses]
    lo, hi = min(all_vals), max(all_vals)
    span = (hi - lo) or 1.0
    pad = 10

    def sx(step: float) -> float:
        return pad + (width - 2 * pad) * step / max(len(train_losses) - 1, 1)

    def sy(loss: float) -> float:
        return pad + (height - 2 * pad) * (1.0 - (loss - lo) / span)

    train_pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(train_losses))
    lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<polyline points="{train_pts}" fill="none" stroke="#4059ad" stroke-width="1.5"/>',
    ]
    if val_losses:
        val_pts = " ".join(f"{sx(s - 1):.1f},{sy(v):.1f}" for s, v in val_losses)
        lines.append(
            f'<polyline points="{val_pts}" fill="none" stroke="#d1495b" stroke-width="1.5"/>'
        )
    lines.append(
        f'<text x="{pad}" y="{pad + 8}" font-size="10" fill="#555">'
        f"loss {lo:.3f}–{hi:.3f} (blue: train, red: val)</text>"
    )
    lines.append("</svg>")
    return "".join(lines)


def write_report(
    path: Path | str,
    title: str,
    train_losses: list[float],
    val_losses: list[tuple[int, float]],
    attention_maps: list[mx.array],
    tokens: list[str],
    samples: list[str],
) -> None:
    """Single self-contained HTML: loss curve + per-layer attention + generated samples."""
    heatmaps = "\n".join(
        f"<h3>Layer {i}</h3>" + attention_heatmap_svg(m, tokens, layer_index=i)
        for i, m in enumerate(attention_maps)
    )
    sample_blocks = "\n".join(f"<pre>{html.escape(s)}</pre>" for s in samples)
    doc = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 960px;
       margin: 2rem auto; padding: 0 1rem; }}
figure {{ display: inline-block; margin: 0.5rem; }}
figcaption {{ font-size: 12px; color: #555; }}
pre {{ background: #f6f6f6; padding: 1rem; white-space: pre-wrap; border-radius: 6px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<h2>Loss curve</h2>
{loss_curve_svg(train_losses, val_losses)}
<h2>Attention patterns</h2>
<p>行 = 出力位置（query）、列 = 参照位置（key）。濃いほど強く注目。上三角が空白 = 因果マスク。</p>
{heatmaps}
<h2>Generated samples</h2>
{sample_blocks}
</body>
</html>
"""
    Path(path).write_text(doc, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Build the HTML report from a checkpoint directory."""
    from tiny_lm.sample import generate
    from tiny_lm.train import load_checkpoint

    parser = argparse.ArgumentParser(description="Render attention/loss report for a checkpoint")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True, help="prompt for attention viz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    model, tokenizer, config = load_checkpoint(args.ckpt)
    prompt = args.prompt[: config.ctx_len]
    ids = mx.array([tokenizer.encode(prompt)])
    maps = model.attention_maps(ids)
    history = json.loads((args.ckpt / "history.json").read_text(encoding="utf-8"))
    samples = [
        generate(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.sample_tokens,
            temperature=t,
            top_k=40,
            seed=args.seed,
        )
        for t in (0.5, 0.8, 1.0)
    ]
    write_report(
        args.out,
        title=f"tiny-lm report — {args.ckpt.name}",
        train_losses=history["train_losses"],
        val_losses=[tuple(v) for v in history["val_losses"]],
        attention_maps=maps,
        tokens=list(prompt),
        samples=samples,
    )
    print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
