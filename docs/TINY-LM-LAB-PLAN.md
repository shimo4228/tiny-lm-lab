# tiny-lm-lab 実装プラン

日付: 2026-07-07 / 種別: `feat` / 発端: `/goal` 自律セッション（成長プロジェクト選定コンペの勝者）

## What

GPT 型 decoder-only transformer を **MLX の低レベル基盤（テンソル演算・autograd・optimizer）だけ**を使ってスクラッチ実装し、自分の Zenn 記事コーパス（日本語、276本）で char-level 言語モデルを学習する学習用ラボ。

成果物:
1. `tiny_lm` パッケージ — tokenizer / model / data / train / sample / viz
2. 学習済みチェックポイント + loss curve + 生成サンプル
3. attention パターン可視化（依存ゼロの自作 HTML/SVG 出力）
4. pytest テストスイート（カバレッジ 80%+）
5. README（アーキテクチャ図 + 実行結果）

## Why

- 審査員3レンズ（成長度 9/10・完成可能性 8/10・持続価値 8/10、全会一致1位）: 「LLM を運用する側 → attention・学習ループを書く側」への質的跳躍。base-model-lab は mlx_lm 依存の LoRA のみで from-scratch 実装経験は不在（審査時に実地検証済み）
- 自分のコーパスで学習することで研究プログラム（authorship / maker としての実践）に接続
- 維持すべき常駐システムを残さず認知負荷を増やさない

## External Research Findings (Phase 0)

`/search-first` 実行済み。

- nanoGPT (60.9k★, MIT, 2025-11): PyTorch 依存 ~2GB。設計リファレンスとしてのみ参照
- mlx-examples (8.8k★, MIT, 2026-04 active): 同上
- mlx 0.31.2: cp310–cp314 wheel、活発維持 → 低レベル基盤として採用。`mlx.nn.MultiHeadAttention` 等の既製 transformer 部品は学習目的のため不使用

**Verdict: Build** — custom (mlx 基盤、nanoGPT アーキテクチャ参照) — 学習目的により自作自体が価値。

## Alternatives

- 候補コンペ次点: retrieval-lab (BM25+評価スイート、20点)、mech-interp lab (16点) — 審査記録は scratchpad/candidates.md 参照
- numpy + 手動 backprop 案: 学習は最深になるが1セッションの完成リスク大 → autograd は mlx に委譲し、forward 側（attention 数式）の自作に学習価値を集中

## 設計判断

| 項目 | 決定 | 理由 |
|---|---|---|
| 自作境界 | Embedding / Linear / LayerNorm / CausalSelfAttention / MLP / Block / GPT / 学習ループ / サンプリング / tokenizer を自作。autograd・AdamW・`nn.Module`(パラメータ登録のみ) は mlx | 学習価値の核は forward の数式と学習ダイナミクス。パラメータ管理は本質でない |
| Tokenizer | char-level（コーパスから語彙構築、`<unk>` 付き、JSON save/load） | 日本語で BPE 自作はスコープ過大。char-level で数分学習が成立 |
| モデル規模 | ctx 256 / d_model 256 / 4 layers / 4 heads (~数M params) | 16GB M系 Mac で数分〜十数分の学習に収まる |
| 可視化 | matplotlib 不使用。純 Python で HTML/SVG 生成（attention heatmap + loss curve） | 依存ゼロ・決定論テスト可能・Artifact 化容易 |
| チェックポイント | mx.save_safetensors + tokenizer JSON + loss history JSON | 標準的・再ロード検証をテスト化 |

## Chain (feat)

- Plan ✓ / Phase 0 ✓ (Build)
- TDD: tokenizer → data → model (causal mask 正当性が最重要テスト) → train (loss 減少 sanity) → sample → viz
- 実装後: python-reviewer + security-reviewer + codex-review を**並列**起動
- Verify: ruff / pyright / pytest (cov≥80%) / secret scan / git status → commit

```
Parallel Group 1: [planner, scout]  # 実施済み (審査コンペ + search-first)
Parallel Group 2: [python-reviewer, security-reviewer, codex-review]
Sequential: TDD (Plan 後) → Verify (全レビュー後)
```

## Done 条件

1. 全テスト PASS、カバレッジ ≥80%
2. Zenn コーパスで学習が走り loss が明確に減少（学習前より生成品質が目視で向上）
3. attention heatmap HTML と loss curve が生成される
4. README + 本プラン文書 + git commit 完了
