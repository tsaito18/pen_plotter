# Pen Plotter — 手書き風文字生成ペンプロッタ

## プロジェクト概要
手書き風の文字を生成してペンプロッタで出力する制御プログラム。
ユーザーの筆跡を学習し、同じ文字でも毎回異なる形で生成する。

## 技術スタック
- Python 3.11+
- PyTorch (ML)
- Matplotlib (プレビュー)
- Gradio (Web UI)
- GRBL (ファームウェア)

## 開発コマンド
- テスト: `pytest`
- リント: `ruff check src/ tests/`
- フォーマット: `ruff format src/ tests/`

## アーキテクチャ
- `src/gcode/` — G-code生成・最適化・プレビュー
- `src/collector/` — 手書きサンプル収集
- `src/model/` — ML モデル（LSTM+MDN）
- `src/layout/` — 組版エンジン
- `src/comm/` — GRBL シリアル通信
- `src/ui/` — Gradio Web UI

## コーディング規約
- ruff でフォーマット・リント
- 型ヒント必須
- docstring は Google style
- テストは tests/ に配置、pytest で実行

## TDD 開発フロー
このプロジェクトはテスト駆動開発（TDD）で進める。

### 原則
1. **Red**: 実装コードを書く前に、まず失敗するテストを書く
2. **Green**: テストが通る最小限の実装を書く
3. **Refactor**: テストが通った状態を維持しつつリファクタリング

### ルール
- 新機能追加時は必ずテストから着手する
- テストなしの実装コードを書かない
- 全テストがパスした状態を常に維持する

### テスト構成
- テストファイル: `tests/test_{module}.py`
- 共通フィクスチャ: `tests/conftest.py`
- フレームワーク: pytest

### 実行コマンド
- 全テスト: `pytest`
- 詳細出力: `pytest -v`
- 特定ファイル: `pytest tests/test_gcode_generator.py`
- 特定テスト: `pytest -k "test_square"`
- 高速テストのみ: `pytest -m "not slow and not hardware"`

### マーカー
- `@pytest.mark.slow` — 実行に時間がかかるテスト（ML訓練など）
- `@pytest.mark.hardware` — 実機接続が必要なテスト

## 現在の進捗
- Phase 1 完了: G-code生成・プレビュー・最適化の基盤
- Phase 3 完了: シリアル通信（GRBL ストリーミング・コントローラ）
- Phase 4 完了: 組版エンジン（ページレイアウト・禁則処理・数式・表）
- Phase 5 完了: サンプル収集基盤（データ形式・KanjiVGパーサー・iPad Web UI・ストローク正規化）
- Phase 6 完了: MLモデル（スタイルエンコーダ・LSTM+MDN・訓練・推論）
- Phase 7 部分完了: パイプライン統合（テキスト→G-code→プレビュー）、仮ストローク描画
- 160テスト全パス

## 実装計画
詳細は [plan.md](plan.md) を参照。
