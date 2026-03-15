# Pen Plotter — 手書き風文字生成ペンプロッタ

## プロジェクト概要
手書き風の文字を生成してペンプロッタで出力する制御プログラム。
ユーザーの筆跡を学習し、同じ文字でも毎回異なる形で生成する。

## 技術スタック
- Python 3.11+ (venvは3.12で作成、IPEX互換性のため)
- PyTorch (ML) — XPU版でIntel Arc GPU対応予定
- Matplotlib (プレビュー)
- Gradio (Web UI)
- GRBL (ファームウェア)
- パッケージマネージャー: uv

## 開発環境
- Intel Core Ultra 7 258V (Lunar Lake) — GPU: Intel Arc統合, NPU: AI Boost
- WSL2 (Ubuntu) で開発、ただしWSLではGPU未認識 (/dev/dri/ なし)
- GPU訓練はWindowsネイティブで実行（PyTorch XPU版、XPU: True確認済み）

## 開発コマンド
- テスト: `pytest`
- リント: `ruff check src/ tests/`
- フォーマット: `ruff format src/ tests/`

## アーキテクチャ

### ディレクトリ構成
- `src/gcode/` — G-code生成・最適化・プレビュー
- `src/collector/` — 手書きサンプル収集（iPad UI, KanjiVGパーサー, CASIAパーサー）
- `src/model/` — ML モデル（V2: CharEncoder + StyleEncoder + LSTM+MDN）
- `src/layout/` — 組版エンジン
- `src/comm/` — GRBL シリアル通信
- `src/ui/` — Gradio Web UI
- `scripts/` — CLI スクリプト群

### MLモデル V2アーキテクチャ（文字条件付き生成）
```
CharEncoder(KanjiVG_skeleton) → char_embedding（何の文字か）
StyleEncoder(user_samples) → style_vector（どんな書き癖か）
StrokeGenerator(char_embedding + style_vector) → strokes
```
- 事前訓練: CASIA-OLHWDB or KanjiVGで CharEncoder + Generator を訓練
- ファインチューニング: ユーザー20-30文字で StyleEncoder のみ更新（Generator凍結）
- V1チェックポイント（char_dimなし）との後方互換性あり

### スクリプト一覧
| スクリプト | 用途 |
|-----------|------|
| `scripts/prepare_kanjivg.py --download` | KanjiVGデータ取得・変換（6,699文字） |
| `scripts/pretrain.py` | 事前訓練（CharEncoder+StyleEncoder+Generator） |
| `scripts/finetune.py` | ファインチューニング（StyleEncoderのみ） |
| `scripts/train_model.py` | V1訓練（旧方式、char_dimなし） |
| `scripts/collect_strokes.py` | ガイド付き手書きサンプル収集（292文字セット） |
| `scripts/run_ui.py` | Gradio Web UI起動 |

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
- スクリプト作成後は必ず実際に実行して動作確認する（テストだけでは不十分）

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
- 265テスト全パス

## 実装計画
詳細は [plan.md](plan.md) を参照。

## メモリ（別デバイスでの開発継続用）

### ユーザープロフィール
- 電子工作・機械工作の経験が豊富（Arduino、回路設計、機械加工に精通）
- 利用可能機材: ノートPC（Intel Core Ultra 7 258V / Lunar Lake）、ラズパイ、Arduino、ESP32、3Dプリンター、iPad（Apple Pencil）
- WSL2で/dev/dri/が認識されずIntel GPU使用不可（2026-03時点）→ Windowsネイティブで GPU 訓練を試行中
- パッケージマネージャー: uv を使用（pip より優先）
- Python: システムは 3.14 だが venv は 3.12 で作成（IPEX 互換性のため）
- 日本語が母語

### 開発方針フィードバック
- TDD厳守（Red→Green→Refactor）
- スクリプトは実際に実行確認してからコミット（importエラー、データ不在時のエラー等、テストでは見つからない問題が頻発した）

### 手書き生成V2 方針
- レポート提出に使える「バレない」品質が目標
- CASIA-OLHWDB（160万サンプル）で事前訓練 → ユーザー20-30字でファインチューニング
- Step A-G 全実装完了、GPU(XPU)対応完了、CASIAデータ取得・.pot直接読み込み対応完了
- WSLでIntel GPU未認識 → WindowsネイティブでPyTorch XPU版を使用（XPU: True確認済み）
- pretrain.py/finetune.py にデバイス自動検出・--device指定を実装済み

### 全体進捗（2026-03-15時点）
- 265テスト全パス
- KanjiVG 6,699文字変換済み
- 3段階フォールバック（ML推論→KanjiVG→矩形）動作
- ガイド付きストローク収集UI（292文字セット）実装済み
- ユーザーサンプルは data/user_strokes/ に保存（KanjiVGデータと分離）
- CASIA-OLHWDB .pot 直接読み込み対応（JSON中間変換不要）
- GPU(XPU) 自動検出・--device指定を pretrain/finetune に実装済み
- CASIA Pot1.0/1.1/1.2 Train/Test データ取得済み（data/casia_raw/）
