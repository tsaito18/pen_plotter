# AGENTS.md

このファイルは、このリポジトリで作業するエージェント向けの作業メモです。
詳細な背景は `CLAUDE.md` と `plan.md` を参照してください。

## プロジェクト概要

このプロジェクトは、手書き風の日本語文字を生成し、xDraw A4 ペンプロッタで紙に描画するための Python アプリケーションです。

主な処理は、iPad/Apple Pencil などで集めた筆跡サンプルを使い、KanjiVG の参照ストロークにユーザーの筆跡スタイルを転写し、レポート用紙向けに組版して、プレビュー画像または G-code を生成する流れです。

主要なワークフローは `src/ui/web_app.py` の `PlotterPipeline` が束ねています。

## 技術スタック

- Python 3.11+
- パッケージ管理: `uv`
- ML: PyTorch
- UI: Gradio、Tkinter
- プレビュー: Matplotlib
- 実機: xDraw A4 ペンプロッタ、GRBL 互換 DrawCore firmware
- テスト: pytest
- リント/フォーマット: ruff

## 主要コマンド

通常はリポジトリルートで実行します。

```sh
make test
make lint
make format
make ui
make collect
make preview
```

個別実行例:

```sh
pytest
pytest -m "not slow and not hardware"
pytest tests/test_gcode_generator.py
ruff check src/ tests/ scripts/
ruff format src/ tests/ scripts/
python scripts/run_ui.py
python scripts/collect_strokes.py
```

Windows 側で xDraw A4 送信 GUI を起動する場合:

```sh
python scripts/run_plotter_gui.py
python -m src.plotter_gui
```

Windows 用 exe を作る場合は `docs/plotter_gui_build.md` を確認してください。

## ディレクトリ構成

- `src/collector/`: 手書きサンプル収集、KanjiVG/CASIA パーサー、iPad UI、データ形式
- `src/model/`: PyTorch モデル、訓練、推論、StyleEncoder、StrokeDeformer、StrokeAligner
- `src/layout/`: レポート用紙向け組版、改行、数式、表
- `src/gcode/`: G-code 生成、最適化、プレビュー、プロッタ設定
- `src/comm/`: GRBL シリアル通信、ポート検出、コントローラ
- `src/ui/`: Web/Gradio UI、生成パイプライン、プレビュー、ストローク描画
- `src/plotter_gui/`: xDraw A4 へ G-code を送る Tkinter デスクトップ GUI
- `scripts/`: 開発・訓練・変換・起動用 CLI
- `tests/`: pytest テスト
- `docs/`: 実機 GUI のチェックリスト、Windows exe ビルド手順
- `data/`: KanjiVG ストローク、ユーザー筆跡、学習済みモデル、レポート用紙画像

## アーキテクチャの要点

中心の流れ:

1. `collector` でユーザー筆跡を収集する。
2. `model` で筆跡スタイルを学習・推論する。
3. `layout` でレポート用紙に合わせて文字、数式、表を配置する。
4. `ui` の `PlotterPipeline` がストローク生成、プレビュー、G-code 生成を統合する。
5. `gcode` が xDraw A4 向けの G-code を生成・最適化する。
6. `comm` または `plotter_gui` が GRBL 互換機へ送信する。

`PlotterPipeline` は薄いオーケストレータとして扱い、重い処理は `Typesetter`、`StrokeRenderer`、`PreviewRenderer`、`GCodeGenerator` など各責務のクラスへ寄せてください。

## 手書き生成モデル

現在の主方針は V3 系のスタイル転写です。

- KanjiVG の正しい参照ストロークを骨格として使う。
- ユーザー筆跡から `StyleEncoder` でスタイルベクトルを作る。
- `StrokeDeformer` / `TransformerDeformer` / `TwoStageDeformer` で per-point offset を予測する。
- 本番系は TwoStageDeformer を使う想定です。
- CASIA は日本語筆跡との不一致が大きいため、基本的にユーザーデータ中心で扱います。
- V2 の LSTM+MDN 方式は品質不足で撤退済みです。新規実装で戻さないでください。

## 座標系の注意

座標系は過去にバグの原因になっているため、変更時は必ず確認してください。

```text
KanjiVG JSON / data/strokes/       : Y-UP
User strokes / data/user_strokes/  : Y-DOWN
G-code / plotter                   : Y-UP, (0,0)=左下, (210,297)=右上
matplotlib default                 : Y-UP
```

- 訓練時はユーザーストロークを Y 反転して KanjiVG に合わせます。
- 推論出力は Y-UP として扱います。
- プレビューでは不要に `invert_yaxis()` を入れないでください。
- `src/ui/stroke_renderer.py` の正規化処理は Y-DOWN から Y-UP への変換を含みます。

## xDraw A4 / GRBL 実機注意

xDraw A4 は Windows PC に USB 接続して扱う前提です。

- USB: CH340
- ボーレート: 115200
- ペン制御は Z 軸です。M3/M5 ではありません。
- ペンアップ: `G1G90 Z0.5 F5000`
- ペンダウン: `G1G90 Z5 F5000`
- ホーミング: `$H`
- ホーミング後の紙座標設定: `G92 X0 Y297 Z0`
- 紙座標: `(0,0)=左下`, `(210,297)=右上`

WSL では Tkinter GUI や実機 USB 制御が期待通り動かないことがあります。xDraw A4 の送信 GUI、実機チェック、exe ビルドは Windows ネイティブ Python で実行してください。

実機操作を伴う変更では `docs/plotter_gui_checklist.md` を確認し、危険な自動送信やペンダウン動作を勝手に追加しないでください。

## 開発方針

このプロジェクトは TDD を重視します。

1. 先に失敗するテストを書く。
2. テストを通す最小限の実装を入れる。
3. テストが通った状態でリファクタリングする。

守ること:

- 新機能・バグ修正は原則として対応するテストを追加または更新する。
- `tests/` に `test_{module}.py` 形式で置く。
- 型ヒントを付ける。
- docstring は必要な場所に Google style で書く。
- ruff の整形・リント方針に合わせる。
- スクリプトを追加・変更した場合は、可能なら実際に起動またはドライランして import error やデータ不在時の失敗を確認する。

## テスト方針

pytest markers:

- `slow`: ML 訓練など時間がかかるテスト
- `hardware`: 実機接続が必要なテスト

通常確認:

```sh
pytest -m "not slow and not hardware"
ruff check src/ tests/ scripts/
```

実機や長時間訓練が絡むテストは、明示的な依頼や実行環境の確認なしに走らせないでください。

## データと生成物の扱い

- `data/user_strokes/` はユーザー筆跡サンプルです。形式や座標系を壊さないでください。
- `data/strokes/` は KanjiVG 由来の参照ストロークです。
- `data/models/` は学習済み checkpoint 置き場です。
- `data/report_paper.jpg` はレポート用紙背景プレビューで使われます。
- 大きなデータやモデルを無関係に再生成・削除しないでください。

## よく触る入口

- Web UI: `scripts/run_ui.py`, `src/ui/gradio_app.py`
- 生成パイプライン: `src/ui/web_app.py`
- ストローク描画: `src/ui/stroke_renderer.py`
- プレビュー: `src/ui/preview_renderer.py`
- 組版: `src/layout/typesetter.py`, `src/layout/page_layout.py`
- 数式: `src/layout/math_layout.py`
- G-code: `src/gcode/generator.py`, `src/gcode/config.py`, `src/gcode/optimizer.py`
- 実機通信: `src/comm/grbl_controller.py`, `src/comm/serial_sender.py`
- 送信 GUI: `src/plotter_gui/app.py`, `src/plotter_gui/worker.py`
- 訓練: `src/model/pretrain.py`, `src/model/finetune.py`, `scripts/pretrain.py`, `scripts/finetune.py`

## 作業時の優先順位

1. 既存の責務分割を尊重する。
2. 座標系と実機コマンドは慎重に扱う。
3. テストを先に見て、必要なら追加する。
4. UI、生成、G-code、通信の境界を混ぜすぎない。
5. 実機依存の変更は mock テストと手動チェックリストの両方を意識する。
6. 既存データ、checkpoint、ユーザー筆跡を壊さない。

