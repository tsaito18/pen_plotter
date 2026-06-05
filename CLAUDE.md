# Pen Plotter — 手書き風文字生成ペンプロッタ

## プロジェクト概要
手書き風の文字を生成してペンプロッタで出力する制御プログラム。
ユーザーの筆跡を学習し、同じ文字でも毎回異なる形で生成する。

## 技術スタック
- Python 3.11+ (venvは3.12で作成、IPEX互換性のため)
- PyTorch (ML) — XPU版でIntel Arc GPU対応予定
- Matplotlib (プレビュー)
- Gradio (Web UI)
- xDraw A4 ペンプロッタ（GRBL互換 DrawCore ファームウェア）
- パッケージマネージャー: uv

## 開発環境
- 開発サーバー: 192.168.86.100（ローカルネット）
- Intel Core Ultra 7 258V (Lunar Lake) — GPU: Intel Arc統合, NPU: AI Boost
- WSL2 (Ubuntu) で開発、ただしWSLではGPU未認識 (/dev/dri/ なし)
- GPU訓練はWindowsネイティブで実行（PyTorch XPU版、XPU: True確認済み）
- xDraw A4 はWindows PCにUSB接続、G-code送信はWindows側で `python scripts/run_plotter_gui.py` または `python -m src.plotter_gui` で GUI 起動（src/plotter_gui/）

## 開発コマンド
- テスト: `pytest`
- リント: `ruff check src/ tests/`
- フォーマット: `ruff format src/ tests/`

## アーキテクチャ

### ディレクトリ構成
- `src/gcode/` — G-code生成・最適化・プレビュー
- `src/collector/` — 手書きサンプル収集（iPad UI, KanjiVGパーサー, CASIAパーサー）
- `src/model/` — ML モデル（V3: StrokeDeformer/TransformerDeformer per-point offset + StyleEncoder + Contrastive Learning）
- `src/layout/` — 組版エンジン
- `src/comm/` — GRBL シリアル通信
- `src/ui/` — Web UI（`web_app.py`: パイプライン、`gradio_app.py`: Gradio UI）
- `scripts/` — CLI スクリプト群

### MLモデル V3アーキテクチャ（スタイル転写 / per-point offset）
```
KanjiVG参照ストローク → リサンプリング（32点）
StyleEncoder(user_samples) → style_vector（どんな書き癖か）
  + ProjectionHead → SupCon対照学習（訓練時のみ、推論時は不使用）
StrokeDeformer or TransformerDeformer → オフセット (N, 2)
  MLP版: smooth_offsets(kernel=11) → clamp(±0.6)
  Transformer版: Self-Attention(2層) + Cross-Attention(style) → clamp(±0.6)（smooth不要）
StrokeAligner(Hungarian + MHD) → ストローク順序/方向のアライメント（マージ/スプリット検出付き）
+ ストローク単位の幾何バリエーション（回転・スケール・シフト）
+ 文字配置の揺らぎ（ベースライン・字間・サイズ・傾き）+ ストローク太さ変化
```
- KanjiVGの正しい形を前提に、per-point offsetで変形（生成ではなく転写）
- ユーザーデータのみで訓練（CASIA不使用 — 中国語/日本語ストロークの不一致）
- 381文字 / 925サンプル収集済み（data/user_strokes/）
- V2 (LSTM+MDN) は300k samples, A100でも読めない品質で断念
- train-inference gap解消済み: smooth+clampを訓練/推論で統一（共有定数/関数）
- **deformer_type**: "twostage"(本番採用), "transformer", "offset"(MLP), "affine" の4種
- 本番モデル: 381文字/925サンプルで訓練したTwoStageDeformer（AffineStrokeDeformer + TransformerDeformer）
- 2段階変形: Stage1 Affine(per-stroke 大域変形=回転/スケール/シーア/平行移動) + Stage2 Transformer(per-point 細部変形)
- AffineMultipliers: theta=0.05, scale=0.10, shear=0.05, translation=0.30 (TwoStage用に拡大)
- **Contrastive Learning**: SupCon loss + ProjectionHead でスタイル空間を識別的に（temp=0.07, weight=0.1）

### 座標系の注意事項（重要）
```
KanjiVG JSON (data/strokes/)   : Y-UP（prepare_kanjivg.py で y = target_size - y 反転済み）
User strokes (data/user_strokes/): Y-DOWN（iPad Canvas座標そのまま）
G-code / plotter               : Y-UP（0,0=左下、210,297=右上）
matplotlib デフォルト           : Y-UP（invert_yaxis() 不要）
```
- **訓練時**: FinetuneDeformationDataset でユーザーストロークのY軸を反転してKanjiVGに統一
- **推論時**: 出力はY-UP座標（KanjiVG参照と同じ）
- **プレビュー**: matplotlib で invert_yaxis() は呼ばない（Y-UP同士で一致）
- **stroke_renderer.py**: `_normalize_strokes_to_unit()` で `1 - y` 反転（Y-DOWN→Y-UP変換）

### スクリプト一覧
| スクリプト | 用途 |
|-----------|------|
| `scripts/prepare_kanjivg.py --download` | KanjiVGデータ取得・変換（6,699文字） |
| `scripts/pretrain.py` | 事前訓練（CharEncoder+StyleEncoder+Generator） |
| `scripts/finetune.py` | ファインチューニング（StyleEncoderのみ） |
| `scripts/train_model.py` | V1訓練（旧方式、char_dimなし） |
| `scripts/collect_strokes.py` | ガイド付き手書きサンプル収集（381文字セット） |
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
- Phase 6 完了: MLモデル V2→V3移行完了（V2 LSTM+MDN断念、V3 StrokeDeformer採用）
- Phase 7 部分完了: V3スタイル転写動作、組版改善、幾何バリエーション、リアルさ改善（太さ変化・密度変動・段落インデント・局所曲率・クランプ±1.2）、数式レイアウト統合（インライン$...$, ブロック$$...$$, ギリシャ文字, 分数線）、直接ストローク使用・ストローク合成・弾性変形・tremor、Web UI改善（Gradio タブUI・設定パネル・プログレス・文字カバレッジ・ヘルプ・例文）、アライメントキャッシュ
- Phase 9 進行中: 少量サンプル対応（Contrastive StyleEncoder + TransformerDeformer実装済み、訓練・推論パイプライン統合済み）
- 訓練: ユーザーデータのみ（381文字/925サンプル）、CASIA不使用
- ストロークアライメント（Hungarian + MHD + マージ/スプリット検出）実装済み — 訓練時use_aligner=True対応
- 624+テスト（ML関連124テスト全パス確認済み）

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

### 手書き生成V3 方針
- レポート提出に使える「バレない」品質が目標
- V2 (LSTM+MDN) は300k samples, A100でも読めない品質で断念
- V3: KanjiVG参照ストロークにper-point offsetを適用するスタイル転写方式
- ユーザーデータのみで訓練（CASIA不使用 — 中国語/日本語ストロークの不一致でノイズ）
- オフセットクランプ±1.2 + スムージングkernel=11（訓練/推論で統一）
- ストローク単位の幾何バリエーション（回転・スケール・シフト）で自然さ追加
- 局所曲率特徴追加でストロークの曲がり角に大きなオフセット許容
- augmentation設定（baseline_drift=0.3, spacing=0.2, size=0.05）＋文字単位の傾き(slant_variation=0.02)有効 — 手書きの揺らぎ。slantはCharPlacement.slant経由で_position_strokesが文字中心回転
- **汚さスライダー（GUI）**: `UISettings.messiness`（0=整った字, 1=標準=上記の素値, 2=大きく乱れる）で baseline_drift/字間/サイズ/傾きを一括スケール。`web_app._scaled_augment_config()` が単一ソース。GUIの「温度」（=ML per-point offsetの字形揺らぎ）とは別軸
- **人らしさ3スライダー（GUI）**: 定幅ペン感を消し「人が書いた」感を出す調整軸。すべて実機キャリブ前提でデフォルトは控えめ。
  - `pressure_variation`(筆圧変化, 既定0.35): 画内の濃淡を `stroke_finishing.pressure_modulation()` で変調（下ろし濃く・上げ薄く＋低周波揺らぎ）。contact に乗算するので preview線幅と実機Zが連動。0=均一(定幅ペン感)
  - `instance_variation`(字のばらつき, 既定0.5): 同一字の繰り返しで形を変える per-stroke ランダムaffine（`StrokeRenderer._apply_instance_variation`）。`augmenter.enabled` と多画字の `_waver_scale` に従う
  - `entry_taper`(入筆, 実機既定0): 始筆を軽く入れて立ち上げる `stroke_finishing.entry_modulation()`。収筆(はらい/はね)と対。**実機注意**: 始筆でZを動かすためかすれ得る→実機は0
  - `connection_strength`(連綿, 既定0): 同字の近い画を確率的に薄いつなぎ画(`CONNECT`)で結ぶ `stroke_finishing.insert_connections()`。**近いほど高確率＋乱数**(`prob=strength*(1-gap/max_gap)`, `max_gap=strength*0.6*scale`)。generatorは`continue_from_prev`でペンを上げず継続=真の連綿。つなぎ画はZ一定(`CONNECT_CONTACT`)なので点線化しない。はらい/はねの後ろには付けない
  - **実機の制約**: 単線シャーペンは描画中にZを上下に振るとペンがバウンドして点線化する。よって筆圧変化・入筆(描画中Z変動)は実機デフォルト0。連綿はZ一定なのでOK。終端リフト(はらい/はね)は単調変化なのでOK
- ストローク太さ変化はプレビューのみ。実機は終端Zリフト（contact_profile, 距離mmベース）で払い・はねを表現
- GPU(XPU) 自動検出・--device指定を pretrain/finetune に実装済み

### 全体進捗（2026-04-01時点）
- 621+テスト全パス
- KanjiVG 6,699文字変換済み（SVGパーサー: smooth cubic bezier s/S対応済み）
- 直接ストローク使用（サンプル単位ランダム選択 + 幾何バリエーション）
- ガイド付きストローク収集UI（381文字セット、KanjiVGお手本表示、Apple Pencilのみ入力）
- ユーザーサンプル: 381文字 / 925+サンプル収集済み（data/user_strokes/）
- V3 StrokeDeformer（per-point offset MLP + 局所曲率）でユーザーデータ訓練完了
- レポート用紙実測レイアウト: 罫線7.16mm、余白48/34/5/5mm、font_size 4.5mm
- 文字バランス: 漢字100%、ひらがな/カタカナ個別調整（85%-68%）、半角70%、小書き55%
- 数式レイアウト統合: インライン$...$, ブロック$$...$$, ギリシャ文字, 分数線, ^/_ブレースなし記法
- **表（Markdownパイプ表）**: `| a | b |`＋区切り`|---|---|`＋データ行を `table_layout.detect_pipe_table()` で検出し、`typesetter._place_table()` が罫線(line_segment)＋手書きセル文字に組版（ブロック数式と同じ「複数行消費＋次ページ送り」方式、`ParsedDocument.table_blocks`）。列幅は中身の最大文字数から決め本文幅に収める。**本文幅の中央寄せ**。横罫線は用紙の罫線(line_positions)に一致。表の直後の `: タイトル` 行はキャプションとして表の下に中央寄せ描画（`table_captions`）
- セクション見出し: #/##/### → 階層インデント（15/25/35/45/55mm）
- **入力書式の総まとめ**: [docs/書式リファレンス.md](docs/書式リファレンス.md)（見出し・数式・表・キャプション・記号・スライダー）。アプリ内ヘルプ（`gradio_app._HELP_MARKDOWN`）とも同期
- マルチページプレビュー + 手書きページ番号
- レポート用紙背景プレビュー（data/report_paper.jpg自動ロード）
- リファクタリング済み: HTML分離、StrokeRenderer/PreviewRenderer分割、BaseFinetuner
- **xDraw A4 ペンプロッタ実機テスト成功**（ホーミング・ペン制御・描画動作確認済み）
- 幾何ストローク生成: 、。・（）／ASCII数式記号(+,-,=,<,>,*,/,%,:,;,!,?)は`_ascii_math_strokes`で幾何描画（矩形フォールバック回避）
- 数字(0-9)はML変形を**スキップ**しKanjiVG素の参照字形を直接使う（`_is_ml_deformable`）。モデルはCJKのみ訓練のため数字にper-point offsetを当てると字形が壊れる（例: 「2」の下の横線が崩れる）
- 訓練サーバー: homesrv (i5-9600K, GTX 1050 Ti 4GB, CUDA 12.1, PyTorch 2.5.1) — mise + uv でパッケージ管理
- Colab Pro: A100 40GB, AMP対応

### xDraw A4 ペンプロッタ
- 機種: xDraw A4（iDraw互換、Inkscape extension制御）
- USB: CH340（VID:PID=1A86:7523/8040）、115200bps
- ペン制御: Z軸（ダウン: Z5 F5000、アップ: Z0.5 F5000）※M3/M5ではない
- **筆遣いの終端Zリフト**: 払い・はねの終端区間で Z を接触(pen_down_z=5.0=最大筆圧)→半浮き(finish_lift_z=2.6)へ漸減し、シャーペンの接触圧を抜いて線を尻すぼみにする（`G1 XYZ`同時補間）。とめ＝変化なし。実機計測で芯の浮き始め≈2.7、それ以上下げても線は変わらず上がりが過大になるだけなので finish_lift_z=2.6（浮き始め直下＝最小リフト）。リフト区間は `contact_profile` が全長の `max_lift_fraction`(=0.5)で頭打ちし、短い画で全体が薄くなるのを防ぐ。接触率は `stroke_finishing.contact_profile()` が単一ソースで、G-codeのZ補間とプレビュー線幅(`compute_stroke_widths`)が連動。パラメータは `PlotterConfig`（finish_lift_z, finish_lift_length_mm, harai/hane_speed_factor）で実機キャリブ。手順は docs/plotter_gui_checklist.md
- ホーミング: `$H`（左上角に移動）→ `G92 X0 Y297 Z0`（左上角を紙座標(0,297)に設定）
- 紙座標: (0,0)=左下、(210,297)=右上
- G-code送信: Windows側で `python scripts/run_plotter_gui.py` または `python -m src.plotter_gui` で GUI を起動（src/plotter_gui/）。CH340 自動検出・ホーミング・ペンテスト・進捗表示・緊急停止が GUI 操作で可能。実機チェックリストは docs/plotter_gui_checklist.md
- 開発サーバー(192.168.86.100)からの .gcode 取得は、当面は手動 scp。GUI への取込みは Phase 2 で予定
- Extension ソース: git@github.com:tsaito18/xdraw_inkscape_extension.git
