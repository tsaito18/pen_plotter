# ペンプロッタ制御プログラム 実装計画

## Context

手書き風の文字を生成してペンプロッタで出力する制御プログラムを自作する。利用者の筆跡を学習し、同じ文字でも毎回異なる形で生成することで、本物の手書きに近い出力を実現する。主な用途はレポート作成（レポート用紙の罫線に沿った出力、数式・回路図・表も含む）。

### ユーザー環境
- **電子工作経験**: 豊富（Arduino・回路設計・機械加工に精通）
- **入力デバイス**: iPad（Apple Pencil）
- **利用可能機材**: ノートPC、ラズパイ（タッチスクリーン付）、Arduino、ESP32、3Dプリンター
- **対象文字**: 日本語（漢字・ひらがな・カタカナ）＋英数字＋数式・回路図・表

---

## システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                      LAPTOP (メイン)                      │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ サンプル収集  │  │  ML訓練       │  │  Web UI        │  │
│  │ (iPad連携)   │→│  (PyTorch)    │  │  (Gradio)      │  │
│  └─────────────┘  └──────┬───────┘  └───────┬────────┘  │
│                          │                   │            │
│                  ┌───────▼───────────────────▼─────────┐ │
│                  │  手書き生成エンジン (推論+バリエーション) │ │
│                  └──────────────┬──────────────────────┘ │
│                                │                         │
│                  ┌─────────────▼──────────────────────┐  │
│                  │  組版エンジン                        │  │
│                  │  - レポート用紙レイアウト (罫線対応)   │  │
│                  │  - 数式レイアウト (LaTeX式 → 配置)    │  │
│                  │  - 表・罫線描画                      │  │
│                  │  - 回路図記号配置                     │  │
│                  └──────────────┬─────────────────────┘  │
│                                │                         │
│                  ┌─────────────▼──────────────────────┐  │
│                  │  G-code生成 + パス最適化             │  │
│                  └──────────────┬─────────────────────┘  │
└─────────────────────────────────┼─────────────────────────┘
                                  │ G-code (USB/WiFi)
┌─────────────────────────────────▼─────────────────────────┐
│  ARDUINO UNO + CNC Shield                                  │
│  GRBL (grbl-servo) + CoreXY + A4988×2 + SG90サーボ         │
└────────────────────────────────────────────────────────────┘
```

### 役割分担
| デバイス | 役割 |
|---------|------|
| **ノートPC** | ML訓練・推論、UI、G-code生成、全ソフトウェア処理 |
| **Arduino UNO** | GRBL firmware でモーター制御のみ |
| **ラズパイ** | オプション: スタンドアロン印刷コントローラ（後回し） |
| **3Dプリンター** | ペンホルダー・サーボマウント等のカスタムパーツ製作 |

---

## ハードウェア設計

### 機種: xDraw A4 ペンプロッタ
市販のペンプロッタ xDraw A4 を使用。Inkscape extension（iDraw互換）で制御。

#### 基本仕様
| 項目 | 値 |
|------|-----|
| ワーキングエリア | 210 × 297mm（A4サイズ） |
| USBチップ | CH340（VID:PID = 1A86:7523 or 1A86:8040） |
| ボーレート | 115200 |
| ファームウェア | GRBL互換（DrawCore） |
| ステップ解像度 | X/Y: 100 steps/mm, Z: 85.8 steps/mm |
| 最大速度 | X: 15000, Y: 12000, Z: 15000 mm/min |
| 加速度 | 3000 mm/s² (全軸) |

#### GRBL設定（`$$` で取得）
```
$22=1   (ホーミング有効)
$23=5   (ホーミング方向: X+ Y+)
$100=100, $101=100, $102=85.8  (steps/mm)
$110=15000, $111=12000, $112=15000  (最大速度 mm/min)
$120=3000, $121=3000, $122=3000  (加速度 mm/s²)
$130=210, $131=297, $132=10  (最大移動量 mm)
```

#### ペン制御
- **ペンアップ**: `G1G90 Z0.5 F5000`（Z軸を0.5mm位置に）
- **ペンダウン**: `G1G90 Z5 F5000`（Z軸を5mm位置に）
- ※ M3/M5（スピンドルPWM）方式ではなくZ軸制御

#### ホーミング＆座標系
ホーミング（`$H`）で紙の左上角に移動。その後 X+210mm に相対移動して右上角（作業原点）へ。
紙座標で正の値を使うために `G92` でオフセットを設定する。

**初期化シーケンス:**
```gcode
$H                          ; ホーミング（左上角のリミットスイッチまで移動）
G4 P1                       ; 安定待ち
G92 X0 Y297 Z0              ; 現在位置（左上角）を紙座標(0,297)に設定
G90                          ; 絶対座標モード
G1G90 Z0.5 F5000            ; ペンアップ
```

**紙座標系:**
```
(0,297) 左上 ← $H    (210,297) 右上
  ┌──────────────────────┐
  │       紙 (A4)         │
  │    X→ 210mm           │
  │    Y↑ 297mm           │
  └──────────────────────┘
(0,0) 左下            (210,0) 右下
```

#### Extension ソースコード
- リポジトリ: git@github.com:tsaito18/xdraw_inkscape_extension.git
- 主要ファイル: `idraw_deps/drawcore_plotink/drawcore_motion.py`（G-code送信）
- 設定: `idraw2_0_conf.py`（速度・ペン位置等のデフォルト値）

---

## 日本語手書き生成戦略

### KanjiVG + スタイル転写アプローチ
数千の漢字をゼロから生成モデルで学習するのは非現実的。代わりに：

1. **KanjiVG** (オープンソース): 6,800以上の漢字のストロークデータ（SVG）をスケルトンとして利用
2. **スタイル転写モデル**: ユーザーの筆跡特徴（傾き・太さの変化・運筆の癖）を学習
3. **推論時**: 正規ストローク → スタイル転写 → ランダムノイズ注入 → 毎回異なる出力

### 必要なサンプル数
- ひらがな46字 + カタカナ46字 + 濁音・記号（各3回）
- 理工系レポート頻出漢字150字以上（各3回）
- 英数字・記号（各3回）
- 合計: 381文字セット、目標1,000+ストロークサンプル

### サンプル収集方法
- **iPad + Apple Pencil**: 専用Web UI（横画面対応）でストロークデータ(x,y,pressure,time)を記録
- KanjiVGお手本表示（色分け書き順ガイド）、自動進行、優先度順収集
- ※スキャン画像からのストローク復元は品質不足で断念（iPadデータのみ使用）

### MLモデルアーキテクチャ（V3 StrokeDeformer）
- **StyleEncoder**: ユーザーサンプルから128次元スタイルベクトルを抽出（Bi-LSTM）
- **StrokeDeformer**: MLP per-point offset予測（参照点+正規化位置t+局所曲率+style+stroke_embed → offset(N,2)）
- **後処理**: smooth_offsets(kernel=11) → clamp(±0.6)
- **フレームワーク**: PyTorch
- **訓練**: ユーザーデータのみ（CASIA不使用）、BaseFinetunerテンプレートメソッドパターン

---

## 組版エンジン設計

### レポート用紙対応
- レポート用紙の罫線間隔7.16mm（実測値）に文字サイズ4.5mmを合わせる
- 余白: 上48mm、下34mm、左5mm、右5mm（実際の用紙に合わせて自動検出）
- 背景にスキャンしたレポート用紙画像を使用

### 文字バランス
- 漢字: 100%、ひらがな/カタカナ: 85%、小書き: 55%、半角: 55%
- 配置間隔: 全角 font_size×0.9（詰め）、半角 font_size×0.5

### 見出し・インデント
- Markdown見出し対応: # h1(1.15倍), ## h2(1.08倍), ### h3(1.0倍)
- 階層インデント: h1見出し15mm/本文25mm、h2見出し25mm/本文35mm、h3見出し35mm/本文45mm
- 右端: 200mm（右から10mm）
- ページ番号: 手書きストロークで「P.」欄の右に描画

### 数式サポート
- LaTeX記法の簡易パーサー（\frac, ^, _, \sqrt, \symbol, ブレースなし記法対応）
- インライン数式 $...$、ブロック数式 $$...$$ (中央配置)
- ギリシャ文字ストローク生成（ω, φ, π, θ, α, Δ）
- 分数線の自動描画（numerator/denominator roleで検出）

### 回路図・表
- 回路図: 基本記号（抵抗、コンデンサ、ダイオード等）をSVGテンプレートとして用意、手書き風変形を適用
- 表: 罫線描画 + セル内文字配置

### 禁則処理
- 日本語禁則処理（行頭禁止文字: 。、）」等、行末禁止文字: （「等）

---

## ディレクトリ構成

```
pen_plotter/
├── CLAUDE.md
├── pyproject.toml
├── plan.md
├── Makefile
├── src/
│   ├── collector/          # サンプル収集
│   │   ├── ipad_sync.py
│   │   ├── templates/collector.html
│   │   ├── stroke_recorder.py
│   │   ├── data_format.py
│   │   ├── kanjivg_parser.py
│   │   └── casia_parser.py
│   ├── model/              # ML モデル
│   │   ├── stroke_deformer.py
│   │   ├── stroke_aligner.py
│   │   ├── style_encoder.py
│   │   ├── dataset.py
│   │   ├── data_utils.py
│   │   ├── finetune.py
│   │   ├── pretrain.py
│   │   ├── inference.py
│   │   ├── augmentation.py
│   │   ├── stroke_model.py
│   │   ├── char_encoder.py
│   │   └── train.py
│   ├── layout/             # 組版エンジン
│   │   ├── typesetter.py
│   │   ├── page_layout.py
│   │   ├── math_layout.py
│   │   ├── table_layout.py
│   │   └── line_breaking.py
│   ├── gcode/              # G-code生成
│   │   ├── generator.py
│   │   ├── optimizer.py
│   │   ├── preview.py
│   │   └── config.py
│   ├── comm/               # シリアル通信
│   │   ├── serial_sender.py
│   │   └── grbl_controller.py
│   └── ui/                 # ユーザーインターフェース
│       ├── web_app.py
│       ├── gradio_app.py
│       ├── stroke_renderer.py
│       └── preview_renderer.py
├── data/
│   ├── user_strokes/       # ユーザー手書きサンプル
│   ├── models/             # 訓練済みチェックポイント
│   ├── strokes/            # KanjiVGストロークデータ
│   └── report_paper.jpg    # レポート用紙スキャン（プレビュー背景）
├── scripts/
│   ├── collect_strokes.py
│   ├── run_ui.py
│   ├── pretrain.py
│   ├── finetune.py
│   ├── prepare_kanjivg.py
│   └── train_model.py
├── hardware/
│   ├── cad/
│   └── firmware/
└── tests/
```

---

## 開発フェーズ

### Phase 1: 基盤構築 — G-code生成 + プレビュー ✅
**目標**: ハードコードしたストロークデータからG-codeを生成し、画面上でプレビューできる

- [x] プロジェクト初期化（pyproject.toml, git, ディレクトリ構成）
- [x] `src/gcode/config.py` — 機械パラメータ定義
- [x] `src/gcode/generator.py` — ストローク列 → G-code変換（G0/G1/M3/M5）+ S字加速
- [x] `src/gcode/preview.py` — Matplotlibでパス可視化（太さ変調付き）
- [x] `src/gcode/optimizer.py` — ペンアップ移動距離の最小化（nearest-neighbor）
- [x] テスト: 簡単な図形（四角、円、文字）のG-code生成・プレビュー確認

### Phase 2: ハードウェア製作（ソフトと並行）
**目標**: G-codeファイルを入力して実際に描画できるプロッタの完成

- [ ] CAD設計（フレーム、ペンホルダー、モーターマウント）
- [ ] フレーム組み立て（2020アルミ + リニアレール）
- [ ] CoreXYベルト配線
- [ ] Arduino + CNC Shield配線（A4988×2 + サーボ）
- [ ] `grbl-servo` firmware フラッシュ（CoreXY有効化）
- [ ] GRBL設定（steps/mm、最大速度、加速度）
- [ ] ペンホルダー3Dプリント・取付
- [ ] キャリブレーション: 100mm四角描画 → 測定 → steps/mm調整

### Phase 3: シリアル通信 ✅
**目標**: PCからG-codeをストリーミングして描画

- [x] `src/comm/serial_sender.py` — GRBLストリーミングプロトコル（character-counting方式）
- [x] `src/comm/grbl_controller.py` — ホーミング、ステータス、設定
- [ ] テスト: Phase 1のG-codeを実機で描画

### Phase 4: 組版エンジン ✅
**目標**: テキストを受け取り、ページ上に文字を配置

- [x] `src/layout/page_layout.py` — 用紙サイズ、余白、レポート罫線（実測7.16mm間隔）
- [x] `src/layout/typesetter.py` — 文字配置（漢字/かなサイズバランス、見出しインデント、段落字下げ）
- [x] `src/layout/line_breaking.py` — 改行・禁則処理
- [x] `src/layout/math_layout.py` — LaTeX簡易数式レイアウト（分数・上下付き・ブロック/インライン）
- [x] `src/layout/table_layout.py` — 表・罫線描画
- [x] テスト: サンプルテキストのレイアウト結果をMatplotlibで確認

### Phase 5: 手書きサンプル収集 ✅
**目標**: ユーザーの筆跡をストロークデータとして取得・保存

- [x] `src/collector/ipad_sync.py` — iPad向けWeb UI（横画面対応、KanjiVGお手本表示、自動進行、優先度順収集）
- [x] `src/collector/stroke_recorder.py` — ストロークの正規化・保存
- [x] `src/collector/data_format.py` — JSON形式定義
- [x] KanjiVGデータのダウンロード・パース（6,699文字変換済み）
- [x] `src/collector/casia_parser.py` — CASIA-OLHWDB .pot ファイルパーサー
- [x] ガイド付きストローク収集UI（381文字セット、Tier1/2/3優先度、進捗バー）
- [x] Apple Pencilのみ入力受付（パームリジェクション）
- [x] HTML/CSS/JSテンプレート分離（templates/collector.html）
- [x] テスト: 全収集機能

### Phase 6: MLモデル ✅
**目標**: 正規ストローク → ユーザースタイルのストローク生成

- [x] `src/model/dataset.py` — KanjiVG正規データ + ユーザーサンプルのペアリング
- [x] `src/model/style_encoder.py` — スタイル特徴抽出（Bi-LSTM → 128dim）
- [x] `src/model/stroke_deformer.py` — MLP per-point offset予測（局所曲率特徴付き）
- [x] `src/model/stroke_aligner.py` — Hungarian + MHD ストロークアライメント
- [x] `src/model/finetune.py` — BaseFinetuner + 3サブクラス（Finetuner/DeformationFinetuner/UserDeformationTrainer）
- [x] `src/model/pretrain.py` — 事前訓練
- [x] `src/model/inference.py` — 推論（V1/V2/V3自動検出、バッチ推論）
- [x] `src/model/augmentation.py` — 弾性変形、tremor、ベースライン揺らぎ等
- [x] train-inference gap解消（smooth+clampを共有定数/関数で統一）

### Phase 7: エンドツーエンド統合 ✅
**目標**: テキスト入力 → 手書き生成 → 組版 → G-code → 印刷の一気通貫

- [x] `src/ui/web_app.py` — PlotterPipeline（薄いオーケストレータ）
- [x] `src/ui/stroke_renderer.py` — StrokeRenderer（文字→ストローク生成、直接ストローク使用）
- [x] `src/ui/preview_renderer.py` — PreviewRenderer（レポート用紙背景プレビュー）
- [x] `src/ui/gradio_app.py` — Gradio Web UI（タブUI・設定パネル・プログレス・ヘルプ・例文）
- [x] 直接ストローク使用（収集済み文字は実ストロークをサンプル単位でランダム選択）
- [x] ML推論フォールバック（未収集文字のみStrokeDeformer使用）
- [x] 数式レイアウト統合（インライン/ブロック、ギリシャ文字、分数線）
- [x] セクション見出し（#/##/### → 階層インデント）
- [x] マルチページプレビュー（Gallery表示）
- [x] ページ番号（手書きストローク）
- [x] レポート用紙背景（data/report_paper.jpg自動ロード）
- [x] Makefile（make ui/collect/train/preview/test/lint/format）
- [ ] パイプライン統合テスト（実データでのエンドツーエンド）
- [ ] 実機テスト（G-code出力→ペンプロッタ印刷）

### Phase 8: 拡張機能（後回し）
- [ ] 回路図記号テンプレート + 手書き風変形
- [ ] ラズパイスタンドアロンコントローラ
- [ ] 複数ペン対応（色替え）
- [ ] 筆圧シミュレーション（サーボ角度微調整で太さ変化）

---

## 手書き生成 V3 アーキテクチャ（スタイル転写）

### V2 からの方針転換
V2（LSTM+MDN自己回帰生成）はストロークをゼロから生成するアプローチだったが、
CJK文字の複雑さに対して根本的に限界があった（300k samples, A100, 80 epochs でも文字として読めない品質）。
V3ではKanjiVGの正しい形を前提に、スタイル転写（変形）に切り替える。

### アーキテクチャ
```
KanjiVG参照ストローク → リサンプリング（32点）
StyleEncoder(user_samples) → style_vector（どんな書き癖か）
StrokeDeformer(参照点 + 正規化位置t + 局所曲率 + style_vector + stroke_embed) → オフセット (N, 2)
    ↓ smooth_offsets(kernel=11) → clamp(±0.6)
参照点 + オフセット → 変形済みストローク
+ ストローク単位の幾何バリエーション（回転±0.9°・スケール±0.9%・シフト±3%）
+ 文字配置の揺らぎ（ベースライン・字間・サイズ・傾き）
+ ストローク太さ変化（始筆太→終筆細）
```

### 推論時の文字生成フォールバック
1. **句読点幾何生成**（、。・）— 常に最優先
2. **ユーザー直接ストローク** — 収集済み文字はサンプル単位でランダム選択
3. **ML推論（StrokeDeformer）** — 未収集文字のみ
4. **括弧・数学記号の幾何生成**
5. **KanjiVGフォールバック**
6. **矩形フォールバック**

### 座標系（重要：過去にバグの原因）
- KanjiVG JSON (`data/strokes/`): **Y-UP**（`prepare_kanjivg.py` で `y = target_size - y` 反転済み）
- ユーザーストローク (`data/user_strokes/`): **Y-DOWN**（iPad Canvas座標そのまま保存）
- 訓練時は `FinetuneDeformationDataset` でユーザーストロークのY軸を反転しKanjiVGと統一
- 推論出力・G-code: Y-UP（0,0=左下）。matplotlibプレビューで `invert_yaxis()` は不要

### 訓練戦略
- **ユーザーデータ直接訓練**: CASIA事前訓練は中国語/日本語ストロークの不一致によりノイズが大きく断念
- ユーザーの手書きサンプル（381文字セット）でStrokeDeformerを直接訓練
- CASIAを使わないため、ユーザーサンプルの量が品質に直結
- ※スキャン画像からのストローク復元も試行したが品質不足で断念

### 滑らかさの確保
- **スムージング**: オフセット予測後に1D移動平均（kernel_size=11）を訓練/推論で統一適用
- **オフセットクランプ**: ±0.6に制限（±0.3→±0.8→±1.2→±0.6と調整）
- **訓練時 smoothness loss**: 隣接点オフセット差の二乗和を正則化項として追加
- **train-inference gap解消**: smooth_offsets()とOFFSET_CLAMPを共有定数/関数として統一

### リアルさの要素
- ストローク単位の幾何バリエーション: 回転(±0.9°)・スケール(±0.9%)・シフト(±3%)をストロークごとに適用
- augmentation: ベースライン揺らぎ(0.15mm)、サイズ変動(2%)、ジッター(0.03mm)、傾き(0.02rad)、字間変動(0.05mm)
- ストローク太さ変化: 始筆1.0→終筆0.7（プレビュー: LineCollection、G-code: S字フィードレート変調）
- 弾性変形(amplitude=0.2%)、手の震え(tremor=0.01mm, 3-5Hz)
- 行密度変動: 行ごとにspacing密度スケールを変動（±5%）

### 実装ステップ
- [x] A. リサンプリング・アライメント (`data_utils.py`)
- [x] B. StrokeDeformer (`stroke_deformer.py`) — MLP per-point offset予測 + 局所曲率
- [x] C. 事前訓練パイプライン更新 (`pretrain.py`)
- [x] D. ファインチューニング更新 (`finetune.py`) — BaseFinetunerテンプレートメソッド
- [x] E. 推論更新 (`inference.py` V3検出、バッチ推論)
- [x] F. スムージング（訓練/推論統一: 移動平均kernel=11 + クランプ±0.6）
- [x] G. SVGパーサー修正（smooth cubic bezier s/S コマンド対応）
- [x] H. ユーザーデータ直接訓練（CASIA不使用）
- [x] I. 組版改善（レポート用紙実測値、文字サイズバランス、見出しインデント）
- [x] J. ストローク単位の幾何バリエーション（回転・スケール・シフト）
- [x] K. ストロークアライメント（Hungarian + MHD + マージ/スプリット検出）
- [x] L. 直接ストローク使用 + 弾性変形 + tremor
- [x] M. 数式レイアウト統合（インライン$...$, ブロック$$...$$, ギリシャ文字, 分数線）
- [x] N. リファクタリング（HTML分離、web_app分割、typesetter分割、finetune重複削除）

### V2 (LSTM+MDN) の結論
V2は300k samples, A100, 80 epochsでも文字として読めない品質だった。
LSTM+MDN自己回帰生成はCJK文字に対して根本的に限界がある。
AffineStrokeDeformer（ストローク単位アフィン変換）も試したが、歪みが大きすぎて断念。

### ストローク合成の結論
ストローク単位の合成（異なるサンプルのストロークを混ぜる）は、画数の違い・書き順の不一致により
二重ストローク・濁点欠落等のアーティファクトが頻発し断念。
KanjiVGアンカー方式（StrokeAlignerで座標系を統一）も試したが根本解決には至らず。
サンプル単位のランダム選択 + 幾何バリエーションで十分なバリエーションが確保できる。

### 現在の状態（2026-04-01）
- v3-user モデル動作中（局所曲率特徴、クランプ±0.6）
- 381文字 / 925+サンプル収集済み（data/user_strokes/）
- 直接ストローク使用（サンプル単位ランダム選択 + 幾何バリエーション）
- レポート用紙実測レイアウト（罫線7.16mm、余白48/34/5/5mm、font_size 4.5mm）
- 数式レイアウト統合済み（インライン/ブロック/ギリシャ文字/分数線）
- セクション見出し・階層インデント・マルチページ・ページ番号実装済み
- レポート用紙背景プレビュー（data/report_paper.jpg）
- リファクタリング済み（HTML分離、責務分割、重複削除）
- 621+テスト全パス

### 本番モデル（2026-04-14時点）
- **deformer_type=transformer** を採用（381文字/925サンプル訓練）
- MLPより文字構造の整合性が高い（点間協調変形）
- OFFSET_CLAMP=0.5, SMOOTHING_KERNEL_SIZE=15
- 推論時のスムージング: 角検出（cos<0.85）+ セグメント内CubicSpline補間

### 完了TODO
- [x] **少量サンプル対応（Phase 9）** — 完了
  - [x] A. Contrastive StyleEncoder（SupCon loss + ProjectionHead）
  - [x] B. Transformer Deformer（Self-Attention + Cross-Attention multi-token KV）
  - [x] C. 訓練パイプライン統合（contrastive warmup + transformer dispatch）
  - [x] D. 推論パイプライン更新（deformer_type="transformer"対応）
  - [x] E. DeformationFinetuner互換性
  - [x] F. 少量データ（30文字）+ 381文字でのABテスト → Transformer採用

### 残りTODO
- [ ] **手書きらしさ改善（将来検討）**
  - [ ] Clamp値の段階的緩和（訓練初期±0.6→後期±1.2）or 学習可能clamp
  - [ ] 2段階変形（Affine per-stroke → per-point offset）で大域+局所変形を分離
  - [ ] KanjiVGテンプレートに代わる手書きプロトタイプ学習（構造保証との両立が課題）
- [ ] ユーザーサンプル追加収集（全文字3サンプル化）
- [ ] 実機テスト（G-code出力→ペンプロッタで印刷）
- [ ] 回路図記号テンプレート（Phase 8）

### 次のアクション
1. **少量サンプル対応（Phase 9）** — Contrastive StyleEncoder + Transformer Deformer 実装
2. **実機テスト** — G-code出力→ペンプロッタで印刷
3. **文字バランスの継続調整** — 参考レポートとの比較で微調整

### GPU環境情報
- Intel Core Ultra 7 258V (Lunar Lake) — Intel Arc 統合GPU
- WSL2では/dev/dri/が未認識（2026-03時点）
- **Windowsネイティブ**: PyTorch XPU版で `torch.xpu.is_available() = True` 確認済み
- IPEX (Intel Extension for PyTorch) はWindows非対応、PyTorch XPU版単体で使用
- Python 3.12 (uv venv)、WSLからのファイルアクセス: `\\wsl$\Ubuntu\home\taiga\Personal\pen_plotter`
- **homesrv**: i5-9600K, GTX 1050 Ti 4GB, CUDA 12.1, PyTorch 2.5.1
- **Colab Pro**: A100 40GB, AMP対応

---

## リスクと対策

| リスク | 対策 |
|-------|------|
| 日本語手書き品質が訓練データ不足で低い | KanjiVGスケルトン＋スタイル転写で必要データ量を削減。品質不足時は直接ストロークにフォールバック |
| CoreXYのベルトアライメント精度 | リニアレール（ロッド+ベアリングではなく）で高精度化。大きい文字から始めて徐々に小さく |
| ストローク合成のアーティファクト | 合成は断念し、サンプル単位ランダム選択を採用 |
| スキャンからのストローク復元精度 | iPadでの直接入力のみ使用（スキャン復元は品質不足で断念） |
| 数式・回路図の手書き風と正確性のバランス | 構造配置は正確に、線の描画のみ手書き風変形を適用 |

---

## 検証方法

1. **G-code検証**: 生成G-codeをプレビューで描画し、期待パスと一致確認
2. **機械精度**: キャリブレーションパターン描画 → ノギスで実測
3. **手書き品質**: 生成文字とユーザー実筆の目視比較
4. **バリエーション**: 同一テキスト5回生成、全出力が視覚的に異なることを確認
5. **統合テスト**: A4レポート1ページ分生成 → 通信エラーなし・品質一貫性確認
