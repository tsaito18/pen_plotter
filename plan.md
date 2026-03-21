# ペンプロッタ制御プログラム 実装計画

## Context

手書き風の文字を生成してペンプロッタで出力する制御プログラムを自作する。利用者の筆跡を学習し、同じ文字でも毎回異なる形で生成することで、本物の手書きに近い出力を実現する。主な用途はレポート作成（レポート用紙の罫線に沿った出力、数式・回路図・表も含む）。

### ユーザー環境
- **電子工作経験**: 豊富（Arduino・回路設計・機械加工に精通）
- **入力デバイス**: iPad（Apple Pencil）、紙に書いてスキャン
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
│  │ (iPad連携/   │→│  (PyTorch)    │  │  (Gradio)      │  │
│  │  スキャン取込)│  │              │  │  テキスト入力   │  │
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

### 機械構成: CoreXY方式
CoreXYを採用する理由：モーターが固定のため慣性が小さく、高速・高精度。振動が少ない。

#### 軸定義
- **X軸**: ガントリー上のキャリッジ移動方向（左右）— 1本のリニアレール
- **Y軸**: ガントリー自体の移動方向（前後）— 2本の平行リニアレール
- **Z軸**: ペン昇降（サーボ駆動）

#### トラベル量の設計根拠
A4用紙（297×210mm）を余裕をもって収めるための寸法設計：

| 項目 | 値 | 計算 |
|------|-----|------|
| Y軸レール長 | 400mm | — |
| Y軸トラベル | 355mm | 400 - 45 (MGN12Hキャリッジ長) |
| Y方向作業エリア | 300mm | A4長辺 297mm + マージン |
| **Y方向余裕** | **55mm** | 355 - 300（ホーミングオフセット含め十分） |
| X軸レール長 | 350mm | — |
| X軸トラベル | 305mm | 350 - 45 |
| X方向作業エリア | 220mm | A4短辺 210mm + マージン |
| **X方向余裕** | **85mm** | 305 - 220（十分な余裕） |

#### フレーム寸法
| 方向 | 外寸 | 内寸 | 収容物 |
|------|------|------|--------|
| 長辺（Y方向） | 500mm | 460mm | Y軸レール 400mm + 取付余裕 |
| 短辺（X方向） | 450mm | 410mm | ガントリービーム + X軸レール 350mm |

#### 部品構成（BOM）
| カテゴリ | 部品 | 数量 |
|---------|------|------|
| **フレーム** | 2020アルミフレーム 500mm | 2本 (外枠・Y方向通し) |
| | 2020アルミフレーム 460mm | 2本 (クロス・Y方向) |
| | 2020アルミフレーム 410mm | 4本 (外枠X方向 2本 + クロスX方向 2本) |
| | コーナーブラケット | 8個 |
| **駆動系** | NEMA17 ステッピングモーター | 2個 |
| | GT2タイミングベルト 6mm (5m) | 1本 |
| | GT2プーリー 20T (5mm軸) | 2個 |
| | GT2アイドラーベアリング | 6個 |
| **ガイド** | MGN12Hリニアレール 400mm | 2本 (Y軸・平行レール) |
| | MGN12Hリニアレール 350mm | 1本 (X軸・ガントリー) |
| **ペン機構** | SG90 (またはMG90S) マイクロサーボ | 1個 |
| | 3Dプリント製ペンホルダー+サーボマウント | 1式 |
| **電子部品** | Arduino UNO | 1個 |
| | CNC Shield V3 | 1枚 |
| | A4988ステッピングドライバ | 2個 |
| | 12V 5A電源 | 1個 |
| | リミットスイッチ | 3個 |

#### CAD設計
- **使用ソフト**: Autodesk Fusion
- **設計ポイント**:
  - 作業エリア: A4用紙対応（300mm × 220mm、用紙 297×210mm + マージン）
  - ベルトパス: CoreXY標準配置（4コーナーアイドラー＋2モータープーリー）
  - ペンホルダー: ペンの太さ調整可能な構造（8-12mm径対応）、サーボによる3-5mm昇降
  - 詳細は [hardware/cad_advice.md](hardware/cad_advice.md) を参照

#### GRBL設定
- firmware: `grbl-servo`（スピンドルPWMでサーボ制御）
- CoreXYキネマティクスを`config.h`で有効化してからフラッシュ
- `M3 S255` = ペンダウン、`M5` = ペンアップ
- ステップ/mm: GT2 20T + 1.8°ステッパー → 約80 steps/mm

---

## 日本語手書き生成戦略

### KanjiVG + スタイル転写アプローチ
数千の漢字をゼロから生成モデルで学習するのは非現実的。代わりに：

1. **KanjiVG** (オープンソース): 6,800以上の漢字のストロークデータ（SVG）をスケルトンとして利用
2. **スタイル転写モデル**: ユーザーの筆跡特徴（傾き・太さの変化・運筆の癖）を学習
3. **推論時**: 正規ストローク → スタイル転写 → ランダムノイズ注入 → 毎回異なる出力

### 必要なサンプル数
- ひらがな46字 + カタカナ46字（各3-5回）
- 常用漢字の代表100-200字（主要な部首をカバー、各3-5回）
- 英数字・記号（各3-5回）
- 合計: 約1,000-2,500ストロークサンプル

### サンプル収集方法
1. **iPad + Apple Pencil**: 専用アプリまたはWeb UIでストロークデータ(x,y,pressure,time)を記録 → PCにエクスポート
2. **紙スキャン**: 紙に書いた文字をスキャン → 画像からストローク軌跡を復元（Online Trajectory Recovery）

### MLモデルアーキテクチャ
- **エンコーダ**: Bi-LSTM — KanjiVGの正規ストロークを処理
- **スタイルエンコーダ**: ユーザーサンプルから128次元スタイルベクトルを抽出
- **デコーダ**: LSTM + MDN（混合密度ネットワーク） — スタイル条件付きストローク生成
- **出力**: (dx, dy) の2変量ガウス混合分布 + ペン状態のベルヌーイ分布
- **フレームワーク**: PyTorch

---

## 組版エンジン設計

### レポート用紙対応
- レポート用紙の罫線間隔（通常8mm）に文字サイズを合わせる
- 余白設定（上下左右）
- ヘッダー・フッター領域

### 数式サポート
- LaTeX記法の簡易パーサー（分数、上付き・下付き、ルート、積分記号など）
- 数式は手書き風ではなく正確な配置が必要 → KanjiVG的なテンプレート＋スタイル適用
- MathJax/KaTeXのレイアウトロジックを参考に独自実装

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
├── src/
│   ├── collector/          # サンプル収集
│   │   ├── ipad_sync.py
│   │   ├── scan_import.py
│   │   ├── stroke_recorder.py
│   │   └── data_format.py
│   ├── model/              # ML モデル
│   │   ├── stroke_model.py
│   │   ├── style_encoder.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── inference.py
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
│       └── web_app.py
├── data/
│   ├── samples/
│   ├── models/
│   ├── kanjivg/
│   └── symbols/
├── hardware/
│   ├── cad/
│   └── firmware/
├── tests/
└── notebooks/
```

---

## 開発フェーズ

### Phase 1: 基盤構築 — G-code生成 + プレビュー ✅
**目標**: ハードコードしたストロークデータからG-codeを生成し、画面上でプレビューできる

- [x] プロジェクト初期化（pyproject.toml, git, ディレクトリ構成）
- [x] `src/gcode/config.py` — 機械パラメータ定義
- [x] `src/gcode/generator.py` — ストローク列 → G-code変換（G0/G1/M3/M5）
- [x] `src/gcode/preview.py` — Matplotlibでパス可視化
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

- [x] `src/layout/page_layout.py` — 用紙サイズ、余白、レポート罫線
- [x] `src/layout/typesetter.py` — 文字配置（全角グリッド、ベースライン）
- [x] `src/layout/line_breaking.py` — 改行・禁則処理
- [x] `src/layout/math_layout.py` — LaTeX簡易数式レイアウト（分数・上下付き等）
- [x] `src/layout/table_layout.py` — 表・罫線描画
- [x] テスト: サンプルテキストのレイアウト結果をMatplotlibで確認

### Phase 5: 手書きサンプル収集 ✅
**目標**: ユーザーの筆跡をストロークデータとして取得・保存

- [x] `src/collector/ipad_sync.py` — iPad向けWeb UI（Canvas API）でストローク収集
- [ ] `src/collector/scan_import.py` — スキャン画像からストローク軌跡復元
- [x] `src/collector/stroke_recorder.py` — ストロークの正規化・保存
- [x] `src/collector/data_format.py` — JSON形式定義
- [x] KanjiVGデータのダウンロード・パース（6,699文字変換済み）
- [x] `src/collector/casia_parser.py` — CASIA-OLHWDB .pot ファイルパーサー
- [x] ガイド付きストローク収集UI（292文字セット、進捗表示）
- [x] テスト: 全収集機能

### Phase 6: MLモデル ✅
**目標**: 正規ストローク → ユーザースタイルのストローク生成

- [x] `src/model/dataset.py` — KanjiVG正規データ + ユーザーサンプルのペアリング（複数ディレクトリ対応）
- [x] `src/model/style_encoder.py` — スタイル特徴抽出（Bi-LSTM → 128dim）
- [x] `src/model/stroke_model.py` — LSTM + MDN（char_embedding対応）
- [x] `src/model/char_encoder.py` — KanjiVGスケルトン→文字埋め込み（Bi-LSTM → 128dim）
- [x] `src/model/train.py` — 訓練ループ（勾配クリップ、学習率スケジューラ）
- [x] `src/model/pretrain.py` — 事前訓練（CharEncoder+StyleEncoder+Generator同時訓練）
- [x] `src/model/finetune.py` — ファインチューニング（StyleEncoderのみ、Generator凍結）
- [x] `src/model/inference.py` — 推論（V1/V2自動検出、温度パラメータでバリエーション制御）
- [x] `src/model/augmentation.py` — リアルさ追加（ベースライン揺らぎ、サイズ変動、ジッター、傾き）

### Phase 7: エンドツーエンド統合（部分完了）
**目標**: テキスト入力 → 手書き生成 → 組版 → G-code → 印刷の一気通貫

- [x] `src/ui/web_app.py` — Gradio Web UI（テキスト入力、プレビュー、印刷ボタン）
- [x] 3段階フォールバック（ML推論→KanjiVG→矩形）
- [x] V2パイプライン（CharEncoder+文字条件付き推論）統合
- [x] GPU(XPU)デバイス対応（pretrain.py/finetune.pyに--device引数実装済み、CUDA/XPU/CPU自動検出）
- [ ] CASIA-OLHWDB事前訓練（ストローク単位生成アーキテクチャで100k samples, 80 epochs完了。300k samples訓練をColab Pro (A100)で実行中）
- [ ] パイプライン統合テスト（実データでのエンドツーエンド）
- [ ] 全体の品質調整（文字サイズ、間隔、速度）

### Phase 8: 拡張機能（後回し）
- [ ] 回路図記号テンプレート + 手書き風変形
- [ ] ラズパイスタンドアロンコントローラ
- [ ] 複数ペン対応（色替え）
- [ ] 筆圧シミュレーション（サーボ角度微調整で太さ変化）

---

## 手書き生成 V2 アーキテクチャ

### 現状の課題と改修方針
V1モデルは文字を区別できない（スタイルサンプルから「何かストローク」を生成するだけ）。
V2では文字条件付き生成に改修し、レポート提出に使える品質を目指す。

### アーキテクチャ
```
CharEncoder(KanjiVG_skeleton) → char_embedding（何の文字か）
StyleEncoder(user_samples) → style_vector（どんな書き癖か）
StrokeGenerator(char_embedding + style_vector) → strokes
```

### 訓練戦略
1. **事前訓練**: CASIA-OLHWDB（160万サンプル）or KanjiVG（6,699字）で CharEncoder + Generator を訓練
2. **ファインチューニング**: ユーザー20-30文字で StyleEncoder のみ更新（Generator凍結）

### リアルさの要素（augmentation.py で実装済み）
- 同一文字バリエーション: MDN温度 + ノイズ
- 行の揺らぎ: ベースライン・文字サイズにランダム変動
- ジッター: ストロークに微振動を加える（スムージング付き）
- 傾き: 文字ごとにランダムな傾きを適用

### 実装ステップ（全完了）
- [x] A. CASIA データローダー (`src/collector/casia_parser.py`) — 8テスト
- [x] B. CharEncoder (`src/model/char_encoder.py`) — 9テスト
- [x] C. StrokeGenerator 改修 (`src/model/stroke_model.py` char_dim追加) — 5テスト
- [x] D. 事前訓練パイプライン (`src/model/pretrain.py` + `scripts/pretrain.py`) — 9テスト
- [x] E. ファインチューニング (`src/model/finetune.py` + `scripts/finetune.py`) — 10テスト
- [x] F. リアルさ追加 (`src/model/augmentation.py`) — 14テスト
- [x] G. パイプライン統合V2 (`inference.py` V1/V2自動検出 + `web_app.py`) — 9テスト

### 次のアクション
1. ~~GPU対応~~ ✅ 完了
2. ~~CASIA取得~~ ✅ 完了 (train 816 .pot files, test 204 .pot files)
3. **事前訓練** → Colab Pro (A100) で300k samples, 80 epochs実行中
4. **ユーザーサンプル収集**: collect_strokes.py（20-30文字）→ 次のステップ
5. **ファインチューニング → プレビュー確認** → 未着手

### V2 主要バグ修正履歴
- 絶対座標→相対座標(delta)変換（Graves方式）
- ストローク境界のpen_stateエンコード
- mean=0, std=1正規化（チェックポイントにstats保存）
- num_mixtures 5→20、log_softmax、rhoクランプ、pen BCE pos_weight
- エンコーダ崩壊修正（style分離, embedding_variance_loss, packed sequences, LayerNorm, forget gate bias=1.0）
- pen_state問題→ストローク単位生成に全面改修
- char_embeddingをLSTM初期hidden stateに注入
- ストローク位置のオフセット、長さ制限

### GPU環境情報
- Intel Core Ultra 7 258V (Lunar Lake) — Intel Arc 統合GPU
- WSL2では/dev/dri/が未認識（2026-03-14時点）
- **Windowsネイティブ**: PyTorch XPU版で `torch.xpu.is_available() = True` 確認済み
- IPEX (Intel Extension for PyTorch) はWindows非対応、PyTorch XPU版単体で使用
- Python 3.12 (uv venv)、WSLからのファイルアクセス: `\\wsl$\Ubuntu\home\taiga\Personal\pen_plotter`
- **homesrv**: i5-9600K, GTX 1050 Ti 4GB, CUDA 12.1, PyTorch 2.5.1
- **Colab Pro**: A100 40GB, AMP対応

---

## リスクと対策

| リスク | 対策 |
|-------|------|
| 日本語手書き品質が訓練データ不足で低い | KanjiVGスケルトン＋スタイル転写で必要データ量を削減。品質不足時はランダム摂動ベースラインにフォールバック |
| CoreXYのベルトアライメント精度 | リニアレール（ロッド+ベアリングではなく）で高精度化。大きい文字から始めて徐々に小さく |
| MDN訓練の不安定性（NaN loss） | 勾配クリップ(max norm 5.0)、低学習率開始、既知の安定設定を参考に |
| スキャンからのストローク復元精度 | iPadでの直接入力を優先。スキャンは補助的手段として位置づけ |
| 数式・回路図の手書き風と正確性のバランス | 構造配置は正確に、線の描画のみ手書き風変形を適用 |

---

## 検証方法

1. **G-code検証**: 生成G-codeをプレビューで描画し、期待パスと一致確認
2. **機械精度**: キャリブレーションパターン描画 → ノギスで実測
3. **手書き品質**: 生成文字とユーザー実筆の目視比較
4. **バリエーション**: 同一テキスト5回生成、全出力が視覚的に異なることを確認
5. **統合テスト**: A4レポート1ページ分生成 → 通信エラーなし・品質一貫性確認
