# ペンプロッタ機械設計 — Autodesk Fusion CAD 作業アドバイス

## 目次
1. [Fusion プロジェクトの初期設定](#1-fusion-プロジェクトの初期設定)
2. [パラメトリック設計（最重要）](#2-パラメトリック設計最重要)
3. [コンポーネント構成とファイル管理](#3-コンポーネント構成とファイル管理)
4. [2020 アルミフレームのモデリング](#4-2020-アルミフレームのモデリング)
5. [CoreXY ベルト経路設計（最も失敗しやすい）](#5-corexy-ベルト経路設計最も失敗しやすい)
6. [リニアレール（MGN12H）の取り付け設計](#6-リニアレールmgn12hの取り付け設計)
7. [3D プリント部品の設計ルール](#7-3d-プリント部品の設計ルール)
8. [ペンホルダー / Z軸サーボ機構](#8-ペンホルダー--z軸サーボ機構)
9. [フレーム設計と直角精度](#9-フレーム設計と直角精度)
10. [ジョイント（拘束）の使い方](#10-ジョイント拘束の使い方)
11. [標準部品の取り込み](#11-標準部品の取り込み)
12. [STL エクスポートと印刷準備](#12-stl-エクスポートと印刷準備)
13. [設計レビューチェックリスト](#13-設計レビューチェックリスト)
14. [参考リソース](#14-参考リソース)

---

## 1. Fusion プロジェクトの初期設定

### 単位系
- **mm** を使用（`Document Settings > Units > mm`）
- ペンプロッタは mm 単位が自然（GRBL も mm ベース）

### デザイン履歴
- **必ず ON**（タイムライン表示）にしておく
- パラメトリック設計の恩恵を受けるために必須
- 重くなったら個別コンポーネントだけ履歴停止（右クリック > Do Not Capture Design History）

### バージョン管理
- 大きな変更の前に Fusion 内で **バージョン保存**（`File > Save` でバージョンが作られる）
- 命名規則例: `PenPlotter_v1_frame`, `PenPlotter_v2_gantry`

---

## 2. パラメトリック設計（最重要）

### なぜ重要か
寸法をハードコードすると、フレーム長を 500→450mm に変えたいとき数十箇所を手動修正する羽目になる。**最初にパラメータテーブルを作る**ことで、1箇所変えれば全体が追従する。

### パラメータテーブルの作成
`Modify > Change Parameters` で以下を定義：

| パラメータ名 | 値 | 説明 |
|---|---|---|
| `extrusion_size` | `20 mm` | 2020 フレームの断面寸法 |
| `frame_x` | `450 mm` | X 方向フレーム外寸（短辺） |
| `frame_y` | `500 mm` | Y 方向フレーム外寸（長辺） |
| `work_area_x` | `220 mm` | 作業エリア X（A4 短辺 210mm + マージン） |
| `work_area_y` | `300 mm` | 作業エリア Y（A4 長辺 297mm + マージン） |
| `rail_length_y` | `400 mm` | Y 軸リニアレール長（平行 2 本） |
| `rail_length_x` | `350 mm` | X 軸リニアレール長（ガントリー） |
| `rail_end_clearance` | `10 mm` | レール端部の余裕 |
| `belt_pitch` | `2 mm` | GT2 ベルトピッチ |
| `pulley_teeth` | `20` | GT2 プーリー歯数 |
| `belt_plane_gap` | `9 mm` | 上下ベルト面の間隔 |
| `print_clearance` | `0.25 mm` | 3D プリント公差（片側） |
| `pen_travel` | `5 mm` | ペン昇降ストローク |
| `nema17_size` | `42 mm` | NEMA17 モーターの幅 |
| `mgn12_carriage_len` | `45 mm` | MGN12H キャリッジ長 |

### 計算式パラメータの活用
```
rail_mount_offset = extrusion_size + rail_end_clearance
travel_x = rail_length_x - mgn12_carriage_len    // 350 - 45 = 305mm
travel_y = rail_length_y - mgn12_carriage_len    // 400 - 45 = 355mm
margin_x = travel_x - work_area_x               // 305 - 220 = 85mm
margin_y = travel_y - work_area_y               // 355 - 300 = 55mm
pulley_diameter = belt_pitch * pulley_teeth / PI
frame_inner_x = frame_x - 2 * extrusion_size    // 450 - 40 = 410mm
frame_inner_y = frame_y - 2 * extrusion_size    // 500 - 40 = 460mm
```

**検証条件**（全て満たすこと）:
- `travel_x >= work_area_x` （X トラベル ≥ 作業エリア X）
- `travel_y >= work_area_y` （Y トラベル ≥ 作業エリア Y）
- `frame_inner_x >= rail_length_x + 2 * rail_end_clearance` （レールがフレーム内に収まる）
- `frame_inner_y >= rail_length_y + 2 * rail_end_clearance` （レールがフレーム内に収まる）
- `margin_x >= 40` （ホーミングオフセット + ベルト固定金具分）
- `margin_y >= 40` （同上）

---

## 3. コンポーネント構成とファイル管理

### 推奨階層
```
PenPlotter (トップレベル)
├── Frame
│   ├── Rail_Y_500mm × 2   (長辺)
│   ├── Rail_X_450mm × 2   (短辺)
│   ├── Cross_Y_460mm × 2  (内寸・長辺方向)
│   ├── Cross_X_410mm × 2  (内寸・短辺方向)
│   └── CornerBracket × 8
├── Y_Left_Assembly
│   ├── MGN12_Rail_Y_400mm
│   ├── MGN12_Carriage
│   └── Rail_Mount_Bracket × 2
├── Y_Right_Assembly
│   └── (同上)
├── X_Gantry
│   ├── Gantry_Beam (2020 extrusion)
│   ├── MGN12_Rail_X_350mm
│   ├── MGN12_Carriage
│   ├── Gantry_Left_Bracket
│   └── Gantry_Right_Bracket
├── Carriage_Assembly
│   ├── Carriage_Plate
│   ├── PenHolder
│   ├── Servo_Mount
│   └── SG90_Servo
├── Motor_Left
│   ├── NEMA17
│   ├── GT2_Pulley_20T
│   └── Motor_Mount
├── Motor_Right
│   └── (同上)
├── Idler_Assemblies
│   ├── Corner_Idler × 4
│   └── Idler_Bracket × 4
├── Belt_Path (参考用)
│   ├── Belt_Upper
│   └── Belt_Lower
└── Electronics_Mount
    ├── Arduino_Mount
    └── PSU_Mount
```

### 設計アプローチ: ハイブリッド方式
- **フレーム**: トップダウン（in-place）で設計。フレーム寸法を基準に他を配置
- **3D プリント部品**: ボトムアップで個別設計。STL エクスポート時に原点が正しくなるよう、部品の中心 or 底面を原点に設定
- **購入部品（モーター、レール等）**: 外部 STEP をインポート

### ヒント
- コンポーネントを作成するとき「**New Component**」を先にクリックしてからスケッチを始める（忘れると Bodies がトップレベルに溜まる）
- 必ず **Component** を右クリック > **Activate** してからそのコンポーネント内で作業する

---

## 4. 2020 アルミフレームのモデリング

### 3つの方法

**方法 A: GrabCAD からダウンロード（推奨）**
- [GrabCAD](https://grabcad.com/library/2020-aluminium-extrusions-1) で "2020 aluminum extrusion" を検索
- STEP 形式でダウンロード → `Insert > Insert into Current Design`
- 長さをパラメータで Extrude / Cut して調整

**方法 B: McMaster-Carr（Fusion 内蔵）**
- `Insert > Insert McMaster-Carr Component` > "t-slotted framing 20mm" で検索
- 正確な断面プロファイルと BOM 用パーツ番号が付属

**方法 C: 自分でモデリング（軽量）**
- 2020 の断面をスケッチ（20mm 角、6mm スロット幅、4.2mm スロット開口、中心 5mm 穴）
- パラメータ `frame_x` 等で Extrude
- **パフォーマンス重視なら簡略版**（外形 + スロットのみ、内部ウェブ省略）で十分

### 実用的なアドバイス
- フレームの内寸計算: `inner_x = frame_x - 2 * extrusion_size`
- 干渉チェック付近以外は簡略モデルを使い、ファイルを軽く保つ
- フレーム同士の接続部は、T ナット + M5 ボルトの穴位置を明確にスケッチしておく

---

## 5. CoreXY ベルト経路設計（最も失敗しやすい）

### 絶対ルール: ベルトセグメントの平行性
CoreXY では、**長さが変化するベルトセグメントは、対応するリニアガイドと完全に平行でなければならない**。これが崩れると：
- キャリッジ位置によってベルトテンションが変化
- ガントリーのラッキング（ねじれ）
- 寸法精度の低下

→ **モーターとプーリーの X 方向位置を mm 単位で正確に設計する**

### ベルト経路の設計手順

1. **2つのベルト面**を定義する（上面・下面、間隔 `belt_plane_gap`）
2. **モーター 2 台はフレーム同じ端（後部）に配置**
3. **4 コーナーにアイドラープーリー配置**
4. Fusion の **3D スケッチ** でベルト経路を描く（実際のベルトとしてモデリングする必要はない。中心線パスで十分）
5. 8の字クロスが発生する箇所で上下面が入れ替わることを確認

### ベルトに関する具体的な注意点

| やること | やってはいけないこと |
|---|---|
| 歯付き面は歯付きプーリーにのみ接触させる | 歯付き面をスムースアイドラーに当てない |
| アイドラーは必要最小限に（各コーナー 1 個） | 不要なアイドラーを追加しない（摩擦増大） |
| 上下ベルトパスを別の平面に分ける | 同一平面でベルトを交差させない |
| ベルトテンション調整機構を最初から設計 | 「後で何とかなる」は禁物 |
| 両ベルトのテンションを均等にする | 片方だけ張るとラッキング発生 |

### テンション調整機構
- **モータースライド方式**（推奨）: モーターマウントを T スロットナットで取り付け、スライドさせてテンション調整
- モーターマウントに M5 ボルトでテンション調整用の長穴（スロット）を設ける
- 調整範囲: 10-15mm あれば十分

### Fusion でのベルトパス確認方法
1. 3D スケッチでプーリー中心を通る経路を描画
2. `Inspect > Measure` で各セグメントの角度を確認（レールと平行 = 0° or 90°）
3. `Inspect > Interference` でベルト同士、ベルトとフレームの干渉チェック

---

## 6. リニアレール（MGN12H）の取り付け設計

### レール取り付け面の精度
- MGN12 レールの取り付け面は **平面度が命**
- 2020 アルミフレーム上面に直接取り付ける場合、フレームの上面は十分に平滑
- 3D プリントブラケット経由で取り付ける場合、**取り付け面をビルドプレート側にして印刷**

### 取り付け穴
- MGN12 レールの取り付け穴: M3、ピッチ 25mm（データシートを確認）
- フレームへの固定: M3 ボルト → 3D プリントブラケット → T ナット → 2020 スロット
- または: レールを直接フレーム上面にネジ止め（M3 タップ穴をフレームに加工）

### 設計のポイント
- 両 Y 軸レールは **完全に平行** でなければならない。平行でないとガントリーがバインドする
- Fusion で 2 本のレール間距離をパラメータ化し、両端で同じ値になるよう拘束
- キャリッジの可動範囲をスライダージョイントで設定し、**端部でフレームと干渉しないこと**をアニメーションで確認

---

## 7. 3D プリント部品の設計ルール

### 公差テーブル（FDM、0.4mm ノズル基準）

| 用途 | 設計寸法 | 備考 |
|---|---|---|
| M3 貫通穴（スライドフィット） | 3.4 mm | 公称 +0.4mm |
| M3 圧入穴（セルフタップ） | 2.8-2.9 mm | |
| M5 貫通穴 | 5.4 mm | 公称 +0.4mm |
| M3 ヒートセットインサート穴 | 4.0-4.2 mm | M3×5×4mm インサート用 |
| 2020 フレーム嵌合ポケット | 20.3-20.5 mm | 片側 0.15-0.25mm クリアランス |
| T スロットキー幅 | 5.8-6.0 mm | 公称 6mm、収縮分を考慮 |
| ベアリング圧入穴 | 公称 -0.1mm | きつめにして圧入 |
| ペンクランプ内径 | 12.5 mm | 8-12mm ペン対応のスプリットクランプ |

### 印刷方向の設計
- **ボルト穴の軸方向 = Z 軸（積層方向）に垂直**にすると穴精度が最も良い
- フレームとの接触面をビルドプレート側に → 最高の平面度
- サポート不要な設計: 45° 以下のオーバーハングを守る、ブリッジは 15mm 以下

### 材質選定
| 材質 | 推奨用途 | 理由 |
|---|---|---|
| **PETG** | モーターマウント、ブラケット全般 | 層間接着良好、割れにくい、適度な柔軟性 |
| PLA | プロトタイプ・試作 | 寸法精度は高いが脆い、熱に弱い |
| ABS/ASA | モーター近くの高温部品 | 耐熱性だがソリが大きい |

### スライサー設定（精度重視）
- レイヤー高さ: **0.16-0.2mm**（機能部品）
- 壁数: **4 以上**（モーターマウント等の高荷重部品）
- インフィル: **40% 以上**（構造部品）
- Outer wall first: **ON**（外径精度向上）
- Elephant's foot compensation: **0.15-0.25mm**
- フロー率をキャリブレーションしてから印刷

### ヒートセットインサート
- M3 ネジを 3D プリント部品に繰り返し使う箇所には、**M3×5×4mm 真鍮ヒートセットインサート**を使用
- はんだごてで圧入。スレッドが確実で繰り返し脱着に耐える
- Fusion では穴コマンドで「ヒートセットインサート」用の穴径をパラメータ化

---

## 8. ペンホルダー / Z軸サーボ機構

### 推奨構造: サーボ + スプリング方式

```
[MGN12H キャリッジ（X軸ガントリー上）]
        │
   ┌────┴────┐
   │ キャリッジ  │ ← X軸方向に移動
   │ プレート   │
   │          │
   │ ┌──────┐ │
   │ │SG90  │ │ ← サーボ固定
   │ │ ┌─┐  │ │
   │ └─┤↕├──┘ │    サーボアームが上下
   │   └─┘    │
   │   │      │
   │ ┌─▼────┐ │
   │ │ペン   │ │ ← リニアガイド（6mmロッド×2 or MGN9）で上下
   │ │ホルダー│ │
   │ │  🖊   │ │ ← スプリングで下方向に付勢
   │ └──────┘ │
   └──────────┘
```

### 設計のポイント

**サーボ配置**
- SG90 サーボのアームがペンホルダーを**押し上げる**構造（ペンアップ）
- スプリングまたはゴムバンドが**引き下げる**（ペンダウン）
- サーボ角度: ペンダウン ≈ 0°、ペンアップ ≈ 30-45°（3-5mm リフト）

**スプリング力**
- 目標: 50-100g の下向き力（ペン先の接触圧）
- 強すぎると: サーボが保持できない、紙にペンが食い込む
- 弱すぎると: 高速移動時にペンが跳ねる
- → **スプリング交換可能な構造**にしておく

**ペンクランプ**
- スプリットカラー式: 円筒を縦に割って、ネジで締める
- 対応ペン径: 8-12mm（ボールペン〜サインペン）
- サムスクリュー（M3 蝶ボルト）で工具なし交換

**ペン先位置の最適化**
- ペン先をキャリッジの**真下**に配置（モーメントアーム最小化）
- ペン先〜キャリッジ間の距離が大きいと、慣性でペン先がぶれる
- 理想: キャリッジプレートから 30-40mm 以内

---

## 9. フレーム設計と直角精度

### フレーム寸法の検証
フレーム 500×450mm に対する A4 作業エリアの収まり：

| 方向 | フレーム外寸 | 内寸 | レール長 | トラベル | 作業エリア | 余裕 |
|------|------------|------|---------|---------|-----------|------|
| Y（長辺） | 500mm | 460mm | 400mm | 355mm | 300mm | **55mm** ✓ |
| X（短辺） | 450mm | 410mm | 350mm | 305mm | 220mm | **85mm** ✓ |

- 余裕 40mm 以上あれば、ホーミングオフセット（5-10mm）＋ ベルト固定金具を含めても安全
- パラメトリック設計にしておけば、組み立て後に微調整可能

### 直角精度の確保
- 対角線の差が **0.5mm 以下**になるよう組み立て
- コーナーブラケットは **90° を強制する形状**に設計（直角治具として機能させる）
- 可能であれば、**対角ブレース**（斜めの補強材）を追加して剛性向上

### コーナーブラケット設計
- 2020 フレームの 2 面に嵌合するポケットを設ける
- M5 ボルト 2 本ずつ（各フレームに対して）で固定
- 3D プリント品でも PETG で十分な強度

---

## 10. ジョイント（拘束）の使い方

### Fusion のジョイントタイプと用途

| ジョイント | 用途 | 適用箇所 |
|---|---|---|
| **Rigid** | 完全固定 | フレームコーナー、ブラケット固定 |
| **Slider** | 1 軸スライド | レールキャリッジ、ペン昇降 |
| **Revolute** | 回転のみ | プーリー、サーボアーム |
| **Cylindrical** | 回転 + スライド | （今回は不使用） |
| **Pin-Slot** | ピン + 溝 | テンション調整スロット |

### 活用テクニック
- Y 軸キャリッジに **Slider ジョイント**を設定し、`Inspect > Motion Study` でガントリーの可動範囲をアニメーション確認
- 干渉チェック: `Inspect > Interference` を移動後の位置で実行
- **Contact Sets** を有効にすると、動かしたときに部品同士がぶつかる箇所を自動検出

### ジョイント設定のコツ
- ジョイントは **Component 間**でのみ作成可能（Body では不可）
- ジョイントの原点は **穴の中心**や**面の中心**を選ぶと楽
- Slider の移動範囲を制限（Motion Limits）しておくと、可動範囲外に動かないようにできる

---

## 11. 標準部品の取り込み

### 入手先と優先順位

| 部品 | 入手先 | 形式 |
|---|---|---|
| 2020 アルミフレーム | GrabCAD / McMaster-Carr | STEP |
| MGN12H レール + キャリッジ | GrabCAD | STEP |
| NEMA17 モーター | GrabCAD | STEP |
| GT2 プーリー 20T | GrabCAD | STEP |
| GT2 アイドラー | GrabCAD | STEP |
| SG90 サーボ | GrabCAD | STEP |
| M3/M5 ボルト・ナット | McMaster-Carr（Fusion 内蔵） | 直接挿入 |
| T ナット（2020 用） | McMaster-Carr | 直接挿入 |
| Arduino UNO | GrabCAD | STEP |
| CNC Shield V3 | GrabCAD | STEP |

### McMaster-Carr の使い方（Fusion 内蔵）
1. `Insert > Insert McMaster-Carr Component`
2. カテゴリから探すか検索
3. サイズ選択後「Insert」で直接配置
4. パーツ番号が自動的に BOM に反映

### GrabCAD からのインポート
1. [grabcad.com](https://grabcad.com) で検索・ダウンロード（STEP 推奨）
2. Fusion で `Insert > Insert into Current Design`
3. コンポーネントとして配置

### ファイルサイズ対策
- 詳細な購入部品（ネジ穴のねじ山など）はファイルを重くする
- 多数のファストナーは **Component Pattern** で複製
- 確認が不要な内部構造は簡略化モデルに差し替え

---

## 12. STL エクスポートと印刷準備

### エクスポート手順
1. 3D プリント対象のコンポーネントを右クリック
2. `Save As STL`
3. Refinement: **High**（曲面がある場合）
4. 座標系: コンポーネントのローカル座標が使われるので、**底面が XY 平面に来るよう**設計時に原点を設定しておく

### 印刷方向の決定（設計段階で考慮）
以下を **Fusion での設計中に**決めておく：

| 部品 | 推奨印刷方向 | 理由 |
|---|---|---|
| コーナーブラケット | フレーム接触面を下 | 平面精度 |
| モーターマウント | ボルト穴軸が Z 方向 | 穴精度 |
| ペンホルダー | クランプ方向が Z 方向 | 締め付け強度 |
| レールマウント | レール取り付け面を下 | 平面度 |
| アイドラーブラケット | 軸穴が Z 方向 | 穴精度 |

### 命名規則
```
PenPlotter_CornerBracket_x4.stl     ← 必要個数を明記
PenPlotter_MotorMount_Left_x1.stl
PenPlotter_MotorMount_Right_x1.stl  ← ミラー部品は別ファイル
PenPlotter_PenHolder_v2_x1.stl      ← バージョン番号
```

---

## 13. 設計レビューチェックリスト

### フレーム
- [ ] 対角線寸法が等しい（直角）
- [ ] 全ての T ナットアクセス穴が確保されている
- [ ] フレーム内寸が作業エリア + マージンを満たす

### CoreXY ベルト
- [ ] 全ての可変長ベルトセグメントが対応レールと平行
- [ ] 上下ベルト面が分離されている（干渉なし）
- [ ] テンション調整機構が両モーターにある
- [ ] 歯付き面は歯付きプーリーのみに接触
- [ ] アイドラーの数が必要最小限

### リニアレール
- [ ] 両 Y 軸レールが完全に平行
- [ ] キャリッジの全可動範囲で干渉なし
- [ ] `travel_x >= work_area_x` かつ `travel_y >= work_area_y`

### ペン機構
- [ ] ペン先がキャリッジ真下
- [ ] 昇降ストローク 3-5mm 確保
- [ ] スプリング力調整可能
- [ ] ペン径 8-12mm に対応

### 3D プリント部品
- [ ] 全部品の印刷方向が決定済み
- [ ] サポート不要な形状（またはサポート箇所が最小限）
- [ ] 公差テーブルに従った穴径設計
- [ ] ヒートセットインサートの穴径が正しい

### 組み立て性
- [ ] ボルトにレンチがアクセスできる
- [ ] 組み立て順序が論理的（内側の部品が後から付けられない等がない）
- [ ] 配線ルートの空間が確保されている

---

## 14. 参考リソース

### CoreXY 設計の必読資料
- [Mark Rehorst — CoreXY Mechanism Layout and Belt Tensioning](https://drmrehorst.blogspot.com/2018/08/corexy-mechanism-layout-and-belt.html) — ベルト経路設計の決定版
- [Mark Rehorst — The CoreXY Belt "Tuning" Myth](https://drmrehorst.blogspot.com/2022/07/the-corexy-belt-tuning-myth.html) — テンション調整の真実

### オープンソース CoreXY プロッタ（参考設計）
- [Plotyx](https://diogosergio.com/posts/plotyx/) — CoreXY ペンプロッタ
- [PlotteRXY](https://github.com/jamescarruthers/PlotteRXY) — CoreXY ペンプロッタ
- [CyberPlotter](https://maniacallabs.com/2020/09/03/cyberplotter/) — CoreXY ペンプロッタ
- [HowToMechatronics — DIY Pen Plotter](https://howtomechatronics.com/projects/diy-pen-plotter-with-automatic-tool-changer-cnc-drawing-machine/)

### Fusion テクニック
- [Autodesk — Parametric Modeling Tutorial](https://www.autodesk.com/products/fusion-360/blog/parametric-modeling-in-fusion-360-tutorial/)
- [Autodesk — Assembly Fundamentals](https://www.autodesk.com/products/fusion-360/blog/autodesk-fusion-360-basics-assemblies/)
- [Fusion Forum — Top Down Assembly Best Practice](https://forums.autodesk.com/t5/fusion-design-validate-document/best-practice-for-top-down-assembly-design-and-component-origins/td-p/7955325)

### 3D プリント公差
- [Formlabs — Guide to 3D Printing Tolerances](https://formlabs.com/blog/understanding-accuracy-precision-tolerance-in-3d-printing/)
- [3DChimera — Tolerances & Fits](https://3dchimera.com/blogs/connecting-the-dots/3d-printing-tolerances-fits)

### 標準部品 CAD データ
- [GrabCAD Library](https://grabcad.com/library) — NEMA17, MGN12, 2020 extrusion 等
- [McMaster-Carr](https://www.mcmaster.com/) — ファストナー・T ナット等（Fusion 内蔵）
