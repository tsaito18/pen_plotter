# 文字レンダリング不具合 修正計画

## Context

文字生成で 2 種類の崩れが報告されている (Gradio プレビュー / xDraw 実機 共通):
- **症状 A**: `「」.()（）` が「左下→右上の対角線 1 本」で描画される
- **症状 B**: 漢字 `均満` が点々の集まりに見える

CLAUDE.md 上の品質目標は「レポート提出に使える"バレない"品質」。両症状とも致命的なので同セッションで A+B 同時修正する。ユーザー方針: ML 推論は全文字で維持する。

---

## 原因分析

### A. `「」.()（）` → 対角線 1 本

**確定**。ユーザー証言「対角線 1 本 (左下→右上)」が `inference.py:413-414` の `[[0.0,0.0],[1.0,1.0]]` フォールバック直線と完全一致。

致命的バグの構造 (`src/ui/stroke_renderer.py:176-252` `generate_char_strokes`):
```
順 1. _simple_punct_strokes       (、。・,. のみ)
順 2. _ascii_math_strokes         (+-=<>*/:; など)
順 3. _direct_stroke              (user_strokes/<char>)
順 4. self._inference.generate(reference=...)   ← 全文字をここで吸収
順 5. _simple_paren_strokes       (（）()) ← 到達しない!
順 6. _math_symbol_strokes
順 7. _load_kanjivg_json
順 8. _rect_fallback
```

`（）()「」` は KanjiVG (`data/strokes/`) にも user_strokes (`data/user_strokes/`) にも存在せず、`_load_reference_strokes` は None を返す。ML 推論は reference=None で `[[0,0],[1,1]]` (対角直線) を返す → `_position_strokes` で文字セルに配置 → セル左下から右上への対角線として描画される。step 5 の幾何ストロークは永久に到達しない。

### B. 均満 → 点々

**仮説段階**。確定には実機/プレビューでの段階別ログが必要。

経路: user_strokes に均満なし → KanjiVG に均満あり (7/12 ストローク, 座標範囲 0-10) → ML 推論 → TwoStageDeformer。

ML 推論パイプライン (`src/model/inference.py:_generate_v3`) の各段階で点々を生み得る箇所:
1. **TwoStageDeformer の暴走** (`stroke_deformer.py:408-433`): Stage1 Affine (per-stroke 大域変形, translation_mult=0.30) + Stage2 Transformer。但し最終 offset は `OFFSET_CLAMP=0.4` (`finetune.py:32`) で抑えられている (KanjiVG 0-10 範囲の 4%)。clamp は効いているはずだが、訓練分布外の文字で異常 offset が出る可能性。
2. **per-stroke variation** (`inference.py:457-467`): 推論後に各ストロークの mean を中心に rotation/scale/translation 加算。`noise_scale=0.3`, translation = ±0.03 (KanjiVG 単位)。複数ストロークで独立に作用 → 相対位置が崩れる可能性。
3. **augmentation** (`augmentation.py:elastic_distort`, `apply_tremor`): bbox の 0.2% / 0.01mm 程度で穏当 → 単独では犯人になりにくい。

→ 修正前に B-1 (再現スクリプト) で段階特定が必須。

---

## 修正方針

### 修正 1 (A 対応): フォールバック順序入れ替え + 安全網

**変更点**:

1. `src/ui/stroke_renderer.py:176-252` `generate_char_strokes` のフォールバック順を以下に変更:
   ```
   1. _simple_punct_strokes
   2. _ascii_math_strokes
   3. _simple_paren_strokes      ← ML 推論より前へ移動
   4. _math_symbol_strokes       ← ML 推論より前へ移動
   5. _direct_stroke
   6. _load_reference_strokes + _inference.generate (KanjiVG にある文字のみ実質的に動く)
   7. _load_kanjivg_json         ← inference 失敗時の安全網
   8. _rect_fallback
   ```
   `_simple_paren_strokes` の引数から `placement` 依存を確認 (現状: 受け取るが未使用 — そのまま順序入れ替えで動く)。

2. `_simple_paren_strokes` に `「」『』` を追加:
   - `「` `『`: 単位正方形内で右上から左下、左下から下右へ折れる 2 セグメント (L 字を上下反転)
   - `」` `』`: その上下/左右対称形
   - 既存の `( ) （ ）` と同じ規約 (np.array, 単位正方形, 後段で `_position_strokes` がスケール)

3. **安全網**: `src/model/inference.py:413-414` の `reference is None → [[0,0],[1,1]]` フォールバックを削除し `raise ValueError("V3 requires reference_strokes")` に変更。`stroke_renderer.py:220` の try/except が補足 → cov.rect_fallback に落ち、`_rect_fallback` で「明らかにおかしい四角」として表示される (= 対角線で誤魔化されるよりデバッグ可能)。

### 修正 2 (B 対応): 段階特定 → パラメータ調整

**B-1: 再現と段階特定スクリプト** (一時, 後で削除可)

`scripts/debug_render_char.py` を新規作成:
- 引数で 1 文字受け取り、ML 推論パイプラインの各段階で stroke 配列をダンプ + matplotlib プレビュー保存
- ダンプ段階:
  1. KanjiVG reference (原型)
  2. TwoStageDeformer 出力 `(affined - ref) + per_point`
  3. smooth + clamp 後
  4. per-stroke variation 後 (`inference.py:457-467` の出力)
  5. `_smooth_stroke` 後
  6. `_position_strokes` 後
  7. `_apply_distortion` (elastic + tremor) 後
- 出力先: `/tmp/debug_render_<char>_<stage>_<N>.png` (sequential番号、上書き禁止)
- 比較対象: 均満 (崩れる) + 一二 / 川 / 中 (user_strokes にあるはず) を並べる

**B-2: B-1 の結果に応じた対処** (B-1 後に確定)

崩れる段階別の対処:
- TwoStageDeformer 出力で既に崩れている → `translation_mult` を 0.30 → 0.15 に縮小 + `OFFSET_CLAMP` を 0.4 → 0.2 へ
- per-stroke variation で崩れる → `noise_scale * 0.1` (translation) を `* 0.03` へ縮小、または「ストローク数 ≥ 5 の文字では per-stroke variation を 1/2」に減衰
- augmentation で崩れる → 多ストローク文字で amplitude を減衰
- どれでもない場合 → ML 推論モデル自体の問題 (再訓練が必要) と判明 → ユーザー報告し B-2 で別途相談

ML 推論は全文字で維持 (ユーザー方針) なので、未訓練文字を KanjiVG 直描画にする選択肢は採らない。Step 6 で `_load_kanjivg_json` を安全網として残すのみ。

---

## 修正対象ファイル

- `src/ui/stroke_renderer.py` — フォールバック順序入れ替え、`_simple_paren_strokes` に `「」『』` 追加
- `src/model/inference.py` — `reference is None` 時 raise (フォールバック直線削除)
- `tests/test_stroke_renderer.py` — 以下のテスト追加:
  - `「」『』（）().` 各文字で `_rect_fallback` に到達せず正しいカテゴリ (geometric / punct) に分類されること
  - reference=None で `_inference.generate` が ValueError を投げ、上位 except が rect_fallback に落とすこと
- `scripts/debug_render_char.py` — B-1 用ダンプスクリプト (新規)
- (B-2 確定後) `src/model/inference.py` または `src/model/stroke_deformer.py` のパラメータ調整

## 検証方法

修正 1 (A):
1. `pytest tests/test_stroke_renderer.py -v` で追加テスト通過
2. Gradio UI 起動: `python -m src.ui.gradio_app`
3. 入力例: `これは「テスト」です。(サンプル)（全角）` を生成し、対角線にならず角括弧形状になることをプレビューで目視確認

修正 2 (B):
1. `python scripts/debug_render_char.py 均` 実行 → `/tmp/debug_render_均_*.png` を Read tool で確認
2. 既知文字 (例: `中` or `川` が user_strokes にあれば) と比較し、崩れ段階を特定
3. B-2 パラメータ調整後、再度ダンプして均満が読める形状になることを確認
4. 全テスト回帰なし: `pytest -m "not slow and not hardware"` (CLAUDE.md「テストは1つずつ実行、ポーリング禁止」遵守 → 多重起動しない)

## リスク・留意

- 修正 1 で順序入れ替え後、ML が KanjiVG にある `（` `）` 文字 (もしあれば) を学習している場合に幾何描画に切替わる → 但し現状はそもそも対角線しか出していないので実質劣化なし
- 修正 1 の `_simple_paren_strokes` への `「」` 追加で、新規描画が浮く可能性 → 同関数内の既存 `（）` と同等の太さ・滑らかさで描く
- 修正 2 のパラメータ縮小は ML 表現の多様性低下 → B-1 結果で最小限の調整に留める
- TDD 原則 (CLAUDE.md): Red → Green → Refactor を厳守。テスト先行で実装
