# Plan 003: 数式レンダリング崩壊バグ修正

## Context

ユーザーから「数式レンダリングがかなり崩壊している」報告。サンプル `## 解法 特性方程式 $\lambda^2 + 4 = 0$ より $\lambda = \pm 2i$ ... $$y = C_1 \cos 2x + C_2 \sin 2x$$ ここで $C_1$, $C_2$ は任意定数` で具体的に:

1. **ギリシャ文字（λ等）がレンダリングされない**（ノイズ描画 or 出ない）
2. **分数の横棒がずれる**
3. **`\bar`, `\overline`, `\left`, `\right`, `\mathrm` などが反映されない**（リテラル文字 `left(` `mathrm` として描画される）

CLAUDE.md では「数式レイアウト統合完了（Phase 7）」と記載されているが、最新コミット `b4ccd453 fix(layout): balance handwriting spacing` で typesetter.py の改行ロジックが大きく変わったこと、および `dbce141e` で大幅追加されたコマンドカバレッジが網羅的でないことが症状の原因。レポート品質（バレない手書き）を目標に、原因 6 箇所を段階的修正。

## 確定原因

| # | 場所 | 症状 | 原因 |
|---|------|------|------|
| 1 | `src/layout/math_layout.py:11-65` | `\cos`, `\sin`, ギリシャ一部, `\bar`, `\left` などが反映されない | `_LATEX_SYMBOL_MAP` に未登録 |
| 2 | `src/layout/math_layout.py:204-206` | `\left(` → `left(` 等のリテラル描画 | 未知コマンドのフォールバックで `cmd` 文字列をそのまま `symbol.content` に入れている |
| 3 | `src/ui/stroke_renderer.py:218-242` | λ 等のギリシャ文字がノイズに | ML inference が reference=None でも実行され、その後の `_math_symbol_strokes` (geometric) まで到達しない |
| 4 | `src/ui/stroke_renderer.py:212-252` | `cos`, `sin`, `dx` 等の英字が rect_fallback で四角になる | a-z, A-Z の glyph 経路なし |
| 5 | `src/layout/typesetter.py:443-462` | 数式の baseline が通常文字とずれる（分数線ずれの一因） | augmenter で通常文字 y を `line_y` に補正するが、`_place_math` には augment 前の `y` をそのまま渡す |
| 6 | `src/layout/typesetter.py:375-394` | 行折返し位置がおかしくなる（長行で数式を含むとき） | `break_paragraph_by_width` に数式中身を **通常文字の文字単位幅** で渡している。実配置は `_inline_math_width()` で正確に測るため不一致 |

## 修正タスク（優先度順）

### P0-1 ML inference を参照ストロークなしで呼ばない

- **対象**: `src/ui/stroke_renderer.py:218-232`
- **内容**: `reference = self._load_reference_strokes(...)` が `None` のとき ML 分岐を **スキップ** して次のフォールバックへ。さらに `_math_symbol_strokes` と `_simple_paren_strokes` を ML より **前** に移動（ギリシャ・数式記号を ML より優先）。
- **新規テスト**: λ が `cov.geometric` に分類されること、`cov.ml_inference` に含まれないこと。
- **影響**: 既存日本語文字（KanjiVG 有）は経路変わらず。

### P0-2 未知 LaTeX コマンドの安全フォールバック

- **対象**: `src/layout/math_layout.py:200-206`
- **内容**: 未知コマンドを `symbol(content=cmd)` で吐かず、警告ログ＋空 emit（描画なし）に変更。
- **新規テスト**: `\foo` が placement を生まないこと。

### P0-3 `\left` / `\right` 透過処理

- **対象**: `src/layout/math_layout.py` の `parse_elements` cmd 分岐
- **内容**: `\left` / `\right` を読んだら **次の 1 文字**（`(`, `)`, `[`, `]`, `|`, `.`）を消費し、`.` 以外はその文字を text として emit。サイズ自動調整は行わない（通常カッコと同じ）。
- **新規テスト**: `\left( \frac{1}{2} \right)` が `(` + frac + `)` を生むこと。

### P0-4 `\mathrm` / `\text` / `\mathbf` / `\mathit` の引数透過

- **対象**: 同上
- **内容**: これらコマンドは `_read_braced()` で引数を取り、`_ParserState(arg).parse_elements()` の結果を `group` として展開（フォント情報は破棄）。
- **新規テスト**: `\mathrm{abc}` が `a` + `b` + `c` text を生むこと。

### P0-5 `_LATEX_SYMBOL_MAP` 拡張

- **対象**: `src/layout/math_layout.py:11-65`
- **内容**:
  - 不足ギリシャ: `\iota`, `\kappa`, `\xi`, `\upsilon`, `\varepsilon`, `\varphi`, `\Xi`, `\Upsilon`
  - 関数名は **`operator` 専用 type** として content に文字列保持（例: `\cos` → `operator("cos")`）。typesetter で 1 つの placement にまとめて P2-1（英字 geometric）へ流す。
- **新規テスト**: `MathParser.parse(r"\xi")` → `symbol("ξ")`、`MathParser.parse(r"\cos")` → `operator("cos")`。

### P1-1 augmenter と数式の baseline 整合

- **対象**: `src/layout/typesetter.py:443-462`
- **内容**: `_place_math(seg_content, x, y, ...)` の `y` 引数を `line_y` に変更。`_place_block_math` も同様に baseline 補正後の y を使う。数式は augmenter 対象外（整然と保つ＝レポート品質優先）— 設計原則として明示。
- **新規テスト**: augmenter ON 時、同じ行内 text placement の y と math placement の y が一致すること（公差 1e-6）。

### P1-2 行折返し計算と実配置の幅基準を統一

- **対象**: `src/layout/typesetter.py:375-394`（`_parse_paragraphs`）
- **内容**: 現状 `stripped = _INLINE_MATH_RE.sub(...)` で数式中身を通常文字扱い → 数式部を **不可分プレースホルダ**（`\x01<idx>\x01` 形式）に置換。`width_fn` ラッパーでプレースホルダ検出時に `_inline_math_width()` を返す。`_rebuild_lines_with_math` も対応修正。
- **新規テスト**: `$\frac{1}{2}$` を含む長行で `_inline_math_width` ベースの折返し位置になること（`len("frac12") * char_width` ではない）。
- **影響**: 既存折返しテストの数件で期待値更新が必要（事前スキャン）。

### P2-1 英字 (`cos`, `sin`, `log`, `dx` 等) の geometric 描画

- **対象**: `src/ui/stroke_renderer.py` に `_math_word_strokes()` 新設、`_math_symbol_strokes` の隣
- **内容**: P0-5 で導入した `operator` MathElement を typesetter で 1 placement（`text="cos"`）に変換し、renderer で word 単位のテンプレートストロークを返す。最小対応: `cos`, `sin`, `tan`, `log`, `ln`, `exp`, `lim`, `dx`, `dy`, `dt`。
- **新規テスト**: `\cos` placement が `cov.geometric` に分類、3 文字分の strokes を返すこと。

### P2-2 修飾子 `\bar`, `\overline`, `\hat`, `\widehat`, `\tilde`, `\vec`, `\dot`, `\ddot`

- **対象**: `src/layout/math_layout.py` に `accent` type 追加 + `MathLayoutEngine._layout_accent()` 実装
- **内容**: 引数を子 layout した後、ascent + small_gap 位置に **line_segment** を 1 本（or `\vec` は短い 2 線分の矢印）追加。frac の line_segment と同じパターン。
- **新規テスト**: `\bar{x}` が `x` placement + 上部の line_segment を 1 本返すこと。

## TDD 実行順

1. **Cycle 1 (P0-1)**: λ がノイズ生成される failing test → ML スキップ → 通る
2. **Cycle 2 (P0-2, P0-3, P0-4)**: `\left( x \right)`, `\mathrm{abc}` リテラル出力 failing → パーサ修正
3. **Cycle 3 (P0-5)**: 不足ギリシャ・operator type の failing → マップ追加
4. **Cycle 4 (P1-1)**: augmenter ON で y 不一致 failing → `line_y` 渡し
5. **Cycle 5 (P1-2)**: 長数式での折返し位置 failing → width_fn ラッパー化
6. **Cycle 6 (P2-1, P2-2)**: `\cos`, `\bar{x}` の expected placement failing → 実装

各 Cycle 後に `pytest -q` で 624+ 全パス確認。

## 動作確認方法

1. `pytest -q` 全テストパス
2. `python scripts/check_math_render.py` 実行 → 出力 PNG 目視。ユーザ報告サンプル（`\lambda^2 + 4 = 0`, `$$y = C_1 \cos 2x + C_2 \sin 2x$$`）を追加し、`output/check_math.png` に λ・分数横棒・cos/sin がレポート品質で出ること
3. 単独確認: `python -c "from src.layout.math_layout import MathParser; print(MathParser.parse(r'\left(\frac{1}{2}\right)'))"`
4. End-to-end: `python scripts/preview_batch.py` で実際の数式段落を A4 PNG にレンダリング → 100% 表示で目視

## リスク・代替案

- **修飾子実装 (P2-2)**: `\overrightarrow{ABC}` 等の複数文字対象は accent box の幅計算が複雑 → 最小対応（`\bar` `\hat` `\vec` 単文字）で良ければ P0 完了時点で「リテラル文字出ない」だけでもレポート品質改善大。P2-2 は後送り可。
- **ML inference スキップ判断**: `reference is None` で即スキップは安全だが、将来「ML で英字を覚えさせる」拡張時に経路が無くなる → ckpt メタデータに `supports_unreferenced=True` フラグの余地を残し、デフォルト False（今は常にスキップ）。
- **augmenter を数式に適用するか**: 整然と vs 手書き感統一 → 今回は **レポート品質優先で前者**（数式は augmenter 対象外）。逆方向を選ぶなら `HandwritingAugmenter` に `math_mode` 弱化パラメータ追加し `_place_math` に渡す設計余地あり。
- **P1-2 既存テスト破壊**: 短い数式中心なら影響軽微。`tests/test_typesetter.py` の長行+数式ケースを事前スキャンし、変更が「より正確な改行」と確認できるものだけ期待値更新。
- **PR 分割**: P0-1 → P0-2,3,4 → P0-5 → P1-1 → P1-2 → (P2-1) → (P2-2) の 5-7 PR が現実的。

## Critical Files

- `src/layout/math_layout.py`（パーサ・レイアウトエンジン）
- `src/ui/stroke_renderer.py`（ストローク生成フォールバック順）
- `src/layout/typesetter.py`（行折返し・baseline 整合）
- `tests/test_math_layout.py`, `tests/test_stroke_renderer.py`, `tests/test_typesetter.py`
- `scripts/check_math_render.py`（動作確認）
