# 手書きストローク品質改善 — 崩れ抑制 + 字間ロジック修正

## Context

ユーザーから生成結果に対する2つのフィードバックが入った:

1. **文字が少し崩れすぎている** — 特に「ストローク位置が元の字形からずれる」、全体的にもう少し整えたい。
2. **文字間スペースが不自然** — ひらがな・小書き仮名の周りが空きすぎ、行内の字間が均等すぎてメカニカル、漢字まわりは詰まりすぎ。

調査の結果、原因は次の3点に集約された:

- **per-point offset の上限 (`OFFSET_CLAMP=0.5`)** が大きめで、端の極端な変形を許している。
- **`_NOISE_SCALE=0.3`** がユーザーストローク直接描画パスの per-stroke 揺らぎを過大にしている。
- **`char_advance` が `font_size * 0.95` 固定** で `size_scale`(0.55〜1.0) と連動していないため、字形サイズと字間がちぐはぐ（漢字は詰まり、ひらがな・小書きは空きすぎ）。さらに **行ごとに1回しか `density_scale` が変動しない** ため字間が均等に並びメカニカルに見える。

なお調査中に、ML 経由の `inference.generate()` 呼び出し時の実効 `noise_scale` は外側デフォルト 0.02（小さい）が使われており、当初想定の 0.3 ではなかったことが判明（内側 `_generate_v3` の 0.3 デフォルトは到達不能パス）。よってML推論側の per-stroke 揺らぎは触らない。

## 修正方針（確定）

### A. 崩れ抑制
- **A-1**: `OFFSET_CLAMP` を `0.5 → 0.4` に下げる。per-point offset の端の極端変形を抑制。訓練/推論で共有定数なので両方が同時に変わる。
- **A-2**: 見送り（実効 noise_scale が既に 0.02 と十分小さいため、変更不要）。
- **A-3**: `_NOISE_SCALE` を `0.3 → 0.15` に下げる。ユーザーストローク直接描画パスの per-stroke 回転/スケール/平行移動を半減。

### B. 字間ロジック改善
- **B-1**: `char_advance` を `size_scale` に連動させる。
  - 新式: `char_advance = font_size * (0.45 + 0.55 * size_scale)`
  - 漢字(1.0) → 1.00（旧0.95→広がる、詰まり改善）
  - ひらがな標準(0.85) → 0.9175（旧0.95→詰まる、空きすぎ改善）
  - ひらがな個別(0.72) → 0.846
  - カタカナ(0.68) → 0.824
  - 小書き仮名(0.55) → 0.7525（旧0.55→広がる、glyph周りに余白）
  - 小書き句読点(0.35) → 0.6425
  - 半角文字は別ロジック (`font_size * 0.55`) のまま不変
- **B-2**: 文字単位の密度微変動を追加。
  - `AugmentConfig` に `char_density_variation: float = 0.02` を追加。
  - `HandwritingAugmenter.get_char_density_scale()` メソッド追加 — `1.0 + uniform(-0.02, +0.02)`。
  - typesetter のループ内で `line_density × char_density` を用い、`spacing_factor` と `char_width` の両方に反映。
  - 既存の `line_density_variation` (±0.05) との二重がけにより、行内でも字間が文字ごとに微変動しメカニカル感が解消する。

## 修正対象ファイル

### `src/model/finetune.py:32`
- `OFFSET_CLAMP = 0.5` → `OFFSET_CLAMP = 0.4`

### `src/ui/stroke_renderer.py:54`
- `_NOISE_SCALE = 0.3` → `_NOISE_SCALE = 0.15`

### `src/layout/typesetter.py:412-418`
旧:
```python
char_font_size = self.font_size * size_scale
if cur_halfwidth:
    char_advance = self.font_size * 0.55
elif ch in _SMALL_KANA or ch in _SMALL_PUNCT:
    char_advance = char_font_size
else:
    char_advance = self.font_size * 0.95
```
新:
```python
char_font_size = self.font_size * size_scale
if cur_halfwidth:
    char_advance = self.font_size * 0.55
else:
    # size_scale 連動: 字形サイズが小さい文字ほど advance も小さくし、
    # 視覚的な詰まり方を均す。漢字(1.0)→1.00、ひらがな(0.85)→0.9175、
    # 小書き仮名(0.55)→0.7525
    char_advance = self.font_size * (0.45 + 0.55 * size_scale)
```

### `src/layout/typesetter.py:420-428`(augmenter 適用ブロック)
旧:
```python
spacing_factor = density_scale
if prev_halfwidth and cur_halfwidth:
    spacing_factor *= 0.5
aug_x = x + (aug_x - x) * spacing_factor
char_width = char_advance * density_scale
```
新:
```python
char_density = self._augmenter.get_char_density_scale()
effective_density = density_scale * char_density
spacing_factor = effective_density
if prev_halfwidth and cur_halfwidth:
    spacing_factor *= 0.5
aug_x = x + (aug_x - x) * spacing_factor
char_width = char_advance * effective_density
```

### `src/model/augmentation.py`
`AugmentConfig` (11-21行) に1フィールド追加:
```python
char_density_variation: float = 0.02
```
`get_line_density_scale()` の直下に新メソッド追加:
```python
def get_char_density_scale(self) -> float:
    """文字ごとの密度微変動。line_density に重ねがけして字間の均一感を崩す。"""
    if not self._config.enabled:
        return 1.0
    cfg = self._config
    return 1.0 + self._rng.uniform(
        -cfg.char_density_variation, cfg.char_density_variation
    )
```

### `tests/test_finetune.py:635`
`assert OFFSET_CLAMP == 0.5` → `assert OFFSET_CLAMP == 0.4`

## 実装ステップ（TDD準拠）

### Phase 1: 数値変更（A系、独立）
1. A-3 (`_NOISE_SCALE 0.3 → 0.15`) + 関連テスト確認。
2. A-1 (`OFFSET_CLAMP 0.5 → 0.4`) + `tests/test_finetune.py:635` 更新。`pytest tests/test_finetune.py tests/test_inference.py` 実行。

### Phase 2: B-1（advance 連動）
3. typesetter.py の advance 式を変更。
4. 既存テストの影響箇所を更新（後述）。
5. 新規 `TestCharAdvanceLinkedToSizeScale` を先に書いて Red → 実装 → Green。
6. `pytest tests/test_typesetter.py tests/test_line_breaking.py tests/test_page_layout.py`。

### Phase 3: B-2（char density）
7. `tests/test_augmentation.py` に `TestCharDensityScale` を先に追加（Red）。
8. `augmentation.py` に `char_density_variation` フィールド + `get_char_density_scale()` 実装（Green）。
9. typesetter のループに `char_density` 呼び出しを追加。
10. `tests/test_typesetter.py` に `TestCharDensityIntegration` 追加（同一行内 spacing 変動の検証）。
11. `pytest tests/test_augmentation.py tests/test_typesetter.py`。

## テスト戦略

### 更新が必要な既存テスト
- `tests/test_finetune.py:635` — `assert OFFSET_CLAMP == 0.4`
- `tests/test_typesetter.py` 中の `test_consecutive_halfwidth_spacing_reduced` (281-310行) は漢字 advance が 0.95 → 1.00 になることで比率が変わるが、`avg_half_dev < avg_full_dev * 0.75` のアサーションは依然成立する見込み。実行して値確認。
- `tests/test_typesetter.py` の `test_multipage` (75-86行) は `0.9` 仮値を `1.0` に更新可（任意、安全側なので必須ではない）。
- `tests/test_augmentation.py` の `test_augment_config_defaults` に `char_density_variation == 0.02` のアサーション追加。
- 注意: B-2 で augmenter の RNG 消費パターンが変わるため、`seed=42` で固定値をアサートしている既存テストがあれば値を再測定。`grep -rn "seed=42" tests/test_typesetter.py tests/test_augmentation.py` で事前確認。

### 新規追加テスト
- `tests/test_typesetter.py::TestCharAdvanceLinkedToSizeScale` — 漢字 advance = font_size、ひらがな advance < 漢字 advance、小書き advance > 字形幅、半角は不変、size_scale 順に単調増加。
- `tests/test_typesetter.py::TestCharDensityIntegration` — `spacing_variation`/`line_density_variation` 等を0にし `char_density_variation` のみ有効化して、連続漢字の spacing が複数値を取ることを検証。
- `tests/test_augmentation.py::TestCharDensityScale` — 戻り値型、変動の存在、振幅範囲 [0.98, 1.02]、line より小、`enabled=False` で 1.0、デフォルト値検証。

### 回帰テスト
```
pytest tests/test_typesetter.py tests/test_augmentation.py tests/test_stroke_renderer.py
pytest tests/test_finetune.py tests/test_inference.py
pytest tests/test_line_breaking.py tests/test_page_layout.py tests/test_preview_renderer.py
```

## 検証方法（Gradio UI 目視）

`make gradio` または `python -m src.ui.gradio_app` で起動。以下サンプルテキストで確認:

1. **ひらがな+漢字混合**: 「私の名前は太郎です。今日は学校に行きました。」 — 漢字間が広がりひらがな間が詰まる方向の改善
2. **小書き連続**: 「ちょっとまって、しゃべってください。」 — 「ょ」「っ」周りの余白
3. **カタカナ+漢字**: 「コンピュータの計算速度は飛躍的に向上した。」
4. **半角英数混合**: 「abc は 123 倍の効率です。」 — 半角/全角境界の違和感
5. **句読点**: 「こんにちは、世界。これはテストです。」
6. **同文複数生成**: 同じテキストを5回生成して、行内字間が均等でないこと（メカニカル感解消）を確認

観察ポイント:
- 漢字同士の間隔がやや広く、ひらがな同士は詰まり気味になっているか
- 「っ」「ょ」など小書き仮名の左右に適度な余白
- ストローク形状が極端に崩れず元字形を保つ
- ユーザーストローク登録済み文字 (`_direct_stroke` 経路) の揺らぎが減って安定

## リスク・留意点

- **OFFSET_CLAMP 変更の訓練済みモデル整合性**: 訓練と推論で同じ定数を共有。既存 checkpoint は ±0.5 想定の重みだが、推論時 0.4 に絞ると outlier 的な大変形のみ頭打ちで平均的な offset 分布への影響は少ない。再訓練は短期不要、長期的には新 clamp 値で再 finetune が理想。
- **半角→小書きの境界**: advance が 0.55→0.7525 と急変するが、現実的にこの並びはまれ。Gradio で「abc っ」のようなテキストで違和感の有無を確認。
- **絵文字・未知 Unicode**: `_char_size_scale` のデフォルト 1.0 → advance 1.00 で安全（全角と同じ扱い）。
- **A-2 は実装しない**: stroke_renderer の `generate()` 呼び出しは現状通り `noise_scale` 暗黙 0.02 で動作。将来の混乱を防ぐため、別 PR で内側 `_generate_v3` のデフォルト 0.3 を 0.02 と揃える（docstring 改善）改修を検討してもよいが、本スコープ外。
- **連続半角の `spacing_factor *= 0.5`** は触らない（ユーザー指摘なし）。
- **数式の `_CHAR_WIDTH_RATIO = 0.6`** も触らない（ユーザー指摘なし）。

## Critical Files
- `/home/taiga/Personal/pen_plotter/src/layout/typesetter.py`
- `/home/taiga/Personal/pen_plotter/src/model/augmentation.py`
- `/home/taiga/Personal/pen_plotter/src/model/finetune.py`
- `/home/taiga/Personal/pen_plotter/src/ui/stroke_renderer.py`
- `/home/taiga/Personal/pen_plotter/tests/test_typesetter.py`
- `/home/taiga/Personal/pen_plotter/tests/test_augmentation.py`
- `/home/taiga/Personal/pen_plotter/tests/test_finetune.py`
