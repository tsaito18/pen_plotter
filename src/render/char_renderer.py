"""1 つの配置要素（文字・罫線・数式）を手書きストロークへ変換する。

文字は次の優先順で字形を決める（先に見つかったものを使う）:

1. 数式: matplotlib の配置へ手書き字形を貼る（:mod:`src.render.math_handwriting`）。
   組版できない式は matplotlib 描画 → 細線化（:mod:`src.render.math_image`）
2. 幾何字形: 記号・句読点・ギリシャ文字・数式記号・丸数字（括弧は本人サンプルを優先）
3. ユーザー筆跡: 本人が書いたサンプル（英字も含む）
4. 幾何英字: 英字サンプルが無いとき
5. 部品合成: 本人が書いた字の部品で組み立てる（:mod:`src.glyphs.compose`、CJK のみ）
6. ML 変形: KanjiVG 参照字形をユーザーの書き癖で変形（CJK のみ）
7. KanjiVG 参照字形そのもの（数字、または ML が使えないとき）

KanjiVG 由来の字（5〜7）には本人の書き癖（:mod:`src.glyphs.style`）を掛ける。

経路ごとに揺らぎ（弾性変形・手ブレ）の強さを変え、ページ内の質感を揃える。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from src.geometry import Stroke, bbox_span, rotation_matrix
from src.glyphs.compose import ComponentComposer, load_components
from src.glyphs.geometric import (
    CIRCLED_NUMBERS,
    circled_number_glyph,
    latin_glyph,
    symbol_glyph,
)
from src.glyphs.sources import KanjiVGStore, UserStrokeDB
from src.glyphs.style import WritingStyle, estimate_writing_style
from src.handwriting.augmentation import HandwritingAugmenter
from src.handwriting.finishing import (
    HARAI,
    NONE,
    classify_finishes,
    infer_finishes,
)
from src.layout.line_breaking import SUPERSCRIPTS
from src.layout.mathtext import (
    MATH_BLOCK_CAP_RATIO,
    MATH_INLINE_CAP_RATIO,
    detect_top_level_fraction_bar,
    extract_math_layout,
    math_scale,
)
from src.layout.placement import CharPlacement
from src.render.math_handwriting import render_math_handwritten
from src.render.math_image import glyph_skeleton, render_latex_to_strokes
from src.render.positioning import position_strokes

logger = logging.getLogger(__name__)

# 描画しない空白
_SKIP_CHARS = frozenset(" \t　")

# 全角記号 → 字形を持つ半角記号
_CHAR_SUBSTITUTIONS: dict[str, str] = {
    "｛": "{",
    "｝": "}",
    "［": "[",
    "］": "]",
    "！": "!",
    "？": "?",
    "：": ":",
    "；": ";",
    "＝": "=",
    "＋": "+",
    "－": "-",
    "／": "/",
    "＜": "<",
    "＞": ">",
    "−": "-",  # 数学マイナス
    "〜": "~",  # 波ダッシュ
    "～": "~",  # 全角チルダ
    **{chr(0xFF10 + i): str(i) for i in range(10)},  # 全角数字
}

# 本人サンプルがあれば幾何字形より優先する記号（括弧）と、その代用サンプル
_USER_FIRST_SYMBOLS = frozenset("（）()「」『』［］[]｛｝{}【】")
_USER_SAMPLE_ALIASES = {"(": "（", ")": "）", "[": "［", "]": "］", "{": "｛", "}": "｝"}

# 揺らぎを乗せない（形が崩れやすい）句読点・長音・括弧類
_SMOOTH_CHARS = frozenset("、。，．・ー～—―()（）「」『』【】〈〉《》〔〕")

# 「日本語だけ描く」モードでも描く字（句読点・数字）
_JAPANESE_MODE_EXTRA = frozenset("、。，．,.")

# 上付き文字は元の字をこの倍率で描き、下端を大文字高さのこの割合まで上げる
_SUPERSCRIPT_SCALE = 0.6
_SUPERSCRIPT_RISE = 0.45

# 読点は終端を払う
_COMMA_CHARS = frozenset("、,，")

# --- 経路ごとの揺らぎ強度（ML/ユーザー筆跡=1.0 基準） ---
# 経路間で質感が大きく違うと「定規直線と乱れた字が混在」して見えるため差を縮める。
WAVER_GEOMETRIC = 1.5  # 幾何英字（直線/円できれいすぎる）
WAVER_MATH_IMAGE = 2.5  # matplotlib 数式（元から平滑）
WAVER_SYMBOL = 0.4  # 記号・句読点・括弧（ツルツル感だけ消す）
# 記号の揺らぎを当てる最小ストローク長。点など極短の画は揺らぎで破綻するため除外する。
_SYMBOL_WAVER_MIN_SPAN = 0.08  # 字形 bbox 長辺に対する比
_SYMBOL_WAVER_MIN_LEN_MM = 1.2

# 画数による揺らぎ逓減。多画字は画間が狭く、揺らぎで隣の画へはみ出して固まるため。
_WAVER_FULL_STROKES = 10
_WAVER_MIN_STROKES = 20
_WAVER_FLOOR = 0.5

# ユーザー筆跡に掛ける画ごとの微小 affine の強さ
_DIRECT_NOISE_SCALE = 0.15

# 横棒の右上がり下限保証（日本語手書きの習性）。揺らぎが全てゼロ平均なので放置すると
# 約半数の横棒が右下がりになる。概ね水平で十分長い画だけを下限角まで起こす。
_RISE_MIN_ANGLE = np.deg2rad(2.0)
_RISE_MIN_X_RANGE_RATIO = 0.25
_RISE_MAX_Y_X_RATIO = 0.35


@dataclass
class CharCoverageReport:
    """どの経路で描いたかの集計（UI の「文字カバレッジ」表示用）。"""

    user_strokes: list[str] = field(default_factory=list)
    composed: list[str] = field(default_factory=list)
    ml_inference: list[str] = field(default_factory=list)
    kanjivg: list[str] = field(default_factory=list)
    geometric: list[str] = field(default_factory=list)
    missing_glyphs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass
class RenderedChar:
    """1 要素ぶんの描画結果。``finishes[i]`` が ``strokes[i]`` の筆法。"""

    strokes: list[Stroke] = field(default_factory=list)
    finishes: list[str] = field(default_factory=list)


def waver_scale(n_strokes: int) -> float:
    """画数に応じた揺らぎ倍率 ∈ [_WAVER_FLOOR, 1.0]（10 画以下は満額、20 画以上で下限）。"""
    lo, hi = _WAVER_FULL_STROKES, _WAVER_MIN_STROKES
    if n_strokes <= lo:
        return 1.0
    if n_strokes >= hi:
        return _WAVER_FLOOR
    frac = (n_strokes - lo) / (hi - lo)
    return 1.0 - frac * (1.0 - _WAVER_FLOOR)


def is_japanese_char(char: str) -> bool:
    """かな・漢字（CJK 統合漢字の各拡張を含む）なら True。"""
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF66 <= code <= 0xFF9D
        or 0x20000 <= code <= 0x3134F
    )


def _is_ml_deformable(char: str) -> bool:
    """モデルは CJK のみで訓練しているため、数字には変形を掛けない（字形が壊れる）。"""
    return char not in "0123456789"


def _normalize_user_strokes(strokes: list[Stroke]) -> list[Stroke]:
    """ユーザー筆跡（Y-DOWN）を単位正方形・Y-UP へ正規化する（アスペクト比保持）。"""
    all_pts = np.concatenate(strokes, axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    span = (maxs - mins).max()
    if span < 1e-6:
        return strokes
    center = (mins + maxs) / 2
    result = []
    for s in strokes:
        normalized = (s - center) / span + 0.5
        normalized[:, 1] = 1.0 - normalized[:, 1]
        result.append(normalized)
    return result


class CharRenderer:
    """配置要素を手書きストロークへ変換する。

    Args:
        kanjivg_dir: KanjiVG 参照字形のディレクトリ。
        user_strokes_dir: ユーザー筆跡の文字ディレクトリ群の親（1 プロファイル分）。
        inference: ML 推論エンジン（:class:`src.model.inference.StrokeInference`）。
        augmenter: 揺らぎの生成器。None なら弾性変形・手ブレ・字のばらつきなし。
        line_spacing: 行間(mm)。
        temperature: ML 変形の字形揺らぎ。
        instance_variation: 同じ字を書くたびに変える画ごとの微小 affine の強さ。
        japanese_only: かな・漢字・句読点・数字以外を描かない。
        writing_style: 本人サンプルの無いかな・漢字に掛ける書き癖。None なら本人サンプルと
            KanjiVG から推定する（データが足りなければ恒等）。
    """

    def __init__(
        self,
        *,
        kanjivg_dir: Path | str | None = None,
        user_strokes_dir: Path | str | None = None,
        inference: object | None = None,
        augmenter: HandwritingAugmenter | None = None,
        line_spacing: float = 8.0,
        temperature: float = 0.2,
        instance_variation: float = 0.1,
        japanese_only: bool = False,
        writing_style: WritingStyle | None = None,
    ) -> None:
        self.kanjivg = KanjiVGStore(kanjivg_dir)
        self.user_db = UserStrokeDB(user_strokes_dir)
        self.inference = inference
        self.augmenter = augmenter
        self.line_spacing = line_spacing
        self.temperature = temperature
        self.instance_variation = instance_variation
        self.japanese_only = japanese_only
        self.writing_style = (
            writing_style
            if writing_style is not None
            else estimate_writing_style(self.user_db, self.kanjivg)
        )
        self.composer = ComponentComposer(load_components(kanjivg_dir), self.user_db, self.kanjivg)
        self.coverage = CharCoverageReport()
        self._ink_width_cache: dict[str, float | None] = {}
        # 全ての乱数はこの 1 系列から引く（augmenter の seed で全体が再現できる）
        self.rng = augmenter.rng if augmenter is not None else np.random.default_rng()

    def ink_width_ratio(self, char: str) -> float | None:
        """この字を描く字形のインク幅 / font_size（組版の字送り用）。字形が無ければ None。

        本人サンプル → KanjiVG 参照字形（書き癖つき）の順で、描画と同じ配置計算をして
        測る（揺らぎ・ML 変形は含まない）。字ごとにキャッシュする。
        """
        if char not in self._ink_width_cache:
            self._ink_width_cache[char] = self._measure_ink_width(char)
        return self._ink_width_cache[char]

    def _measure_ink_width(self, char: str) -> float | None:
        sample = self.user_db.best_sample(char)
        if sample is not None:
            glyph = _normalize_user_strokes(sample)
        elif (geometric := symbol_glyph(char)) is not None:
            glyph = geometric
        elif (composed := self._composed_glyph(char)) is not None:
            glyph = composed
        else:
            reference, _ = self.kanjivg.load(char)
            if reference is None:
                return None
            glyph = self.writing_style.apply(reference) if is_japanese_char(char) else reference
        probe = CharPlacement(char, 0.0, 0.0, 1.0)
        pts = np.concatenate(position_strokes(glyph, probe, self.line_spacing), axis=0)
        return float(pts[:, 0].max() - pts[:, 0].min())

    @property
    def has_reference_source(self) -> bool:
        """ML か KanjiVG の参照字形を持つか（無ければ幾何字形・ユーザー筆跡のみ）。"""
        return self.inference is not None or self.kanjivg.available

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def render(self, placement: CharPlacement) -> RenderedChar:
        """配置要素を描画する。描くものが無ければ空の :class:`RenderedChar`。"""
        cov = self.coverage
        if placement.line_segment is not None:
            x1, y1, x2, y2 = placement.line_segment
            return RenderedChar([np.array([[x1, y1], [x2, y2]], dtype=np.float64)], [NONE])

        if placement.math is not None:
            if self.japanese_only:
                cov.skipped.append(placement.char)
                return RenderedChar()
            return self._render_math(placement)

        original = placement.char
        if self.japanese_only and not (
            is_japanese_char(original) or original in _JAPANESE_MODE_EXTRA or _is_digit(original)
        ):
            cov.skipped.append(original)
            return RenderedChar()
        if original in _SKIP_CHARS:
            cov.skipped.append(original)
            return RenderedChar()

        if original in SUPERSCRIPTS:
            return self._render_superscript(placement)
        return self._render_glyph(placement, original)

    def _render_glyph(self, placement: CharPlacement, original: str) -> RenderedChar:
        """文字 1 字を経路の優先順に描き、使った経路を ``original`` の名で記録する。"""
        cov = self.coverage
        char = _CHAR_SUBSTITUTIONS.get(placement.char, placement.char)
        if char != placement.char:
            placement = replace(placement, char=char)
        smooth = original in _SMOOTH_CHARS or char in _SMOOTH_CHARS

        rendered = (
            (char in _USER_FIRST_SYMBOLS and self._render_user_strokes(placement, smooth))
            or self._render_symbol(placement)
            or self._render_user_strokes(placement, smooth)
            or self._render_latin(placement)
            or self._render_composed(placement, smooth)
            or self._render_ml(placement, smooth)
            or self._render_kanjivg(placement, smooth)
        )
        if rendered is None:
            cov.missing_glyphs.append(original)
            return RenderedChar()
        source, result = rendered
        getattr(cov, source).append(original)
        return result

    def _render_superscript(self, placement: CharPlacement) -> RenderedChar:
        """上付き文字（² ⁻ 等）: 元の字を小さく描き、大文字高さの中ほどより上へ置く。"""
        cap = placement.font_size
        small = cap * _SUPERSCRIPT_SCALE
        baseline = placement.y + (self.line_spacing - cap) / 2
        # 小さい字は行ボックスの縦中央に描かれるので、その中央を上付きの位置へ合わせる
        center = baseline + cap * _SUPERSCRIPT_RISE + small / 2
        sub = replace(
            placement,
            char=SUPERSCRIPTS[placement.char],
            y=center - self.line_spacing / 2,
            font_size=small,
            advance=None,
        )
        return self._render_glyph(sub, placement.char)

    # ------------------------------------------------------------------
    # 経路
    # ------------------------------------------------------------------

    def _render_math(self, placement: CharPlacement) -> RenderedChar:
        spec = placement.math
        assert spec is not None
        if spec.handwritten:
            strokes = self._render_math_handwritten(placement)
            if strokes is not None:
                self.coverage.geometric.append(spec.source)
                return RenderedChar(strokes, [NONE] * len(strokes))
        bbox = spec.bbox
        if spec.align == "baseline":
            # 本文字形は行ボックス内で縦中央寄せされるため、数式のベースラインも
            # 本文字形の下端ラインへ揃える
            x0, y0, w, h = bbox
            bbox = (x0, y0 + (self.line_spacing - placement.font_size) / 2, w, h)
        strokes = render_latex_to_strokes(spec.source, bbox, spec.align)
        if not strokes:
            return RenderedChar()
        self.coverage.geometric.append(spec.source)
        strokes = self._distort(strokes, WAVER_MATH_IMAGE)
        return RenderedChar(strokes, [NONE] * len(strokes))

    def _render_math_handwritten(self, placement: CharPlacement) -> list[Stroke] | None:
        """matplotlib の配置へ手書き字形を貼る。組版できなければ None（活字の細線化へ）。"""
        spec = placement.math
        assert spec is not None
        layout = extract_math_layout(spec.source)
        if layout is None:
            return None
        inline = spec.align == "baseline"
        cap_ratio = MATH_INLINE_CAP_RATIO if inline else MATH_BLOCK_CAP_RATIO
        scale = math_scale(placement.font_size, cap_ratio)
        x0, y0, w, h = spec.bbox
        bar_pt = detect_top_level_fraction_bar(layout) if spec.fraction_bar_y is not None else None
        if inline:
            # 本文の英字と同じベースライン（行ボックスに大文字の高さを縦中央寄せ）
            baseline = y0 + (self.line_spacing - placement.font_size * cap_ratio) / 2
        elif spec.fraction_bar_y is not None and bar_pt is not None:
            baseline = spec.fraction_bar_y - bar_pt * scale
        else:
            ink_h = (layout.height + layout.depth) * scale
            baseline = y0 + h / 2 - ink_h / 2 + layout.depth * scale
        return render_math_handwritten(
            layout,
            scale=scale,
            x_left=x0 + (w - layout.width * scale) / 2,
            baseline=baseline,
            glyph_source=self._math_glyph,
            distort=self._distort,
        )

    def _math_glyph(self, char: str, is_large: bool) -> tuple[list[Stroke], float] | None:
        """数式中の 1 字の字形（単位系）と揺らぎ倍率。

        本文と同じ順（括弧は本人サンプル → 幾何記号 → 本人サンプル → 幾何英字 → KanjiVG）。
        本人サンプルは傾くと細い字（1・l）が崩れるので揺らぎを足さない。どれも無い字・
        大型記号は活字を細線化する。
        """
        if not is_large:
            char = _CHAR_SUBSTITUTIONS.get(char, char)
            if char in _USER_FIRST_SYMBOLS and (sample := self._user_sample(char)) is not None:
                return _normalize_user_strokes(sample), 0.0
            if (glyph := symbol_glyph(char)) is not None:
                return self._hand_drawn(glyph), WAVER_SYMBOL
            if (sample := self._user_sample(char)) is not None:
                return _normalize_user_strokes(sample), 0.0
            if (glyph := latin_glyph(char)) is not None:
                return self._hand_drawn(glyph), WAVER_GEOMETRIC
            reference, _ = self.kanjivg.load(char)
            if reference is not None:
                return reference, 1.0
        skeleton = glyph_skeleton(char)
        return (skeleton, WAVER_MATH_IMAGE) if skeleton else None

    def _user_sample(self, char: str) -> list[Stroke] | None:
        """本人サンプル（半角の括弧は全角のサンプルで代用）。"""
        sample = self.user_db.best_sample(char)
        if sample is None and char in _USER_SAMPLE_ALIASES:
            sample = self.user_db.best_sample(_USER_SAMPLE_ALIASES[char])
        return sample

    def _render_symbol(self, placement: CharPlacement) -> tuple[str, RenderedChar] | None:
        char = placement.char
        if char in CIRCLED_NUMBERS:
            glyph = circled_number_glyph(
                [self.kanjivg.load(d)[0] or [] for d in CIRCLED_NUMBERS[char]]
            )
        else:
            glyph = symbol_glyph(char)
        if glyph is None:
            return None
        glyph = self._hand_drawn(glyph)
        positioned = self._symbol_distort(position_strokes(glyph, placement, self.line_spacing))
        finish = HARAI if char in _COMMA_CHARS else NONE
        return "geometric", RenderedChar(positioned, [finish] * len(positioned))

    def _render_user_strokes(
        self, placement: CharPlacement, smooth: bool
    ) -> tuple[str, RenderedChar] | None:
        sample = self._user_sample(placement.char)
        if sample is None:
            return None
        glyph = _jitter_strokes(_normalize_user_strokes(sample), self.rng)
        positioned = position_strokes(glyph, placement, self.line_spacing)
        if not smooth:
            extent = bbox_span(positioned)
            distorted = self._distort(positioned, waver_scale(len(positioned)))
            positioned = _rescale_to_extent(distorted, extent)
        return "user_strokes", RenderedChar(positioned, [NONE] * len(positioned))

    def _render_latin(self, placement: CharPlacement) -> tuple[str, RenderedChar] | None:
        glyph = latin_glyph(placement.char)
        if glyph is None:
            return None
        glyph = self._hand_drawn(glyph)
        positioned = position_strokes(glyph, placement, self.line_spacing)
        positioned = self._distort(positioned, WAVER_GEOMETRIC)
        return "geometric", RenderedChar(positioned, [NONE] * len(positioned))

    def _composed_glyph(self, char: str) -> list[Stroke] | None:
        if not (is_japanese_char(char) and _is_ml_deformable(char)):
            return None
        return self.composer.compose(char, style=self.writing_style)

    def _render_composed(
        self, placement: CharPlacement, smooth: bool
    ) -> tuple[str, RenderedChar] | None:
        """本人の書いた部品で組み立てた字（本人サンプルの無いかな・漢字）。"""
        glyph = self._composed_glyph(placement.char)
        if glyph is None:
            return None
        _, kvg_types = self.kanjivg.load(placement.char)
        return "composed", self._finish_reference_glyph(
            glyph,
            kvg_types,
            placement,
            smooth=smooth,
            waver=waver_scale(len(glyph)),
            deformable=True,
            styled=True,
        )

    def _render_ml(self, placement: CharPlacement, smooth: bool) -> tuple[str, RenderedChar] | None:
        if self.inference is None or not _is_ml_deformable(placement.char):
            return None
        reference, kvg_types = self.kanjivg.load(placement.char)
        if reference is None:
            return None
        waver = waver_scale(len(reference))
        try:
            raw = self.inference.generate(  # type: ignore[attr-defined]
                reference, temperature=self.temperature, deform_scale=waver, rng=self.rng
            )
        except Exception:
            logger.warning("ML inference failed for %r", placement.char, exc_info=True)
            return None
        return "ml_inference", self._finish_reference_glyph(
            raw, kvg_types, placement, smooth=smooth, waver=waver, deformable=True
        )

    def _render_kanjivg(
        self, placement: CharPlacement, smooth: bool
    ) -> tuple[str, RenderedChar] | None:
        reference, kvg_types = self.kanjivg.load(placement.char)
        if reference is None:
            return None
        return "kanjivg", self._finish_reference_glyph(
            reference,
            kvg_types,
            placement,
            smooth=smooth,
            waver=waver_scale(len(reference)),
            deformable=_is_ml_deformable(placement.char),
        )

    def _finish_reference_glyph(
        self,
        strokes: list[Stroke],
        kvg_types: list[str],
        placement: CharPlacement,
        *,
        smooth: bool,
        waver: float,
        deformable: bool,
        styled: bool = False,
    ) -> RenderedChar:
        """KanjiVG 由来の字形（ML 変形後を含む）に書き癖を掛けて配置し、筆遣いと揺らぎを乗せる。

        揺らぎで bbox が膨らむと経路ごとに字の大きさが変わるため、
        最後に配置直後の大きさへ戻す。数字（``deformable=False``）は終端加工・右上がり
        矯正で下線や等号が歪むため掛けない。``styled=True`` は書き癖を掛け済み（部品合成）。
        """
        if deformable and not styled and is_japanese_char(placement.char):
            strokes = self.writing_style.apply(strokes)
        positioned = position_strokes(strokes, placement, self.line_spacing)
        extent = bbox_span(positioned)
        finishes = (
            _resolve_finishes(kvg_types, positioned) if deformable else [NONE] * len(positioned)
        )
        positioned = self._instance_variation(positioned, waver)
        if not smooth:
            positioned = self._distort(positioned, waver)
        positioned = _rescale_to_extent(positioned, extent)
        if deformable:
            positioned = _enforce_horizontal_rise(positioned, placement.font_size)
        return RenderedChar(positioned, finishes)

    # ------------------------------------------------------------------
    # 揺らぎ
    # ------------------------------------------------------------------

    def _distort(self, strokes: list[Stroke], waver: float) -> list[Stroke]:
        """弾性変形（bbox 比 0.002）と手ブレ（0.01mm）を ``waver`` 倍で乗せる。"""
        aug = self.augmenter
        if aug is None:
            return strokes
        strokes = [aug.elastic_distort(s, amplitude=0.002 * waver) for s in strokes]
        return [aug.apply_tremor(s, amplitude=0.01 * waver) for s in strokes]

    def _hand_drawn(self, glyph: list[Stroke]) -> list[Stroke]:
        """幾何字形（単位系）の定規の直線・鋭い角を、手で引いた線にする。"""
        return self.augmenter.hand_drawn(glyph) if self.augmenter is not None else glyph

    def _symbol_distort(self, strokes: list[Stroke]) -> list[Stroke]:
        """記号に微量の揺らぎを乗せる（点などの極短の画は素のまま）。"""
        if self.augmenter is None or not strokes:
            return strokes
        span = bbox_span(strokes)
        if span < 1e-9:
            return strokes
        threshold = max(span * _SYMBOL_WAVER_MIN_SPAN, _SYMBOL_WAVER_MIN_LEN_MM)
        out: list[Stroke] = []
        for s in strokes:
            if bbox_span([s]) < threshold:
                out.append(s)
            else:
                out.extend(self._distort([s], WAVER_SYMBOL))
        return out

    def _instance_variation(self, strokes: list[Stroke], waver: float) -> list[Stroke]:
        """同じ字でも書くたびに形が変わるよう、画ごとに微小な回転・拡縮・移動を掛ける。"""
        strength = self.instance_variation * waver
        aug = self.augmenter
        if aug is None or not aug.enabled or strength <= 0 or not strokes:
            return strokes
        span = bbox_span(strokes)
        if span < 1e-9:
            return strokes
        rng = self.rng
        out: list[Stroke] = []
        for s in strokes:
            c = s.mean(axis=0)
            ang = rng.normal(0, strength * 0.04)
            sc = 1.0 + rng.normal(0, strength * 0.03)
            shift = rng.normal(0, strength * 0.04, size=2) * span
            out.append((s - c) @ rotation_matrix(ang) * sc + c + shift)
        return out


def _rescale_to_extent(strokes: list[Stroke], extent: float) -> list[Stroke]:
    """bbox の長辺が ``extent`` になるよう、bbox 中心で等倍率に拡縮する。"""
    current = bbox_span(strokes)
    if current < 1e-9 or extent < 1e-9:
        return strokes
    pts = np.concatenate(strokes, axis=0)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    k = extent / current
    return [(s - center) * k + center for s in strokes]


def _is_digit(char: str) -> bool:
    return len(char) == 1 and (0x30 <= ord(char) <= 0x39 or 0xFF10 <= ord(char) <= 0xFF19)


def _resolve_finishes(kvg_types: list[str], positioned: list[Stroke]) -> list[str]:
    """kvg:type があれば分類し、無い字（かな等）は軌跡から筆法を推定する。"""
    finishes = classify_finishes(kvg_types)
    if all(f == NONE for f in finishes):
        return infer_finishes(positioned)
    return finishes


def _jitter_strokes(strokes: list[Stroke], rng: np.random.Generator) -> list[Stroke]:
    """ユーザー筆跡の各画に微小な回転・拡縮・移動を掛ける（単位系）。"""
    ns = _DIRECT_NOISE_SCALE
    result = []
    for stroke in strokes:
        center = stroke.mean(axis=0)
        rotated = (stroke - center) @ rotation_matrix(rng.normal(0, ns * 0.05))
        scale = 1.0 + rng.normal(0, ns * 0.03, size=2)
        shift = rng.normal(0, ns * 0.1, size=2)
        result.append(rotated * scale + center + shift)
    return result


def _enforce_horizontal_rise(strokes: list[Stroke], font_size: float) -> list[Stroke]:
    """概ね水平で十分長い画（横棒）だけを、右上がり ``_RISE_MIN_ANGLE`` まで起こす。

    既に十分右上がりの画・縦画・斜め画・曲がった画・短い画はそのまま。
    """
    min_x_range = font_size * _RISE_MIN_X_RANGE_RATIO
    out: list[Stroke] = []
    for s in strokes:
        if s.shape[0] < 2:
            out.append(s)
            continue
        x_range = float(s[:, 0].max() - s[:, 0].min())
        y_range = float(s[:, 1].max() - s[:, 1].min())
        if not (x_range > min_x_range and y_range < x_range * _RISE_MAX_Y_X_RATIO):
            out.append(s)
            continue
        order = np.argsort(s[:, 0])
        start, end = s[order[0]], s[order[-1]]
        theta = float(np.arctan2(end[1] - start[1], end[0] - start[0]))
        if theta >= _RISE_MIN_ANGLE:
            out.append(s)
            continue
        center = s.mean(axis=0)
        out.append((s - center) @ rotation_matrix(_RISE_MIN_ANGLE - theta).T + center)
    return out
