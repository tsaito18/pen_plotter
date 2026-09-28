"""1 つの配置要素（文字・罫線・数式）を手書きストロークへ変換する。

文字は次の優先順で字形を決める（先に見つかったものを使う）:

1. 数式: matplotlib 描画 → 細線化（:mod:`src.render.math_image`）
2. 幾何字形: 記号・句読点・括弧・ギリシャ文字・数式記号・丸数字
3. ユーザー筆跡: 本人が書いたサンプル（英字も含む）
4. 幾何英字: 英字サンプルが無いとき
5. ML 変形: KanjiVG 参照字形をユーザーの書き癖で変形（CJK のみ）
6. KanjiVG 参照字形そのもの（数字、または ML が使えないとき）

経路ごとに揺らぎ（弾性変形・手ブレ）の強さを変え、ページ内の質感を揃える。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from src.geometry import Stroke, bbox_span, rotation_matrix
from src.glyphs.geometric import (
    CIRCLED_NUMBERS,
    circled_number_glyph,
    latin_glyph,
    symbol_glyph,
)
from src.glyphs.sources import KanjiVGStore, UserStrokeDB
from src.handwriting.augmentation import HandwritingAugmenter
from src.handwriting.finishing import (
    HARAI,
    NONE,
    apply_finishing,
    classify_finishes,
    infer_finishes,
)
from src.layout.placement import CharPlacement
from src.render.math_image import render_latex_to_strokes
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

# 揺らぎを乗せない（形が崩れやすい）句読点・長音・括弧類
_SMOOTH_CHARS = frozenset("、。，．・ー～—―()（）「」『』【】〈〉《》〔〕")

# 「日本語だけ描く」モードでも描く字（句読点・数字）
_JAPANESE_MODE_EXTRA = frozenset("、。，．,.")

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
        augmenter: 揺らぎの乱数源。None なら揺らぎなし。
        line_spacing: 行間(mm)。
        temperature: ML 変形の字形揺らぎ。
        instance_variation: 同じ字を書くたびに変える画ごとの微小 affine の強さ。
        japanese_only: かな・漢字・句読点・数字以外を描かない。
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
    ) -> None:
        self.kanjivg = KanjiVGStore(kanjivg_dir)
        self.user_db = UserStrokeDB(user_strokes_dir)
        self.inference = inference
        self.augmenter = augmenter
        self.line_spacing = line_spacing
        self.temperature = temperature
        self.instance_variation = instance_variation
        self.japanese_only = japanese_only
        self.coverage = CharCoverageReport()

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

        char = _CHAR_SUBSTITUTIONS.get(original, original)
        if char != original:
            placement = replace(placement, char=char)
        smooth = original in _SMOOTH_CHARS or char in _SMOOTH_CHARS

        rendered = (
            self._render_symbol(placement)
            or self._render_user_strokes(placement, smooth)
            or self._render_latin(placement)
            or self._render_ml(placement, smooth)
            or self._render_kanjivg(placement, smooth)
        )
        if rendered is None:
            cov.missing_glyphs.append(original)
            return RenderedChar()
        source, result = rendered
        getattr(cov, source).append(original)
        return result

    # ------------------------------------------------------------------
    # 経路
    # ------------------------------------------------------------------

    def _render_math(self, placement: CharPlacement) -> RenderedChar:
        spec = placement.math
        assert spec is not None
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
        positioned = self._symbol_distort(position_strokes(glyph, placement, self.line_spacing))
        finish = HARAI if char in _COMMA_CHARS else NONE
        return "geometric", RenderedChar(positioned, [finish] * len(positioned))

    def _render_user_strokes(
        self, placement: CharPlacement, smooth: bool
    ) -> tuple[str, RenderedChar] | None:
        sample = self.user_db.best_sample(placement.char)
        if sample is None:
            return None
        glyph = _jitter_strokes(_normalize_user_strokes(sample))
        positioned = position_strokes(glyph, placement, self.line_spacing)
        if not smooth:
            positioned = self._distort(positioned, waver_scale(len(positioned)))
        return "user_strokes", RenderedChar(positioned, [NONE] * len(positioned))

    def _render_latin(self, placement: CharPlacement) -> tuple[str, RenderedChar] | None:
        glyph = latin_glyph(placement.char)
        if glyph is None:
            return None
        positioned = position_strokes(glyph, placement, self.line_spacing, logical_latin=True)
        positioned = self._distort(positioned, WAVER_GEOMETRIC)
        return "geometric", RenderedChar(positioned, [NONE] * len(positioned))

    def _render_ml(self, placement: CharPlacement, smooth: bool) -> tuple[str, RenderedChar] | None:
        if self.inference is None or not _is_ml_deformable(placement.char):
            return None
        reference, kvg_types = self.kanjivg.load(placement.char)
        if reference is None:
            return None
        waver = waver_scale(len(reference))
        try:
            raw = self.inference.generate(  # type: ignore[attr-defined]
                reference, temperature=self.temperature, deform_scale=waver
            )
        except Exception:
            logger.warning("ML inference failed for %r", placement.char, exc_info=True)
            return None
        positioned = position_strokes(raw, placement, self.line_spacing)
        finishes = _resolve_finishes(kvg_types, positioned)
        positioned = apply_finishing(positioned, finishes, scale=placement.font_size)
        positioned = self._instance_variation(positioned, waver)
        if not smooth:
            positioned = self._distort(positioned, waver)
        positioned = _enforce_horizontal_rise(positioned, placement.font_size)
        return "ml_inference", RenderedChar(positioned, finishes)

    def _render_kanjivg(
        self, placement: CharPlacement, smooth: bool
    ) -> tuple[str, RenderedChar] | None:
        reference, kvg_types = self.kanjivg.load(placement.char)
        if reference is None:
            return None
        deformable = _is_ml_deformable(placement.char)
        positioned = position_strokes(reference, placement, self.line_spacing)
        # 数字は終端加工・右上がり矯正で下線や等号が歪むため無効化する
        finishes = (
            _resolve_finishes(kvg_types, positioned) if deformable else [NONE] * len(positioned)
        )
        positioned = apply_finishing(positioned, finishes, scale=placement.font_size)
        waver = waver_scale(len(reference))
        positioned = self._instance_variation(positioned, waver)
        if not smooth:
            positioned = self._distort(positioned, waver)
        if deformable:
            positioned = _enforce_horizontal_rise(positioned, placement.font_size)
        return "kanjivg", RenderedChar(positioned, finishes)

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
        rng = aug.rng
        out: list[Stroke] = []
        for s in strokes:
            c = s.mean(axis=0)
            ang = rng.normal(0, strength * 0.04)
            sc = 1.0 + rng.normal(0, strength * 0.03)
            shift = rng.normal(0, strength * 0.04, size=2) * span
            out.append((s - c) @ rotation_matrix(ang) * sc + c + shift)
        return out


def _is_digit(char: str) -> bool:
    return len(char) == 1 and (0x30 <= ord(char) <= 0x39 or 0xFF10 <= ord(char) <= 0xFF19)


def _resolve_finishes(kvg_types: list[str], positioned: list[Stroke]) -> list[str]:
    """kvg:type があれば分類し、無い字（かな等）は軌跡から筆法を推定する。"""
    finishes = classify_finishes(kvg_types)
    if all(f == NONE for f in finishes):
        return infer_finishes(positioned)
    return finishes


def _jitter_strokes(strokes: list[Stroke]) -> list[Stroke]:
    """ユーザー筆跡の各画に微小な回転・拡縮・移動を掛ける（単位系）。"""
    ns = _DIRECT_NOISE_SCALE
    result = []
    for stroke in strokes:
        center = stroke.mean(axis=0)
        angle = np.random.normal(0, ns * 0.05)
        rotated = (stroke - center) @ rotation_matrix(angle)
        sx = 1.0 + np.random.normal(0, ns * 0.03)
        sy = 1.0 + np.random.normal(0, ns * 0.03)
        scaled = rotated * np.array([sx, sy])
        dx = np.random.normal(0, ns * 0.1)
        dy = np.random.normal(0, ns * 0.1)
        result.append(scaled + center + np.array([dx, dy]))
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
