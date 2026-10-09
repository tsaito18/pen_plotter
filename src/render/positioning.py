"""単位系の字形を配置情報（mm）へスケール・移動する。

縦方向の基準: ``placement.y`` は行ボックスの下端。通常字形は行間 ``line_spacing`` の
帯に縦中央寄せし、英字はベースラインを漢字の下端に揃える。

英字と括弧は字形の bbox をセルいっぱいに広げず、字種ごとの帯（英字: x-height・
ディセンダ、括弧: 中身に寄せた位置）へ合わせる。本人サンプルは基準枠なしで 1 字ずつ
書いたため大きさがバラバラ（s が A より大きい等）で、bbox の大きさは当てにならない。
寸法は本人の手書きレポートのスキャン実測に合わせている。
"""

from __future__ import annotations

import numpy as np

from src.geometry import Stroke, rotate_about
from src.layout.char_metrics import char_type_scale, effective_char_scale, halfwidth_advance
from src.layout.line_breaking import is_halfwidth
from src.layout.placement import CharPlacement

# 全角セル幅 = font_size × この係数（半角は字ごとの字送り :func:`halfwidth_advance`）
FULLWIDTH_CELL_FACTOR = 0.95

# 英字・ギリシャ文字の縦の帯（ベースライン=0、大文字の高さ=1）。実測: x-height ≈ 大文字の 0.5〜0.55
X_HEIGHT = 0.55
_DESCENDER = -0.3
_LATIN_BANDS: dict[str, tuple[float, float]] = {
    **dict.fromkeys("acemnorsuvwxz", (0.0, X_HEIGHT)),
    **dict.fromkeys("gpqy", (_DESCENDER, X_HEIGHT)),
    **dict.fromkeys("bdfhkl", (0.0, 1.02)),
    "t": (0.0, 0.8),
    "i": (0.0, 0.85),
    "j": (_DESCENDER, 0.85),
    "Q": (-0.1, 1.0),
    # ギリシャ小文字
    **dict.fromkeys("αεικνοπστυωϵ", (0.0, X_HEIGHT)),
    **dict.fromkeys("γημρχς", (_DESCENDER, X_HEIGHT)),
    **dict.fromkeys("δθλϑ", (0.0, 1.02)),
    **dict.fromkeys("βζξφψϕ", (_DESCENDER, 1.02)),
}
_CAP_BAND = (0.0, 1.0)
# セル幅に対する英字の最大インク幅（両側に字間を残す）
_LATIN_MAX_INK = 0.92

# 演算子は大文字高さ比の正方枠（字形の単位正方形をそのまま写す）に描き、枠の中心を
# 数式の軸（ベースラインからの高さ）に置く。セル幅いっぱいに伸ばすと隣の字に触れる。
_OPERATORS = frozenset("+-=<>×÷±≠≈≤≥~")
_OPERATOR_BOX = 0.6
_OPERATOR_AXIS = 0.3

# 括弧。高さは本文サイズ比（実測: 丸括弧 ≈ 漢字の 0.8〜0.85）、中身の側へ寄せる
_OPENING_BRACKETS = frozenset("（([｛{「『【〈《〔")
_CLOSING_BRACKETS = frozenset("）)]｝}」』】〉》〕")
_CORNER_BRACKETS = frozenset("「」『』")
_BRACKET_HEIGHT = 0.85
_CORNER_BRACKET_HEIGHT = 0.55
_BRACKET_INNER_GAP = 0.12  # 中身の側に空ける距離（本文サイズ比）
_BRACKET_MAX_WIDTH = 0.5  # セル幅に対する最大インク幅
_HALFWIDTH_BRACKET_BAND = (-0.22, 1.05)  # 英文中の ( ) [ ] { } は大文字より上下に伸びる

_COMMA_CHARS = ("、", ",", "，")
_PERIOD_CHARS = (".", "。", "．")
_MIDDLE_DOT = "・"


def position_strokes(
    strokes: list[Stroke], placement: CharPlacement, line_spacing: float
) -> list[Stroke]:
    """字形を実 bbox フィットで配置する（句読点・中黒・英字・括弧は専用の配置）。

    Args:
        strokes: 単位系字形（Y-UP）。
        placement: 配置情報。
        line_spacing: 行間(mm)。
    """
    if not strokes:
        return []
    char = placement.char
    if char in _COMMA_CHARS:
        return [_position_comma(strokes, placement, line_spacing)]
    if char in _PERIOD_CHARS:
        return [_position_period(placement, line_spacing)]
    if char == _MIDDLE_DOT:
        return _position_middle_dot(strokes, placement, line_spacing)
    if char.isalpha() and is_halfwidth(char):
        return _position_latin(strokes, placement, line_spacing)
    if char in _OPENING_BRACKETS or char in _CLOSING_BRACKETS:
        return _position_bracket(strokes, placement, line_spacing)
    if char in _OPERATORS:
        return _position_operator(strokes, placement, line_spacing)

    all_pts = np.concatenate(strokes, axis=0)
    mins = all_pts.min(axis=0)
    ranges = all_pts.max(axis=0) - mins

    # サイズ倍率(字種×密度)は組版が font_size に 1 回だけ焼く。ここでは再適用しない。
    fs = placement.font_size
    cell_width = _cell_width(placement)

    scale_w = cell_width / ranges[0] if ranges[0] > 1e-6 else float("inf")
    scale_h = fs / ranges[1] if ranges[1] > 1e-6 else float("inf")
    scale = min(scale_w, scale_h)

    rendered_w = ranges[0] * scale
    rendered_h = ranges[1] * scale
    slot = placement.advance if placement.advance is not None else cell_width
    x_offset = placement.x + (slot - rendered_w) / 2
    # 小書き仮名は行ボックス中央だと浮くため、周りの字の下端に揃える
    if char_type_scale(char) < 0.6:
        y_offset = placement.y + (line_spacing - _body_size(placement)) / 2
    else:
        y_offset = placement.y + (line_spacing - rendered_h) / 2

    offset = np.array([x_offset, y_offset])
    positioned = [(stroke - mins) * scale + offset for stroke in strokes]
    if placement.slant:
        center = (x_offset + rendered_w / 2, y_offset + rendered_h / 2)
        positioned = rotate_about(positioned, placement.slant, center)
    return positioned


def _body_size(placement: CharPlacement) -> float:
    """字種・密度の倍率を外した本文のフォントサイズ(mm)（組版の字送りの基準）。"""
    return placement.font_size / effective_char_scale(placement.char)


def _cell_width(placement: CharPlacement) -> float:
    """組版が予約した字送りに相当するセル幅(mm)。"""
    if is_halfwidth(placement.char):
        return _body_size(placement) * halfwidth_advance(placement.char)
    return placement.font_size * FULLWIDTH_CELL_FACTOR


def _fit_band(
    strokes: list[Stroke],
    placement: CharPlacement,
    *,
    x_left: float,
    y_bottom: float,
    height: float,
    max_width: float,
) -> list[Stroke]:
    """bbox を高さ ``height`` に合わせ、左下を (x_left, y_bottom) に置く（幅超過は横だけ縮小）。"""
    all_pts = np.concatenate(strokes, axis=0)
    mins = all_pts.min(axis=0)
    w, h = all_pts.max(axis=0) - mins
    sy = height / h if h > 1e-9 else 1.0
    sx = min(sy, max_width / w) if w > 1e-9 else sy
    scale = np.array([sx, sy])
    origin = np.array([x_left, y_bottom])
    positioned = [(s - mins) * scale + origin for s in strokes]
    if placement.slant:
        center = (x_left + w * sx / 2, y_bottom + height / 2)
        positioned = rotate_about(positioned, placement.slant, center)
    return positioned


def _ink_width(strokes: list[Stroke], height: float, max_width: float) -> float:
    """:func:`_fit_band` で配置したときのインク幅。"""
    all_pts = np.concatenate(strokes, axis=0)
    w, h = all_pts.max(axis=0) - all_pts.min(axis=0)
    if w <= 1e-9:
        return 0.0
    return min(w * height / h if h > 1e-9 else w, max_width)


def _position_latin(
    strokes: list[Stroke], placement: CharPlacement, line_spacing: float
) -> list[Stroke]:
    """英字・ギリシャ文字を字種の帯（x-height・アセンダ・ディセンダ）へ合わせ、セル中央に置く。

    大文字の高さ = ``font_size``、ベースライン = 漢字の下端付近。
    """
    cap = placement.font_size
    lo, hi = _LATIN_BANDS.get(placement.char, _CAP_BAND)
    baseline_y = placement.y + (line_spacing - cap) / 2
    cell = _cell_width(placement)
    height = (hi - lo) * cap
    max_width = cell * _LATIN_MAX_INK
    ink = _ink_width(strokes, height, max_width)
    return _fit_band(
        strokes,
        placement,
        x_left=placement.x + (cell - ink) / 2,
        y_bottom=baseline_y + lo * cap,
        height=height,
        max_width=max_width,
    )


def _position_operator(
    strokes: list[Stroke], placement: CharPlacement, line_spacing: float
) -> list[Stroke]:
    """演算子を、字形の単位正方形を大文字高さ比の正方枠へ写して数式の軸に置く。"""
    cap = placement.font_size
    size = cap * _OPERATOR_BOX
    baseline_y = placement.y + (line_spacing - cap) / 2
    center = np.array([placement.x + _cell_width(placement) / 2, baseline_y + cap * _OPERATOR_AXIS])
    positioned = [(s - 0.5) * size + center for s in strokes]
    if placement.slant:
        positioned = rotate_about(positioned, placement.slant, tuple(center))
    return positioned


def _position_bracket(
    strokes: list[Stroke], placement: CharPlacement, line_spacing: float
) -> list[Stroke]:
    """括弧を中身の側へ寄せて置く。

    丸括弧などは漢字よりやや小さく行の中央に、「」『』は漢字の上端・下端に揃える。
    """
    char = placement.char
    body = _body_size(placement)
    cell = placement.advance if placement.advance is not None else _cell_width(placement)
    center_y = placement.y + line_spacing / 2
    opening = char in _OPENING_BRACKETS
    if char in _CORNER_BRACKETS:
        height = body * _CORNER_BRACKET_HEIGHT
        kanji_half = body / 2
        y_bottom = center_y + kanji_half - height if opening else center_y - kanji_half
    elif is_halfwidth(char):
        cap = placement.font_size
        lo, hi = _HALFWIDTH_BRACKET_BAND
        height = (hi - lo) * cap
        y_bottom = placement.y + (line_spacing - cap) / 2 + lo * cap
    else:
        height = body * _BRACKET_HEIGHT
        y_bottom = center_y - height / 2

    max_width = cell * _BRACKET_MAX_WIDTH
    ink = _ink_width(strokes, height, max_width)
    gap = min(body * _BRACKET_INNER_GAP, (cell - ink) / 2)
    x_left = placement.x + cell - gap - ink if opening else placement.x + gap
    return _fit_band(
        strokes, placement, x_left=x_left, y_bottom=y_bottom, height=height, max_width=max_width
    )


def _position_comma(strokes: list[Stroke], placement: CharPlacement, line_spacing: float) -> Stroke:
    all_pts = np.concatenate(strokes, axis=0)
    mins = all_pts.min(axis=0)
    ranges = all_pts.max(axis=0) - mins

    fs = placement.font_size
    cell_width = _cell_width(placement)
    target_h = fs * 0.3675
    target_w = target_h * 0.72
    scale_w = target_w / ranges[0] if ranges[0] > 1e-6 else float("inf")
    scale_h = target_h / ranges[1] if ranges[1] > 1e-6 else float("inf")
    scale = min(scale_w, scale_h)

    rendered_w = ranges[0] * scale
    rendered_h = ranges[1] * scale
    x_offset = placement.x + (cell_width - rendered_w) / 2
    y_offset = placement.y + line_spacing * 0.1
    stroke = (strokes[0] - mins) * scale + np.array([x_offset, y_offset])
    if placement.slant:
        center = (x_offset + rendered_w / 2, y_offset + rendered_h / 2)
        stroke = rotate_about([stroke], placement.slant, center)[0]
    return stroke


def _position_middle_dot(
    strokes: list[Stroke], placement: CharPlacement, line_spacing: float
) -> list[Stroke]:
    cell_width = placement.font_size * FULLWIDTH_CELL_FACTOR
    scale = cell_width * 0.5 / 0.3
    center = np.array([placement.x + cell_width / 2, placement.y + line_spacing / 2])
    raw_center = np.array([0.5, 0.5])
    positioned = [(stroke - raw_center) * scale + center for stroke in strokes]
    if placement.slant:
        positioned = rotate_about(positioned, placement.slant, center)
    return positioned


def _position_period(placement: CharPlacement, line_spacing: float) -> Stroke:
    """句点は丸ではなく短い斜めのダッシュ（ピリオド風の点）として描く。"""
    fs = placement.font_size
    dot_w = min(0.58, max(0.38, fs * 0.08))
    dot_h = dot_w * 0.75
    cx = placement.x + (_cell_width(placement) if is_halfwidth(placement.char) else fs * 0.55) / 2
    cy = placement.y + line_spacing * 0.14
    return np.array(
        [[cx - dot_w / 2, cy + dot_h / 2], [cx + dot_w / 2, cy - dot_h / 2]],
        dtype=np.float64,
    )
