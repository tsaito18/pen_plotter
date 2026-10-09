"""構造式を手書きで描く（matplotlib の配置 + 本文と同じ手書き字形）。

:func:`src.layout.mathtext.extract_math_layout` の配置へ、字ごとの手書き字形（本人サンプル・
幾何字形など。何を使うかは呼び出し側の ``glyph_source`` が決める）を貼る。

- 通常のグリフ: matplotlib のインク矩形に縦横比を保って収める
- 分数線: 直線
- 根号: 「入り → 谷 → 屋根の左端 → 屋根の右端」の 1 本の折れ線（チェック形は字の大きさで固定し、
  中身を屋根の左端へ寄せる）
- 中身の高さに合わせた大括弧: ( ) は弧、[ ] は折れ線、{ } は幾何字形を引き伸ばす
- アクセント（x̄ x̂ x⃗ ẋ x̃）: 下の字の上に短い線・山・矢印・点・波
- プライム f′: 短い斜めの線
- それ以外の大型記号（∑ ∫ など）: ``glyph_source`` に任せる（細線化した活字など）
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.geometry import Stroke
from src.layout.mathtext import MathGlyph, MathLayout, MathRect, ink_bbox, sqrt_roofs

GlyphSource = Callable[[str, bool], "tuple[list[Stroke], float] | None"]
"""``(字, 大型記号か) → (単位系の字形 Y-UP, 揺らぎ倍率)``。字形が無ければ None。"""
Distort = Callable[[list[Stroke], float], list[Stroke]]
"""``(mm のストローク, 揺らぎ倍率) → 揺らぎを乗せたストローク``。"""

# 根号のチェック形（字の大きさ比）と、屋根を中身の上端からどれだけ離すか
_CHECK_WIDTH = 0.45
_CHECK_HEIGHT = 0.55
_CHECK_VALLEY = 0.3  # 谷の位置（チェック幅比）
_ROOF_CLEARANCE = 0.08
# 根号の折れ線の揺らぎ（matplotlib 由来の線と同じ強さ）
ROOT_WAVER = 2.5
# 上付きのマイナス（10^{-4}）を数字の中ほどへ上げる量（字の大きさ比）
_SUPERSCRIPT_MINUS_RISE = 0.18
_DOT_CHARS = frozenset("·⋅・")
_PRIME_CHARS = frozenset("′'")
_OPENING = "([{"
_CLOSING = ")]}"
# アクセント（結合文字）。matplotlib は下の字の右端に置くので、位置は下の字から決める
_ACCENTS = {
    "\u0304": "bar",
    "\u0305": "bar",
    "\u0302": "hat",
    "\u0303": "tilde",
    "\u0307": "dot",
    "\u0308": "ddot",
    "\u20d7": "vec",
    "\u0301": "acute",
    "\u0300": "grave",
}
_ACCENT_GAP = 0.12  # 下の字の上端からの隙間（字の大きさ比）
_POINT_CHARS = frozenset(".,")
_MINUS_CHARS = frozenset("-−")


def render_math_handwritten(
    layout: MathLayout,
    *,
    scale: float,
    x_left: float,
    baseline: float,
    glyph_source: GlyphSource,
    distort: Distort,
) -> list[Stroke]:
    """配置 ``layout``（pt）を mm のストロークにする。

    Args:
        layout: 数式の配置。
        scale: pt → mm の縮尺。
        x_left: 式の左端(mm)。
        baseline: 式のベースライン(mm, Y-UP)。
        glyph_source: 字形の取り出し方。
        distort: 揺らぎの乗せ方。
    """

    def to_mm(pts: list[tuple[float, float]] | np.ndarray) -> Stroke:
        arr = np.asarray(pts, dtype=np.float64)
        return np.column_stack([x_left + arr[:, 0] * scale, baseline + arr[:, 1] * scale])

    out: list[Stroke] = []
    roofs = sqrt_roofs(layout)
    shifts: dict[int, float] = {}  # 根号の中身の左寄せ量（glyph 番号 → pt）
    rect_shifts: dict[int, float] = {}
    for gi, ri in roofs.items():
        stroke, glyph_ids, rect_ids, shift = _root_polyline(layout, gi, ri)
        out.extend(distort([to_mm(stroke)], ROOT_WAVER))
        for gj in glyph_ids:
            shifts[gj] = shift
        for rj in rect_ids:
            rect_shifts[rj] = shift

    spans = _bracket_spans(layout)
    base_size = max((g.fontsize for g in layout.glyphs), default=1.0)
    base_line = min(
        (g.baseline_y for g in layout.glyphs if g.fontsize >= base_size * 0.95), default=0.0
    )
    for gi, g in enumerate(layout.glyphs):
        if g.char == "√" and gi in roofs:
            continue
        if g.char in _ACCENTS:
            accent = _accent(layout, gi, shifts)
            if accent is not None:
                out.append(to_mm(accent))
            continue
        ink = ink_bbox(g)
        if ink is None:
            continue
        gx, gy, gw, gh = ink
        x0, y0 = g.x + gx + shifts.get(gi, 0.0), g.baseline_y + gy
        if gi in spans:
            low, high = spans[gi]
            if g.char in "()":
                out.append(to_mm(_parenthesis_arc(g.char, x0, gw, low, high)))
                continue
            if g.char in "[]":
                out.append(to_mm(_square_bracket(g.char, x0, gw, low, high)))
                continue
            source = glyph_source(g.char, False)
            if source is not None:  # { } は字形を中身の高さへ引き伸ばす
                out.extend(to_mm(s) for s in _stretch_to_box(source[0], x0, low, gw, high - low))
                continue
        if g.char in _PRIME_CHARS:
            out.append(to_mm([(x0 + gw * 0.9, y0 + gh), (x0 + gw * 0.2, y0 + gh * 0.35)]))
            continue
        if g.char in _DOT_CHARS or g.char in _POINT_CHARS:
            # 点は字形を引き伸ばすと読点のように大きくなるので、短い斜めの点にする
            cx, cy = x0 + gw / 2, y0 + gh / 2
            d = g.fontsize * 0.03
            out.append(to_mm([(cx - d, cy + d * 0.8), (cx + d, cy - d * 0.8)]))
            continue
        if (
            g.char in _MINUS_CHARS
            and g.fontsize < base_size * 0.95
            and g.baseline_y > base_line + base_size * 0.1
        ):
            y0 += g.fontsize * _SUPERSCRIPT_MINUS_RISE
        source = glyph_source(g.char, g.is_large)
        if source is None:
            continue
        unit, waver = source
        placed = [to_mm(s) for s in _fit_to_box(unit, x0, y0, gw, gh)]
        out.extend(distort(placed, waver) if waver > 0 else placed)

    for ri, r in enumerate(layout.rects):
        if ri in roofs.values():
            continue
        dx = rect_shifts.get(ri, 0.0)
        out.append(to_mm([(r.x + dx, r.center_y), (r.x + r.width + dx, r.center_y)]))
    return out


def _fit_to_box(unit: list[Stroke], x0: float, y0: float, w: float, h: float) -> list[np.ndarray]:
    """字形を矩形 (x0, y0, w, h) の中央へ縦横比を保って収める（細い 1・l を横に伸ばさない）。"""
    pts = np.concatenate(unit, axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    uw, uh = max(hi[0] - lo[0], 1e-6), max(hi[1] - lo[1], 1e-6)
    k = min(w / uw, h / uh)
    offset = np.array([x0 + (w - uw * k) / 2, y0 + (h - uh * k) / 2])
    return [(s - lo) * k + offset for s in unit]


def _stretch_to_box(unit: list[Stroke], x0: float, y0: float, w: float, h: float) -> list[Stroke]:
    """字形を矩形いっぱいに縦横別々に引き伸ばす。"""
    pts = np.concatenate(unit, axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return [(s - lo) / span * np.array([w, h]) + np.array([x0, y0]) for s in unit]


def _accent(
    layout: MathLayout, gi: int, shifts: dict[int, float]
) -> list[tuple[float, float]] | None:
    """アクセントの線（pt）。下の字 = アクセントより左に始まり、右端が最も近い字。"""
    g = layout.glyphs[gi]
    best: tuple[float, int, tuple[float, float, float, float]] | None = None
    for j, other in enumerate(layout.glyphs):
        if j == gi or other.char in _ACCENTS or other.x > g.x + 1e-6:
            continue
        ink = ink_bbox(other)
        if ink is None:
            continue
        distance = abs(other.x + ink[0] + ink[2] - g.x)
        if best is None or distance < best[0]:
            best = (distance, j, ink)
    if best is None:
        return None
    _, j, (ix, iy, iw, ih) = best
    base = layout.glyphs[j]
    left = base.x + ix + shifts.get(j, 0.0) + iw * 0.1
    w = iw * 0.8
    y = base.baseline_y + iy + ih + g.fontsize * _ACCENT_GAP
    fs = g.fontsize
    kind = _ACCENTS[g.char]
    if kind == "bar":
        return [(left, y), (left + w, y)]
    if kind == "hat":
        return [(left, y), (left + w / 2, y + fs * 0.2), (left + w, y)]
    if kind == "vec":
        tip = (left + w, y)
        return [
            (left, y),
            tip,
            (tip[0] - fs * 0.12, y + fs * 0.08),
            tip,
            (tip[0] - fs * 0.12, y - fs * 0.08),
        ]
    if kind == "tilde":
        t = np.linspace(0.0, 1.0, 7)
        return list(zip(left + w * t, y + fs * 0.06 * np.sin(t * 2 * np.pi), strict=True))
    if kind in ("acute", "grave"):
        lo, hi = (left + w * 0.35, left + w * 0.65)
        return [(lo, y), (hi, y + fs * 0.18)] if kind == "acute" else [(hi, y), (lo, y + fs * 0.18)]
    d = fs * 0.03  # 点
    cx = left + w / 2
    if kind == "dot":
        return [(cx - d, y + d), (cx + d, y)]
    return [(cx - w * 0.25 - d, y + d), (cx - w * 0.25 + d, y)]  # ddot の左の点（右は省略）


def _inside_roof(layout: MathLayout, roof: MathRect, glyph: MathGlyph) -> bool:
    """グリフが根号の屋根の下（中身）にあるか。x だけで選ぶと分数の分子まで拾う。"""
    if not roof.x <= glyph.x <= roof.x + roof.width:
        return False
    ink = ink_bbox(glyph)
    return ink is not None and glyph.baseline_y + ink[1] + ink[3] / 2 < roof.center_y


def _root_polyline(
    layout: MathLayout, gi: int, ri: int
) -> tuple[list[tuple[float, float]], list[int], list[int], float]:
    """根号の折れ線（pt）と、左へ寄せる中身（glyph・rect の番号）と寄せ量を返す。"""
    g = layout.glyphs[gi]
    roof = layout.rects[ri]
    gx, gy, gw, gh = ink_bbox(g) or (0.0, 0.0, 0.0, 0.0)
    left, bottom = g.x + gx, g.baseline_y + gy

    glyph_ids = [
        j
        for j, other in enumerate(layout.glyphs)
        if j != gi and other.char != "√" and _inside_roof(layout, roof, other)
    ]
    rect_ids = [
        j
        for j, r in enumerate(layout.rects)
        if j != ri
        and r.x >= roof.x
        and r.x + r.width <= roof.x + roof.width
        and r.center_y < roof.center_y
    ]
    lefts, rights, tops = [], [], [bottom]
    for j in glyph_ids:
        other = layout.glyphs[j]
        ix, iy, iw, ih = ink_bbox(other) or (0.0, 0.0, 0.0, 0.0)
        lefts.append(other.x + ix)
        rights.append(other.x + ix + iw)
        tops.append(other.baseline_y + iy + ih)
    for j in rect_ids:
        r = layout.rects[j]
        lefts.append(r.x)
        rights.append(r.x + r.width)
        tops.append(r.y + r.height)

    # チェック形（入り→谷）は字の大きさで固定し、谷から屋根へは根号の幅いっぱいに上がる
    check_w = min(gw, g.fontsize * _CHECK_WIDTH)
    check_h = min(gh * _CHECK_HEIGHT, g.fontsize * _CHECK_HEIGHT)
    peak_x = left + max(gw, check_w)
    roof_y = min(roof.center_y, max(tops) + g.fontsize * _ROOF_CLEARANCE)
    shift = min(0.0, peak_x - min(lefts)) if lefts else 0.0
    right = max(rights) + shift if rights else roof.x + roof.width
    stroke = [
        (left, bottom + check_h),  # 入り
        (left + check_w * _CHECK_VALLEY, bottom),  # 谷
        (peak_x, roof_y),  # 屋根の左端
        (right, roof_y),
    ]
    return stroke, glyph_ids, rect_ids, shift


def _bracket_spans(layout: MathLayout) -> dict[int, tuple[float, float]]:
    """大括弧の対ごとに、中身（グリフ・分数線）の下端〜上端 (pt) を少し広げた範囲。

    matplotlib の大括弧は分数の全高に届かないことがあるので、縦の範囲は中身から決める。
    """
    spans: dict[int, tuple[float, float]] = {}
    stack: list[int] = []
    for gi, g in enumerate(layout.glyphs):
        if not g.is_large:
            continue
        if g.char in _OPENING:
            stack.append(gi)
            continue
        if g.char not in _CLOSING or not stack:
            continue
        oi = stack.pop()
        x_open, x_close = layout.glyphs[oi].x, g.x
        lows, highs = [], []
        for inner in layout.glyphs[oi + 1 : gi]:
            ink = ink_bbox(inner)
            if ink is not None:
                lows.append(inner.baseline_y + ink[1])
                highs.append(inner.baseline_y + ink[1] + ink[3])
        for r in layout.rects:
            if x_open < r.x < x_close:
                lows.append(r.y)
                highs.append(r.y + r.height)
        if lows:
            pad = (max(highs) - min(lows)) * 0.08
            spans[oi] = spans[gi] = (min(lows) - pad, max(highs) + pad)
    return spans


def _square_bracket(
    char: str, x0: float, w: float, low: float, high: float
) -> list[tuple[float, float]]:
    """中身の高さまで伸ばした [ ] の折れ線（pt）。"""
    if char == "[":
        return [(x0 + w, high), (x0 + w * 0.3, high), (x0 + w * 0.3, low), (x0 + w, low)]
    return [(x0, high), (x0 + w * 0.7, high), (x0 + w * 0.7, low), (x0, low)]


def _parenthesis_arc(
    char: str, x0: float, w: float, low: float, high: float
) -> list[tuple[float, float]]:
    """中身の高さまで伸ばした ( ) の弧（pt）。"""
    h = high - low
    mid = (low + high) / 2
    if char == "(":
        return [
            (x0 + w, high),
            (x0 + 0.35 * w, high - 0.18 * h),
            (x0, mid),
            (x0 + 0.35 * w, low + 0.18 * h),
            (x0 + w, low),
        ]
    return [
        (x0, high),
        (x0 + 0.65 * w, high - 0.18 * h),
        (x0 + w, mid),
        (x0 + 0.65 * w, low + 0.18 * h),
        (x0, low),
    ]
