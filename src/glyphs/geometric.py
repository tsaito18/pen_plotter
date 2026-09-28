"""単位正方形 [0,1]x[0,1]（Y-UP）で描く幾何字形の定義。

ML・KanjiVG・ユーザー筆跡のいずれも持たない記号・句読点・括弧・ギリシャ文字・
数式記号・英字を、直線や円弧の組み合わせで描く。字形は ``文字 → ストローク列``
を返す関数として登録し、:func:`symbol_glyph` / :func:`latin_glyph` で引く。

英字(LATIN)は他と座標の約束が異なる: 大文字は cap height ``y∈[0, 0.95]``、小文字は
x-height 0.70 を上端、ディセンダは ``y<0`` を使う（配置側で論理座標フィットする）。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.geometry import Stroke

GlyphFn = Callable[[], list[Stroke]]

SYMBOL_GLYPHS: dict[str, GlyphFn] = {}
LATIN_GLYPHS: dict[str, GlyphFn] = {}

# 英字字形の cap height 上端 y（論理座標フィットの基準）
LATIN_CAP_TOP = 0.95


def _register(table: dict[str, GlyphFn], *chars: str) -> Callable[[GlyphFn], GlyphFn]:
    def deco(fn: GlyphFn) -> GlyphFn:
        for c in chars:
            table[c] = fn
        return fn

    return deco


def _symbol_glyph(*chars: str) -> Callable[[GlyphFn], GlyphFn]:
    return _register(SYMBOL_GLYPHS, *chars)


def _latin_glyph(*chars: str) -> Callable[[GlyphFn], GlyphFn]:
    return _register(LATIN_GLYPHS, *chars)


def symbol_glyph(char: str) -> list[Stroke] | None:
    """記号・句読点・括弧・ギリシャ文字・数式記号の字形。未登録なら None。"""
    fn = SYMBOL_GLYPHS.get(char)
    return fn() if fn else None


def latin_glyph(char: str) -> list[Stroke] | None:
    """英字の幾何字形。未登録なら None。"""
    fn = LATIN_GLYPHS.get(char)
    return fn() if fn else None


def small_dot(cx: float, cy: float, r: float = 0.06) -> Stroke:
    """小円（コロン・i の点など）。"""
    angles = np.linspace(0, 2 * np.pi, 12)
    return np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1).astype(np.float64)


def unit_circle(cx: float, cy: float, r: float, n: int = 24) -> Stroke:
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1).astype(np.float64)


def middle_dot_spiral() -> Stroke:
    """中黒「・」: 塗りつぶし風の渦巻き。"""
    t = np.linspace(0.0, 12.0 * np.pi, 145)
    r = np.linspace(0.15, 0.0, t.size)
    return np.stack([0.5 + r * np.cos(t), 0.5 + r * np.sin(t)], axis=1).astype(np.float64)


def fit_into_box(strokes: list[Stroke], x0: float, y0: float, x1: float, y1: float) -> list[Stroke]:
    """ストローク群をアスペクト比保持で [x0,x1]x[y0,y1] の中央に収める。"""
    if not strokes:
        return []
    pts = np.concatenate(strokes, axis=0)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    sx = (x1 - x0) / span[0] if span[0] > 1e-6 else 1.0
    sy = (y1 - y0) / span[1] if span[1] > 1e-6 else 1.0
    s = min(sx, sy)
    w, h = span[0] * s, span[1] * s
    ox = x0 + ((x1 - x0) - w) / 2 - mn[0] * s
    oy = y0 + ((y1 - y0) - h) / 2 - mn[1] * s
    return [stroke * s + np.array([ox, oy]) for stroke in strokes]


# 丸数字 ①..⑳ → 内側に入れる数字文字列（数字字形は KanjiVG 参照から取る）
CIRCLED_NUMBERS: dict[str, str] = {chr(0x2460 + i): str(i + 1) for i in range(20)}


def circled_number_glyph(digit_strokes: list[Stroke] | None) -> list[Stroke]:
    """丸数字: 外周の円＋内側に縮小した数字字形。"""
    circle = unit_circle(0.5, 0.5, 0.46)
    if not digit_strokes:
        return [circle]
    return [circle, *fit_into_box(digit_strokes, 0.3, 0.28, 0.7, 0.72)]


# =============================================================================
# 記号・句読点・括弧・ギリシャ文字・数式記号
# =============================================================================


@_symbol_glyph("、", ",", "，")
def _ideographic_comma() -> list[Stroke]:
    # 正規化後の本文読点はすべて「，」(U+FF0C)に寄せるため同一形にする
    return [np.array([[0.58, 0.48], [0.42, 0.26]], dtype=np.float64)]


@_symbol_glyph("。", ".", "．")
def _ideographic_full_stop() -> list[Stroke]:
    # 句点はレポート体裁に合わせ、丸(円)ではなくピリオド風の短い点(描けるドット)
    # 正規化後の本文句点はすべて「．」(U+FF0E)に寄せるため同一形にする
    return [np.array([[0.475, 0.245], [0.525, 0.205]], dtype=np.float64)]


@_symbol_glyph("・")
def _katakana_middle_dot() -> list[Stroke]:
    return [middle_dot_spiral()]


@_symbol_glyph("+")
def _plus_sign() -> list[Stroke]:
    h = np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float64)
    v = np.array([[0.5, 0.2], [0.5, 0.8]], dtype=np.float64)
    return [h, v]


@_symbol_glyph("-")
def _hyphen_minus() -> list[Stroke]:
    return [np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float64)]


@_symbol_glyph("=")
def _equals_sign() -> list[Stroke]:
    top = np.array([[0.2, 0.4], [0.8, 0.4]], dtype=np.float64)
    bot = np.array([[0.2, 0.6], [0.8, 0.6]], dtype=np.float64)
    return [top, bot]


@_symbol_glyph("<")
def _less_than_sign() -> list[Stroke]:
    return [np.array([[0.8, 0.2], [0.2, 0.5], [0.8, 0.8]], dtype=np.float64)]


@_symbol_glyph(">")
def _greater_than_sign() -> list[Stroke]:
    return [np.array([[0.2, 0.2], [0.8, 0.5], [0.2, 0.8]], dtype=np.float64)]


@_symbol_glyph("*")
def _asterisk() -> list[Stroke]:
    cx, cy, r = 0.5, 0.5, 0.3
    arms: list[Stroke] = []
    for k in range(3):
        ang = np.pi * k / 3.0
        dx, dy = r * np.cos(ang), r * np.sin(ang)
        arms.append(np.array([[cx - dx, cy - dy], [cx + dx, cy + dy]], dtype=np.float64))
    return arms


@_symbol_glyph("/")
def _solidus() -> list[Stroke]:
    return [np.array([[0.2, 0.2], [0.8, 0.8]], dtype=np.float64)]


@_symbol_glyph("%")
def _percent_sign() -> list[Stroke]:
    # 斜線（右上がり）＋左上・右下の小円
    diag = np.array([[0.18, 0.12], [0.82, 0.88]], dtype=np.float64)
    t = np.linspace(0.0, 2.0 * np.pi, 13)
    tl = np.stack([0.28 + 0.13 * np.cos(t), 0.72 + 0.13 * np.sin(t)], axis=1).astype(np.float64)
    br = np.stack([0.72 + 0.13 * np.cos(t), 0.28 + 0.13 * np.sin(t)], axis=1).astype(np.float64)
    return [diag, tl, br]


@_symbol_glyph(":")
def _colon() -> list[Stroke]:
    return [small_dot(0.5, 0.7), small_dot(0.5, 0.3)]


@_symbol_glyph(";")
def _semicolon() -> list[Stroke]:
    tail = np.array([[0.55, 0.3], [0.4, 0.0]], dtype=np.float64)
    return [small_dot(0.5, 0.7), tail]


@_symbol_glyph("!")
def _exclamation_mark() -> list[Stroke]:
    stem = np.array([[0.5, 0.85], [0.5, 0.25]], dtype=np.float64)
    return [stem, small_dot(0.5, 0.05)]


@_symbol_glyph("?")
def _question_mark() -> list[Stroke]:
    t = np.linspace(np.pi, 0.0, 16)
    arc = np.stack([0.5 + 0.2 * np.cos(t), 0.725 + 0.125 * np.sin(t)], axis=1).astype(np.float64)
    stem = np.array([[0.7, 0.6], [0.55, 0.35]], dtype=np.float64)
    return [arc, stem, small_dot(0.55, 0.1)]


@_symbol_glyph("[")
def _left_square_bracket() -> list[Stroke]:
    return [np.array([[0.62, 0.9], [0.4, 0.9], [0.4, 0.1], [0.62, 0.1]], dtype=np.float64)]


@_symbol_glyph("]")
def _right_square_bracket() -> list[Stroke]:
    return [np.array([[0.38, 0.9], [0.6, 0.9], [0.6, 0.1], [0.38, 0.1]], dtype=np.float64)]


@_symbol_glyph("~")
def _tilde() -> list[Stroke]:
    t = np.linspace(0.0, 1.0, 20)
    x = 0.2 + 0.6 * t
    y = 0.5 + 0.12 * np.sin(2.0 * np.pi * t)
    return [np.stack([x, y], axis=1).astype(np.float64)]


@_symbol_glyph("(", "（")
def _left_parenthesis() -> list[Stroke]:
    # 「(」は左寄りで中央が左に凸（開口は右向き）
    points = []
    for i in range(20):
        t = i / 19
        x = 0.40 - 0.25 * np.cos(np.pi * (t - 0.5))
        y = 0.1 + 0.8 * t
        points.append([x, y])
    return [np.array(points)]


@_symbol_glyph(")", "）")
def _right_parenthesis() -> list[Stroke]:
    # 「)」は右寄りで中央が右に凸（開口は左向き）
    points = []
    for i in range(20):
        t = i / 19
        x = 0.60 + 0.25 * np.cos(np.pi * (t - 0.5))
        y = 0.1 + 0.8 * t
        points.append([x, y])
    return [np.array(points)]


@_symbol_glyph("「")
def _left_corner_bracket() -> list[Stroke]:
    return [
        np.array([[0.8, 0.15], [0.25, 0.15]], dtype=np.float64),
        np.array([[0.25, 0.15], [0.25, 0.45]], dtype=np.float64),
    ]


@_symbol_glyph("」")
def _right_corner_bracket() -> list[Stroke]:
    return [
        np.array([[0.75, 0.55], [0.75, 0.85]], dtype=np.float64),
        np.array([[0.75, 0.85], [0.2, 0.85]], dtype=np.float64),
    ]


@_symbol_glyph("『")
def _left_white_corner_bracket() -> list[Stroke]:
    return [
        np.array([[0.8, 0.15], [0.25, 0.15], [0.25, 0.45]], dtype=np.float64),
        np.array([[0.65, 0.25], [0.35, 0.25], [0.35, 0.45]], dtype=np.float64),
    ]


@_symbol_glyph("』")
def _right_white_corner_bracket() -> list[Stroke]:
    return [
        np.array([[0.75, 0.55], [0.75, 0.85], [0.2, 0.85]], dtype=np.float64),
        np.array([[0.65, 0.55], [0.65, 0.75], [0.35, 0.75]], dtype=np.float64),
    ]


@_symbol_glyph("ω")
def _greek_omega() -> list[Stroke]:
    # ω: 左右の半円を底辺でつないだ形
    t_left = np.linspace(np.pi, 2 * np.pi, 16)
    left = np.stack([0.28 + 0.20 * np.cos(t_left), 0.45 + 0.30 * np.sin(t_left)], axis=1)
    t_right = np.linspace(np.pi, 2 * np.pi, 16)
    right = np.stack([0.72 + 0.20 * np.cos(t_right), 0.45 + 0.30 * np.sin(t_right)], axis=1)
    bridge = np.array([[0.20, 0.45], [0.80, 0.45]], dtype=np.float64)
    return [left, right, bridge]


@_symbol_glyph("φ")
def _greek_phi() -> list[Stroke]:
    angles = np.linspace(0, 2 * np.pi, 24)
    r = 0.3
    circle = np.stack([0.5 + r * np.cos(angles), 0.55 + r * np.sin(angles)], axis=1)
    stem = np.array([[0.5, 0.1], [0.5, 0.9]])
    return [circle, stem]


@_symbol_glyph("π")
def _greek_pi() -> list[Stroke]:
    # π: バーが上、脚が下（逆U型ではなくπ型）
    top = np.array([[0.10, 0.80], [0.90, 0.80]], dtype=np.float64)
    left_leg = np.array([[0.25, 0.80], [0.20, 0.05]], dtype=np.float64)
    right_leg = np.array([[0.75, 0.80], [0.80, 0.05]], dtype=np.float64)
    return [top, left_leg, right_leg]


@_symbol_glyph("θ")
def _greek_theta() -> list[Stroke]:
    angles = np.linspace(0, 2 * np.pi, 24)
    rx, ry = 0.3, 0.4
    ellipse = np.stack([0.5 + rx * np.cos(angles), 0.5 + ry * np.sin(angles)], axis=1)
    bar = np.array([[0.2, 0.5], [0.8, 0.5]])
    return [ellipse, bar]


@_symbol_glyph("α")
def _greek_alpha() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 30)
    x = 0.5 + 0.3 * np.cos(t) - 0.1 * np.sin(2 * t)
    y = 0.5 + 0.35 * np.sin(t)
    return [np.stack([x, y], axis=1)]


@_symbol_glyph("Δ")
def _greek_capital_delta() -> list[Stroke]:
    triangle = np.array([[0.5, 0.1], [0.1, 0.9], [0.9, 0.9], [0.5, 0.1]])
    return [triangle]


@_symbol_glyph("±")
def _plus_minus_sign() -> list[Stroke]:
    h_top = np.array([[0.15, 0.2], [0.85, 0.2]])
    h_mid = np.array([[0.15, 0.5], [0.85, 0.5]])
    v_mid = np.array([[0.5, 0.2], [0.5, 0.8]])
    return [h_top, h_mid, v_mid]


@_symbol_glyph("≈")
def _almost_equal_to() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 20)
    x = np.linspace(0.1, 0.9, 20)
    wave1 = np.stack([x, 0.35 + 0.08 * np.sin(t)], axis=1)
    wave2 = np.stack([x, 0.65 + 0.08 * np.sin(t)], axis=1)
    return [wave1, wave2]


@_symbol_glyph("∞")
def _infinity() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 40)
    x = 0.5 + 0.35 * np.cos(t) / (1 + np.sin(t) ** 2)
    y = 0.5 + 0.25 * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
    return [np.stack([x, y], axis=1)]


@_symbol_glyph("β")
def _greek_beta() -> list[Stroke]:
    stem = np.array([[0.25, 0.05], [0.25, 0.95]], dtype=np.float64)
    t = np.linspace(0, 1, 30)
    upper = np.stack([0.25 + 0.45 * np.sin(np.pi * t), 0.5 + 0.4 * (1 - t)], axis=1)
    lower = np.stack([0.25 + 0.5 * np.sin(np.pi * t), 0.5 - 0.45 * t], axis=1)
    return [stem, upper, lower]


@_symbol_glyph("γ")
def _greek_gamma() -> list[Stroke]:
    left = np.array([[0.15, 0.85], [0.5, 0.4]], dtype=np.float64)
    right = np.array([[0.85, 0.85], [0.4, 0.05]], dtype=np.float64)
    return [left, right]


@_symbol_glyph("δ")
def _greek_delta() -> list[Stroke]:
    t = np.linspace(0.2 * np.pi, 1.8 * np.pi, 30)
    body = np.stack([0.5 + 0.3 * np.cos(t), 0.4 + 0.3 * np.sin(t)], axis=1)
    tail = np.array([[0.7, 0.85], [0.55, 0.95]], dtype=np.float64)
    return [body, tail]


@_symbol_glyph("ε")
def _greek_epsilon() -> list[Stroke]:
    t = np.linspace(0.5 * np.pi, 1.5 * np.pi, 16)
    top = np.stack([0.55 - 0.3 * np.sin(t), 0.7 + 0.18 * np.cos(t)], axis=1)
    bot = np.stack([0.55 - 0.3 * np.sin(t), 0.3 + 0.18 * np.cos(t)], axis=1)
    mid = np.array([[0.3, 0.5], [0.55, 0.5]], dtype=np.float64)
    return [top, mid, bot]


@_symbol_glyph("ζ")
def _greek_zeta() -> list[Stroke]:
    top = np.array([[0.2, 0.9], [0.8, 0.9], [0.3, 0.4]], dtype=np.float64)
    t = np.linspace(0, np.pi, 16)
    tail = np.stack([0.5 + 0.25 * np.sin(t), 0.2 - 0.18 * (1 - np.cos(t))], axis=1)
    return [top, tail]


@_symbol_glyph("η")
def _greek_eta() -> list[Stroke]:
    stem_left = np.array([[0.2, 0.7], [0.2, 0.05]], dtype=np.float64)
    t = np.linspace(np.pi, 0, 16)
    arch = np.stack([0.5 + 0.3 * np.cos(t), 0.55 + 0.15 * np.sin(t)], axis=1)
    stem_right = np.array([[0.8, 0.7], [0.8, 0.2]], dtype=np.float64)
    return [stem_left, arch, stem_right]


@_symbol_glyph("λ")
def _greek_lamda() -> list[Stroke]:
    left_leg = np.array([[0.15, 0.05], [0.45, 0.90]], dtype=np.float64)
    right_leg = np.array([[0.45, 0.90], [0.85, 0.05]], dtype=np.float64)
    return [left_leg, right_leg]


@_symbol_glyph("μ")
def _greek_mu() -> list[Stroke]:
    left_stem = np.array([[0.2, 0.85], [0.2, 0.0]], dtype=np.float64)
    right_stem = np.array([[0.8, 0.85], [0.8, 0.48]], dtype=np.float64)
    t = np.linspace(np.pi, 2 * np.pi, 16)
    u_curve = np.stack([0.5 + 0.3 * np.cos(t), 0.48 + 0.28 * np.sin(t)], axis=1)
    return [left_stem, u_curve, right_stem]


@_symbol_glyph("ν")
def _greek_nu() -> list[Stroke]:
    return [np.array([[0.15, 0.85], [0.5, 0.1], [0.85, 0.85]], dtype=np.float64)]


@_symbol_glyph("ρ")
def _greek_rho() -> list[Stroke]:
    stem = np.array([[0.3, 0.5], [0.3, 0.0]], dtype=np.float64)
    t = np.linspace(0, 2 * np.pi, 24)
    circle = np.stack([0.5 + 0.25 * np.cos(t), 0.55 + 0.25 * np.sin(t)], axis=1)
    return [stem, circle]


@_symbol_glyph("σ")
def _greek_sigma() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 24)
    circle = np.stack([0.4 + 0.25 * np.cos(t), 0.4 + 0.25 * np.sin(t)], axis=1)
    top = np.array([[0.4, 0.7], [0.85, 0.7]], dtype=np.float64)
    return [circle, top]


@_symbol_glyph("τ")
def _greek_tau() -> list[Stroke]:
    top = np.array([[0.1, 0.75], [0.9, 0.75]], dtype=np.float64)
    t = np.linspace(0, np.pi / 2, 16)
    stem = np.stack([0.5 + 0.2 * np.sin(t), 0.75 - 0.7 * t / (np.pi / 2)], axis=1)
    return [top, stem]


@_symbol_glyph("χ")
def _greek_chi() -> list[Stroke]:
    d1 = np.array([[0.15, 0.1], [0.85, 0.85]], dtype=np.float64)
    d2 = np.array([[0.85, 0.1], [0.15, 0.85]], dtype=np.float64)
    return [d1, d2]


@_symbol_glyph("ψ")
def _greek_psi() -> list[Stroke]:
    v = np.array([[0.15, 0.75], [0.5, 0.35], [0.85, 0.75]], dtype=np.float64)
    stem = np.array([[0.5, 0.95], [0.5, 0.05]], dtype=np.float64)
    return [v, stem]


@_symbol_glyph("Γ")
def _greek_capital_gamma() -> list[Stroke]:
    top = np.array([[0.15, 0.9], [0.85, 0.9]], dtype=np.float64)
    left = np.array([[0.15, 0.9], [0.15, 0.1]], dtype=np.float64)
    return [top, left]


@_symbol_glyph("Λ")
def _greek_capital_lamda() -> list[Stroke]:
    return [np.array([[0.1, 0.1], [0.5, 0.9], [0.9, 0.1]], dtype=np.float64)]


@_symbol_glyph("Θ")
def _greek_capital_theta() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 30)
    ellipse = np.stack([0.5 + 0.35 * np.cos(t), 0.5 + 0.4 * np.sin(t)], axis=1)
    bar = np.array([[0.3, 0.5], [0.7, 0.5]], dtype=np.float64)
    return [ellipse, bar]


@_symbol_glyph("Π")
def _greek_capital_pi() -> list[Stroke]:
    top = np.array([[0.1, 0.9], [0.9, 0.9]], dtype=np.float64)
    left = np.array([[0.2, 0.9], [0.2, 0.1]], dtype=np.float64)
    right = np.array([[0.8, 0.9], [0.8, 0.1]], dtype=np.float64)
    return [top, left, right]


@_symbol_glyph("Σ", "∑")
def _greek_capital_sigma() -> list[Stroke]:
    top = np.array([[0.1, 0.9], [0.9, 0.9]], dtype=np.float64)
    diag1 = np.array([[0.1, 0.9], [0.5, 0.5]], dtype=np.float64)
    diag2 = np.array([[0.5, 0.5], [0.1, 0.1]], dtype=np.float64)
    bot = np.array([[0.1, 0.1], [0.9, 0.1]], dtype=np.float64)
    return [top, diag1, diag2, bot]


@_symbol_glyph("Φ")
def _greek_capital_phi() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 24)
    circle = np.stack([0.5 + 0.3 * np.cos(t), 0.5 + 0.3 * np.sin(t)], axis=1)
    stem = np.array([[0.5, 0.95], [0.5, 0.05]], dtype=np.float64)
    return [circle, stem]


@_symbol_glyph("Ψ")
def _greek_capital_psi() -> list[Stroke]:
    t = np.linspace(np.pi, 2 * np.pi, 16)
    cup = np.stack([0.5 + 0.35 * np.cos(t), 0.55 + 0.25 * np.sin(t)], axis=1)
    stem = np.array([[0.5, 0.95], [0.5, 0.05]], dtype=np.float64)
    base = np.array([[0.25, 0.05], [0.75, 0.05]], dtype=np.float64)
    return [cup, stem, base]


@_symbol_glyph("Ω")
def _greek_capital_omega() -> list[Stroke]:
    t = np.linspace(np.pi, 2 * np.pi, 24)
    arch = np.stack([0.5 + 0.35 * np.cos(t), 0.4 + 0.45 * np.sin(t)], axis=1)
    left_foot = np.array([[0.15, 0.4], [0.05, 0.1]], dtype=np.float64)
    right_foot = np.array([[0.85, 0.4], [0.95, 0.1]], dtype=np.float64)
    base_l = np.array([[0.05, 0.1], [0.25, 0.1]], dtype=np.float64)
    base_r = np.array([[0.75, 0.1], [0.95, 0.1]], dtype=np.float64)
    return [arch, left_foot, right_foot, base_l, base_r]


@_symbol_glyph("×")
def _multiplication_sign() -> list[Stroke]:
    d1 = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float64)
    d2 = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float64)
    return [d1, d2]


@_symbol_glyph("÷")
def _division_sign() -> list[Stroke]:
    bar = np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float64)
    return [bar, small_dot(0.5, 0.75), small_dot(0.5, 0.25)]


@_symbol_glyph("≠")
def _not_equal_to() -> list[Stroke]:
    top = np.array([[0.2, 0.4], [0.8, 0.4]], dtype=np.float64)
    bot = np.array([[0.2, 0.6], [0.8, 0.6]], dtype=np.float64)
    slash = np.array([[0.7, 0.2], [0.3, 0.8]], dtype=np.float64)
    return [top, bot, slash]


@_symbol_glyph("≤")
def _less_than_or_equal_to() -> list[Stroke]:
    v = np.array([[0.8, 0.25], [0.2, 0.55], [0.8, 0.85]], dtype=np.float64)
    bar = np.array([[0.2, 0.15], [0.8, 0.15]], dtype=np.float64)
    return [v, bar]


@_symbol_glyph("≥")
def _greater_than_or_equal_to() -> list[Stroke]:
    v = np.array([[0.2, 0.25], [0.8, 0.55], [0.2, 0.85]], dtype=np.float64)
    bar = np.array([[0.2, 0.15], [0.8, 0.15]], dtype=np.float64)
    return [v, bar]


@_symbol_glyph("·")
def _middle_dot() -> list[Stroke]:
    return [small_dot(0.5, 0.5)]


@_symbol_glyph("…")
def _horizontal_ellipsis() -> list[Stroke]:
    return [small_dot(0.2, 0.2), small_dot(0.5, 0.2), small_dot(0.8, 0.2)]


@_symbol_glyph("→")
def _rightwards_arrow() -> list[Stroke]:
    shaft = np.array([[0.1, 0.5], [0.85, 0.5]], dtype=np.float64)
    head_top = np.array([[0.85, 0.5], [0.65, 0.65]], dtype=np.float64)
    head_bot = np.array([[0.85, 0.5], [0.65, 0.35]], dtype=np.float64)
    return [shaft, head_top, head_bot]


@_symbol_glyph("←")
def _leftwards_arrow() -> list[Stroke]:
    shaft = np.array([[0.15, 0.5], [0.9, 0.5]], dtype=np.float64)
    head_top = np.array([[0.15, 0.5], [0.35, 0.65]], dtype=np.float64)
    head_bot = np.array([[0.15, 0.5], [0.35, 0.35]], dtype=np.float64)
    return [shaft, head_top, head_bot]


@_symbol_glyph("⇒")
def _rightwards_double_arrow() -> list[Stroke]:
    top = np.array([[0.1, 0.55], [0.8, 0.55]], dtype=np.float64)
    bot = np.array([[0.1, 0.45], [0.8, 0.45]], dtype=np.float64)
    head_top = np.array([[0.85, 0.5], [0.65, 0.7]], dtype=np.float64)
    head_bot = np.array([[0.85, 0.5], [0.65, 0.3]], dtype=np.float64)
    return [top, bot, head_top, head_bot]


@_symbol_glyph("∂")
def _partial_differential() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 30)
    body = np.stack([0.5 + 0.3 * np.cos(t), 0.4 + 0.35 * np.sin(t)], axis=1)
    tail = np.array([[0.55, 0.75], [0.85, 0.95]], dtype=np.float64)
    return [body, tail]


@_symbol_glyph("∇")
def _nabla() -> list[Stroke]:
    return [np.array([[0.1, 0.9], [0.9, 0.9], [0.5, 0.1], [0.1, 0.9]], dtype=np.float64)]


@_symbol_glyph("∫")
def _integral() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 40)
    x = 0.5 + 0.18 * np.sin(t * 0.5 + np.pi)
    y = np.linspace(0.05, 0.95, 40)
    stroke = np.stack([x, y], axis=1)
    top_hook = np.array([[stroke[-1, 0], stroke[-1, 1]], [0.7, 0.95]], dtype=np.float64)
    bot_hook = np.array([[0.3, 0.05], [stroke[0, 0], stroke[0, 1]]], dtype=np.float64)
    return [bot_hook, stroke, top_hook]


@_symbol_glyph("∏")
def _n_ary_product() -> list[Stroke]:
    top = np.array([[0.1, 0.9], [0.9, 0.9]], dtype=np.float64)
    left = np.array([[0.2, 0.9], [0.2, 0.1]], dtype=np.float64)
    right = np.array([[0.8, 0.9], [0.8, 0.1]], dtype=np.float64)
    return [top, left, right]


@_symbol_glyph("°")
def _degree_sign() -> list[Stroke]:
    return [unit_circle(0.5, 0.78, 0.13)]


@_symbol_glyph("℃")
def _degree_celsius() -> list[Stroke]:
    deg = unit_circle(0.22, 0.82, 0.1)
    t = np.linspace(0.35 * np.pi, 1.65 * np.pi, 22)
    c_arc = np.stack([0.62 + 0.3 * np.cos(t), 0.42 + 0.34 * np.sin(t)], axis=1).astype(np.float64)
    return [deg, c_arc]


# =============================================================================
# 英字（x-height 0.70 / cap 0.95 / ディセンダ y<0）
# =============================================================================


@_latin_glyph("c")
def _small_c() -> list[Stroke]:
    # x-height 統一: 上端を 0.70 に揃え「大文字混じり」に見えるのを防ぐ。
    t = np.linspace(0.25 * np.pi, 1.75 * np.pi, 24)
    return [
        np.stack(
            [0.55 + 0.27 * np.cos(t), 0.42 + 0.27 * np.sin(t)],
            axis=1,
        ).astype(np.float64)
    ]


@_latin_glyph("o")
def _small_o() -> list[Stroke]:
    # x-height 統一: 上端 0.70（旧 0.80）に揃える。
    t = np.linspace(0, 2 * np.pi, 28)
    return [
        np.stack(
            [0.5 + 0.27 * np.cos(t), 0.43 + 0.27 * np.sin(t)],
            axis=1,
        ).astype(np.float64)
    ]


@_latin_glyph("s")
def _small_s() -> list[Stroke]:
    t = np.linspace(0, 1, 30)
    # 上が左・下が右に膨らむ正しい S 字（+sin だと左右反転 Ƨ になる）
    # x-height 統一: 上端 0.70（旧 0.85）に揃える。
    x = 0.5 - 0.28 * np.sin(2 * np.pi * t)
    y = 0.70 - 0.55 * t
    return [np.stack([x, y], axis=1).astype(np.float64)]


@_latin_glyph("i")
def _small_i() -> list[Stroke]:
    stem = np.array([[0.5, 0.25], [0.5, 0.7]], dtype=np.float64)
    return [stem, small_dot(0.5, 0.88)]


@_latin_glyph("n")
def _small_n() -> list[Stroke]:
    # 左の縦棒＋上に膨らむアーチ(∩)＋右脚。中央が下がる ν との混同を解消
    # x-height 統一: アーチ上端 0.70 に揃える。
    stem = np.array([[0.22, 0.2], [0.22, 0.7]], dtype=np.float64)
    t = np.linspace(np.pi, 0, 16)
    arch = np.stack([0.5 + 0.28 * np.cos(t), 0.48 + 0.22 * np.sin(t)], axis=1).astype(np.float64)
    right = np.array([[0.78, 0.48], [0.78, 0.2]], dtype=np.float64)
    return [stem, arch, right]


@_latin_glyph("t")
def _small_t() -> list[Stroke]:
    # 縦棒の下端を右へ軽くはらい、クロスバーは上寄り → "+" と区別
    # アセンダ統一: ステム上端は cap より少し低い 0.80（t の伝統的高さ）。
    # クロスバーは x-height 帯（≈0.62）に据えて他小文字と高さを揃える。
    stem = np.array([[0.48, 0.80], [0.48, 0.12], [0.62, 0.07]], dtype=np.float64)
    cross = np.array([[0.28, 0.62], [0.72, 0.62]], dtype=np.float64)
    return [stem, cross]


@_latin_glyph("a")
def _small_a() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 24)
    body = np.stack(
        [0.45 + 0.25 * np.cos(t), 0.45 + 0.25 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    tail = np.array([[0.7, 0.2], [0.7, 0.7]], dtype=np.float64)
    return [body, tail]


@_latin_glyph("l")
def _small_l() -> list[Stroke]:
    return [np.array([[0.45, 0.15], [0.45, 0.85]], dtype=np.float64)]


@_latin_glyph("g")
def _small_g() -> list[Stroke]:
    # x-height 統一: ボウル上端 0.70 に揃える（旧 0.74）。
    t = np.linspace(0, 2 * np.pi, 24)
    body = np.stack(
        [0.48 + 0.24 * np.cos(t), 0.48 + 0.22 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    # 右から下降しベースライン下(y<0)で左へカールする descender → "9" と区別
    tail = np.array([[0.72, 0.48], [0.72, -0.05], [0.5, -0.2], [0.28, -0.1]], dtype=np.float64)
    return [body, tail]


@_latin_glyph("e")
def _small_e() -> list[Stroke]:
    # x-height 統一: 上端 0.70 に揃える（旧 0.78）。バーは円の縦中央に追従。
    t = np.linspace(0.2 * np.pi, 1.8 * np.pi, 24)
    body = np.stack(
        [0.52 + 0.28 * np.cos(t), 0.42 + 0.28 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    bar = np.array([[0.25, 0.42], [0.75, 0.42]], dtype=np.float64)
    return [body, bar]


@_latin_glyph("x")
def _small_x() -> list[Stroke]:
    # x-height 統一: 上端 0.70 に揃える（旧 0.80）。
    a = np.array([[0.25, 0.18], [0.75, 0.7]], dtype=np.float64)
    b = np.array([[0.75, 0.18], [0.25, 0.7]], dtype=np.float64)
    return [a, b]


@_latin_glyph("p")
def _small_p() -> list[Stroke]:
    # x-height 統一: ボウル/ステム上端を 0.70 に揃え、ステム下端を
    # ベースライン下(-0.2)へ伸ばす真のディセンダ体にする。旧字形は
    # ステム上端 0.85・下端 0.0 で「縦に巨大な大文字混じり」に見えた。
    stem = np.array([[0.25, -0.2], [0.25, 0.7]], dtype=np.float64)
    t = np.linspace(-np.pi / 2, np.pi / 2, 18)
    loop = np.stack(
        [0.25 + 0.42 * np.cos(t), 0.47 + 0.23 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    return [stem, loop]


@_latin_glyph("m")
def _small_m() -> list[Stroke]:
    # x-height 統一: 山の上端 0.70 に揃える（旧 0.75）。
    return [
        np.array(
            [[0.15, 0.2], [0.15, 0.7], [0.38, 0.32], [0.6, 0.7], [0.85, 0.2]],
            dtype=np.float64,
        )
    ]


@_latin_glyph("d")
def _small_d() -> list[Stroke]:
    stem = np.array([[0.75, 0.1], [0.75, 0.9]], dtype=np.float64)
    t = np.linspace(0, 2 * np.pi, 24)
    body = np.stack(
        [0.48 + 0.25 * np.cos(t), 0.45 + 0.25 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    return [stem, body]


@_latin_glyph("y")
def _small_y() -> list[Stroke]:
    # x-height 統一: V の上端 0.70 に揃え、右枝の尾をベースライン下(-0.2)へ
    # 伸ばす真のディセンダ体にする（旧字形は上端 0.75・尾が y=0 止まり）。
    return [
        np.array([[0.2, 0.7], [0.48, 0.35], [0.75, 0.7]], dtype=np.float64),
        np.array([[0.48, 0.35], [0.3, -0.2]], dtype=np.float64),
    ]


@_latin_glyph("b")
def _small_b() -> list[Stroke]:
    stem = np.array([[0.25, 0.0], [0.25, 0.95]], dtype=np.float64)
    t = np.linspace(0, 2 * np.pi, 24)
    body = np.stack(
        [0.48 + 0.25 * np.cos(t), 0.25 + 0.25 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    return [stem, body]


@_latin_glyph("f")
def _small_f() -> list[Stroke]:
    t = np.linspace(0, np.pi / 2, 12)
    hook = np.stack(
        [0.45 + 0.25 * np.sin(t), 0.7 + 0.2 * (1 - np.cos(t))],
        axis=1,
    ).astype(np.float64)
    stem = np.array([[0.45, 0.7], [0.45, 0.0]], dtype=np.float64)
    cross = np.array([[0.25, 0.5], [0.6, 0.5]], dtype=np.float64)
    return [hook, stem, cross]


@_latin_glyph("h")
def _small_h() -> list[Stroke]:
    stem = np.array([[0.25, 0.0], [0.25, 0.95]], dtype=np.float64)
    arch = np.array([[0.25, 0.5], [0.5, 0.7], [0.75, 0.5], [0.75, 0.0]], dtype=np.float64)
    return [stem, arch]


@_latin_glyph("j")
def _small_j() -> list[Stroke]:
    stem = np.array([[0.55, 0.7], [0.55, -0.05], [0.4, -0.15]], dtype=np.float64)
    return [stem, small_dot(0.55, 0.88)]


@_latin_glyph("k")
def _small_k() -> list[Stroke]:
    stem = np.array([[0.25, 0.0], [0.25, 0.95]], dtype=np.float64)
    upper = np.array([[0.25, 0.35], [0.7, 0.7]], dtype=np.float64)
    lower = np.array([[0.4, 0.45], [0.75, 0.0]], dtype=np.float64)
    return [stem, upper, lower]


@_latin_glyph("q")
def _small_q() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 24)
    body = np.stack(
        [0.45 + 0.25 * np.cos(t), 0.45 + 0.25 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    tail = np.array([[0.7, 0.45], [0.7, -0.05]], dtype=np.float64)
    return [body, tail]


@_latin_glyph("r")
def _small_r() -> list[Stroke]:
    stem = np.array([[0.3, 0.0], [0.3, 0.7]], dtype=np.float64)
    hook = np.array([[0.3, 0.55], [0.5, 0.7], [0.7, 0.55]], dtype=np.float64)
    return [stem, hook]


@_latin_glyph("u")
def _small_u() -> list[Stroke]:
    t = np.linspace(np.pi, 2 * np.pi, 16)
    cup = np.stack(
        [0.5 + 0.3 * np.cos(t), 0.3 + 0.3 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    right = np.array([[0.8, 0.3], [0.8, 0.0]], dtype=np.float64)
    top_left = np.array([[0.2, 0.7], [0.2, 0.3]], dtype=np.float64)
    return [top_left, cup, right]


@_latin_glyph("v")
def _small_v() -> list[Stroke]:
    return [np.array([[0.2, 0.7], [0.5, 0.0], [0.8, 0.7]], dtype=np.float64)]


@_latin_glyph("w")
def _small_w() -> list[Stroke]:
    return [
        np.array(
            [[0.1, 0.7], [0.3, 0.0], [0.5, 0.45], [0.7, 0.0], [0.9, 0.7]],
            dtype=np.float64,
        )
    ]


@_latin_glyph("z")
def _small_z() -> list[Stroke]:
    return [np.array([[0.2, 0.7], [0.8, 0.7], [0.2, 0.05], [0.8, 0.05]], dtype=np.float64)]


@_latin_glyph("A")
def _capital_a() -> list[Stroke]:
    left = np.array([[0.2, 0.0], [0.5, 0.95]], dtype=np.float64)
    right = np.array([[0.5, 0.95], [0.8, 0.0]], dtype=np.float64)
    cross = np.array([[0.3, 0.35], [0.7, 0.35]], dtype=np.float64)
    return [left, right, cross]


@_latin_glyph("B")
def _capital_b() -> list[Stroke]:
    stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
    t = np.linspace(-np.pi / 2, np.pi / 2, 14)
    upper = np.stack(
        [0.2 + 0.4 * np.cos(t), 0.7 + 0.25 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    lower = np.stack(
        [0.2 + 0.45 * np.cos(t), 0.225 + 0.25 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    return [stem, upper, lower]


@_latin_glyph("C")
def _capital_c() -> list[Stroke]:
    t = np.linspace(0.25 * np.pi, 1.75 * np.pi, 28)
    return [
        np.stack(
            [0.55 + 0.38 * np.cos(t), 0.5 + 0.45 * np.sin(t)],
            axis=1,
        ).astype(np.float64)
    ]


@_latin_glyph("D")
def _capital_d() -> list[Stroke]:
    stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
    t = np.linspace(-np.pi / 2, np.pi / 2, 20)
    arc = np.stack(
        [0.2 + 0.55 * np.cos(t), 0.475 + 0.475 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    return [stem, arc]


@_latin_glyph("E")
def _capital_e() -> list[Stroke]:
    stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
    top = np.array([[0.2, 0.95], [0.8, 0.95]], dtype=np.float64)
    mid = np.array([[0.2, 0.5], [0.7, 0.5]], dtype=np.float64)
    bot = np.array([[0.2, 0.0], [0.8, 0.0]], dtype=np.float64)
    return [stem, top, mid, bot]


@_latin_glyph("F")
def _capital_f() -> list[Stroke]:
    stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
    top = np.array([[0.2, 0.95], [0.8, 0.95]], dtype=np.float64)
    mid = np.array([[0.2, 0.5], [0.7, 0.5]], dtype=np.float64)
    return [stem, top, mid]


@_latin_glyph("G")
def _capital_g() -> list[Stroke]:
    t = np.linspace(0.25 * np.pi, 1.75 * np.pi, 28)
    arc = np.stack(
        [0.55 + 0.38 * np.cos(t), 0.5 + 0.45 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    inner = np.array([[0.55, 0.5], [0.85, 0.5], [0.85, 0.1]], dtype=np.float64)
    return [arc, inner]


@_latin_glyph("H")
def _capital_h() -> list[Stroke]:
    left = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
    right = np.array([[0.8, 0.0], [0.8, 0.95]], dtype=np.float64)
    cross = np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float64)
    return [left, right, cross]


@_latin_glyph("I")
def _capital_i() -> list[Stroke]:
    stem = np.array([[0.5, 0.0], [0.5, 0.95]], dtype=np.float64)
    top = np.array([[0.3, 0.95], [0.7, 0.95]], dtype=np.float64)
    bot = np.array([[0.3, 0.0], [0.7, 0.0]], dtype=np.float64)
    return [stem, top, bot]


@_latin_glyph("J")
def _capital_j() -> list[Stroke]:
    stem = np.array([[0.65, 0.95], [0.65, 0.2]], dtype=np.float64)
    t = np.linspace(0, np.pi, 14)
    curl = np.stack(
        [0.45 + 0.2 * np.cos(t), 0.2 - 0.15 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    return [stem, curl]


@_latin_glyph("K")
def _capital_k() -> list[Stroke]:
    stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
    upper = np.array([[0.2, 0.5], [0.8, 0.95]], dtype=np.float64)
    lower = np.array([[0.4, 0.6], [0.85, 0.0]], dtype=np.float64)
    return [stem, upper, lower]


@_latin_glyph("L")
def _capital_l() -> list[Stroke]:
    stem = np.array([[0.2, 0.95], [0.2, 0.0]], dtype=np.float64)
    bot = np.array([[0.2, 0.0], [0.8, 0.0]], dtype=np.float64)
    return [stem, bot]


@_latin_glyph("M")
def _capital_m() -> list[Stroke]:
    return [
        np.array(
            [[0.15, 0.0], [0.15, 0.95], [0.5, 0.3], [0.85, 0.95], [0.85, 0.0]],
            dtype=np.float64,
        )
    ]


@_latin_glyph("N")
def _capital_n() -> list[Stroke]:
    left = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
    diag = np.array([[0.2, 0.95], [0.8, 0.0]], dtype=np.float64)
    right = np.array([[0.8, 0.0], [0.8, 0.95]], dtype=np.float64)
    return [left, diag, right]


@_latin_glyph("O")
def _capital_o() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 32)
    return [
        np.stack(
            [0.5 + 0.35 * np.cos(t), 0.5 + 0.45 * np.sin(t)],
            axis=1,
        ).astype(np.float64)
    ]


@_latin_glyph("P")
def _capital_p() -> list[Stroke]:
    stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
    t = np.linspace(-np.pi / 2, np.pi / 2, 14)
    loop = np.stack(
        [0.2 + 0.45 * np.cos(t), 0.7 + 0.25 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    return [stem, loop]


@_latin_glyph("Q")
def _capital_q() -> list[Stroke]:
    t = np.linspace(0, 2 * np.pi, 32)
    body = np.stack(
        [0.5 + 0.35 * np.cos(t), 0.5 + 0.45 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    tail = np.array([[0.55, 0.2], [0.85, -0.05]], dtype=np.float64)
    return [body, tail]


@_latin_glyph("R")
def _capital_r() -> list[Stroke]:
    stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
    t = np.linspace(-np.pi / 2, np.pi / 2, 14)
    loop = np.stack(
        [0.2 + 0.45 * np.cos(t), 0.7 + 0.25 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    leg = np.array([[0.4, 0.45], [0.85, 0.0]], dtype=np.float64)
    return [stem, loop, leg]


@_latin_glyph("S")
def _capital_s() -> list[Stroke]:
    t = np.linspace(0, 1, 32)
    # 上が左・下が右に膨らむ正しい S 字（+sin だと左右反転 Ƨ になる）
    x = 0.5 - 0.32 * np.sin(2 * np.pi * t)
    y = 0.95 - 0.9 * t
    return [np.stack([x, y], axis=1).astype(np.float64)]


@_latin_glyph("T")
def _capital_t() -> list[Stroke]:
    top = np.array([[0.15, 0.95], [0.85, 0.95]], dtype=np.float64)
    stem = np.array([[0.5, 0.95], [0.5, 0.0]], dtype=np.float64)
    return [top, stem]


@_latin_glyph("U")
def _capital_u() -> list[Stroke]:
    left = np.array([[0.2, 0.95], [0.2, 0.3]], dtype=np.float64)
    t = np.linspace(np.pi, 2 * np.pi, 18)
    cup = np.stack(
        [0.5 + 0.3 * np.cos(t), 0.3 + 0.3 * np.sin(t)],
        axis=1,
    ).astype(np.float64)
    right = np.array([[0.8, 0.3], [0.8, 0.95]], dtype=np.float64)
    return [left, cup, right]


@_latin_glyph("V")
def _capital_v() -> list[Stroke]:
    return [np.array([[0.15, 0.95], [0.5, 0.0], [0.85, 0.95]], dtype=np.float64)]


@_latin_glyph("W")
def _capital_w() -> list[Stroke]:
    return [
        np.array(
            [[0.1, 0.95], [0.3, 0.0], [0.5, 0.6], [0.7, 0.0], [0.9, 0.95]],
            dtype=np.float64,
        )
    ]


@_latin_glyph("X")
def _capital_x() -> list[Stroke]:
    d1 = np.array([[0.2, 0.0], [0.8, 0.95]], dtype=np.float64)
    d2 = np.array([[0.8, 0.0], [0.2, 0.95]], dtype=np.float64)
    return [d1, d2]


@_latin_glyph("Y")
def _capital_y() -> list[Stroke]:
    v = np.array([[0.2, 0.95], [0.5, 0.5], [0.8, 0.95]], dtype=np.float64)
    stem = np.array([[0.5, 0.5], [0.5, 0.0]], dtype=np.float64)
    return [v, stem]


@_latin_glyph("Z")
def _capital_z() -> list[Stroke]:
    return [np.array([[0.2, 0.95], [0.8, 0.95], [0.2, 0.0], [0.8, 0.0]], dtype=np.float64)]
