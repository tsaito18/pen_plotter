"""matplotlib の数式組版（mathtext）から、グリフと罫線の配置を取り出す。

構造式（分数・根号・添字・上付き）を手書きで描くときの骨組み。matplotlib に正しい配置を
させ、その位置へ本文と同じ手書き字形を貼る（貼るのは :mod:`src.render.math_handwriting`）。

座標は pt（dpi=72 で pt=px）、ベースライン原点・上向き正。手書き字形を本文の大文字と
同じ大きさにするため、縮尺は「基準サイズの大文字 M のインク高」を本文の大文字高さへ
写す一定値にする（式の高さで割ると、分数・上付きのある式ほど字が縮む）。

ブロック数式用に、トップレベルの分数線の検出（分子を上の行・分母を下の行へ置く罫線揃え）、
``\\frac`` → ``\\dfrac`` の昇格（分数の字が小さくならないように）、本文幅を超える式の
関係演算子・加減での分割もここに置く。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# 配置を取るときの基準サイズ(pt)
_LAYOUT_PT = 28.0
# 手書きで描かず構造線として扱うグリフのフォント（√・大括弧・∑・∫ 等は STIXSize* で出る）
_BODY_FONT_FAMILY = "DejaVu Sans"

MATH_INLINE_CAP_RATIO = 0.8
"""インライン数式の大文字高さ / 本文フォントサイズ（本文の英数字と同じ 0.8）。"""
MATH_BLOCK_CAP_RATIO = 0.85
"""ブロック数式の大文字高さ / 本文フォントサイズ（インラインより一回り大きく）。"""

# 記号フォント cmsy10 の字番号 → 本来の字（プライム f' は cmsy10 の 0x30 で出る）
_CMSY_CHARS = {0x30: "′"}

# 罫線揃えの分数線とみなす帯（最も幅広い分数線の中心からの距離 pt）
_FRACTION_AXIS_TOL_PT = 4.0
# この大型記号に囲まれた分数は罫線揃えしない（括弧が行をまたいで崩れる）
_ENCLOSING_LARGE_CHARS = frozenset("()∑∫")


@dataclass(frozen=True)
class MathGlyph:
    """mathtext が配置した 1 グリフ。"""

    char: str
    x: float  # 描画原点の x（pt）
    baseline_y: float  # ベースラインの y（pt）
    fontsize: float  # 実サイズ（添字・分数では縮む, pt）
    is_large: bool  # 本文に字形の無い構造記号（√・大括弧・∑・∫ 等）
    font_file: str = ""  # 大型記号のフォント（伸ばした √ や大きい ∑ はこのフォントで測る）


@dataclass(frozen=True)
class MathRect:
    """分数線・根号の屋根（pt、``y`` は下端）。"""

    x: float
    y: float
    width: float
    height: float

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class MathLayout:
    """mathtext の組版結果（pt）。"""

    width: float
    height: float  # ベースラインから上
    depth: float  # ベースラインから下
    glyphs: tuple[MathGlyph, ...]
    rects: tuple[MathRect, ...]


def _escape_percent(src: str) -> str:
    return re.sub(r"(?<!\\)%", r"\\%", src)


@lru_cache(maxsize=512)
def extract_math_layout(math_src: str) -> MathLayout | None:
    """LaTeX 数式（``$`` なし、``\\tag`` 除去済み）を組版し、配置を返す。失敗時は None。

    素の ``( )`` は ``\\left( \\right)`` に昇格して中身の高さまで伸ばす（括弧の対応が
    崩れて解析できない式は昇格しない）。
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser

    safe = _escape_percent(math_src)
    sized = re.sub(r"(?<!\\left)\(", r"\\left(", safe)
    sized = re.sub(r"(?<!\\right)\)", r"\\right)", sized)
    parser = MathTextParser("path")
    prop = FontProperties(size=_LAYOUT_PT)
    for candidate in (sized, safe):
        try:
            vp = parser.parse(f"${candidate}$", dpi=72, prop=prop)
            break
        except Exception:  # noqa: BLE001 — 解析できない式は次の候補・呼び出し側の代替へ
            continue
    else:
        logger.warning("math layout failed: %r", math_src)
        return None

    glyphs = tuple(
        MathGlyph(
            char=_CMSY_CHARS.get(num, chr(num)) if font.family_name == "cmsy10" else chr(num),
            x=float(ox),
            baseline_y=float(oy),
            fontsize=float(size),
            is_large=font.family_name != _BODY_FONT_FAMILY,
            font_file=font.fname if font.family_name != _BODY_FONT_FAMILY else "",
        )
        for font, size, num, ox, oy in vp.glyphs
    )
    rects = tuple(MathRect(float(x), float(y), float(w), float(h)) for x, y, w, h in vp.rects)
    return MathLayout(float(vp.width), float(vp.height), float(vp.depth), glyphs, rects)


# 数式記法として解釈される字（1 字だけ測るときにエスケープする）
_MATH_ESCAPES = {c: "\\" + c for c in "{}%#&_$"}


def ink_bbox(glyph: MathGlyph) -> tuple[float, float, float, float] | None:
    """配置されたグリフのインク範囲 ``(x, y, w, h)``（描画原点基準の pt）。"""
    return glyph_ink_bbox(glyph.char, glyph.fontsize, glyph.font_file)


@lru_cache(maxsize=1024)
def glyph_ink_bbox(
    char: str, fontsize: float, font_file: str = ""
) -> tuple[float, float, float, float] | None:
    """1 グリフを ``fontsize`` pt で描いたインク範囲 ``(x, y, w, h)``（描画原点基準）。

    mathtext の配置はグリフの描画原点なので、字形を貼る矩形はここで求める。``font_file`` を
    渡すとそのフォントの字（伸ばした √・大きい ∑ 等）で測る。墨が無ければ None。
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    try:
        if font_file:
            prop = FontProperties(fname=font_file)
            v = TextPath((0, 0), char, size=fontsize, prop=prop).vertices
        else:
            v = TextPath((0, 0), f"${_MATH_ESCAPES.get(char, char)}$", size=fontsize).vertices
    except Exception:  # noqa: BLE001 — 描けない字は貼らない
        return None
    if len(v) == 0:
        return None
    x0, y0 = v.min(axis=0)
    x1, y1 = v.max(axis=0)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


@lru_cache(maxsize=1)
def ref_cap_height_pt() -> float:
    """基準サイズの大文字 M のインク高(pt)。pt→mm の縮尺の基準。"""
    ink = glyph_ink_bbox("M", _LAYOUT_PT)
    return ink[3] if ink is not None else 0.7 * _LAYOUT_PT


def math_scale(font_size: float, cap_ratio: float) -> float:
    """pt → mm の縮尺（基準の大文字を ``font_size * cap_ratio`` mm にする）。"""
    return font_size * cap_ratio / ref_cap_height_pt()


def handwrite_draw_width_mm(
    math_src: str, font_size: float, cap_ratio: float = MATH_INLINE_CAP_RATIO
) -> float | None:
    """手書きで描いたときの式の幅(mm)。組版の予約幅と描画幅の単一ソース。組版できなければ None。"""
    layout = extract_math_layout(math_src)
    if layout is None or layout.width <= 0:
        return None
    return layout.width * math_scale(font_size, cap_ratio)


def sqrt_roofs(layout: MathLayout) -> dict[int, int]:
    """根号グリフの番号 → その屋根（横棒）の rect 番号。

    屋根は「√ のインク中央より右から始まり、√ の下端より上にある、最も左の rect」。
    √ が分母にあるとき外側の分数線は √ より左から始まるので、屋根と取り違えない。
    """
    roofs: dict[int, int] = {}
    for gi, g in enumerate(layout.glyphs):
        if g.char != "√":
            continue
        ink = ink_bbox(g)
        if ink is None:
            continue
        gx, gy, gw, gh = ink
        left, bottom = g.x + gx, g.baseline_y + gy
        best: int | None = None
        for ri, r in enumerate(layout.rects):
            if ri in roofs.values():
                continue
            in_x = left + gw * 0.5 <= r.x <= left + gw * 3.5
            above = r.center_y >= bottom + gh * 0.3
            if in_x and above and (best is None or r.x < layout.rects[best].x):
                best = ri
        if best is not None:
            roofs[gi] = best
    return roofs


def detect_top_level_fraction_bar(layout: MathLayout) -> float | None:
    """罫線に乗せるトップレベルの分数線の中心 y(pt)。対象外なら None。

    分子を上の行・分数線を罫線・分母を下の行に書けるのは、主分数が括弧・∑・∫ に
    囲まれていない式。最も幅広い分数線と同じ高さ（数式の軸）に並ぶ分数線の平均を返す
    （横並びの分数も同じ罫線に乗る）。入れ子の小さい分数は軸から外れるので含まれない。
    """
    if any(g.is_large and g.char in _ENCLOSING_LARGE_CHARS for g in layout.glyphs):
        return None
    roofs = set(sqrt_roofs(layout).values())
    bars = [r for ri, r in enumerate(layout.rects) if ri not in roofs]
    if not bars:
        return None
    axis = max(bars, key=lambda r: r.width).center_y
    near = [r.center_y for r in bars if abs(r.center_y - axis) <= _FRACTION_AXIS_TOL_PT]
    return sum(near) / len(near)


def _scan_commands(src: str):
    """``(位置, 文字 or コマンド名)`` を順に返す（``\\frac`` などは 1 トークン）。"""
    i, n = 0, len(src)
    while i < n:
        if src[i] == "\\":
            j = i + 1
            while j < n and src[j].isalpha():
                j += 1
            if j == i + 1 and j < n:
                j += 1  # \, \; \{ などの 1 文字コマンド
            yield i, src[i:j]
            i = j
        else:
            yield i, src[i]
            i += 1


def promote_top_level_frac_to_dfrac(src: str) -> str:
    """どの ``{...}`` の中にも無い（√ の中は可）``\\frac`` を ``\\dfrac`` にする。

    mathtext は ``\\frac`` の分子・分母を小さく描くため、ブロック数式の主分数は ``\\dfrac``
    （本文と同じ大きさ）にする。上付き・添字・分子分母の中の分数は小さいままにする。
    """
    out: list[str] = []
    braces: list[bool] = []  # 開いている { が √ の引数か
    sqrt_state = ""  # "await": \sqrt の直後 / "index": \sqrt[n] の n の中
    for _, tok in _scan_commands(src):
        if sqrt_state == "index":
            sqrt_state = "await" if tok == "]" else "index"
        elif tok == "{":
            braces.append(sqrt_state == "await")
            sqrt_state = ""
        elif tok == "}":
            if braces:
                braces.pop()
        elif tok == "\\sqrt":
            sqrt_state = "await"
        elif sqrt_state == "await" and tok == "[":
            sqrt_state = "index"
        elif tok == "\\frac" and all(braces) and len(braces) <= 1:
            tok = "\\dfrac"
        elif not tok.isspace():
            sqrt_state = ""
        out.append(tok)
    return "".join(out)


def space_adjacent_fractions(src: str) -> str:
    """隣り合う分数（``\\dfrac{l}{d}\\dfrac{v^2}{2g}``）の間を空ける（1 つの分数に見えるため）。"""
    return re.sub(r"\}\s*\\(d?frac)", r"}\\;\\\1", src)


# 分割に使う演算子（関係演算子が第一候補、足りなければ加減）
_RELATION_COMMANDS = frozenset(
    ["\\leq", "\\geq", "\\le", "\\ge", "\\neq", "\\ne", "\\approx", "\\sim", "\\simeq", "\\equiv"]
)
_RELATION_CHARS = frozenset("=<>")
_ADDITIVE_CHARS = frozenset("+-")


def _split_at_top_level(src: str, commands: frozenset[str], chars: frozenset[str]) -> list[str]:
    """括弧・``{}``・``\\left \\right`` の外にある演算子の直前で切る（演算子は次の片へ）。"""
    pieces: list[str] = []
    depth = 0
    start = 0
    for i, tok in _scan_commands(src):
        if tok in ("{", "\\left"):
            depth += 1
        elif tok in ("}", "\\right"):
            depth -= 1
        elif depth == 0 and (tok in commands or tok in chars) and src[start:i].strip():
            pieces.append(src[start:i])
            start = i
    pieces.append(src[start:])
    return [p for p in pieces if p.strip()]


def split_math_for_width(src: str, font_size: float, max_width_mm: float) -> list[str]:
    """本文幅を超えるブロック数式を、関係演算子（次に加減）の前で複数行に分ける。

    縮小して収めると本文より小さくなって読めないため、意味の切れ目で改行する。
    分けられなければ ``[src]``。
    """

    def width(s: str) -> float:
        return handwrite_draw_width_mm(s, font_size, MATH_BLOCK_CAP_RATIO) or 0.0

    if width(src) <= max_width_mm:
        return [src]
    lines: list[str] = []
    for piece in _split_at_top_level(src, _RELATION_COMMANDS, _RELATION_CHARS):
        if width(piece) <= max_width_mm:
            lines.append(piece)
            continue
        current = ""
        for term in _split_at_top_level(piece, frozenset(), _ADDITIVE_CHARS):
            if current and width(current + term) > max_width_mm:
                lines.append(current)
                current = term
            else:
                current += term
        lines.append(current)
    return lines if len(lines) > 1 else [src]
