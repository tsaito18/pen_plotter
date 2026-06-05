from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# LaTeX コマンド → Unicode 記号の変換マップ。
# 旧実装ではコマンド名がそのまま symbol.content に入りプレビューで
# "omega" のような文字列が描画されていたため、ここで Unicode に置換する。
# `\frac` `\sqrt` `\tag` は構造を持つため意図的に含めない（特殊処理を維持）。
_LATEX_SYMBOL_MAP: dict[str, str] = {
    # ギリシャ小文字
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "varepsilon": "ε",
    "varphi": "φ",
    # ギリシャ大文字
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    # 演算子・関係子
    "pm": "±",
    "approx": "≈",
    "simeq": "≃",
    "infty": "∞",
    "times": "×",
    "div": "÷",
    "neq": "≠",
    "leq": "≤",
    "geq": "≥",
    "cdot": "·",
    "ldots": "…",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "Rightarrow": "⇒",
    "partial": "∂",
    "nabla": "∇",
    "int": "∫",
    "sum": "∑",
    "prod": "∏",
    # スペース系（\quad は連続スペース 2 つ、\qquad は 4 つで近似）
    "quad": "  ",
    "qquad": "    ",
}

# `\,` `\;` `\:` のような 1 文字スペースコマンド。`_read_command` は
# isalpha のみ拾うので、これらは別経路で処理する必要がある。
_LATEX_SHORT_SPACES: set[str] = {",", ";", ":"}
_LATEX_OPERATORS: dict[str, str] = {
    "cos": "cos",
    "sin": "sin",
    "tan": "tan",
    "log": "log",
    "ln": "ln",
    "exp": "exp",
    "lim": "lim",
    "max": "max",
    "min": "min",
}
_LATEX_TEXT_COMMANDS: set[str] = {"mathrm", "text", "mathbf", "mathit"}
_LATEX_ACCENTS: set[str] = {
    "bar",
    "overline",
    "hat",
    "widehat",
    "tilde",
    "vec",
    "dot",
    "ddot",
}


@dataclass
class MathElement:
    type: str  # "text", "frac", "sup", "sub", "sqrt", "symbol", "group", "operator", "accent"
    content: str = ""
    numerator: Optional[MathElement] = None
    denominator: Optional[MathElement] = None
    children: list[MathElement] = field(default_factory=list)


@dataclass
class MathPlacement:
    text: str
    x: float
    y: float
    font_size: float
    role: Optional[str] = None
    line_segment: tuple[float, float, float, float] | None = None


@dataclass
class MathBox:
    """再帰レイアウトの結果: 配置とベースライン基準の高さ情報。

    ascent/descent はベースライン (layout の引数 y) を基準とした
    上方向・下方向の余白で、親側でネスト時の高さ確保に使う。
    """

    placements: list[MathPlacement]
    width: float
    ascent: float
    descent: float


class MathParser:
    """LaTeX数式の簡易パーサー"""

    @staticmethod
    def parse(source: str) -> list[MathElement]:
        parser = _ParserState(source)
        return parser.parse_elements()


class _ParserState:
    def __init__(self, source: str) -> None:
        self._src = source
        self._pos = 0

    @property
    def _remaining(self) -> str:
        return self._src[self._pos :]

    def _peek(self) -> str | None:
        if self._pos < len(self._src):
            return self._src[self._pos]
        return None

    def _advance(self) -> str:
        ch = self._src[self._pos]
        self._pos += 1
        return ch

    def _read_braced(self) -> str:
        """'{' ... '}' の中身を返す（ネスト対応）"""
        assert self._peek() == "{"
        self._advance()
        depth = 1
        start = self._pos
        while self._pos < len(self._src) and depth > 0:
            ch = self._src[self._pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            self._pos += 1
        return self._src[start : self._pos - 1]

    def parse_elements(self) -> list[MathElement]:
        elements: list[MathElement] = []
        text_buf: list[str] = []

        def flush_text() -> None:
            if text_buf:
                elements.append(MathElement(type="text", content="".join(text_buf)))
                text_buf.clear()

        while self._pos < len(self._src):
            ch = self._peek()

            if ch == "\\":
                flush_text()
                self._advance()
                # `\\` は数式中の強制改行（ブロック数式で多段組みに使用）
                if self._peek() == "\\":
                    self._advance()
                    elements.append(MathElement(type="linebreak"))
                    continue
                cmd = self._read_command()
                # `\,` `\;` `\:` は alpha でないため _read_command は空文字を返す。
                # 次の 1 文字を覗いてスペースに変換する。
                if not cmd:
                    next_ch = self._peek()
                    if next_ch in _LATEX_SHORT_SPACES:
                        self._advance()
                        elements.append(MathElement(type="symbol", content=" "))
                        continue
                    # 未対応のエスケープ（例: \%）はその文字をそのままテキスト扱い
                    if next_ch is not None:
                        text_buf.append(self._advance())
                    continue
                if cmd == "frac":
                    num_src = self._read_braced()
                    den_src = self._read_braced()
                    num_children = _ParserState(num_src).parse_elements()
                    den_children = _ParserState(den_src).parse_elements()
                    elements.append(
                        MathElement(
                            type="frac",
                            numerator=MathElement(type="group", children=num_children),
                            denominator=MathElement(type="group", children=den_children),
                        )
                    )
                elif cmd == "sqrt":
                    inner_src = self._read_braced()
                    inner_children = _ParserState(inner_src).parse_elements()
                    elements.append(MathElement(type="sqrt", children=inner_children))
                elif cmd == "tag":
                    tag_inner = self._read_braced()
                    elements.append(MathElement(type="tag", content=f"({tag_inner})"))
                elif cmd in ("left", "right"):
                    delimiter = self._peek()
                    if delimiter is not None:
                        self._advance()
                        if delimiter != ".":
                            elements.append(MathElement(type="text", content=delimiter))
                elif cmd in _LATEX_TEXT_COMMANDS:
                    if self._peek() == "{":
                        inner_src = self._read_braced()
                    else:
                        logger.warning("LaTeX command \\%s missing braced argument", cmd)
                        continue
                    inner_children = _ParserState(inner_src).parse_elements()
                    elements.append(MathElement(type="group", children=inner_children))
                elif cmd in _LATEX_ACCENTS:
                    inner_src = self._read_script_argument()
                    inner_children = _ParserState(inner_src).parse_elements()
                    elements.append(
                        MathElement(
                            type="accent",
                            content=cmd,
                            children=inner_children,
                        )
                    )
                elif cmd in _LATEX_OPERATORS:
                    elements.append(
                        MathElement(type="operator", content=_LATEX_OPERATORS[cmd])
                    )
                elif cmd in _LATEX_SYMBOL_MAP:
                    elements.append(
                        MathElement(type="symbol", content=_LATEX_SYMBOL_MAP[cmd])
                    )
                else:
                    logger.warning("Unsupported LaTeX command ignored: \\%s", cmd)

            elif ch == "^":
                flush_text()
                self._advance()
                sup_src = self._read_script_argument()
                sup_children = _ParserState(sup_src).parse_elements()
                elements.append(MathElement(type="sup", children=sup_children))

            elif ch == "_":
                flush_text()
                self._advance()
                sub_src = self._read_script_argument()
                sub_children = _ParserState(sub_src).parse_elements()
                elements.append(MathElement(type="sub", children=sub_children))

            elif ch == "{" or ch == "}":
                self._advance()

            elif ch == "\n":
                self._advance()

            else:
                text_buf.append(self._advance())

        flush_text()
        return elements

    def _read_script_argument(self) -> str:
        """^/_ の後ろの引数を読む。{ なら波括弧の中、それ以外は1文字。"""
        if self._peek() == "{":
            return self._read_braced()
        if self._pos < len(self._src):
            return self._advance()
        return ""

    def _read_command(self) -> str:
        start = self._pos
        while self._pos < len(self._src) and self._src[self._pos].isalpha():
            self._pos += 1
        return self._src[start : self._pos]


# 文字幅推定の係数（font_size に対する比率）
_CHAR_WIDTH_RATIO = 0.6


class MathLayoutEngine:
    """MathElement リストを座標付き MathPlacement と高さ情報を含む MathBox に変換する。

    ネスト数式 (例: \\frac{\\frac{1}{2}}{x}) を扱うため、frac/sup/sub/sqrt の
    子要素を再帰的にレイアウトしてから親 box の幅・高さを組み立てる。
    """

    @staticmethod
    def layout(
        elements: list[MathElement],
        x: float,
        y: float,
        font_size: float,
    ) -> MathBox:
        """要素リストを横一列にレイアウトし MathBox を返す。

        Args:
            elements: 横並びに配置する要素列。
            x: 開始 X 座標。
            y: ベースライン Y 座標。
            font_size: 基準フォントサイズ。
        """
        placements: list[MathPlacement] = []
        cursor_x = x
        max_ascent = 0.0
        max_descent = 0.0

        # text/symbol の高さは目視バランス用の概算（font_size 基準）
        default_ascent = font_size * 0.75
        default_descent = font_size * 0.25

        for elem in elements:
            child_box = MathLayoutEngine._layout_element(elem, cursor_x, y, font_size)
            placements.extend(child_box.placements)
            cursor_x += child_box.width
            if child_box.ascent > max_ascent:
                max_ascent = child_box.ascent
            if child_box.descent > max_descent:
                max_descent = child_box.descent

        if not elements:
            max_ascent = default_ascent
            max_descent = default_descent

        return MathBox(
            placements=placements,
            width=cursor_x - x,
            ascent=max_ascent,
            descent=max_descent,
        )

    @staticmethod
    def _layout_element(
        elem: MathElement, x: float, y: float, font_size: float
    ) -> MathBox:
        """単一要素を box として返す。横方向の起点は x、ベースラインは y。"""
        if elem.type == "text" or elem.type == "symbol" or elem.type == "tag":
            text = elem.content
            width = len(text) * font_size * _CHAR_WIDTH_RATIO
            return MathBox(
                placements=[MathPlacement(text=text, x=x, y=y, font_size=font_size)],
                width=width,
                ascent=font_size * 0.75,
                descent=font_size * 0.25,
            )

        if elem.type == "operator":
            text = elem.content
            width = len(text) * font_size * _CHAR_WIDTH_RATIO
            return MathBox(
                placements=[
                    MathPlacement(
                        text=text,
                        x=x,
                        y=y,
                        font_size=font_size,
                        role="operator",
                    )
                ],
                width=width,
                ascent=font_size * 0.75,
                descent=font_size * 0.25,
            )

        if elem.type == "group":
            return MathLayoutEngine.layout(elem.children, x, y, font_size)

        if elem.type == "frac":
            return MathLayoutEngine._layout_frac(elem, x, y, font_size)

        if elem.type == "sup":
            return MathLayoutEngine._layout_script(elem, x, y, font_size, is_sup=True)

        if elem.type == "sub":
            return MathLayoutEngine._layout_script(elem, x, y, font_size, is_sup=False)

        if elem.type == "sqrt":
            return MathLayoutEngine._layout_sqrt(elem, x, y, font_size)

        if elem.type == "accent":
            return MathLayoutEngine._layout_accent(elem, x, y, font_size)

        return MathBox(placements=[], width=0.0, ascent=0.0, descent=0.0)

    @staticmethod
    def _layout_frac(
        elem: MathElement, x: float, y: float, font_size: float
    ) -> MathBox:
        frac_font = font_size * 0.7
        gap = font_size * 0.15
        # 分子は y より上、分母は下に配置するため、各々の box は仮ベースラインで計算してから平行移動する
        num_children = elem.numerator.children if elem.numerator else []
        den_children = elem.denominator.children if elem.denominator else []

        num_box = MathLayoutEngine.layout(num_children, x=0.0, y=0.0, font_size=frac_font)
        den_box = MathLayoutEngine.layout(den_children, x=0.0, y=0.0, font_size=frac_font)

        frac_width = max(num_box.width, den_box.width)
        # 子の placement に role を付与（外側 frac であることを明示）
        num_role = "numerator"
        den_role = "denominator"

        num_offset_x = x + (frac_width - num_box.width) / 2
        den_offset_x = x + (frac_width - den_box.width) / 2
        # 分子・分母の中心はベースライン y から ±(font_size*0.5 + gap) 離す
        num_offset_y = y + font_size * 0.5 + gap
        den_offset_y = y - font_size * 0.5 - gap

        placements: list[MathPlacement] = []
        for i, p in enumerate(num_box.placements):
            placements.append(
                MathPlacement(
                    text=p.text,
                    x=p.x + num_offset_x,
                    y=p.y + num_offset_y,
                    font_size=p.font_size,
                    role=num_role if i == 0 else p.role,
                    line_segment=p.line_segment,
                )
            )
        for i, p in enumerate(den_box.placements):
            placements.append(
                MathPlacement(
                    text=p.text,
                    x=p.x + den_offset_x,
                    y=p.y + den_offset_y,
                    font_size=p.font_size,
                    role=den_role if i == 0 else p.role,
                    line_segment=p.line_segment,
                )
            )

        # 分数の横線はベースライン y を分子・分母の中間として 1 本引く
        placements.append(
            MathPlacement(
                text="",
                x=x,
                y=y,
                font_size=font_size,
                role="frac_bar",
                line_segment=(x, y, x + frac_width, y),
            )
        )

        # ascent/descent は分子・分母の box 高さをオフセット込みで合成
        ascent = num_offset_y + num_box.ascent - y
        descent = y - (den_offset_y - den_box.descent)

        return MathBox(
            placements=placements,
            width=frac_width + font_size * 0.2,
            ascent=ascent,
            descent=descent,
        )

    @staticmethod
    def _layout_script(
        elem: MathElement, x: float, y: float, font_size: float, *, is_sup: bool
    ) -> MathBox:
        script_font = font_size * 0.65
        if is_sup:
            offset_y = font_size * 0.4
            role = "superscript"
        else:
            offset_y = -font_size * 0.3
            role = "subscript"

        child_box = MathLayoutEngine.layout(
            elem.children, x=0.0, y=0.0, font_size=script_font
        )

        placements: list[MathPlacement] = []
        for i, p in enumerate(child_box.placements):
            placements.append(
                MathPlacement(
                    text=p.text,
                    x=p.x + x,
                    y=p.y + y + offset_y,
                    font_size=p.font_size,
                    role=role if i == 0 else p.role,
                    line_segment=p.line_segment,
                )
            )

        if is_sup:
            ascent = offset_y + child_box.ascent
            descent = max(0.0, child_box.descent - offset_y)
        else:
            ascent = max(0.0, child_box.ascent + offset_y)
            descent = -offset_y + child_box.descent

        return MathBox(
            placements=placements,
            width=child_box.width,
            ascent=ascent,
            descent=descent,
        )

    @staticmethod
    def _layout_sqrt(
        elem: MathElement, x: float, y: float, font_size: float
    ) -> MathBox:
        prefix = font_size * 0.4
        suffix = font_size * 0.2
        child_box = MathLayoutEngine.layout(
            elem.children, x=x + prefix, y=y, font_size=font_size
        )
        # 既存テスト互換のため、子の先頭 placement に sqrt_body role を付与する
        placements: list[MathPlacement] = []
        for i, p in enumerate(child_box.placements):
            placements.append(
                MathPlacement(
                    text=p.text,
                    x=p.x,
                    y=p.y,
                    font_size=p.font_size,
                    role="sqrt_body" if i == 0 else p.role,
                    line_segment=p.line_segment,
                )
            )

        # 中身が空の sqrt でも視認可能な大きさを保つためのフォールバック
        ascent = child_box.ascent if child_box.ascent > 1e-6 else font_size * 0.5
        descent = child_box.descent if child_box.descent > 1e-6 else font_size * 0.5
        inner_right = x + prefix + child_box.width
        roof_y = y + ascent + font_size * 0.05
        check_low_y = y - descent * 0.7
        check_mid_x = x + prefix * 0.5
        check_start = (x, y + ascent * 0.4)
        check_bottom = (check_mid_x, check_low_y)
        roof_start = (x + prefix, roof_y)
        roof_end = (inner_right + suffix * 0.5, roof_y)

        for seg in (
            (*check_start, *check_bottom),
            (*check_bottom, *roof_start),
            (*roof_start, *roof_end),
        ):
            placements.append(
                MathPlacement(
                    text="",
                    x=seg[0],
                    y=seg[1],
                    font_size=font_size,
                    role="sqrt_radical",
                    line_segment=seg,
                )
            )

        return MathBox(
            placements=placements,
            width=prefix + child_box.width + suffix,
            ascent=child_box.ascent,
            descent=child_box.descent,
        )

    @staticmethod
    def _layout_accent(
        elem: MathElement, x: float, y: float, font_size: float
    ) -> MathBox:
        child_box = MathLayoutEngine.layout(elem.children, x=x, y=y, font_size=font_size)
        width = max(child_box.width, font_size * 0.5)
        gap = font_size * 0.12
        accent_y = y + child_box.ascent + gap
        left = x
        right = x + width
        mid = (left + right) / 2
        role = f"accent_{elem.content}"

        placements = list(child_box.placements)

        def add_segment(seg: tuple[float, float, float, float]) -> None:
            placements.append(
                MathPlacement(
                    text="",
                    x=seg[0],
                    y=seg[1],
                    font_size=font_size,
                    role=role,
                    line_segment=seg,
                )
            )

        if elem.content in ("bar", "overline"):
            add_segment((left, accent_y, right, accent_y))
        elif elem.content in ("hat", "widehat"):
            add_segment((left, accent_y - gap, mid, accent_y + gap))
            add_segment((mid, accent_y + gap, right, accent_y - gap))
        elif elem.content == "vec":
            add_segment((left, accent_y, right, accent_y))
            add_segment((right, accent_y, right - font_size * 0.18, accent_y + gap))
            add_segment((right, accent_y, right - font_size * 0.18, accent_y - gap))
        elif elem.content == "ddot":
            dot_w = font_size * 0.08
            add_segment((mid - font_size * 0.12, accent_y, mid - font_size * 0.12 + dot_w, accent_y))
            add_segment((mid + font_size * 0.12, accent_y, mid + font_size * 0.12 + dot_w, accent_y))
        elif elem.content == "dot":
            dot_w = font_size * 0.08
            add_segment((mid, accent_y, mid + dot_w, accent_y))
        else:
            add_segment((left, accent_y, right, accent_y))

        ascent = max(child_box.ascent, accent_y + gap - y)
        return MathBox(
            placements=placements,
            width=width,
            ascent=ascent,
            descent=child_box.descent,
        )

    @staticmethod
    def total_width(placements: list[MathPlacement]) -> float:
        if not placements:
            return 0.0
        rightmost = max(
            p.x + len(p.text) * p.font_size * _CHAR_WIDTH_RATIO for p in placements
        )
        leftmost = min(p.x for p in placements)
        return rightmost - leftmost
