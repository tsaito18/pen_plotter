from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MathElement:
    type: str  # "text", "frac", "sup", "sub", "sqrt", "symbol"
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
                cmd = self._read_command()
                if cmd == "frac":
                    num_src = self._read_braced()
                    den_src = self._read_braced()
                    num_elements = _ParserState(num_src).parse_elements()
                    den_elements = _ParserState(den_src).parse_elements()
                    num_el = num_elements[0] if len(num_elements) == 1 else MathElement(type="text", content=num_src, children=num_elements)
                    den_el = den_elements[0] if len(den_elements) == 1 else MathElement(type="text", content=den_src, children=den_elements)
                    elements.append(MathElement(type="frac", numerator=num_el, denominator=den_el))
                elif cmd == "sqrt":
                    inner = self._read_braced()
                    elements.append(MathElement(type="sqrt", content=inner))
                else:
                    elements.append(MathElement(type="symbol", content=cmd))

            elif ch == "^":
                flush_text()
                self._advance()
                sup_content = self._read_braced()
                elements.append(MathElement(type="sup", content=sup_content))

            elif ch == "_":
                flush_text()
                self._advance()
                sub_content = self._read_braced()
                elements.append(MathElement(type="sub", content=sub_content))

            elif ch == "{" or ch == "}":
                self._advance()

            else:
                text_buf.append(self._advance())

        flush_text()
        return elements

    def _read_command(self) -> str:
        start = self._pos
        while self._pos < len(self._src) and self._src[self._pos].isalpha():
            self._pos += 1
        return self._src[start : self._pos]


# 文字幅推定の係数（font_size に対する比率）
_CHAR_WIDTH_RATIO = 0.6


class MathLayoutEngine:
    """MathElement リストを座標付き MathPlacement リストに変換する"""

    @staticmethod
    def layout(
        elements: list[MathElement],
        x: float,
        y: float,
        font_size: float,
    ) -> list[MathPlacement]:
        placements: list[MathPlacement] = []
        cursor_x = x

        for elem in elements:
            if elem.type == "text":
                placements.append(
                    MathPlacement(text=elem.content, x=cursor_x, y=y, font_size=font_size)
                )
                cursor_x += len(elem.content) * font_size * _CHAR_WIDTH_RATIO

            elif elem.type == "frac":
                num_text = elem.numerator.content if elem.numerator else ""
                den_text = elem.denominator.content if elem.denominator else ""
                frac_font = font_size * 0.7
                num_width = len(num_text) * frac_font * _CHAR_WIDTH_RATIO
                den_width = len(den_text) * frac_font * _CHAR_WIDTH_RATIO
                frac_width = max(num_width, den_width)

                gap = font_size * 0.15
                num_x = cursor_x + (frac_width - num_width) / 2
                den_x = cursor_x + (frac_width - den_width) / 2

                placements.append(
                    MathPlacement(
                        text=num_text,
                        x=num_x,
                        y=y + font_size * 0.5 + gap,
                        font_size=frac_font,
                        role="numerator",
                    )
                )
                placements.append(
                    MathPlacement(
                        text=den_text,
                        x=den_x,
                        y=y - font_size * 0.5 - gap,
                        font_size=frac_font,
                        role="denominator",
                    )
                )
                cursor_x += frac_width + font_size * 0.2

            elif elem.type == "sup":
                placements.append(
                    MathPlacement(
                        text=elem.content,
                        x=cursor_x,
                        y=y + font_size * 0.4,
                        font_size=font_size * 0.65,
                        role="superscript",
                    )
                )
                cursor_x += len(elem.content) * font_size * 0.65 * _CHAR_WIDTH_RATIO

            elif elem.type == "sub":
                placements.append(
                    MathPlacement(
                        text=elem.content,
                        x=cursor_x,
                        y=y - font_size * 0.3,
                        font_size=font_size * 0.65,
                        role="subscript",
                    )
                )
                cursor_x += len(elem.content) * font_size * 0.65 * _CHAR_WIDTH_RATIO

            elif elem.type == "sqrt":
                inner_width = len(elem.content) * font_size * _CHAR_WIDTH_RATIO
                placements.append(
                    MathPlacement(
                        text=elem.content,
                        x=cursor_x + font_size * 0.4,
                        y=y,
                        font_size=font_size,
                        role="sqrt_body",
                    )
                )
                cursor_x += inner_width + font_size * 0.6

            elif elem.type == "symbol":
                placements.append(
                    MathPlacement(text=elem.content, x=cursor_x, y=y, font_size=font_size)
                )
                cursor_x += len(elem.content) * font_size * _CHAR_WIDTH_RATIO

        return placements

    @staticmethod
    def total_width(placements: list[MathPlacement]) -> float:
        if not placements:
            return 0.0
        rightmost = max(
            p.x + len(p.text) * p.font_size * _CHAR_WIDTH_RATIO for p in placements
        )
        leftmost = min(p.x for p in placements)
        return rightmost - leftmost
