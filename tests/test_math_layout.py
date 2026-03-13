import pytest
from src.layout.math_layout import MathElement, MathParser, MathLayoutEngine


class TestMathParser:
    def test_plain_number(self):
        elements = MathParser.parse("123")
        assert len(elements) == 1
        assert elements[0].type == "text"
        assert elements[0].content == "123"

    def test_fraction(self):
        elements = MathParser.parse(r"\frac{1}{2}")
        assert len(elements) == 1
        assert elements[0].type == "frac"
        assert elements[0].numerator.content == "1"
        assert elements[0].denominator.content == "2"

    def test_superscript(self):
        elements = MathParser.parse("x^{2}")
        assert len(elements) == 2
        assert elements[0].content == "x"
        assert elements[1].type == "sup"
        assert elements[1].content == "2"

    def test_subscript(self):
        elements = MathParser.parse("x_{i}")
        assert len(elements) == 2
        assert elements[0].content == "x"
        assert elements[1].type == "sub"
        assert elements[1].content == "i"

    def test_sqrt(self):
        elements = MathParser.parse(r"\sqrt{x}")
        assert len(elements) == 1
        assert elements[0].type == "sqrt"
        assert elements[0].content == "x"

    def test_combined(self):
        elements = MathParser.parse(r"\frac{x^{2}}{3}")
        assert elements[0].type == "frac"


class TestMathLayoutEngine:
    def test_text_placement(self):
        elements = MathParser.parse("abc")
        placements = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        assert len(placements) > 0
        assert placements[0].x == 0
        assert placements[0].y == 0

    def test_fraction_vertical_layout(self):
        elements = MathParser.parse(r"\frac{1}{2}")
        placements = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        numerator_p = [p for p in placements if p.role == "numerator"]
        denominator_p = [p for p in placements if p.role == "denominator"]
        assert len(numerator_p) > 0
        assert len(denominator_p) > 0
        assert numerator_p[0].y > denominator_p[0].y

    def test_superscript_offset(self):
        elements = MathParser.parse("x^{2}")
        placements = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        base = [p for p in placements if p.text == "x"][0]
        sup = [p for p in placements if p.text == "2"][0]
        assert sup.y > base.y
        assert sup.font_size < base.font_size

    def test_layout_returns_total_width(self):
        elements = MathParser.parse("abc")
        placements = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        total_width = MathLayoutEngine.total_width(placements)
        assert total_width > 0
