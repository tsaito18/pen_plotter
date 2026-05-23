from src.layout.math_layout import MathParser, MathLayoutEngine


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
        # numerator/denominator は group ノードで、children に元の要素を保持する
        assert elements[0].numerator.type == "group"
        assert elements[0].denominator.type == "group"
        assert elements[0].numerator.children[0].content == "1"
        assert elements[0].denominator.children[0].content == "2"

    def test_superscript(self):
        elements = MathParser.parse("x^{2}")
        assert len(elements) == 2
        assert elements[0].content == "x"
        assert elements[1].type == "sup"
        # sup の中身は children として保持される
        assert elements[1].children[0].content == "2"

    def test_subscript(self):
        elements = MathParser.parse("x_{i}")
        assert len(elements) == 2
        assert elements[0].content == "x"
        assert elements[1].type == "sub"
        assert elements[1].children[0].content == "i"

    def test_sqrt(self):
        elements = MathParser.parse(r"\sqrt{x}")
        assert len(elements) == 1
        assert elements[0].type == "sqrt"
        assert elements[0].children[0].content == "x"

    def test_combined(self):
        elements = MathParser.parse(r"\frac{x^{2}}{3}")
        assert elements[0].type == "frac"


class TestMathLayoutEngine:
    def test_text_placement(self):
        elements = MathParser.parse("abc")
        box = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        placements = box.placements
        assert len(placements) > 0
        assert placements[0].x == 0
        assert placements[0].y == 0

    def test_fraction_vertical_layout(self):
        elements = MathParser.parse(r"\frac{1}{2}")
        box = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        placements = box.placements
        numerator_p = [p for p in placements if p.role == "numerator"]
        denominator_p = [p for p in placements if p.role == "denominator"]
        assert len(numerator_p) > 0
        assert len(denominator_p) > 0
        assert numerator_p[0].y > denominator_p[0].y

    def test_superscript_offset(self):
        elements = MathParser.parse("x^{2}")
        box = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        placements = box.placements
        base = [p for p in placements if p.text == "x"][0]
        sup = [p for p in placements if p.text == "2"][0]
        assert sup.y > base.y
        assert sup.font_size < base.font_size

    def test_layout_returns_total_width(self):
        elements = MathParser.parse("abc")
        box = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        total_width = MathLayoutEngine.total_width(box.placements)
        assert total_width > 0

    def test_layout_returns_box(self):
        elements = MathParser.parse("abc")
        box = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        assert hasattr(box, "placements")
        assert hasattr(box, "width")
        assert hasattr(box, "ascent")
        assert hasattr(box, "descent")
        assert box.width > 0
        assert box.ascent > 0

    def test_nested_fraction_width(self):
        nested = MathParser.parse(r"\frac{\frac{1}{2}}{x}")
        plain = MathParser.parse(r"\frac{1}{x}")
        nested_box = MathLayoutEngine.layout(nested, x=0, y=0, font_size=8.0)
        plain_box = MathLayoutEngine.layout(plain, x=0, y=0, font_size=8.0)
        assert nested_box.ascent > plain_box.ascent

    def test_sup_with_frac(self):
        elements = MathParser.parse(r"x^{\frac{1}{2}}")
        box = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        texts = [p.text for p in box.placements]
        assert "x" in texts
        assert "1" in texts
        assert "2" in texts
        base_y = [p.y for p in box.placements if p.text == "x"][0]
        one_y = [p.y for p in box.placements if p.text == "1"][0]
        assert one_y > base_y

    def test_box_ascent_descent_for_fraction(self):
        elements = MathParser.parse(r"\frac{1}{2}")
        box = MathLayoutEngine.layout(elements, x=0, y=0, font_size=8.0)
        assert box.ascent > 0
        assert box.descent > 0


class TestTagParser:
    def test_tag_parses(self):
        elements = MathParser.parse(r"x \tag{1}")
        tag = [e for e in elements if e.type == "tag"]
        assert len(tag) == 1
        assert tag[0].content == "(1)"

    def test_tag_in_complex_expr(self):
        elements = MathParser.parse(r"\frac{a}{b} \tag{2}")
        tag = [e for e in elements if e.type == "tag"]
        assert len(tag) == 1
        assert tag[0].content == "(2)"


class TestLineBreakParser:
    def test_double_backslash_parses_as_linebreak(self):
        elements = MathParser.parse(r"a \\ b")
        types = [e.type for e in elements]
        assert "linebreak" in types
        # linebreak の前後にそれぞれ要素がある
        assert types.index("linebreak") > 0
        assert types.index("linebreak") < len(types) - 1

    def test_multiple_linebreaks(self):
        elements = MathParser.parse(r"a \\ b \\ c")
        breaks = [e for e in elements if e.type == "linebreak"]
        assert len(breaks) == 2


class TestLatexSymbolMap:
    """\\omega などの LaTeX コマンドが Unicode 記号に変換されることを確認する。

    背景: 旧実装ではコマンド名 (例: "omega") が文字列として描画され、
    プレビューに "omega 2pi f" のような表示が出る不具合があった。
    """

    def test_omega_to_unicode(self):
        elements = MathParser.parse(r"\omega")
        assert elements[0].type == "symbol"
        assert elements[0].content == "ω"

    def test_pi_to_unicode(self):
        elements = MathParser.parse(r"\pi")
        assert elements[0].content == "π"

    def test_theta_to_unicode(self):
        elements = MathParser.parse(r"\theta")
        assert elements[0].content == "θ"

    def test_approx_to_unicode(self):
        elements = MathParser.parse(r"\approx")
        assert elements[0].content == "≈"

    def test_capital_sigma_to_unicode(self):
        elements = MathParser.parse(r"\Sigma")
        assert elements[0].content == "Σ"

    def test_unknown_command_is_ignored(self):
        elements = MathParser.parse(r"\unknowncmd")
        assert elements == []

    def test_quad_becomes_space(self):
        elements = MathParser.parse(r"x \quad y")
        contents = [e.content for e in elements]
        # quad が空白 (連続スペース) に変換されている
        assert any(c == "  " for c in contents)

    def test_qquad_becomes_wider_space(self):
        elements = MathParser.parse(r"x \qquad y")
        contents = [e.content for e in elements]
        assert any(c == "    " for c in contents)

    def test_thin_space_backslash_comma(self):
        elements = MathParser.parse(r"x\,y")
        contents = [e.content for e in elements]
        assert any(c == " " for c in contents)

    def test_thin_space_backslash_semicolon(self):
        elements = MathParser.parse(r"x\;y")
        contents = [e.content for e in elements]
        assert any(c == " " for c in contents)

    def test_frac_still_special(self):
        elements = MathParser.parse(r"\frac{1}{2}")
        assert elements[0].type == "frac"

    def test_sqrt_still_special(self):
        elements = MathParser.parse(r"\sqrt{x}")
        assert elements[0].type == "sqrt"

    def test_tag_still_special(self):
        elements = MathParser.parse(r"\tag{1}")
        assert elements[0].type == "tag"
        assert elements[0].content == "(1)"

    def test_real_world_expression(self):
        """元の不具合再現: \\omega = 2\\pi f, \\quad \\theta \\approx 0"""
        elements = MathParser.parse(r"\omega = 2\pi f, \quad \theta \approx 0")
        contents = [e.content for e in elements]
        # コマンド名そのものが残っていない
        assert "omega" not in contents
        assert "pi" not in contents
        assert "theta" not in contents
        assert "approx" not in contents
        assert "quad" not in contents
        # 対応する Unicode が現れる
        assert "ω" in contents
        assert "π" in contents
        assert "θ" in contents
        assert "≈" in contents

    def test_missing_greek_letters_to_unicode(self):
        elements = MathParser.parse(r"\xi")
        assert elements[0].type == "symbol"
        assert elements[0].content == "ξ"

    def test_operator_command(self):
        elements = MathParser.parse(r"\cos")
        assert elements[0].type == "operator"
        assert elements[0].content == "cos"

    def test_left_right_emit_delimiters(self):
        elements = MathParser.parse(r"\left(\frac{1}{2}\right)")
        assert [e.type for e in elements] == ["text", "frac", "text"]
        assert elements[0].content == "("
        assert elements[2].content == ")"

    def test_left_right_dot_emit_nothing(self):
        elements = MathParser.parse(r"\left. x \right.")
        contents = [e.content for e in elements if hasattr(e, "content")]
        assert "left" not in contents
        assert "right" not in contents
        assert "." not in contents

    def test_mathrm_group_passthrough(self):
        elements = MathParser.parse(r"\mathrm{abc}")
        assert len(elements) == 1
        assert elements[0].type == "group"
        assert elements[0].children[0].type == "text"
        assert elements[0].children[0].content == "abc"

    def test_bar_parses_as_accent(self):
        elements = MathParser.parse(r"\bar{x}")
        assert len(elements) == 1
        assert elements[0].type == "accent"
        assert elements[0].content == "bar"
        assert elements[0].children[0].content == "x"

    def test_linebreak_followed_by_newline_does_not_leak_into_text(self):
        elements = MathParser.parse("y = a \\\\\n y = b")
        text_contents = [e.content for e in elements if e.type == "text"]
        for content in text_contents:
            assert "\n" not in content, f"newline leaked into text: {content!r}"


class TestMathLineSegments:
    def test_fraction_emits_bar_segment(self):
        elements = MathParser.parse(r"\frac{1}{2}")
        box = MathLayoutEngine.layout(elements, x=10.0, y=20.0, font_size=8.0)
        bars = [p for p in box.placements if p.role == "frac_bar"]
        assert len(bars) == 1
        seg = bars[0].line_segment
        assert seg is not None
        x1, y1, x2, y2 = seg
        # 分数全体の幅とほぼ一致する水平線
        assert y1 == y2
        assert x2 > x1
        # ベースライン y=20 に水平線が引かれる
        assert abs(y1 - 20.0) < 0.5

    def test_sqrt_emits_radical_segments(self):
        elements = MathParser.parse(r"\sqrt{x}")
        box = MathLayoutEngine.layout(elements, x=0.0, y=0.0, font_size=8.0)
        radicals = [p for p in box.placements if p.role == "sqrt_radical"]
        # チェックマーク 2 本 + 屋根線 1 本 = 3 本
        assert len(radicals) == 3
        for r in radicals:
            assert r.line_segment is not None

    def test_nested_fraction_emits_two_bars(self):
        # \frac{\frac{1}{2}}{x} は外側と内側の 2 本の分数線を持つ
        elements = MathParser.parse(r"\frac{\frac{1}{2}}{x}")
        box = MathLayoutEngine.layout(elements, x=0.0, y=0.0, font_size=8.0)
        bars = [p for p in box.placements if p.role == "frac_bar"]
        assert len(bars) == 2

    def test_operator_layout_uses_single_placement(self):
        elements = MathParser.parse(r"\cos")
        box = MathLayoutEngine.layout(elements, x=0.0, y=0.0, font_size=8.0)
        assert len(box.placements) == 1
        assert box.placements[0].text == "cos"
        assert box.placements[0].role == "operator"

    def test_bar_emits_accent_segment(self):
        elements = MathParser.parse(r"\bar{x}")
        box = MathLayoutEngine.layout(elements, x=0.0, y=0.0, font_size=8.0)
        texts = [p.text for p in box.placements]
        accents = [p for p in box.placements if p.role == "accent_bar"]
        assert "x" in texts
        assert len(accents) == 1
        assert accents[0].line_segment is not None
