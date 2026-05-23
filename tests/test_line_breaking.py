from src.layout.line_breaking import (
    break_paragraph,
    break_paragraph_by_width,
    break_lines,
    is_line_start_prohibited,
    is_line_end_prohibited,
    _char_width,
)


class TestProhibitedChars:
    def test_line_start_prohibited(self):
        for ch in "。、，．）」』】〉》〕!?！？ー":
            assert is_line_start_prohibited(ch), f"'{ch}' should be prohibited at line start"

    def test_line_start_allowed(self):
        for ch in "あア亜aA1（「":
            assert not is_line_start_prohibited(ch), f"'{ch}' should be allowed at line start"

    def test_line_end_prohibited(self):
        for ch in "（「『【〈《〔":
            assert is_line_end_prohibited(ch), f"'{ch}' should be prohibited at line end"

    def test_line_end_allowed(self):
        for ch in "あア亜aA1。、）」":
            assert not is_line_end_prohibited(ch), f"'{ch}' should be allowed at line end"


class TestBreakLines:
    def test_simple_text(self):
        lines = break_lines("あいうえお", chars_per_line=5)
        assert lines == ["あいうえお"]

    def test_wrapping(self):
        lines = break_lines("あいうえおかきくけこ", chars_per_line=5)
        assert lines == ["あいうえお", "かきくけこ"]

    def test_line_start_prohibition(self):
        text = "あいうえ。かきくけこ"
        lines = break_lines(text, chars_per_line=4)
        assert not any(is_line_start_prohibited(line[0]) for line in lines if line)

    def test_line_end_prohibition(self):
        text = "あいうえ「かきくけこ"
        lines = break_lines(text, chars_per_line=5)
        for line in lines:
            if line:
                assert not is_line_end_prohibited(line[-1]), (
                    f"Line '{line}' ends with prohibited char"
                )

    def test_empty_text(self):
        lines = break_lines("", chars_per_line=10)
        assert lines == [""]

    def test_newline_preserved(self):
        lines = break_lines("あいう\nかきく", chars_per_line=10)
        assert lines == ["あいう", "かきく"]

    def test_half_width_counts_as_half(self):
        lines = break_lines("abc", chars_per_line=2)
        assert lines == ["abc"]

    def test_mixed_width(self):
        lines = break_lines("あab", chars_per_line=3)
        assert lines == ["あab"]

    def test_halfwidth_char_width_is_0_5(self):
        """半角文字の幅は0.5（全角の半分）。"""
        assert _char_width("a") == 0.5
        assert _char_width("Z") == 0.5
        assert _char_width("0") == 0.5

    def test_fullwidth_char_width_is_1(self):
        assert _char_width("あ") == 1.0
        assert _char_width("漢") == 1.0

    def test_halfwidth_fitting_more_chars_per_line(self):
        """半角0.5幅なら、chars_per_line=2に半角4文字が収まる。"""
        lines = break_lines("abcd", chars_per_line=2)
        assert lines == ["abcd"]

    def test_halfwidth_overflow(self):
        """半角0.5幅なら、chars_per_line=2に半角5文字は溢れる。"""
        lines = break_lines("abcde", chars_per_line=2)
        assert lines == ["abcd", "e"]

    def test_break_paragraph_keeps_legacy_halfwidth_widths(self):
        lines = break_paragraph("abcde", chars_per_line=2)
        assert lines == ["abcd", "e"]


class TestWidthAwareBreaks:
    def test_custom_widths_control_breaks(self):
        widths = {"広": 2.0, "狭": 0.5}

        lines = break_paragraph_by_width(
            "広狭広",
            max_width=2.5,
            char_width=lambda ch: widths[ch],
        )

        assert lines == ["広狭", "広"]

    def test_line_start_prohibition_with_custom_widths(self):
        lines = break_paragraph_by_width(
            "あいうえ。か",
            max_width=4.0,
            char_width=lambda _: 1.0,
        )

        assert lines == ["あいうえ。", "か"]

    def test_line_end_prohibition_with_custom_widths(self):
        lines = break_paragraph_by_width(
            "あいうえ「か",
            max_width=5.0,
            char_width=lambda _: 1.0,
        )

        assert lines == ["あいうえ", "「か"]
