"""組版: 改行・ページ・表・数式パーサ・字サイズ・Typesetter。"""

from __future__ import annotations

import pytest

from src.handwriting.augmentation import AugmentConfig, HandwritingAugmenter
from src.layout.char_metrics import (
    char_ink_length,
    char_type_scale,
    compute_complexity,
    effective_char_scale,
    normalize_robust,
)
from src.layout.line_breaking import break_paragraph_by_width
from src.layout.math_layout import MathLayoutEngine, MathParser
from src.layout.page_layout import PageConfig, PageLayout
from src.layout.placement import CharPlacement
from src.layout.table_layout import detect_pipe_table, split_pipe_row
from src.layout.typesetter import Typesetter, normalize_body_punctuation

FS = 5.0


def _typesetter(augmenter: HandwritingAugmenter | None = None, **page: float) -> Typesetter:
    return Typesetter(PageConfig(**page), font_size=FS, augmenter=augmenter)


def _text(page: list[CharPlacement]) -> str:
    return "".join(p.char for p in page if p.is_text)


def _rows(page: list[CharPlacement]) -> list[str]:
    """y ごとの本文文字列（上の行から）。"""
    rows: dict[float, list[CharPlacement]] = {}
    for p in page:
        if p.is_text:
            rows.setdefault(round(p.y, 3), []).append(p)
    return [
        "".join(p.char for p in sorted(r, key=lambda p: p.x))
        for _, r in sorted(rows.items(), reverse=True)
    ]


# --- 改行（禁則処理） ---


def test_break_by_width():
    assert break_paragraph_by_width("あいうえお", 3, lambda c: 1.0) == ["あいう", "えお"]
    assert break_paragraph_by_width("abcdef", 2, lambda c: 0.5) == ["abcd", "ef"]


def test_kinsoku_pulls_closing_punctuation_back_and_pushes_opening_forward():
    assert break_paragraph_by_width("あいう。え", 3, lambda c: 1.0) == ["あいう。", "え"]
    assert break_paragraph_by_width("あい「うえ", 3, lambda c: 1.0) == ["あい", "「うえ"]


# --- ページ ---


def test_page_layout_lines_are_inside_content_area():
    cfg = PageConfig(margin_top=48, margin_bottom=34, line_spacing=7.14)
    layout = PageLayout(cfg)
    area = layout.content_area()
    rows = layout.line_positions()
    assert rows[0] == pytest.approx(297 - 48)
    assert all(area.y <= y <= area.y + area.height for y in rows)
    assert len(layout.ruled_line_strokes()) == len(rows)
    assert PageLayout(PageConfig(line_spacing=0)).line_positions() == []


# --- パイプ表の検出 ---


def test_pipe_table_detection():
    assert split_pipe_row("| a | b |") == ["a", "b"] == split_pipe_row("a | b")
    rows, consumed = detect_pipe_table(["| a | b |", "|---|:-:|", "| 1 |", "本文"], 0)
    assert rows == [["a", "b"], ["1", ""]]  # 列数は最大に揃える
    assert consumed == 3
    assert detect_pipe_table(["| a | b |", "| 1 | 2 |"], 0) is None  # 区切り行が必須


# --- 数式パーサ / レイアウト ---


def test_math_parser_structures():
    types = [e.type for e in MathParser.parse(r"x^2 + \frac{a}{b} - \sqrt{c}_0 \tag{1}")]
    assert types == ["text", "sup", "text", "frac", "text", "sqrt", "sub", "text", "tag"]


@pytest.mark.parametrize(
    ("src", "content"),
    [(r"\omega", "ω"), (r"\Sigma", "Σ"), (r"\approx", "≈"), (r"\quad", "  "), (r"\,", " ")],
)
def test_latex_symbols_become_unicode(src: str, content: str):
    (elem,) = MathParser.parse(src)
    assert elem.content == content


def test_latex_commands_are_not_rendered_literally():
    text = "".join(e.content for e in MathParser.parse(r"\left( x \right. \unknown"))
    assert "left" not in text and "unknown" not in text and "(" in text
    assert MathParser.parse(r"\cos x")[0].type == "operator"
    for name in ("arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "det", "arg"):
        (op, _) = MathParser.parse(rf"\{name} x")  # 未対応だと関数名が式から丸ごと抜ける
        assert (op.type, op.content) == ("operator", name)
    assert [e.type for e in MathParser.parse(r"\bar{x}")] == ["accent"]
    assert [e.type for e in MathParser.parse(r"a \\ b")] == ["text", "linebreak", "text"]


def test_fraction_layout_stacks_numerator_above_denominator():
    box = MathLayoutEngine.layout(MathParser.parse(r"\frac{a}{b}"), x=0, y=0, font_size=FS)
    by_role = {p.role: p for p in box.placements}
    assert by_role["numerator"].y > 0 > by_role["denominator"].y
    assert by_role["frac_bar"].line_segment is not None
    nested = MathLayoutEngine.layout(MathParser.parse(r"\frac{\frac{1}{2}}{x}"), 0, 0, FS)
    assert nested.ascent > box.ascent


# --- 字サイズ ---


def test_char_type_scale():
    assert char_type_scale("漢") == 1.0
    assert char_type_scale("ぬ") == char_type_scale("ロ") == 0.8  # 字種の代表値（個別表なし）
    assert char_type_scale("a") == 0.8
    assert char_type_scale("っ") == 0.55
    assert char_type_scale("、") == 0.35


def test_char_size_follows_kind_and_complexity():
    """漢字は大きく画数が多いほど大きい。かなは小さく、単純な形ほど小さい（実物の傾向）。"""
    s = effective_char_scale
    assert s("一") < s("国") < s("験") <= 1.1
    assert 0.6 <= s("ン") < s("と") < s("あ") <= 0.88 < s("一")  # どのかなも漢字より小さい
    assert s("く") < s("わ")
    assert s("a") == 0.8 and s("っ") == 0.55 and s("、") == 0.35
    assert s("\ue000") == char_type_scale("\ue000")  # 複雑度データの無い字は字種の代表値


def test_complexity_helpers():
    assert char_ink_length([[(0, 0), (3, 4)], [{"x": 0, "y": 0}, {"x": 0, "y": 1}]]) == 6.0
    norm = normalize_robust(list(range(101)))
    assert norm[0] == 0.0 and norm[-1] == 1.0 and norm == sorted(norm)
    assert normalize_robust([3.0, 3.0]) == [0.0, 0.0]
    assert compute_complexity(stroke_norm=1.0, ink_norm=0.0) == pytest.approx(0.5)


# --- Typesetter: 本文 ---


def test_normalize_body_punctuation_skips_math_and_halfwidth_text():
    text = "あ,い.う、え。 0.1 uF, 1,000 Fig. 2 $1.5, 2$"
    # 和文の後は全角、小数点・英文の後（直前が半角文字）は半角のまま
    assert normalize_body_punctuation(text) == "あ，い．う，え． 0.1 uF, 1,000 Fig. 2 $1.5, 2$"


def test_text_wraps_and_paginates():
    ts = _typesetter()
    pages = ts.typeset("あ" * 2000)
    assert len(pages) > 1
    right = ts.layout.content_area().x + ts.layout.content_area().width
    assert all(p.x + ts.body_char_advance("あ") <= right + 1e-6 for p in pages[0])
    assert ts.typeset("") == [[]]


def test_advance_depends_on_char_kind():
    ts = _typesetter()
    assert ts.body_char_advance("a") < ts.body_char_advance("あ") < ts.body_char_advance("漢")
    # 英字は字ごとの幅（実物の手書き実測で、平均は全角の約半分）
    assert ts.body_char_advance("i") < ts.body_char_advance("a") < ts.body_char_advance("m")
    assert ts.body_char_advance("a") < 0.5 * ts.body_char_advance("漢")
    assert ts.body_char_advance("a") < ts.body_char_advance("Z") < ts.body_char_advance("M")
    assert ts.body_char_advance("Z") > 0.6 * ts.body_char_advance("漢")  # 実測: 大文字 ≈0.6〜0.67
    assert ts.body_char_advance("ω") < 0.6 * ts.body_char_advance("漢")  # ギリシャ文字も欧文幅


def test_ink_aware_advance_keeps_the_gap_between_glyphs_even():
    """字送り = 字形のインク幅 + 一定の隙間。細い字で空き、太い字で詰まるのを防ぐ。"""
    widths = {"り": 0.5, "漢": 0.95, "あ": 0.8}
    ts = Typesetter(PageConfig(), font_size=FS, ink_width=widths.get)
    ink = {c: w * FS * effective_char_scale(c) for c, w in widths.items()}
    gaps = {c: ts.body_char_advance(c) - ink[c] for c in widths}
    assert max(gaps.values()) - min(gaps.values()) < 1e-9
    assert ts.body_char_advance("り") < ts.body_char_advance("漢")
    line = ts.typeset("漢りあ")[0]
    assert [p.advance for p in line] == pytest.approx([ts.body_char_advance(c) for c in "漢りあ"])
    assert line[1].x == pytest.approx(line[0].x + line[0].advance)
    # インク幅の分からない字は従来の字送り
    assert ts.body_char_advance("無") == _typesetter().body_char_advance("無")


def test_paragraph_indent_rules():
    ts = _typesetter()
    page = ts.typeset("最初\n二段落目\n\\noindent 字下げなし")[0]
    first_x = {row[0]: min(p.x for p in page if p.char == row[0]) for row in ["最", "二", "字"]}
    area_x = ts.layout.content_area().x
    assert first_x["最"] == pytest.approx(area_x)  # ページ先頭の行は字下げしない
    assert first_x["二"] == pytest.approx(area_x + FS)
    assert first_x["字"] == pytest.approx(area_x)


def test_headings_are_larger_and_preceded_by_blank_line():
    ts = _typesetter()
    page = ts.typeset("# 見出し\n本文\n## 小見出し\n本文")[0]
    heading = next(p for p in page if p.char == "見")
    body = next(p for p in page if p.char == "本")
    assert heading.font_size > body.font_size
    assert "#" not in _text(page)
    body_y = body.y
    sub_heading_y = next(p.y for p in page if p.char == "小")
    assert body_y - sub_heading_y == pytest.approx(2 * PageConfig().line_spacing)  # 空行 1 行


def test_page_break_marker():
    pages = _typesetter().typeset("---\n一ページ目\n-----\n二ページ目")
    assert [_text(p) for p in pages] == ["一ページ目", "二ページ目"]


def test_augmented_placement_varies_but_stays_near_the_line():
    aug = HandwritingAugmenter(AugmentConfig(), seed=0)
    flat = _typesetter().typeset("あいうえおかきくけこ")[0]
    wavy = _typesetter(aug).typeset("あいうえおかきくけこ")[0]
    assert {round(p.y, 6) for p in flat} != {round(p.y, 6) for p in wavy}
    assert all(abs(w.y - f.y) < 2.0 for f, w in zip(flat, wavy, strict=True))
    assert any(w.slant != 0 for w in wavy)


def test_text_sits_a_little_below_the_middle_of_the_ruled_band():
    """実物は罫線の少し上に乗せて書く（帯の中央より約 0.5mm 下。スキャン実測）。"""
    ts = _typesetter()
    row = ts.layout.line_positions()[0]
    (first,) = [p for p in ts.typeset("漢")[0] if p.is_text]
    drop = row - first.y
    assert drop == pytest.approx(0.07 * ts.config.line_spacing, abs=1e-6)


def test_same_seed_same_layout():
    def run() -> list[tuple[float, float, float]]:
        page = _typesetter(HandwritingAugmenter(seed=3)).typeset("手書きの揺らぎ")[0]
        return [(p.x, p.y, p.font_size) for p in page]

    assert run() == run()


# --- Typesetter: 数式 ---


def test_plain_inline_math_uses_body_glyphs():
    page = _typesetter().typeset("電圧 $V = IR$ と $\\sigma$")[0]
    assert all(p.math is None for p in page)
    assert "V=IR" in _text(page).replace(" ", "")
    assert "σ" in _text(page)


def test_structured_inline_math_becomes_one_rendered_placement():
    page = _typesetter().typeset("式 $E=mc^2$ です")[0]
    (math,) = [p for p in page if p.math is not None]
    assert math.math.source == "E=mc^2" and math.math.align == "baseline"
    after = next(p for p in page if p.char == "で")
    x0, _y, width, _h = math.math.bbox
    assert after.x >= x0 + width - 1e-6  # 数式の実描画幅ぶんカーソルが進む


def test_block_math_is_centered_with_tag_after_body():
    ts = _typesetter()
    page = ts.typeset("前\n$$x^2 + y^2 = r^2 \\tag{1}$$\n後")[0]
    (math,) = [p for p in page if p.math is not None]
    x0, _y, w, _h = math.math.bbox
    area = ts.layout.content_area()
    assert x0 + w / 2 == pytest.approx(area.x + area.width / 2)
    tag = [p for p in page if p.char in "(1)" and p.is_text and p.x > x0 + w]
    assert "".join(p.char for p in tag) == "(1)"
    assert "\\tag" not in math.math.source
    assert _rows(page)[0] == "前" and _rows(page)[-1] == "後"


def test_multiline_block_math_consumes_rows_and_moves_to_next_page_when_full():
    ts = _typesetter(margin_top=250, margin_bottom=15)  # 数行しか入らないページ
    rows = len(ts.layout.line_positions())
    pages = ts.typeset("本文\n" * (rows - 1) + "$$\n\\frac{a}{b}\n$$")
    assert len(pages) == 2
    assert any(p.math is not None for p in pages[1])


# --- Typesetter: 表 ---


def test_table_is_centered_with_caption_below():
    ts = _typesetter()
    page = ts.typeset("| 材料 | 値 |\n|---|---|\n| SS400 | 245 |\n: 表1 結果")[0]
    lines = [p for p in page if p.line_segment is not None]
    horizontal = [p for p in lines if p.line_segment[1] == p.line_segment[3]]
    vertical = [p for p in lines if p.line_segment[0] == p.line_segment[2]]
    assert len(horizontal) == 3 and len(vertical) == 3
    left, right = min(p.line_segment[0] for p in lines), max(p.line_segment[2] for p in lines)
    area = ts.layout.content_area()
    assert (left + right) / 2 == pytest.approx(area.x + area.width / 2)
    # セル文字は列の罫線内に収まる
    for p in page:
        if p.is_text and p.char in "SS400245":
            assert left < p.x < right
    caption_y = next(p.y for p in page if p.char == "表")
    assert caption_y < min(p.line_segment[1] for p in horizontal)


def test_zero_width_characters_take_no_space():
    ts = _typesetter()
    plain = ts.typeset("あい")[0]
    with_zwsp = ts.typeset("あ​い﻿")[0]
    assert [(p.char, p.x) for p in with_zwsp] == [(p.char, p.x) for p in plain]


# --- 数式の配置（mathtext）---


def test_mathtext_layout_has_glyphs_and_fraction_bar():
    from src.layout.mathtext import extract_math_layout

    layout = extract_math_layout(r"\frac{a}{b}")
    assert [g.char for g in layout.glyphs] == ["a", "b"]
    assert len(layout.rects) == 1
    a, b = layout.glyphs
    assert a.baseline_y > layout.rects[0].center_y > b.baseline_y
    assert extract_math_layout(r"\frac{a}") is None  # 解析できない式


def test_fraction_bar_for_ruled_alignment():
    from src.layout.mathtext import detect_top_level_fraction_bar, extract_math_layout

    def bar(src: str) -> float | None:
        return detect_top_level_fraction_bar(extract_math_layout(src))

    assert bar(r"L = l + \dfrac{a}{b}") is not None
    assert bar(r"\dfrac{a}{b} + \dfrac{c}{d}") is not None  # 横並びも同じ罫線に乗る
    assert bar(r"x^2 + y^2") is None
    assert bar(r"\sqrt{x}") is None  # 根号の屋根は分数線ではない
    assert bar(r"\left(\frac{a}{b}\right)^2") is None  # 括弧で囲まれた分数


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (r"\frac{a}{b}", r"\dfrac{a}{b}"),
        (r"\sqrt{\frac{a}{b}}", r"\sqrt{\dfrac{a}{b}}"),
        (r"\sqrt[3]{\frac{a}{b}}", r"\sqrt[3]{\dfrac{a}{b}}"),
        (r"e^{\frac{x}{2}}", r"e^{\frac{x}{2}}"),  # 指数の中の分数は小さいまま
        (r"x_{\frac{1}{2}}", r"x_{\frac{1}{2}}"),
        (r"\frac{\frac{a}{b}}{c}", r"\dfrac{\frac{a}{b}}{c}"),  # 分子の中の分数も小さいまま
        (r"\left(\frac{a}{b}\right)", r"\left(\dfrac{a}{b}\right)"),
    ],
)
def test_only_top_level_fractions_are_promoted_to_display_size(src: str, expected: str):
    from src.layout.mathtext import promote_top_level_frac_to_dfrac

    assert promote_top_level_frac_to_dfrac(src) == expected


def test_long_block_math_splits_before_relations_then_terms():
    from src.layout.mathtext import handwrite_draw_width_mm, split_math_for_width

    src = r"E = a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = b_1 + b_2"
    full = handwrite_draw_width_mm(src, FS, 0.85)
    lines = split_math_for_width(src, FS, full * 0.45)
    assert len(lines) >= 3 and "".join(lines) == src
    assert all(handwrite_draw_width_mm(line, FS, 0.85) <= full * 0.45 for line in lines[1:])
    assert lines[1].lstrip().startswith("=")
    assert split_math_for_width(src, FS, full + 1) == [src]
    assert split_math_for_width(r"\frac{a}{b}", FS, 1.0) == [r"\frac{a}{b}"]  # 切れ目が無い


# --- Typesetter: 手書きの構造式 ---


def test_structured_inline_math_is_handwritten_with_its_drawn_width():
    from src.layout.mathtext import MATH_INLINE_CAP_RATIO, handwrite_draw_width_mm

    page = _typesetter().typeset("式 $E=mc^2$ です")[0]
    (math,) = [p for p in page if p.math is not None]
    assert math.math.handwritten
    width = handwrite_draw_width_mm("E=mc^2", FS, MATH_INLINE_CAP_RATIO)
    assert math.math.bbox[2] == pytest.approx(width)
    printed = Typesetter(PageConfig(), font_size=FS, handwrite_math=False).typeset("$E=mc^2$")[0]
    assert not printed[0].math.handwritten


def test_block_fraction_bar_sits_on_a_ruling_line():
    ts = _typesetter()
    page = ts.typeset("前\n$$L = l + \\frac{a}{b} \\tag{2}$$\n後")[0]
    (math,) = [p for p in page if p.math is not None]
    assert math.math.handwritten and "\\dfrac" in math.math.source
    rows = ts.layout.line_positions()
    assert any(math.math.fraction_bar_y == pytest.approx(y) for y in rows)
    _x, y0, _w, h = math.math.bbox
    assert y0 < math.math.fraction_bar_y < y0 + h  # 分子は上の行、分母は下の行
    tag_y = next(p.y for p in page if p.char == "2")
    assert tag_y + ts.config.line_spacing / 2 == pytest.approx(math.math.fraction_bar_y)
    assert _rows(page)[0] == "前" and _rows(page)[-1] == "後"


def test_long_block_math_is_split_into_lines_that_fit():
    ts = _typesetter()
    terms = " + ".join(f"a_{{{i}}}" for i in range(30))
    page = ts.typeset(f"$$E = {terms} \\tag{{3}}$$")[0]
    maths = [p.math for p in page if p.math is not None]
    area = ts.layout.content_area()
    assert len(maths) >= 2
    assert all(area.x - 1e-6 <= m.bbox[0] and m.bbox[0] + m.bbox[2] <= area.x + area.width + 1e-6
               for m in maths)  # fmt: skip
    assert len({m.bbox[1] for m in maths}) == len(maths)  # 別々の行
    tag_y = next(p.y for p in page if p.char == "3")
    assert tag_y < maths[0].bbox[1]  # 式番号は最後の行


def test_split_block_math_that_does_not_fit_moves_whole_to_next_page():
    ts = _typesetter(margin_top=200, margin_bottom=15)  # 11 行のページ
    rows = len(ts.layout.line_positions())
    terms = " + ".join(f"a_{{{i}}}" for i in range(30))
    pages = ts.typeset("本文\n" * (rows - 3) + f"$$E = {terms}$$")
    assert len(pages) == 2
    assert not any(p.math is not None for p in pages[0])  # 途中の行だけ前ページに残らない
    assert len([p for p in pages[1] if p.math is not None]) >= 2
