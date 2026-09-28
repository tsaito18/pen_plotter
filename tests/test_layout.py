"""組版: 改行・ページ・表・数式パーサ・字サイズ・Typesetter。"""

from __future__ import annotations

import pytest

from src.handwriting.augmentation import AugmentConfig, HandwritingAugmenter
from src.layout.char_metrics import (
    char_ink_length,
    char_type_scale,
    compute_complexity,
    density_scale,
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
    assert char_type_scale("ぬ") == 0.85
    assert char_type_scale("ロ") == 0.68  # 個別調整
    assert char_type_scale("a") == 0.8
    assert char_type_scale("っ") == 0.55  # 小書きは個別調整より優先
    assert char_type_scale("、") == 0.35


def test_density_scale_is_bounded_and_defaults_to_one():
    assert density_scale("\ue000") == 1.0
    assert 0.88 <= density_scale("一") <= density_scale("驚") <= 1.12
    assert effective_char_scale("あ") == pytest.approx(char_type_scale("あ") * density_scale("あ"))


def test_complexity_helpers():
    assert char_ink_length([[(0, 0), (3, 4)], [{"x": 0, "y": 0}, {"x": 0, "y": 1}]]) == 6.0
    norm = normalize_robust(list(range(101)))
    assert norm[0] == 0.0 and norm[-1] == 1.0 and norm == sorted(norm)
    assert normalize_robust([3.0, 3.0]) == [0.0, 0.0]
    assert compute_complexity(stroke_norm=1.0, ink_norm=0.0) == pytest.approx(0.5)


# --- Typesetter: 本文 ---


def test_normalize_body_punctuation_skips_math():
    assert normalize_body_punctuation("a,b.c、d。 $1.5, 2$") == "a，b．c，d． $1.5, 2$"


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
