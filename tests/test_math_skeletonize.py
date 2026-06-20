"""math_skeletonize の幅計算（formula_aspect / formula_draw_width_mm）のテスト。"""

from __future__ import annotations

import pytest

from src.ui.math_skeletonize import (
    MATH_BLOCK_CAP_RATIO,
    MATH_INLINE_CAP_RATIO,
    detect_top_level_fraction_bar,
    extract_math_layout,
    formula_aspect,
    formula_draw_width_mm,
    handwrite_draw_width_mm,
)


class TestFormulaAspect:
    """formula_aspect: matplotlib 実描画のアスペクト比（幅/高さ）を返す。"""

    def test_aspect_positive_for_simple_formula(self):
        assert formula_aspect("E=mc^2") > 0.0

    def test_aspect_positive_for_various(self):
        for src in ["V=IR", "x^2+y^2=z^2", r"\frac{a}{b}", r"\sum_{i=1}^{n} i"]:
            assert formula_aspect(src) > 0.0

    def test_wide_formula_has_larger_aspect_than_tall(self):
        wide = formula_aspect("x^2+y^2=z^2")
        tall = formula_aspect(r"\frac{a}{b}")
        assert wide > tall

    def test_empty_or_blank_returns_zero(self):
        # 墨が出ない入力は 0.0（呼び出し側は論理幅 fallback）
        assert formula_aspect("") == 0.0

    def test_draw_width_equals_h_times_aspect(self):
        for src in ["E=mc^2", "V=IR", r"\frac{a}{b}"]:
            h_mm = 6.0
            assert formula_draw_width_mm(src, h_mm) == pytest.approx(h_mm * formula_aspect(src))


class TestDetectTopLevelFractionBar:
    """detect_top_level_fraction_bar: 罫線揃え対象の単一主分数線中心 y(pt) を返す。"""

    @pytest.mark.parametrize(
        "src",
        [
            r"L = l + r + \frac{2r^{2}}{5(l+r)}",  # (1) 混在式の主分数
            r"\alpha = \arcsin\frac{d}{l+r}",  # (2)
            r"m_{w} = \rho\cdot\frac{\pi}{4}d_{w}^{2}\cdot l = 0.50",  # (5) 幅狭だが単一
            (  # (6) 入れ子: 主線が最幅・math axis 上、入れ子は帯外
                r"L' = \frac{\frac{1}{3}m_{w}l^{2} + \frac{2}{5}Mr^{2} + M(l+r)^{2}}"
                r"{\frac{1}{2}m_{w}l + M(l+r)}"
            ),
        ],
    )
    def test_top_level_fraction_returns_axis_y(self, src):
        layout = extract_math_layout(src)
        cy = detect_top_level_fraction_bar(layout)
        assert cy is not None
        # matplotlib の math axis 帯（実測 cy≈6〜7pt）に乗る
        assert 4.0 <= cy <= 9.0

    @pytest.mark.parametrize(
        "src",
        [
            r"\frac{\Delta g}{g} \leq 2\frac{\Delta\pi}{\pi} + 2\frac{\Delta T}{T}",  # (4) 横並び
            r"\gamma = \frac{1}{200\,T}\ln\frac{430}{320}",  # (7) 横並び
            r"g = \left(\frac{2\pi}{T}\right)^{2} L",  # (3) 括弧内のみ
            r"g = 9.80 \pm 0.01",  # 分数なし
        ],
    )
    def test_non_top_level_returns_none(self, src):
        layout = extract_math_layout(src)
        assert detect_top_level_fraction_bar(layout) is None


class TestHandwriteDrawWidthCapRatio:
    """handwrite_draw_width_mm: cap_ratio で実描画幅をスケールする。"""

    def test_block_cap_ratio_wider_than_inline(self):
        fs = 4.5
        inline = handwrite_draw_width_mm(r"\frac{a}{b}", fs, MATH_INLINE_CAP_RATIO)
        block = handwrite_draw_width_mm(r"\frac{a}{b}", fs, MATH_BLOCK_CAP_RATIO)
        assert inline is not None and block is not None
        assert block == pytest.approx(inline * (MATH_BLOCK_CAP_RATIO / MATH_INLINE_CAP_RATIO))

    def test_default_cap_ratio_is_inline(self):
        fs = 4.5
        assert handwrite_draw_width_mm("V=IR", fs) == pytest.approx(
            handwrite_draw_width_mm("V=IR", fs, MATH_INLINE_CAP_RATIO)
        )
