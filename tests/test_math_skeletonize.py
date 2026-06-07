"""math_skeletonize の幅計算（formula_aspect / formula_draw_width_mm）のテスト。"""

from __future__ import annotations

import pytest

from src.ui.math_skeletonize import formula_aspect, formula_draw_width_mm


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
