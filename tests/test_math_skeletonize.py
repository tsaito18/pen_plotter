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
    split_math_for_width,
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
        ],
    )
    def test_side_by_side_fractions_share_common_bar(self, src):
        # 横並びの複数分数（v²/2g + p/ρg + … 等）は全分数線が同一 axis 帯に
        # 並ぶため、共通の帯として平均 cy に揃えられる（分子=上の行・分母=下の行で
        # 自然に手書き展開できる）。
        layout = extract_math_layout(src)
        cy = detect_top_level_fraction_bar(layout)
        assert cy is not None
        assert 4.0 <= cy <= 9.0

    @pytest.mark.parametrize(
        "src",
        [
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


class TestSplitMathForWidth:
    """split_math_for_width: 本文幅を超えるブロック数式を関係/加減で分割する。"""

    def test_short_formula_returns_single_segment(self):
        """max_width 以下なら分割せず元のまま返す。"""
        result = split_math_for_width(r"V=IR", font_size=4.5, max_width_mm=200.0)
        assert result == [r"V=IR"]

    def test_long_formula_with_relation_splits_at_relation(self):
        """関係演算子で 1 次分割される。"""
        # 関係演算子 \leq で2つに分かれる。max_width を非常に小さくすれば分割確定。
        src = r"\frac{\Delta g}{g} \leq 2\frac{\Delta T}{T}"
        result = split_math_for_width(src, font_size=4.5, max_width_mm=10.0)
        assert len(result) >= 2
        # 各セグメントの結合で元式と一致
        assert "".join(result).replace(" ", "") == src.replace(" ", "")
        # 2 番目のセグメントは \leq で始まる
        assert result[1].lstrip().startswith(r"\leq")

    def test_long_formula_with_additive_splits_at_plus(self):
        """関係演算子だけで足りない場合、加減で 2 次分割する。"""
        src = r"a + b + c + d + e + f + g + h"
        # 非常に小さい max_width にすれば加減で分割される
        result = split_math_for_width(src, font_size=4.5, max_width_mm=8.0)
        assert len(result) >= 3
        joined = "".join(result).replace(" ", "")
        assert joined == src.replace(" ", "")

    def test_force_split_multiple_fractions_separates_each_fraction(self):
        """force_split_multiple_fractions=True で複数分数式は1分数/セグメントへ分割（式(4)想定）。"""
        from src.ui.math_skeletonize import _count_top_level_fracs

        # 式(4): δg/g ≤ 2δπ/π + 2δT/T + δL/L + αδα/4/(1+α²/8)
        src = (
            r"\frac{\Delta g}{g} \leq 2\frac{\Delta\pi}{\pi} "
            r"+ 2\frac{\Delta T}{T} + \frac{\Delta L}{L} "
            r"+ \frac{\alpha\,\Delta\alpha/4}{1+\alpha^{2}/8}"
        )
        # 関係演算子で 1 次分割→セグメント2 (4分数) を加減で 2 次分割→5セグメント
        result = split_math_for_width(
            src,
            font_size=4.5,
            max_width_mm=200.0,
            force_split_multiple_fractions=True,
        )
        assert len(result) == 5
        # 各セグメントの結合で元式と一致（順序保存・内容ロスなし）
        joined = "".join(result).replace(" ", "")
        assert joined == src.replace(" ", "")
        # 各セグメントのトップレベル \frac は 1 個以下（罫線揃え対象）
        for seg in result:
            assert _count_top_level_fracs(seg) <= 1

    def test_no_force_split_when_single_fraction(self):
        """force_split=True でも分数が 1 個なら分割しない（既存の単一行を維持）。"""
        src = r"L = l + r + \frac{2r^{2}}{5(l+r)}"
        result = split_math_for_width(
            src,
            font_size=4.5,
            max_width_mm=200.0,
            force_split_multiple_fractions=True,
        )
        assert result == [src]

    def test_no_force_split_when_fractions_inside_sqrt(self):
        """force_split=True でも分数が √ 内（入れ子）のみなら分割しない。"""
        # T_0 = 2π√(I/Mgh) = 2π√(L/g): 2 個の分数があるが両方 √ 内（axis 帯外）
        src = r"T_{0} = 2\pi\sqrt{\frac{I}{Mgh}} = 2\pi\sqrt{\frac{L}{g}}"
        result = split_math_for_width(
            src,
            font_size=4.5,
            max_width_mm=200.0,
            force_split_multiple_fractions=True,
        )
        assert result == [src]

    def test_unsplittable_returns_single_segment(self):
        """分割点がない単一項は1セグメントのまま返す（呼び出し側で縮小フォールバック）。"""
        src = r"\frac{\Delta g}{g}"
        # 強制的に小さい max_width にしても、関係/加減演算子がないので分割不能
        result = split_math_for_width(src, font_size=4.5, max_width_mm=1.0)
        assert result == [src]

    def test_unary_minus_not_split(self):
        """先頭の単項 - は分割しない。"""
        src = r"-x = a + b"
        result = split_math_for_width(src, font_size=4.5, max_width_mm=10.0)
        # `-x ` と `= a + b` または `-x ` `= a ` `+ b` 等。先頭 `-` は切られない。
        assert result[0].lstrip().startswith("-")

    def test_fraction_internal_plus_not_split(self):
        """分数内の + は分割対象外（深さ>0 のため）。"""
        src = r"\frac{a+b}{c+d} = e"
        result = split_math_for_width(src, font_size=4.5, max_width_mm=10.0)
        # `=` だけが分割点。\frac の中の `+` は触らない。
        assert all(r"\frac{a+b}{c+d}" in seg or seg.lstrip().startswith("=") for seg in result)
