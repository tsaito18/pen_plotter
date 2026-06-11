"""統一サイズAPI（種別スケール×密度補正）のテスト。

種別スケールは typesetter.py の値（漢字1.0/ひらがな0.85/カタカナ0.85/半角0.7/
小書き0.55/句読点0.35）を「正」とする。密度補正は data/char_complexity.json に
依存するため、簡単字 < 複雑字 の相対関係のみを検証し絶対値には依存しない。
"""

from __future__ import annotations

import pytest

from src.layout.char_metrics import (
    DENSITY_SCALE_MAX,
    DENSITY_SCALE_MIN,
    char_type_scale,
    density_scale,
    effective_char_scale,
)


class TestCharTypeScale:
    def test_kanji_is_one(self) -> None:
        assert char_type_scale("漢") == 1.0
        assert char_type_scale("験") == 1.0

    def test_hiragana_is_085(self) -> None:
        # renderer 側の 0.88 はバグ。0.85 に統一するのが本タスクの要点
        assert char_type_scale("ぬ") == 0.85

    def test_katakana_is_085(self) -> None:
        assert char_type_scale("ヌ") == 0.85

    def test_halfwidth_is_07(self) -> None:
        assert char_type_scale("A") == 0.7
        assert char_type_scale("z") == 0.7

    def test_small_kana_is_055(self) -> None:
        assert char_type_scale("っ") == 0.55
        assert char_type_scale("ャ") == 0.55

    def test_punctuation_is_035(self) -> None:
        assert char_type_scale("。") == 0.35
        assert char_type_scale("、") == 0.35
        # 正規化後の本文句読点（全角「，」「．」）も同じ小サイズで扱う
        assert char_type_scale("，") == 0.35
        assert char_type_scale("．") == 0.35

    def test_override_table_applied(self) -> None:
        # 個別調整テーブルの値が種別デフォルトより優先される
        assert char_type_scale("の") == 0.78
        assert char_type_scale("ロ") == 0.68

    def test_small_kana_precedes_override(self) -> None:
        # 小書きは override テーブルより優先（小書きにoverrideは無いが順序を固定）
        assert char_type_scale("ぁ") == 0.55


class TestDensityScale:
    def test_in_clamp_range_for_mapped_char(self) -> None:
        # マップに含まれる漢字は連続補正値（clamp範囲内）を返す
        val = density_scale("口")
        assert DENSITY_SCALE_MIN <= val <= DENSITY_SCALE_MAX

    def test_unmapped_char_falls_back_to_one(self) -> None:
        # 記号・幾何字形などマップに無い文字は 1.0 フォールバック
        assert density_scale("☃") == 1.0
        assert density_scale("+") == 1.0

    def test_simple_char_smaller_than_complex_char(self) -> None:
        # 簡単字（少画・短ink）< 複雑字（多画・長ink）の密度補正
        for simple, complex_ in [("口", "験"), ("入", "論"), ("子", "理")]:
            assert density_scale(simple) < density_scale(complex_), (
                f"{simple} should be smaller than {complex_}"
            )


class TestEffectiveCharScale:
    def test_is_product_of_type_and_density(self) -> None:
        for ch in ["験", "口", "あ", "A"]:
            expected = char_type_scale(ch) * density_scale(ch)
            assert effective_char_scale(ch) == pytest.approx(expected)

    def test_kanji_density_modulation_visible(self) -> None:
        # 漢字は種別1.0なので effective がそのまま密度補正＝簡単字<複雑字
        assert effective_char_scale("口") < effective_char_scale("験")

    def test_unmapped_char_equals_type_scale(self) -> None:
        # マップ無し文字は density=1.0 なので effective == type_scale
        assert effective_char_scale("+") == pytest.approx(char_type_scale("+"))
