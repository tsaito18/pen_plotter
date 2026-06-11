"""ink長計算と複雑度正規化ロジックの単体テスト。

`scripts/compute_char_complexity.py` が依存する純粋関数を検証する。データ生成
スクリプトは全6,697字を走査するため重い。ここでは既知の小さな点列で数学的な
正しさ（距離総和・正規化・単調性）を固定し、回帰を防ぐ。
"""

from __future__ import annotations

import math

from src.layout.char_metrics import (
    char_ink_length,
    compute_complexity,
    normalize_robust,
    stroke_ink_length,
)


class TestStrokeInkLength:
    def test_single_point_has_zero_length(self) -> None:
        assert stroke_ink_length([(1.0, 2.0)]) == 0.0

    def test_empty_stroke_has_zero_length(self) -> None:
        assert stroke_ink_length([]) == 0.0

    def test_horizontal_segment(self) -> None:
        assert stroke_ink_length([(0.0, 0.0), (3.0, 0.0)]) == 3.0

    def test_pythagorean_triangle(self) -> None:
        # 3-4-5 直角三角形でユークリッド距離が正しいことを確認
        assert stroke_ink_length([(0.0, 0.0), (3.0, 4.0)]) == 5.0

    def test_multi_segment_sum(self) -> None:
        length = stroke_ink_length([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
        assert math.isclose(length, 2.0)

    def test_accepts_dict_points(self) -> None:
        # JSON 由来の {"x":..,"y":..} 形式も受け付ける
        pts = [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 5.0}]
        assert stroke_ink_length(pts) == 5.0


class TestCharInkLength:
    def test_empty_char(self) -> None:
        assert char_ink_length([]) == 0.0

    def test_sums_all_strokes(self) -> None:
        strokes = [
            [(0.0, 0.0), (2.0, 0.0)],  # 2.0
            [(0.0, 0.0), (0.0, 3.0)],  # 3.0
        ]
        assert math.isclose(char_ink_length(strokes), 5.0)


class TestNormalizeRobust:
    def test_within_percentile_range_maps_to_unit_interval(self) -> None:
        values = [float(i) for i in range(101)]  # 0..100
        norm = normalize_robust(values, low_pct=5.0, high_pct=95.0)
        assert all(0.0 <= v <= 1.0 for v in norm)

    def test_below_low_percentile_clamps_to_zero(self) -> None:
        values = [float(i) for i in range(101)]
        norm = normalize_robust(values, low_pct=5.0, high_pct=95.0)
        # 最小値(0)は5%境界以下なので0にクランプ
        assert norm[0] == 0.0

    def test_above_high_percentile_clamps_to_one(self) -> None:
        values = [float(i) for i in range(101)]
        norm = normalize_robust(values, low_pct=5.0, high_pct=95.0)
        assert norm[-1] == 1.0

    def test_monotonic_non_decreasing(self) -> None:
        values = [float(i) for i in range(101)]
        norm = normalize_robust(values, low_pct=5.0, high_pct=95.0)
        for a, b in zip(norm, norm[1:]):
            assert a <= b

    def test_constant_values_map_to_zero(self) -> None:
        # 分散ゼロ時は0除算を避け全て0を返す
        norm = normalize_robust([7.0, 7.0, 7.0], low_pct=5.0, high_pct=95.0)
        assert norm == [0.0, 0.0, 0.0]

    def test_empty_input(self) -> None:
        assert normalize_robust([], low_pct=5.0, high_pct=95.0) == []


class TestComputeComplexity:
    def test_weighted_average_of_normalized_metrics(self) -> None:
        # complexity = w_stroke*stroke_norm + w_ink*ink_norm
        c = compute_complexity(stroke_norm=1.0, ink_norm=0.0, w_stroke=0.5, w_ink=0.5)
        assert math.isclose(c, 0.5)

    def test_both_max(self) -> None:
        c = compute_complexity(stroke_norm=1.0, ink_norm=1.0, w_stroke=0.5, w_ink=0.5)
        assert math.isclose(c, 1.0)

    def test_both_min(self) -> None:
        c = compute_complexity(stroke_norm=0.0, ink_norm=0.0, w_stroke=0.5, w_ink=0.5)
        assert math.isclose(c, 0.0)

    def test_monotonic_in_stroke_count(self) -> None:
        # ink固定で画数正規化が増えると complexity も増える（単調性）
        low = compute_complexity(stroke_norm=0.2, ink_norm=0.5, w_stroke=0.5, w_ink=0.5)
        high = compute_complexity(stroke_norm=0.8, ink_norm=0.5, w_stroke=0.5, w_ink=0.5)
        assert high > low

    def test_monotonic_in_ink_length(self) -> None:
        low = compute_complexity(stroke_norm=0.5, ink_norm=0.2, w_stroke=0.5, w_ink=0.5)
        high = compute_complexity(stroke_norm=0.5, ink_norm=0.8, w_stroke=0.5, w_ink=0.5)
        assert high > low
