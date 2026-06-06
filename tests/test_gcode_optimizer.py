import numpy as np

from src.gcode.generator import Stroke
from src.gcode.optimizer import (
    calculate_travel_distance,
    optimize_stroke_order,
    optimize_stroke_order_with_finishes,
)


class TestOptimizeTravelDistance:
    """最適化による総移動距離の改善テスト"""

    def test_optimization_reduces_travel(self):
        strokes = [
            np.array([[100.0, 100.0], [110.0, 100.0]]),
            np.array([[0.0, 0.0], [10.0, 0.0]]),
            np.array([[200.0, 200.0], [210.0, 200.0]]),
        ]
        dist_before = calculate_travel_distance(strokes)
        optimized = optimize_stroke_order(strokes)
        dist_after = calculate_travel_distance(optimized)
        assert dist_after <= dist_before

    def test_optimization_with_scattered_strokes(self, scattered_strokes: list[Stroke]):
        dist_before = calculate_travel_distance(scattered_strokes)
        optimized = optimize_stroke_order(scattered_strokes)
        dist_after = calculate_travel_distance(optimized)
        assert dist_after <= dist_before


class TestStrokeReversal:
    """ストローク反転の動作確認"""

    def test_reverses_when_endpoint_is_closer(self):
        strokes = [
            np.array([[0.0, 0.0], [10.0, 0.0]]),
            np.array([[50.0, 0.0], [11.0, 0.0]]),
        ]
        optimized = optimize_stroke_order(strokes)
        assert len(optimized) == 2
        second = optimized[1]
        assert np.allclose(second[0], [11.0, 0.0]), "終点が近い場合、反転されるべき"

    def test_no_reverse_when_start_is_closer(self):
        strokes = [
            np.array([[0.0, 0.0], [10.0, 0.0]]),
            np.array([[11.0, 0.0], [50.0, 0.0]]),
        ]
        optimized = optimize_stroke_order(strokes)
        second = optimized[1]
        assert np.allclose(second[0], [11.0, 0.0]), "始点が近い場合、反転されないべき"


class TestOptimizeWithFinishes:
    """finishes 連動の並べ替えテスト"""

    def test_strokes_ordered_same_as_base(self):
        strokes = [
            np.array([[100.0, 100.0], [110.0, 100.0]]),
            np.array([[0.0, 0.0], [10.0, 0.0]]),
            np.array([[200.0, 200.0], [210.0, 200.0]]),
        ]
        finishes = ["harai", "hane", "tome"]
        base = optimize_stroke_order(strokes)
        ordered, ordered_finishes = optimize_stroke_order_with_finishes(strokes, finishes)
        # strokes は base と同順
        assert len(ordered) == len(base)
        for a, b in zip(ordered, base, strict=True):
            assert np.array_equal(a, b)

    def test_finishes_track_stroke_permutation(self):
        # 既知の最適化結果に対し finish の対応が保たれること
        strokes = [
            np.array([[100.0, 100.0], [110.0, 100.0]]),
            np.array([[0.0, 0.0], [10.0, 0.0]]),
            np.array([[200.0, 200.0], [210.0, 200.0]]),
        ]
        finishes = ["harai", "hane", "tome"]
        ordered, ordered_finishes = optimize_stroke_order_with_finishes(strokes, finishes)
        assert len(ordered_finishes) == len(ordered)
        # start_pos=(0,0) に最も近いストローク[1](finish=hane)が先頭
        assert ordered_finishes[0] == "hane"
        assert np.array_equal(ordered[0], strokes[1])
        # 各 finish が並べ替え後のストロークに正しく対応
        finish_by_first_point = {
            (0.0, 0.0): "hane",
            (100.0, 100.0): "harai",
            (200.0, 200.0): "tome",
        }
        for stroke, finish in zip(ordered, ordered_finishes, strict=True):
            key = (float(stroke[0][0]), float(stroke[0][1]))
            assert finish_by_first_point[key] == finish

    def test_reversed_stroke_keeps_its_finish(self):
        strokes = [
            np.array([[0.0, 0.0], [10.0, 0.0]]),
            np.array([[50.0, 0.0], [11.0, 0.0]]),
        ]
        finishes = ["tome", "harai"]
        ordered, ordered_finishes = optimize_stroke_order_with_finishes(strokes, finishes)
        # 2本目は反転されるが finish は元の harai のまま
        assert np.allclose(ordered[1][0], [11.0, 0.0])
        assert ordered_finishes[1] == "harai"

    def test_short_finishes_padded_with_none(self):
        strokes = [
            np.array([[0.0, 0.0], [10.0, 0.0]]),
            np.array([[50.0, 0.0], [60.0, 0.0]]),
        ]
        finishes = ["tome"]  # 不足
        ordered, ordered_finishes = optimize_stroke_order_with_finishes(strokes, finishes)
        assert len(ordered_finishes) == len(ordered)
        assert set(ordered_finishes) <= {"tome", "none"}
        assert "none" in ordered_finishes

    def test_none_finishes_defaults_to_all_none(self):
        strokes = [
            np.array([[0.0, 0.0], [10.0, 0.0]]),
            np.array([[50.0, 0.0], [60.0, 0.0]]),
        ]
        ordered, ordered_finishes = optimize_stroke_order_with_finishes(strokes, None)
        assert ordered_finishes == ["none", "none"]

    def test_empty(self):
        ordered, ordered_finishes = optimize_stroke_order_with_finishes([], [])
        assert ordered == []
        assert ordered_finishes == []

    def test_single_stroke(self):
        stroke = np.array([[0.0, 0.0], [10.0, 10.0]])
        ordered, ordered_finishes = optimize_stroke_order_with_finishes([stroke], ["harai"])
        assert len(ordered) == 1
        assert np.array_equal(ordered[0], stroke)
        assert ordered_finishes == ["harai"]


class TestEdgeCases:
    def test_empty_list(self):
        result = optimize_stroke_order([])
        assert result == []

    def test_single_stroke(self):
        stroke = np.array([[0.0, 0.0], [10.0, 10.0]])
        result = optimize_stroke_order([stroke])
        assert len(result) == 1
        assert np.array_equal(result[0], stroke)

    def test_empty_travel_distance(self):
        result = calculate_travel_distance([])
        assert result == 0.0
