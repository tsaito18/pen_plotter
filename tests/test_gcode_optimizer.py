import numpy as np

from src.gcode.generator import Stroke
from src.gcode.optimizer import calculate_travel_distance, optimize_stroke_order


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
