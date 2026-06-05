import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.gcode.generator import Stroke
from src.gcode.preview import preview_gcode, preview_strokes


def _line_stroke_mm(length: float, n_points: int) -> np.ndarray:
    """長さ length mm の水平直線を n_points 点で。距離mmベースのテスト用。"""
    xs = np.linspace(0.0, length, n_points)
    return np.column_stack([xs, np.zeros_like(xs)])


class TestPreviewStrokes:
    def test_saves_image_file(self, tmp_path: Path, square_stroke: Stroke):
        save_path = tmp_path / "preview_strokes.png"
        preview_strokes([square_stroke], save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPreviewGcode:
    def test_saves_image_file(self, tmp_path: Path, sample_gcode: list[str]):
        save_path = tmp_path / "preview_gcode.png"
        preview_gcode(sample_gcode, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPreviewStrokesVaryWidth:
    """ストローク太さ変化のテスト"""

    def test_vary_width_default_on_saves_image(self, tmp_path: Path, square_stroke: Stroke):
        """vary_width=True（デフォルト）で画像保存できる"""
        save_path = tmp_path / "vary_width_on.png"
        preview_strokes([square_stroke], save_path=save_path, vary_width=True)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_vary_width_off_saves_image(self, tmp_path: Path, square_stroke: Stroke):
        """vary_width=Falseで画像保存できる（従来動作）"""
        save_path = tmp_path / "vary_width_off.png"
        preview_strokes([square_stroke], save_path=save_path, vary_width=False)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_vary_width_uses_line_collection(self, tmp_path: Path, square_stroke: Stroke):
        """vary_width=Trueの場合、LineCollectionが使われる"""
        save_path = tmp_path / "lc.png"
        with patch("src.gcode.preview.LineCollection", wraps=None) as mock_lc:
            from matplotlib.collections import LineCollection

            mock_lc.side_effect = LineCollection
            preview_strokes([square_stroke], save_path=save_path, vary_width=True)
            assert mock_lc.called

    def test_vary_width_linewidths_decrease(self, tmp_path: Path):
        """none は実機Z一定に連動して幅一定、払いは終端で細る。"""
        from src.gcode.preview import compute_stroke_widths

        stroke = _line_stroke_mm(16.0, 11)
        # 終端Zリフトの無い none は接触一定＝幅一定（実機の単線に忠実）
        widths = compute_stroke_widths(stroke)
        assert len(widths) == len(stroke) - 1
        assert all(abs(w - widths[0]) < 1e-9 for w in widths)
        # 払いは終端で細る（接触圧が抜ける）
        harai = compute_stroke_widths(stroke, "harai")
        assert harai[0] > harai[-1]

    def test_vary_width_with_single_point_stroke(self, tmp_path: Path):
        """1点ストロークでもクラッシュしない"""
        single = np.array([[10.0, 10.0]])
        save_path = tmp_path / "single.png"
        preview_strokes([single], save_path=save_path, vary_width=True)
        assert save_path.exists()

    def test_vary_width_with_two_point_stroke(self, tmp_path: Path, line_stroke: Stroke):
        """2点ストローク（1セグメント）でも動作する"""
        save_path = tmp_path / "two_point.png"
        preview_strokes([line_stroke], save_path=save_path, vary_width=True)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestComputeStrokeWidthsFinish:
    """筆画タイプ別の太さプロファイルのテスト（距離mmベース）。"""

    def test_default_finish_matches_none(self):
        """finish引数省略時は finish='none' と一致する（後方互換）。"""
        from src.gcode.preview import compute_stroke_widths

        s = _line_stroke_mm(16.0, 9)
        assert compute_stroke_widths(s) == compute_stroke_widths(s, "none")

    def test_short_stroke_returns_empty_all_finishes(self):
        """点数2未満はどの finish でも空リスト。"""
        from src.gcode.preview import compute_stroke_widths

        one = np.array([[0.0, 0.0]])
        for finish in ("none", "tome", "hane", "harai"):
            assert compute_stroke_widths(one, finish) == []

    def test_harai_thinner_at_end_than_none(self):
        """払い: 終端が none より細い（強く細くなる）。"""
        from src.gcode.preview import compute_stroke_widths

        s = _line_stroke_mm(16.0, 17)
        harai = compute_stroke_widths(s, "harai")
        none = compute_stroke_widths(s, "none")
        assert harai[-1] < none[-1]

    def test_harai_end_less_than_start(self):
        """払い: 末尾 < 始点。"""
        from src.gcode.preview import compute_stroke_widths

        harai = compute_stroke_widths(_line_stroke_mm(16.0, 17), "harai")
        assert harai[-1] < harai[0]

    def test_hane_end_less_than_start(self):
        """はね: 末尾 < 始点（細め）。"""
        from src.gcode.preview import compute_stroke_widths

        hane = compute_stroke_widths(_line_stroke_mm(16.0, 17), "hane")
        assert hane[-1] < hane[0]

    def test_tome_constant(self):
        """とめ: 全要素が一定値 0.9（接触一定）。"""
        from src.gcode.preview import compute_stroke_widths

        tome = compute_stroke_widths(_line_stroke_mm(16.0, 17), "tome")
        assert len(tome) == 16
        assert all(abs(w - 0.9) < 1e-9 for w in tome)

    def test_size_independent_taper(self):
        """同じ長さmmなら点数が違っても、終端から同距離の太さが一致（サイズ非依存）。"""
        from src.gcode.preview import compute_stroke_widths

        def width_at(length, n, d):
            s = _line_stroke_mm(length, n)
            w = compute_stroke_widths(s, "harai")
            # 各セグメント中点の終端からの距離
            xs = np.linspace(0.0, length, n)
            mid_from_end = length - (xs[:-1] + xs[1:]) / 2.0
            return float(np.interp(d, mid_from_end[::-1], np.array(w)[::-1]))

        coarse = width_at(16.0, 9, 1.25)
        fine = width_at(16.0, 33, 1.25)
        assert abs(coarse - fine) < 0.05  # 同じmm距離なら点数によらず一致

    def test_unknown_finish_falls_back_to_none(self):
        """未知の finish は none と同じプロファイル。"""
        from src.gcode.preview import compute_stroke_widths

        s = _line_stroke_mm(16.0, 11)
        assert compute_stroke_widths(s, "bogus") == compute_stroke_widths(s, "none")
