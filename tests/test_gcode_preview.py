import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.gcode.generator import Stroke
from src.gcode.preview import preview_gcode, preview_strokes


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

        n_segments = 10
        # 終端Zリフトの無い none は接触一定＝幅一定（実機の単線に忠実）
        widths = compute_stroke_widths(n_segments)
        assert len(widths) == n_segments
        assert all(abs(w - widths[0]) < 1e-9 for w in widths)
        # 払いは終端で細る（接触圧が抜ける）
        harai = compute_stroke_widths(n_segments, "harai")
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
    """筆画タイプ別の太さプロファイルのテスト。"""

    def test_default_finish_matches_none(self):
        """finish引数省略時は finish='none' と一致する（後方互換）。"""
        from src.gcode.preview import compute_stroke_widths

        n = 8
        assert compute_stroke_widths(n) == compute_stroke_widths(n, "none")

    def test_zero_segments_returns_empty_all_finishes(self):
        """n_segments<=0 はどの finish でも空リスト。"""
        from src.gcode.preview import compute_stroke_widths

        for finish in ("none", "tome", "hane", "harai"):
            assert compute_stroke_widths(0, finish) == []
            assert compute_stroke_widths(-3, finish) == []

    def test_single_segment_uses_midpoint_all_finishes(self):
        """n_segments==1 は t=[0.5] で1要素を返す。"""
        from src.gcode.preview import compute_stroke_widths

        for finish in ("none", "tome", "hane", "harai"):
            widths = compute_stroke_widths(1, finish)
            assert len(widths) == 1

    def test_harai_thinner_at_end_than_none(self):
        """払い: 終端が none より細い（強く細くなる）。"""
        from src.gcode.preview import compute_stroke_widths

        n = 16
        harai = compute_stroke_widths(n, "harai")
        none = compute_stroke_widths(n, "none")
        assert harai[-1] < none[-1]

    def test_harai_end_less_than_start(self):
        """払い: 末尾 < 始点。"""
        from src.gcode.preview import compute_stroke_widths

        harai = compute_stroke_widths(16, "harai")
        assert harai[-1] < harai[0]

    def test_hane_end_less_than_start(self):
        """はね: 末尾 < 始点（細め）。"""
        from src.gcode.preview import compute_stroke_widths

        hane = compute_stroke_widths(16, "hane")
        assert hane[-1] < hane[0]

    def test_tome_constant(self):
        """とめ: 全要素が一定値 0.9。"""
        from src.gcode.preview import compute_stroke_widths

        tome = compute_stroke_widths(16, "tome")
        assert len(tome) == 16
        assert all(abs(w - 0.9) < 1e-9 for w in tome)

    def test_unknown_finish_falls_back_to_none(self):
        """未知の finish は none と同じプロファイル。"""
        from src.gcode.preview import compute_stroke_widths

        n = 10
        assert compute_stroke_widths(n, "bogus") == compute_stroke_widths(n, "none")
