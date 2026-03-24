import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.gcode.generator import GCodeGenerator, Stroke
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
        """太さが始点で太く終点で細い（減衰する）"""
        from src.gcode.preview import compute_stroke_widths

        n_segments = 10
        widths = compute_stroke_widths(n_segments)
        assert len(widths) == n_segments
        assert widths[0] > widths[-1]
        assert widths[0] > 0.9
        assert widths[-1] < 0.8

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
