"""PreviewRenderer の単体テスト。"""

import numpy as np

from src.gcode.config import PlotterConfig
from src.layout.page_layout import PageConfig
from src.ui.preview_renderer import PreviewRenderer


class TestPreviewRendererInit:
    def test_creates_with_configs(self):
        renderer = PreviewRenderer(
            plotter_config=PlotterConfig(),
            page_config=PageConfig(),
        )
        assert renderer._report_bg_path is None

    def test_creates_with_bg_path(self, tmp_path):
        bg = tmp_path / "bg.png"
        bg.write_bytes(b"fake")
        renderer = PreviewRenderer(
            plotter_config=PlotterConfig(),
            page_config=PageConfig(),
            report_bg_path=bg,
        )
        assert renderer._report_bg_path == bg


class TestPreviewWithRuledLines:
    def test_saves_image(self, tmp_path):
        renderer = PreviewRenderer(
            plotter_config=PlotterConfig(),
            page_config=PageConfig(),
        )
        save_path = tmp_path / "test.png"
        strokes = [np.array([[10.0, 10.0], [20.0, 20.0]])]
        renderer.preview_with_ruled_lines(strokes, [], save_path)
        assert save_path.exists()

    def test_page_number_strokes(self, tmp_path):
        """page_number_strokes が渡された場合に画像が生成される。"""
        renderer = PreviewRenderer(
            plotter_config=PlotterConfig(),
            page_config=PageConfig(),
        )
        save_path = tmp_path / "pn.png"
        pn_strokes = [np.array([[20.0, 5.0], [25.0, 5.0]])]
        renderer.preview_with_ruled_lines(
            [], [], save_path, page_number=5, page_number_strokes=pn_strokes
        )
        assert save_path.exists()

    def test_finishes_none_default_saves_image(self, tmp_path):
        """finishes 省略（None）で従来通り画像が生成される。"""
        renderer = PreviewRenderer(
            plotter_config=PlotterConfig(),
            page_config=PageConfig(),
        )
        save_path = tmp_path / "no_finishes.png"
        strokes = [
            np.array([[10.0, 10.0], [20.0, 20.0]]),
            np.array([[30.0, 30.0], [40.0, 40.0]]),
        ]
        renderer.preview_with_ruled_lines(strokes, [], save_path, finishes=None)
        assert save_path.exists()

    def test_finishes_passed_saves_image(self, tmp_path):
        """finishes を渡しても落ちず画像が生成される。"""
        renderer = PreviewRenderer(
            plotter_config=PlotterConfig(),
            page_config=PageConfig(),
        )
        save_path = tmp_path / "with_finishes.png"
        strokes = [
            np.array([[10.0, 10.0], [20.0, 20.0]]),
            np.array([[30.0, 30.0], [40.0, 40.0]]),
        ]
        finishes = ["harai", "tome"]
        renderer.preview_with_ruled_lines(strokes, [], save_path, finishes=finishes)
        assert save_path.exists()

    def test_finishes_length_mismatch_no_indexerror(self, tmp_path):
        """finishes が strokes より短くても IndexError にならない。"""
        renderer = PreviewRenderer(
            plotter_config=PlotterConfig(),
            page_config=PageConfig(),
        )
        save_path = tmp_path / "mismatch.png"
        strokes = [
            np.array([[10.0, 10.0], [20.0, 20.0]]),
            np.array([[30.0, 30.0], [40.0, 40.0]]),
            np.array([[50.0, 50.0], [60.0, 60.0]]),
        ]
        finishes = ["harai"]  # strokes より短い
        renderer.preview_with_ruled_lines(strokes, [], save_path, finishes=finishes)
        assert save_path.exists()

    def test_finishes_passed_to_draw_stroke(self, tmp_path):
        """finishes の各要素が _draw_stroke_with_width に finish として渡る。"""
        from unittest.mock import patch

        renderer = PreviewRenderer(
            plotter_config=PlotterConfig(),
            page_config=PageConfig(),
        )
        save_path = tmp_path / "dispatch.png"
        strokes = [
            np.array([[10.0, 10.0], [20.0, 20.0]]),
            np.array([[30.0, 30.0], [40.0, 40.0]]),
        ]
        finishes = ["harai", "tome"]
        with patch("src.ui.preview_renderer._draw_stroke_with_width") as mock_draw:
            renderer.preview_with_ruled_lines(strokes, [], save_path, finishes=finishes)
        passed = [call.kwargs.get("finish") for call in mock_draw.call_args_list]
        assert "harai" in passed
        assert "tome" in passed
