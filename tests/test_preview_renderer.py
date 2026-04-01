"""PreviewRenderer の単体テスト。"""
import numpy as np
import pytest

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
