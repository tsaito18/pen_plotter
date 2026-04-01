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

    def test_page_number(self, tmp_path):
        from unittest.mock import patch

        renderer = PreviewRenderer(
            plotter_config=PlotterConfig(),
            page_config=PageConfig(),
        )
        save_path = tmp_path / "pn.png"
        with patch("matplotlib.axes.Axes.text") as mock_text:
            renderer.preview_with_ruled_lines([], [], save_path, page_number=5)
            mock_text.assert_called_once()
            assert "P. 5" in str(mock_text.call_args)
