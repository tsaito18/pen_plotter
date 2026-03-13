import numpy as np
import pytest
from src.layout.page_layout import PageConfig, PageLayout, PaperSize


class TestPaperSize:
    def test_a4_dimensions(self):
        assert PaperSize.A4 == (210.0, 297.0)

    def test_b5_dimensions(self):
        assert PaperSize.B5 == (182.0, 257.0)


class TestPageConfig:
    def test_default_config(self):
        cfg = PageConfig()
        assert cfg.paper_size == PaperSize.A4
        assert cfg.margin_top > 0
        assert cfg.margin_bottom > 0
        assert cfg.margin_left > 0
        assert cfg.margin_right > 0
        assert cfg.line_spacing == 8.0

    def test_custom_margins(self):
        cfg = PageConfig(margin_top=20.0, margin_left=25.0)
        assert cfg.margin_top == 20.0
        assert cfg.margin_left == 25.0


class TestPageLayout:
    def test_content_area(self):
        cfg = PageConfig()
        layout = PageLayout(cfg)
        area = layout.content_area()
        assert area.x >= cfg.margin_left
        assert area.y >= cfg.margin_bottom
        assert area.width > 0
        assert area.height > 0
        assert area.x + area.width <= cfg.paper_size[0] - cfg.margin_right
        assert area.y + area.height <= cfg.paper_size[1] - cfg.margin_top

    def test_line_positions(self):
        cfg = PageConfig(line_spacing=8.0)
        layout = PageLayout(cfg)
        lines = layout.line_positions()
        assert len(lines) > 0
        for i in range(1, len(lines)):
            assert abs(lines[i] - lines[i-1]) == pytest.approx(8.0)

    def test_line_positions_within_content_area(self):
        layout = PageLayout(PageConfig())
        area = layout.content_area()
        lines = layout.line_positions()
        for y in lines:
            assert y >= area.y
            assert y <= area.y + area.height

    def test_ruled_line_strokes(self):
        layout = PageLayout(PageConfig())
        strokes = layout.ruled_line_strokes()
        assert len(strokes) > 0
        area = layout.content_area()
        for stroke in strokes:
            assert isinstance(stroke, np.ndarray)
            assert stroke.shape == (2, 2)  # 2点の直線
            assert stroke[0, 0] == pytest.approx(area.x)  # 左端から
            assert stroke[1, 0] == pytest.approx(area.x + area.width)  # 右端まで

    def test_b5_has_smaller_area(self):
        a4 = PageLayout(PageConfig(paper_size=PaperSize.A4))
        b5 = PageLayout(PageConfig(paper_size=PaperSize.B5))
        assert b5.content_area().width < a4.content_area().width
        assert b5.content_area().height < a4.content_area().height

    def test_no_ruled_lines_when_spacing_zero(self):
        cfg = PageConfig(line_spacing=0.0)
        layout = PageLayout(cfg)
        strokes = layout.ruled_line_strokes()
        assert len(strokes) == 0
