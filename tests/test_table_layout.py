import numpy as np
import pytest
from src.layout.table_layout import TableConfig, TableLayout, CellPlacement


class TestTableConfig:
    def test_defaults(self):
        cfg = TableConfig(rows=3, cols=4)
        assert cfg.rows == 3
        assert cfg.cols == 4
        assert cfg.cell_padding > 0
        assert cfg.row_height > 0

    def test_custom_col_widths(self):
        cfg = TableConfig(rows=2, cols=3, col_widths=[20.0, 30.0, 40.0])
        assert cfg.col_widths == [20.0, 30.0, 40.0]


class TestTableLayout:
    def test_border_strokes(self):
        cfg = TableConfig(rows=2, cols=3, col_widths=[20.0, 30.0, 25.0], row_height=10.0)
        layout = TableLayout(cfg, origin_x=0, origin_y=0)
        strokes = layout.border_strokes()
        assert len(strokes) > 0
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.shape == (2, 2)  # 各罫線は2点の直線

    def test_horizontal_lines_count(self):
        cfg = TableConfig(rows=3, cols=2, row_height=10.0, col_widths=[20.0, 20.0])
        layout = TableLayout(cfg, origin_x=0, origin_y=0)
        h_strokes = layout.horizontal_strokes()
        assert len(h_strokes) == 4  # rows+1 = 4本

    def test_vertical_lines_count(self):
        cfg = TableConfig(rows=3, cols=2, row_height=10.0, col_widths=[20.0, 20.0])
        layout = TableLayout(cfg, origin_x=0, origin_y=0)
        v_strokes = layout.vertical_strokes()
        assert len(v_strokes) == 3  # cols+1 = 3本

    def test_cell_placement(self):
        cfg = TableConfig(rows=2, cols=2, row_height=10.0, col_widths=[30.0, 40.0], cell_padding=2.0)
        layout = TableLayout(cfg, origin_x=5.0, origin_y=10.0)
        placement = layout.cell_position(row=0, col=0)
        assert isinstance(placement, CellPlacement)
        assert placement.x >= 5.0 + 2.0  # origin + padding
        assert placement.y >= 10.0
        assert placement.width == 30.0 - 2 * 2.0  # col_width - 2*padding
        assert placement.height == 10.0 - 2 * 2.0

    def test_cell_out_of_range(self):
        cfg = TableConfig(rows=2, cols=2, row_height=10.0, col_widths=[20.0, 20.0])
        layout = TableLayout(cfg, origin_x=0, origin_y=0)
        with pytest.raises(IndexError):
            layout.cell_position(row=5, col=0)

    def test_total_dimensions(self):
        cfg = TableConfig(rows=3, cols=2, row_height=10.0, col_widths=[20.0, 30.0])
        layout = TableLayout(cfg, origin_x=0, origin_y=0)
        assert layout.total_width() == 50.0
        assert layout.total_height() == 30.0
