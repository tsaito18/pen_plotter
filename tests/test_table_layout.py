import numpy as np
import pytest
from src.layout.table_layout import (
    CellPlacement,
    TableConfig,
    TableLayout,
    detect_pipe_table,
    is_table_separator,
    split_pipe_row,
)


class TestPipeTableParser:
    """Markdownパイプ表のパース。"""

    def test_split_pipe_row(self):
        assert split_pipe_row("| 項目 | 値 |") == ["項目", "値"]
        assert split_pipe_row("項目 | 値") == ["項目", "値"]  # 端の | は任意
        assert split_pipe_row("| a |  | c |") == ["a", "", "c"]

    def test_is_table_separator(self):
        assert is_table_separator("|---|---|")
        assert is_table_separator("| :--- | ---: |")
        assert is_table_separator("|:-:|:-:|")
        assert not is_table_separator("| a | b |")
        assert not is_table_separator("| --- | x |")  # データ混在は区切りでない

    def test_detect_pipe_table_basic(self):
        paras = ["前文", "| 項目 | 値 |", "|---|---|", "| 降伏 | 235 |", "| 引張 | 400 |", "後文"]
        result = detect_pipe_table(paras, 1)
        assert result is not None
        rows, consumed = result
        assert rows == [["項目", "値"], ["降伏", "235"], ["引張", "400"]]
        assert consumed == 4  # ヘッダ+区切り+データ2行

    def test_detect_requires_separator(self):
        # 2行目が区切りでなければ表として認識しない
        assert detect_pipe_table(["| a | b |", "| c | d |"], 0) is None

    def test_detect_non_table_line(self):
        assert detect_pipe_table(["ふつうの文", "次"], 0) is None

    def test_detect_ragged_cells_padded(self):
        # 列数が揃わない行は最大列数にパディング
        paras = ["| a | b | c |", "|---|---|---|", "| 1 | 2 |"]
        rows, consumed = detect_pipe_table(paras, 0)
        assert rows[0] == ["a", "b", "c"]
        assert rows[1] == ["1", "2", ""]
        assert consumed == 3


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
        cfg = TableConfig(
            rows=2, cols=2, row_height=10.0, col_widths=[30.0, 40.0], cell_padding=2.0
        )
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
