from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# パイプ表の区切り行セル（---, :--, --:, :-: のような形）。
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{1,}:?$")


def split_pipe_row(line: str) -> list[str]:
    """Markdown パイプ表の 1 行をセル文字列のリストへ分解する。

    端の ``|`` は任意。``"| a | b |"`` も ``"a | b"`` も ``["a", "b"]``。各セルは
    前後空白を除去する。
    """
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [cell.strip() for cell in s.split("|")]


def is_table_separator(line: str) -> bool:
    """行が表の区切り行（``|---|---|`` 等）かを判定する。

    ``|`` を含み、全セルが ``-`` 主体（``:`` 揃え指定可）であること。
    """
    if "|" not in line:
        return False
    cells = split_pipe_row(line)
    if not cells or any(c == "" for c in cells):
        return False
    return all(_TABLE_SEP_CELL_RE.match(c) is not None for c in cells)


def detect_pipe_table(paragraphs: list[str], start: int) -> tuple[list[list[str]], int] | None:
    """``paragraphs[start]`` から始まるパイプ表を検出して (行データ, 消費行数) を返す。

    表の条件: ``start`` 行がパイプ行、``start+1`` 行が区切り行。以降 ``|`` を含む行を
    データ行として取り込む。行データはヘッダ＋データ（区切りは除く）。列数が揃わない
    行は最大列数まで空セルでパディングする。表でなければ ``None``。

    Returns:
        ``(rows, consumed)``。``rows`` は ``list[list[str]]``（先頭がヘッダ）、
        ``consumed`` は表が占める段落数（区切り行を含む）。
    """
    n = len(paragraphs)
    if start + 1 >= n:
        return None
    header = paragraphs[start]
    if "|" not in header or not is_table_separator(paragraphs[start + 1]):
        return None
    rows: list[list[str]] = [split_pipe_row(header)]
    consumed = 2  # ヘッダ + 区切り
    i = start + 2
    while i < n and "|" in paragraphs[i] and not is_table_separator(paragraphs[i]):
        rows.append(split_pipe_row(paragraphs[i]))
        consumed += 1
        i += 1
    ncols = max(len(r) for r in rows)
    for r in rows:
        if len(r) < ncols:
            r.extend([""] * (ncols - len(r)))
    return rows, consumed


@dataclass
class TableConfig:
    rows: int
    cols: int
    col_widths: list[float] | None = None
    row_height: float = 8.0
    default_col_width: float = 25.0
    cell_padding: float = 1.5

    def __post_init__(self) -> None:
        if self.col_widths is None:
            self.col_widths = [self.default_col_width] * self.cols


@dataclass
class CellPlacement:
    x: float
    y: float
    width: float
    height: float


class TableLayout:
    def __init__(self, config: TableConfig, origin_x: float, origin_y: float) -> None:
        self._cfg = config
        self._ox = origin_x
        self._oy = origin_y

    def total_width(self) -> float:
        return sum(self._cfg.col_widths)  # type: ignore[arg-type]

    def total_height(self) -> float:
        return self._cfg.rows * self._cfg.row_height

    def horizontal_strokes(self) -> list[np.ndarray]:
        w = self.total_width()
        strokes: list[np.ndarray] = []
        for i in range(self._cfg.rows + 1):
            y = self._oy + i * self._cfg.row_height
            strokes.append(np.array([[self._ox, y], [self._ox + w, y]]))
        return strokes

    def vertical_strokes(self) -> list[np.ndarray]:
        h = self.total_height()
        strokes: list[np.ndarray] = []
        x = self._ox
        for i in range(self._cfg.cols + 1):
            strokes.append(np.array([[x, self._oy], [x, self._oy + h]]))
            if i < self._cfg.cols:
                x += self._cfg.col_widths[i]  # type: ignore[index]
        return strokes

    def border_strokes(self) -> list[np.ndarray]:
        return self.horizontal_strokes() + self.vertical_strokes()

    def cell_position(self, row: int, col: int) -> CellPlacement:
        if row < 0 or row >= self._cfg.rows or col < 0 or col >= self._cfg.cols:
            raise IndexError(
                f"Cell ({row}, {col}) is out of range for {self._cfg.rows}x{self._cfg.cols} table"
            )
        pad = self._cfg.cell_padding
        x = self._ox + sum(self._cfg.col_widths[:col]) + pad  # type: ignore[index]
        y = self._oy + row * self._cfg.row_height + pad
        w = self._cfg.col_widths[col] - 2 * pad  # type: ignore[index]
        h = self._cfg.row_height - 2 * pad
        return CellPlacement(x=x, y=y, width=w, height=h)
