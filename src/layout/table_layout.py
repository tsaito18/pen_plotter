from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
                f"Cell ({row}, {col}) is out of range for "
                f"{self._cfg.rows}x{self._cfg.cols} table"
            )
        pad = self._cfg.cell_padding
        x = self._ox + sum(self._cfg.col_widths[:col]) + pad  # type: ignore[index]
        y = self._oy + row * self._cfg.row_height + pad
        w = self._cfg.col_widths[col] - 2 * pad  # type: ignore[index]
        h = self._cfg.row_height - 2 * pad
        return CellPlacement(x=x, y=y, width=w, height=h)
