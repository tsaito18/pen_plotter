"""用紙・余白・罫線の幾何（座標は mm・Y-UP、左下が原点）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry import Stroke


@dataclass
class PageConfig:
    paper_size: tuple[float, float] = (210.0, 297.0)
    margin_top: float = 25.0
    margin_bottom: float = 15.0
    margin_left: float = 20.0
    margin_right: float = 15.0
    line_spacing: float = 8.0


@dataclass
class ContentArea:
    x: float
    y: float
    width: float
    height: float


class PageLayout:
    def __init__(self, config: PageConfig) -> None:
        self._config = config

    def content_area(self) -> ContentArea:
        cfg = self._config
        return ContentArea(
            x=cfg.margin_left,
            y=cfg.margin_bottom,
            width=cfg.paper_size[0] - cfg.margin_left - cfg.margin_right,
            height=cfg.paper_size[1] - cfg.margin_top - cfg.margin_bottom,
        )

    def line_positions(self) -> list[float]:
        """罫線（＝各行ボックスの下端）の y 座標を上から順に返す。"""
        if self._config.line_spacing <= 0:
            return []
        area = self.content_area()
        spacing = self._config.line_spacing
        positions: list[float] = []
        y = area.y + area.height
        while y >= area.y:
            positions.append(y)
            y -= spacing
        return positions

    def ruled_line_strokes(self) -> list[Stroke]:
        """プレビュー背景用の罫線（用紙の全幅）。"""
        width = self._config.paper_size[0]
        return [np.array([[0.0, y], [width, y]]) for y in self.line_positions()]
