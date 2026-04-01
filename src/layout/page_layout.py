from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PaperSize:
    A4: tuple[float, float] = (210.0, 297.0)
    B5: tuple[float, float] = (182.0, 257.0)


@dataclass
class PageConfig:
    paper_size: tuple[float, float] = PaperSize.A4
    margin_top: float = 25.0
    margin_bottom: float = 15.0
    margin_left: float = 20.0
    margin_right: float = 15.0
    line_spacing: float = 8.0
    header_height: float = 0.0
    footer_height: float = 0.0


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
        x = cfg.margin_left
        y = cfg.margin_bottom + cfg.footer_height
        width = cfg.paper_size[0] - cfg.margin_left - cfg.margin_right
        height = cfg.paper_size[1] - cfg.margin_top - cfg.margin_bottom - cfg.header_height - cfg.footer_height
        return ContentArea(x=x, y=y, width=width, height=height)

    def line_positions(self) -> list[float]:
        if self._config.line_spacing <= 0:
            return []
        area = self.content_area()
        top = area.y + area.height
        spacing = self._config.line_spacing
        positions: list[float] = []
        y = top
        while y >= area.y:
            positions.append(y)
            y -= spacing
        return positions

    def ruled_line_strokes(self) -> list[np.ndarray]:
        """罫線ストロークを生成。罫線は用紙のほぼ全幅に渡る。"""
        lines = self.line_positions()
        if not lines:
            return []
        pw = self._config.paper_size[0]
        x_left = 0.0
        x_right = pw
        return [
            np.array([[x_left, y], [x_right, y]])
            for y in lines
        ]
