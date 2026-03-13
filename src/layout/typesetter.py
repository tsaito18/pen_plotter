from __future__ import annotations

from dataclasses import dataclass, field

from src.layout.line_breaking import break_lines, is_halfwidth
from src.layout.page_layout import PageConfig, PageLayout


@dataclass
class CharPlacement:
    char: str
    x: float
    y: float
    font_size: float
    page: int = 0


class Typesetter:
    def __init__(self, page_config: PageConfig, font_size: float | None = None) -> None:
        self._config = page_config
        self._layout = PageLayout(page_config)
        self.font_size = font_size if font_size is not None else page_config.line_spacing * 0.9

    def typeset(self, text: str) -> list[list[CharPlacement]]:
        area = self._layout.content_area()
        line_positions = self._layout.line_positions()

        if not text:
            return [[]]

        chars_per_line = int(area.width / self.font_size)
        lines = break_lines(text, chars_per_line)

        pages: list[list[CharPlacement]] = []
        current_page: list[CharPlacement] = []
        page_idx = 0
        line_idx = 0

        for line_text in lines:
            if line_idx >= len(line_positions):
                pages.append(current_page)
                current_page = []
                page_idx += 1
                line_idx = 0

            y = line_positions[line_idx]
            x = area.x
            for ch in line_text:
                if ch == "\n":
                    continue
                char_width = self.font_size * 0.5 if is_halfwidth(ch) else self.font_size
                current_page.append(CharPlacement(
                    char=ch, x=x, y=y, font_size=self.font_size, page=page_idx,
                ))
                x += char_width

            line_idx += 1

        pages.append(current_page)
        return pages
