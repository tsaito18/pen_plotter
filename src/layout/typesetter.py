from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.layout.line_breaking import break_lines, break_paragraph, is_halfwidth
from src.layout.page_layout import PageConfig, PageLayout

_SMALL_KANA = set('っゃゅょぁぃぅぇぉァィゥェォッャュョヵヶ')
_SMALL_PUNCT = set('・、。，．')


def _char_size_scale(ch: str) -> float:
    """文字種別に応じたサイズスケールを返す。"""
    cp = ord(ch)
    if ch in _SMALL_KANA:
        return 0.55
    if ch in _SMALL_PUNCT:
        return 0.35
    if 0x3040 <= cp <= 0x309F:
        return 0.88
    if 0x30A0 <= cp <= 0x30FF:
        return 0.88
    if is_halfwidth(ch):
        return 0.6
    return 1.0


if TYPE_CHECKING:
    from src.model.augmentation import HandwritingAugmenter


@dataclass
class CharPlacement:
    char: str
    x: float
    y: float
    font_size: float
    page: int = 0


class Typesetter:
    def __init__(
        self,
        page_config: PageConfig,
        font_size: float | None = None,
        augmenter: HandwritingAugmenter | None = None,
    ) -> None:
        self._config = page_config
        self._layout = PageLayout(page_config)
        self.font_size = font_size if font_size is not None else page_config.line_spacing * 0.9
        self._augmenter = augmenter

    @property
    def augmenter(self) -> HandwritingAugmenter | None:
        return self._augmenter

    def typeset(self, text: str) -> list[list[CharPlacement]]:
        area = self._layout.content_area()
        line_positions = self._layout.line_positions()

        if not text:
            return [[]]

        chars_per_line = int(area.width / self.font_size)

        # 段落ごとに改行し、段落先頭行のインデックスを記録
        paragraphs = text.split("\n")
        lines: list[str] = []
        para_start_indices: set[int] = set()
        for para in paragraphs:
            para_start_indices.add(len(lines))
            if not para:
                lines.append("")
            else:
                lines.extend(break_paragraph(para, chars_per_line))

        pages: list[list[CharPlacement]] = []
        current_page: list[CharPlacement] = []
        page_idx = 0
        line_idx = 0

        for global_line_idx, line_text in enumerate(lines):
            if line_idx >= len(line_positions):
                pages.append(current_page)
                current_page = []
                page_idx += 1
                line_idx = 0

            y = line_positions[line_idx]
            x = area.x
            is_page_first_line = (line_idx == 0)
            if global_line_idx in para_start_indices and not is_page_first_line:
                x += self.font_size

            # 行単位のベースライン揺らぎ + 密度スケール
            if self._augmenter is not None:
                _, line_y, _ = self._augmenter.augment_char_placement(
                    x, y, self.font_size
                )
                density_scale = self._augmenter.get_line_density_scale()
            else:
                line_y = y
                density_scale = 1.0

            prev_halfwidth = False
            for ch in line_text:
                if ch == "\n":
                    continue

                cur_halfwidth = is_halfwidth(ch)

                scale = _char_size_scale(ch)
                char_font_size = self.font_size * scale

                if self._augmenter is not None:
                    aug_x, _, aug_size = self._augmenter.augment_char_placement(
                        x, y, char_font_size
                    )
                    spacing_factor = density_scale
                    if prev_halfwidth and cur_halfwidth:
                        spacing_factor *= 0.5
                    aug_x = x + (aug_x - x) * spacing_factor
                    char_width = aug_size
                    char_width *= density_scale
                    current_page.append(CharPlacement(
                        char=ch, x=aug_x, y=line_y, font_size=aug_size, page=page_idx,
                    ))
                else:
                    char_width = char_font_size
                    current_page.append(CharPlacement(
                        char=ch, x=x, y=y, font_size=char_font_size, page=page_idx,
                    ))

                prev_halfwidth = cur_halfwidth

                x += char_width

            line_idx += 1

        pages.append(current_page)
        return pages
