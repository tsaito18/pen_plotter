from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.layout.line_breaking import break_paragraph, is_halfwidth
from src.layout.math_layout import (
    MathLayoutEngine,
    MathParser,
    MathPlacement,
    _CHAR_WIDTH_RATIO,
)
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
        return 0.93
    if 0x30A0 <= cp <= 0x30FF:
        return 0.93
    if is_halfwidth(ch):
        return 0.55
    return 1.0


_INLINE_MATH_RE = re.compile(r'(?<!\$)\$(?!\$)(.*?)\$')
_BLOCK_MATH_RE = re.compile(r'\$\$(.*?)\$\$')


def _split_segments(text: str) -> list[tuple[str, str]]:
    """テキストを通常テキストと数式セグメントに分割する。

    Returns:
        [("text", "通常テキスト"), ("math", "V = IR"), ...] の形式
    """
    segments: list[tuple[str, str]] = []
    last_end = 0
    for m in _INLINE_MATH_RE.finditer(text):
        if m.start() > last_end:
            segments.append(("text", text[last_end:m.start()]))
        math_content = m.group(1)
        if math_content:
            segments.append(("math", math_content))
        last_end = m.end()
    if last_end < len(text):
        segments.append(("text", text[last_end:]))
    return segments


if TYPE_CHECKING:
    from src.model.augmentation import HandwritingAugmenter


@dataclass
class CharPlacement:
    char: str
    x: float
    y: float
    font_size: float
    page: int = 0
    role: str | None = None


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

        page_config = self._config
        default_chars_per_line = int(area.width / self.font_size)

        # 段落ごとに改行し、段落先頭行のインデックスを記録
        # block_math_lines: ブロック数式として中央配置する行のインデックス→数式ソース
        paragraphs = text.split("\n")
        lines: list[str] = []
        para_start_indices: set[int] = set()
        block_math_lines: dict[int, str] = {}

        heading_lines: dict[int, int] = {}  # global_line_idx → heading_level
        heading_font_scales = {1: 1.15, 2: 1.08, 3: 1.0}

        # 見出しレベルに応じたインデント（mm）
        pw = page_config.paper_size[0]
        heading_x: dict[int, float] = {1: 15.0, 2: 25.0, 3: 35.0}
        body_x: dict[int, float] = {1: 25.0, 2: 35.0, 3: 45.0}
        right_x: float = pw - 10.0  # 右端 = 右から10mm
        current_body_level: int = 0  # 現在の本文インデントレベル

        for para in paragraphs:
            # 見出し判定
            heading_level = 0
            display_para = para
            if para.startswith('###'):
                heading_level = 3
                display_para = para[3:].strip()
            elif para.startswith('##'):
                heading_level = 2
                display_para = para[2:].strip()
            elif para.startswith('#'):
                heading_level = 1
                display_para = para[1:].strip()

            if heading_level > 0:
                # 見出し前に空行（ページ先頭でない場合）
                if len(lines) > 0 and lines != ['']:
                    lines.append("")
                heading_lines[len(lines)] = heading_level
                current_body_level = heading_level

            para_start_indices.add(len(lines))
            if not display_para:
                lines.append("")
                continue

            # $$...$$ でブロック数式を分割
            parts = _BLOCK_MATH_RE.split(display_para)
            for part_idx, part in enumerate(parts):
                if part_idx % 2 == 1:
                    # ブロック数式
                    if part.strip():
                        block_math_lines[len(lines)] = part.strip()
                        lines.append("")  # プレースホルダ
                else:
                    # 通常テキスト
                    if not part:
                        continue
                    stripped = _INLINE_MATH_RE.sub(
                        lambda m: m.group(1).replace(' ', '\x00'), part
                    )
                    if heading_level > 0:
                        line_width = right_x - heading_x.get(heading_level, area.x)
                    elif current_body_level > 0:
                        line_width = right_x - body_x.get(current_body_level, area.x)
                    else:
                        line_width = area.width
                    cpl = int(line_width / self.font_size)
                    broken = break_paragraph(stripped, cpl)
                    result_lines = self._rebuild_lines_with_math(part, broken)
                    # 見出しの場合、全行を見出し行として登録
                    if heading_level > 0:
                        for i in range(len(result_lines)):
                            heading_lines[len(lines) + i] = heading_level
                    lines.extend(result_lines)

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

            # ブロック数式行: 中央配置
            if global_line_idx in block_math_lines:
                self._place_block_math(
                    block_math_lines[global_line_idx], y, area, page_idx, current_page
                )
                line_idx += 1
                continue

            # 見出し行のフォントサイズスケール
            is_heading = global_line_idx in heading_lines
            if is_heading:
                h_level = heading_lines[global_line_idx]
                line_font_size = self.font_size * heading_font_scales[h_level]
                x = heading_x.get(h_level, area.x)
            else:
                line_font_size = self.font_size
                if current_body_level > 0:
                    x = body_x.get(current_body_level, area.x)
                else:
                    x = area.x

            is_page_first_line = (line_idx == 0)
            # 段落先頭インデント（見出し行・ページ先頭・見出し直後の最初の段落を除く）
            prev_is_heading = (global_line_idx - 1) in heading_lines or (global_line_idx - 2) in heading_lines
            if global_line_idx in para_start_indices and not is_page_first_line and not is_heading and not prev_is_heading:
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

            segments = _split_segments(line_text)

            prev_halfwidth = False
            for seg_type, seg_content in segments:
                if seg_type == "math":
                    x = self._place_math(
                        seg_content, x, y, page_idx, current_page
                    )
                    prev_halfwidth = False
                else:
                    for ch in seg_content:
                        if ch == "\n":
                            continue

                        cur_halfwidth = is_halfwidth(ch)
                        if is_heading:
                            char_font_size = line_font_size
                        else:
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

    def _place_block_math(
        self,
        math_src: str,
        y: float,
        area: object,
        page_idx: int,
        output: list[CharPlacement],
    ) -> None:
        """ブロック数式を行の中央に配置する。"""
        elements = MathParser.parse(math_src)
        # x=0 で仮レイアウトして幅を測定
        temp = MathLayoutEngine.layout(elements, x=0, y=y, font_size=self.font_size)
        math_width = MathLayoutEngine.total_width(temp) if temp else 0.0
        center_x = area.x + (area.width - math_width) / 2
        placements = MathLayoutEngine.layout(
            elements, x=center_x, y=y, font_size=self.font_size
        )
        self._convert_math_placements(placements, page_idx, output)

    def _place_math(
        self,
        math_src: str,
        x: float,
        y: float,
        page_idx: int,
        output: list[CharPlacement],
    ) -> float:
        """数式をパース・レイアウトしてCharPlacementに変換する。xの次の位置を返す。"""
        elements = MathParser.parse(math_src)
        placements = MathLayoutEngine.layout(elements, x=x, y=y, font_size=self.font_size)
        self._convert_math_placements(placements, page_idx, output)
        total_w = MathLayoutEngine.total_width(placements) if placements else 0.0
        return x + total_w

    @staticmethod
    def _convert_math_placements(
        placements: list[MathPlacement],
        page_idx: int,
        output: list[CharPlacement],
    ) -> None:
        """MathPlacement リストを CharPlacement に変換して output に追加。"""
        for mp in placements:
            if len(mp.text) == 1:
                output.append(CharPlacement(
                    char=mp.text, x=mp.x, y=mp.y,
                    font_size=mp.font_size, page=page_idx,
                    role=mp.role,
                ))
            else:
                for i, ch in enumerate(mp.text):
                    role = mp.role if i == 0 else None
                    output.append(CharPlacement(
                        char=ch,
                        x=mp.x + i * mp.font_size * _CHAR_WIDTH_RATIO,
                        y=mp.y,
                        font_size=mp.font_size,
                        page=page_idx,
                        role=role,
                    ))

    @staticmethod
    def _rebuild_lines_with_math(
        original: str, broken_lines: list[str]
    ) -> list[str]:
        """break_paragraphの結果を元テキスト（$付き）に復元する。

        break_paragraphには$を除去・スペース→\\x00変換したテキストを渡している。
        broken_linesの各行の文字数を使って、元テキストから対応部分を切り出す。
        """
        segments = _split_segments(original)
        # 元テキストをフラットな文字列として、$なしの文字列を作る
        flat_no_dollar: list[str] = []
        # 各位置が元テキストのどのセグメントに属するかマッピング
        char_to_segment: list[tuple[int, int]] = []  # (seg_idx, char_idx_in_seg)
        for seg_idx, (seg_type, seg_content) in enumerate(segments):
            if seg_type == "math":
                content_no_space = seg_content.replace(' ', '\x00')
                for ci, ch in enumerate(content_no_space):
                    flat_no_dollar.append(ch)
                    char_to_segment.append((seg_idx, ci))
            else:
                for ci, ch in enumerate(seg_content):
                    flat_no_dollar.append(ch)
                    char_to_segment.append((seg_idx, ci))

        # broken_linesの各行の文字数に基づいて元テキストを切り出す
        result: list[str] = []
        offset = 0
        for bline in broken_lines:
            line_len = len(bline)
            # この行に含まれるセグメントを再構築
            line_parts: list[str] = []
            in_math_seg: int | None = None
            math_chars: list[str] = []
            for i in range(offset, min(offset + line_len, len(char_to_segment))):
                seg_idx, ci = char_to_segment[i]
                seg_type, seg_content = segments[seg_idx]
                if seg_type == "math":
                    if in_math_seg != seg_idx:
                        if in_math_seg is not None:
                            line_parts.append("$" + "".join(math_chars) + "$")
                            math_chars = []
                        in_math_seg = seg_idx
                    math_chars.append(seg_content[ci] if ci < len(seg_content) else '')
                else:
                    if in_math_seg is not None:
                        line_parts.append("$" + "".join(math_chars) + "$")
                        math_chars = []
                        in_math_seg = None
                    line_parts.append(seg_content[ci] if ci < len(seg_content) else '')
            if in_math_seg is not None:
                line_parts.append("$" + "".join(math_chars) + "$")
            result.append("".join(line_parts))
            offset += line_len

        return result if result else [""]
