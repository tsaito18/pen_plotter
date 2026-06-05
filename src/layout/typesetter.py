from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.layout.line_breaking import break_paragraph_by_width, is_halfwidth
from src.layout.math_layout import (
    MathLayoutEngine,
    MathParser,
    MathPlacement,
    _CHAR_WIDTH_RATIO,
)
from src.layout.page_layout import PageConfig, PageLayout
from src.layout.table_layout import detect_pipe_table

_SMALL_KANA = set("っゃゅょぁぃぅぇぉァィゥェォッャュョヵヶ")
_SMALL_PUNCT = set("・、。，．")
_HEADING_X: dict[int, float] = {1: 15.0, 2: 25.0, 3: 35.0}
_BODY_X: dict[int, float] = {1: 25.0, 2: 35.0, 3: 45.0}
_HEADING_FONT_SCALES: dict[int, float] = {1: 1.15, 2: 1.08, 3: 1.0}
_KANJI_ADVANCE_SCALE = 1.08

# 手書きバランス調整: 画数が少なく視覚的に軽い文字は小さめにする
_KANA_SIZE_OVERRIDES: dict[str, float] = {
    # カタカナ: 画数少・形が小さい文字 → 小さめ
    "ロ": 0.68,
    "ハ": 0.78,
    "ニ": 0.78,
    "ノ": 0.75,
    "ヘ": 0.78,
    "フ": 0.80,
    "ク": 0.80,
    "ワ": 0.80,
    "カ": 0.82,
    "コ": 0.80,
    "ン": 0.78,
    "ソ": 0.78,
    "リ": 0.78,
    "ル": 0.80,
    "レ": 0.78,
    "イ": 0.80,
    "ト": 0.78,
    "チ": 0.82,
    "ラ": 0.82,
    # カタカナ: 画数多・形が大きい文字 → やや大きめ
    "ス": 0.85,
    "テ": 0.85,
    "セ": 0.85,
    "サ": 0.85,
    "タ": 0.85,
    "ナ": 0.85,
    "マ": 0.85,
    "ミ": 0.82,
    "ム": 0.85,
    "メ": 0.82,
    "モ": 0.85,
    "ヤ": 0.85,
    "ユ": 0.82,
    "ヨ": 0.82,
    "キ": 0.85,
    "ケ": 0.82,
    "シ": 0.82,
    "ネ": 0.85,
    "ヌ": 0.85,
    "オ": 0.85,
    "エ": 0.82,
    "ア": 0.85,
    "ウ": 0.85,
    "ダ": 0.88,
    "デ": 0.88,
    "ド": 0.88,
    "バ": 0.85,
    "パ": 0.85,
    "ガ": 0.88,
    "ギ": 0.88,
    "グ": 0.85,
    "ゲ": 0.85,
    "ゴ": 0.85,
    "ザ": 0.88,
    "ジ": 0.85,
    "ズ": 0.88,
    "ゼ": 0.88,
    "ゾ": 0.85,
    "ビ": 0.85,
    "ブ": 0.85,
    "ベ": 0.82,
    "ボ": 0.88,
    "ピ": 0.85,
    "プ": 0.82,
    "ペ": 0.82,
    "ポ": 0.85,
    "ヒ": 0.78,
    "ホ": 0.85,
    # ひらがな: 画数少・形が小さい文字 → 小さめ
    "の": 0.78,
    "く": 0.75,
    "し": 0.78,
    "へ": 0.78,
    "つ": 0.80,
    "り": 0.78,
    "い": 0.80,
    "こ": 0.78,
    "て": 0.72,
    "に": 0.80,
    "と": 0.80,
    "う": 0.80,
    "か": 0.82,
    "る": 0.80,
    "を": 0.80,
    # ひらがな: 標準〜やや大きめ
    "あ": 0.85,
    "お": 0.85,
    "き": 0.85,
    "け": 0.82,
    "さ": 0.82,
    "す": 0.82,
    "せ": 0.85,
    "そ": 0.82,
    "た": 0.85,
    "ち": 0.82,
    "な": 0.85,
    "ぬ": 0.85,
    "ね": 0.85,
    "は": 0.85,
    "ひ": 0.78,
    "ふ": 0.85,
    "ほ": 0.85,
    "ま": 0.85,
    "み": 0.82,
    "む": 0.85,
    "め": 0.82,
    "も": 0.82,
    "や": 0.85,
    "ゆ": 0.85,
    "よ": 0.82,
    "ら": 0.82,
    "れ": 0.82,
    "ろ": 0.80,
    "わ": 0.82,
    "ん": 0.80,
    "え": 0.82,
    # 濁音ひらがな
    "が": 0.88,
    "ぎ": 0.88,
    "ぐ": 0.85,
    "げ": 0.85,
    "ご": 0.85,
    "ざ": 0.88,
    "じ": 0.85,
    "ず": 0.85,
    "ぜ": 0.88,
    "ぞ": 0.85,
    "だ": 0.88,
    "ぢ": 0.85,
    "づ": 0.85,
    "で": 0.85,
    "ど": 0.88,
    "ば": 0.88,
    "び": 0.85,
    "ぶ": 0.88,
    "べ": 0.85,
    "ぼ": 0.88,
    "ぱ": 0.88,
    "ぴ": 0.85,
    "ぷ": 0.85,
    "ぺ": 0.85,
    "ぽ": 0.88,
}


def _char_size_scale(ch: str) -> float:
    """文字種別に応じたサイズスケールを返す。個別調整テーブルあり。"""
    if ch in _SMALL_KANA:
        return 0.55
    if ch in _SMALL_PUNCT:
        return 0.35
    if ch in _KANA_SIZE_OVERRIDES:
        return _KANA_SIZE_OVERRIDES[ch]
    cp = ord(ch)
    if 0x3040 <= cp <= 0x309F:
        return 0.85
    if 0x30A0 <= cp <= 0x30FF:
        return 0.85
    if is_halfwidth(ch):
        return 0.7
    return 1.0


def _is_kanji(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
    )


_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)\$")
# DOTALL: $$\n...\n$$ のように改行を含む複数行ブロック数式を 1 つのマッチで捕捉する
_BLOCK_MATH_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
# ブロック数式を段落分割前に抽出するためのプレースホルダ（NULL文字で前後を囲み通常テキストと衝突しない形に）
_BLOCK_MATH_PLACEHOLDER_PREFIX = "\x00BLK\x00"
_BLOCK_MATH_PLACEHOLDER_SUFFIX = "\x00BLK\x00"
_INLINE_MATH_PLACEHOLDER_BASE = 0xE000
# パイプ表を段落処理前に1段落へ畳むためのプレースホルダ（NULL文字で衝突回避）
_TABLE_PLACEHOLDER_PREFIX = "\x00TBL\x00"
_TABLE_PLACEHOLDER_SUFFIX = "\x00TBL\x00"

# 数式内の . は変換しない（小数点は LaTeX で . のまま使う）
_MATH_OR_BLOCK_RE = re.compile(r"\$\$.*?\$\$|\$[^$]+?\$", re.DOTALL)


def _replace_ascii_period(text: str) -> str:
    """テキスト部分の ASCII . を 。 に変換（$...$, $$...$$ 内は除外）。"""
    result: list[str] = []
    last_end = 0
    for m in _MATH_OR_BLOCK_RE.finditer(text):
        result.append(text[last_end : m.start()].replace(".", "。"))
        result.append(m.group(0))
        last_end = m.end()
    result.append(text[last_end:].replace(".", "。"))
    return "".join(result)


def _split_segments(text: str) -> list[tuple[str, str]]:
    """テキストを通常テキストと数式セグメントに分割する。

    Returns:
        [("text", "通常テキスト"), ("math", "V = IR"), ...] の形式
    """
    segments: list[tuple[str, str]] = []
    last_end = 0
    for m in _INLINE_MATH_RE.finditer(text):
        if m.start() > last_end:
            segments.append(("text", text[last_end : m.start()]))
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
    line_segment: tuple[float, float, float, float] | None = None
    slant: float = 0.0  # 文字単位の微小傾き角(rad)。手書きの揺らぎ表現。
    # skeletonize レンダリング用: 先頭文字のみ設定、残りは math_skip=True
    math_source: str | None = None  # LaTeX ソース文字列
    math_bbox: tuple[float, float, float, float] | None = None  # (x, y, width_mm, height_mm)
    math_skip: bool = False  # True の文字はスキップ（先頭がまとめて描画済み）
    math_align: str = "center"  # "center"=ブロック中央寄せ / "baseline"=インライン本文ベース揃え


@dataclass
class ParsedDocument:
    """第1パス（段落解析）の結果。"""

    lines: list[str]
    heading_lines: dict[int, int]  # global_line_idx → heading_level
    line_body_level: dict[int, int]  # global_line_idx → body indent level
    para_start_indices: set[int]
    block_math_lines: dict[int, str]  # global_line_idx → math source
    # global_line_idx → 表の行データ（先頭ヘッダ）。パイプ表ブロックの起点行に登録。
    table_blocks: dict[int, list[list[str]]] = field(default_factory=dict)


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

    def _body_char_advance(self, ch: str) -> float:
        if is_halfwidth(ch):
            return self.font_size * 0.55
        if _is_kanji(ch):
            return self.font_size * _KANJI_ADVANCE_SCALE
        return self.font_size * (0.45 + 0.55 * _char_size_scale(ch))

    def _char_advance(self, ch: str, is_heading: bool, line_font_size: float) -> float:
        if is_heading:
            return line_font_size
        return self._body_char_advance(ch)

    def _line_right_x(self, area: object, is_heading: bool, body_level: int) -> float:
        if is_heading or body_level > 0:
            return self._config.paper_size[0] - 10.0
        return area.x + area.width

    def _inline_math_width(self, math_src: str) -> float:
        elements = MathParser.parse(math_src)
        elements = [e for e in elements if e.type != "tag"]
        return MathLayoutEngine.layout(elements, x=0.0, y=0.0, font_size=self.font_size).width

    def _line_neutral_width(
        self,
        segments: list[tuple[str, str]],
        is_heading: bool,
        line_font_size: float,
    ) -> float:
        total = 0.0
        for seg_type, seg_content in segments:
            if seg_type == "math":
                total += self._inline_math_width(seg_content)
                continue
            total += sum(
                self._char_advance(ch, is_heading, line_font_size)
                for ch in seg_content
                if ch != "\n"
            )
        return total

    def _inline_math_break_text(self, text: str) -> tuple[str, dict[str, str]]:
        placeholders: dict[str, str] = {}
        parts: list[str] = []
        last_end = 0
        for idx, match in enumerate(_INLINE_MATH_RE.finditer(text)):
            parts.append(text[last_end : match.start()])
            placeholder = chr(_INLINE_MATH_PLACEHOLDER_BASE + idx)
            placeholders[placeholder] = match.group(1)
            parts.append(placeholder)
            last_end = match.end()
        parts.append(text[last_end:])
        return "".join(parts), placeholders

    def typeset(self, text: str) -> list[list[CharPlacement]]:
        if not text:
            return [[]]

        text = _replace_ascii_period(text)

        area = self._layout.content_area()
        line_positions = self._layout.line_positions()
        doc = self._parse_paragraphs(text, area)

        pages: list[list[CharPlacement]] = []
        current_page: list[CharPlacement] = []
        page_idx = 0
        line_idx = 0

        for global_line_idx, line_text in enumerate(doc.lines):
            if line_idx >= len(line_positions):
                pages.append(current_page)
                current_page = []
                page_idx += 1
                line_idx = 0

            y = line_positions[line_idx]

            if global_line_idx in doc.block_math_lines:
                math_src = doc.block_math_lines[global_line_idx]
                consumed = self._place_block_math(
                    math_src,
                    line_idx,
                    line_positions,
                    area,
                    page_idx,
                    current_page,
                )
                if consumed == -1:
                    # 残り行不足 → ページ確定して次ページ先頭に再配置
                    pages.append(current_page)
                    current_page = []
                    page_idx += 1
                    line_idx = 0
                    consumed = self._place_block_math(
                        math_src,
                        0,
                        line_positions,
                        area,
                        page_idx,
                        current_page,
                    )
                    # 新ページでも入らないケースは想定外（required_rows > rows/page）。
                    # 安全策として -1 のまま返ったら 1 行扱いで進めて無限ループを避ける。
                    if consumed == -1:
                        consumed = 1
                line_idx += consumed
                continue

            if global_line_idx in doc.table_blocks:
                rows = doc.table_blocks[global_line_idx]
                consumed = self._place_table(
                    rows, line_idx, line_positions, area, page_idx, current_page
                )
                if consumed == -1:
                    pages.append(current_page)
                    current_page = []
                    page_idx += 1
                    line_idx = 0
                    consumed = self._place_table(
                        rows, 0, line_positions, area, page_idx, current_page
                    )
                    if consumed == -1:
                        consumed = 1  # 1ページに収まらない巨大表は無限ループ回避
                line_idx += consumed
                continue

            is_heading = global_line_idx in doc.heading_lines
            h_level = doc.heading_lines.get(global_line_idx, 0)
            body_level = doc.line_body_level.get(global_line_idx, 0)
            is_para_start = global_line_idx in doc.para_start_indices
            is_page_first = line_idx == 0
            prev_is_heading = (global_line_idx - 1) in doc.heading_lines or (
                global_line_idx - 2
            ) in doc.heading_lines

            placements = self._place_line(
                line_text=line_text,
                y=y,
                page_idx=page_idx,
                area=area,
                is_heading=is_heading,
                heading_level=h_level,
                body_level=body_level,
                is_para_start=is_para_start,
                is_page_first=is_page_first,
                prev_is_heading=prev_is_heading,
                heading_x=_HEADING_X,
                body_x=_BODY_X,
                heading_font_scales=_HEADING_FONT_SCALES,
            )
            current_page.extend(placements)
            line_idx += 1

        pages.append(current_page)
        return pages

    def _parse_paragraphs(self, text: str, area: object) -> ParsedDocument:
        """第1パス: テキストを段落解析し、行リストと各行のメタ情報を返す。"""
        # 改行を含むブロック数式 $$\n...\n$$ は通常 paragraph 分割（\n split）でばらけてしまうため、
        # 先に DOTALL マッチで全体から抽出してプレースホルダ独立段落に置換しておく。
        # これにより 1 行 $$...$$ も複数行 $$...\n...$$ も同じ後段処理で扱える。
        stashed_block_maths: list[str] = []

        def _stash_block_math(m: re.Match) -> str:
            stashed_block_maths.append(m.group(1).strip())
            placeholder = (
                _BLOCK_MATH_PLACEHOLDER_PREFIX
                + str(len(stashed_block_maths) - 1)
                + _BLOCK_MATH_PLACEHOLDER_SUFFIX
            )
            # 前後に \n を挟むことでプレースホルダ単独段落として split される
            return "\n" + placeholder + "\n"

        text = _BLOCK_MATH_RE.sub(_stash_block_math, text)
        placeholder_re = re.compile(
            re.escape(_BLOCK_MATH_PLACEHOLDER_PREFIX)
            + r"(\d+)"
            + re.escape(_BLOCK_MATH_PLACEHOLDER_SUFFIX)
        )

        paragraphs = text.split("\n")

        # プレースホルダ置換時に追加した前後 \n が元の paragraph 区切りと重複して
        # 空段落を生むため、placeholder 段落に隣接する空段落のみ取り除く
        def _is_placeholder_para(s: str) -> bool:
            return bool(placeholder_re.fullmatch(s.strip()))

        cleaned: list[str] = []
        for i, p in enumerate(paragraphs):
            if p == "":
                next_is_placeholder = i + 1 < len(paragraphs) and _is_placeholder_para(
                    paragraphs[i + 1]
                )
                prev_is_placeholder = bool(cleaned) and _is_placeholder_para(cleaned[-1])
                if next_is_placeholder or prev_is_placeholder:
                    continue
            cleaned.append(p)
        paragraphs = cleaned

        # パイプ表を検出して 1 プレースホルダ段落へ畳む（複数行表を後段で一括配置）
        stashed_tables: list[list[list[str]]] = []
        collapsed: list[str] = []
        ti = 0
        while ti < len(paragraphs):
            tbl = detect_pipe_table(paragraphs, ti)
            if tbl is not None:
                rows, consumed = tbl
                stashed_tables.append(rows)
                collapsed.append(
                    _TABLE_PLACEHOLDER_PREFIX
                    + str(len(stashed_tables) - 1)
                    + _TABLE_PLACEHOLDER_SUFFIX
                )
                ti += consumed
            else:
                collapsed.append(paragraphs[ti])
                ti += 1
        paragraphs = collapsed
        table_placeholder_re = re.compile(
            re.escape(_TABLE_PLACEHOLDER_PREFIX) + r"(\d+)" + re.escape(_TABLE_PLACEHOLDER_SUFFIX)
        )

        lines: list[str] = []
        para_start_indices: set[int] = set()
        block_math_lines: dict[int, str] = {}
        table_blocks: dict[int, list[list[str]]] = {}
        heading_lines: dict[int, int] = {}
        line_body_level: dict[int, int] = {}
        current_body_level: int = 0

        for para in paragraphs:
            # パイプ表プレースホルダ: 表ブロック起点行として登録
            table_match = table_placeholder_re.fullmatch(para.strip())
            if table_match is not None:
                para_start_indices.add(len(lines))
                table_blocks[len(lines)] = stashed_tables[int(table_match.group(1))]
                line_body_level[len(lines)] = current_body_level
                lines.append("")
                continue

            # プレースホルダ単独段落を先に処理: ブロック数式行として登録
            placeholder_match = placeholder_re.fullmatch(para.strip())
            if placeholder_match is not None:
                math_src = stashed_block_maths[int(placeholder_match.group(1))]
                if math_src:
                    para_start_indices.add(len(lines))
                    block_math_lines[len(lines)] = math_src
                    line_body_level[len(lines)] = current_body_level
                    lines.append("")
                continue

            heading_level = 0
            display_para = para
            if para.startswith("###"):
                heading_level = 3
                display_para = para[3:].strip()
            elif para.startswith("##"):
                heading_level = 2
                display_para = para[2:].strip()
            elif para.startswith("#"):
                heading_level = 1
                display_para = para[1:].strip()

            if heading_level > 0:
                if len(lines) > 0 and lines != [""]:
                    lines.append("")
                heading_lines[len(lines)] = heading_level
                current_body_level = heading_level

            para_start_indices.add(len(lines))
            if not display_para:
                line_body_level[len(lines)] = current_body_level
                lines.append("")
                continue

            parts = _BLOCK_MATH_RE.split(display_para)
            for part_idx, part in enumerate(parts):
                if part_idx % 2 == 1:
                    if part.strip():
                        block_math_lines[len(lines)] = part.strip()
                        line_body_level[len(lines)] = current_body_level
                        lines.append("")
                else:
                    if not part:
                        continue
                    break_text, math_placeholders = self._inline_math_break_text(part)
                    is_heading = heading_level > 0
                    if is_heading:
                        line_x = _HEADING_X.get(heading_level, area.x)
                        line_font_size = self.font_size * _HEADING_FONT_SCALES[heading_level]

                        def width_fn(ch: str) -> float:
                            if ch in math_placeholders:
                                return self._inline_math_width(math_placeholders[ch])
                            return line_font_size
                    else:
                        line_x = _BODY_X.get(current_body_level, area.x)

                        def width_fn(ch: str) -> float:
                            if ch in math_placeholders:
                                return self._inline_math_width(math_placeholders[ch])
                            return self._body_char_advance(ch)

                    line_width = self._line_right_x(area, is_heading, current_body_level) - line_x
                    broken = break_paragraph_by_width(break_text, line_width, width_fn)
                    result_lines = self._rebuild_lines_with_math(part, broken, math_placeholders)
                    if heading_level > 0:
                        for i in range(len(result_lines)):
                            heading_lines[len(lines) + i] = heading_level
                    for i in range(len(result_lines)):
                        line_body_level[len(lines) + i] = current_body_level
                    lines.extend(result_lines)

        return ParsedDocument(
            lines=lines,
            heading_lines=heading_lines,
            line_body_level=line_body_level,
            para_start_indices=para_start_indices,
            block_math_lines=block_math_lines,
            table_blocks=table_blocks,
        )

    def _place_line(
        self,
        line_text: str,
        y: float,
        page_idx: int,
        area: object,
        is_heading: bool,
        heading_level: int,
        body_level: int,
        is_para_start: bool,
        is_page_first: bool,
        prev_is_heading: bool,
        heading_x: dict[int, float],
        body_x: dict[int, float],
        heading_font_scales: dict[int, float],
    ) -> list[CharPlacement]:
        """第2パス: 1行分の文字配置を計算する。"""
        output: list[CharPlacement] = []

        if is_heading:
            line_font_size = self.font_size * heading_font_scales[heading_level]
            x = heading_x.get(heading_level, area.x)
        else:
            line_font_size = self.font_size
            if body_level > 0:
                x = body_x.get(body_level, area.x)
            else:
                x = area.x

        if is_para_start and not is_page_first and not is_heading and not prev_is_heading:
            x += self.font_size

        if self._augmenter is not None:
            _, line_y, _ = self._augmenter.augment_char_placement(x, y, self.font_size)
            density_scale = self._augmenter.get_line_density_scale()
        else:
            line_y = y
            density_scale = 1.0

        segments = _split_segments(line_text)
        neutral_remaining = self._line_neutral_width(segments, is_heading, line_font_size)
        line_right_x = self._line_right_x(area, is_heading, body_level)

        prev_halfwidth = False
        for seg_type, seg_content in segments:
            if seg_type == "math":
                math_x = x
                x = self._place_math(seg_content, x, line_y, page_idx, output)
                neutral_remaining -= x - math_x
                prev_halfwidth = False
            else:
                for ch in seg_content:
                    if ch == "\n":
                        continue

                    cur_halfwidth = is_halfwidth(ch)
                    size_scale = _char_size_scale(ch)
                    if is_heading:
                        char_font_size = line_font_size
                    else:
                        char_font_size = self.font_size * size_scale
                    char_advance = self._char_advance(ch, is_heading, line_font_size)
                    neutral_remaining -= char_advance

                    if self._augmenter is not None:
                        aug_x, _, aug_size = self._augmenter.augment_char_placement(
                            x, y, char_font_size
                        )
                        char_density_scale = self._augmenter.get_char_density_scale()
                        density_factor = density_scale * char_density_scale
                        spacing_factor = density_factor
                        if prev_halfwidth and cur_halfwidth:
                            spacing_factor *= 0.5
                        aug_x = x + (aug_x - x) * spacing_factor
                        char_width = char_advance * density_factor
                        if density_factor > 1.0:
                            density_width_cap = max(
                                char_advance,
                                line_right_x - x - neutral_remaining,
                            )
                            char_width = min(char_width, density_width_cap)
                        get_slant = getattr(self._augmenter, "get_char_slant", None)
                        output.append(
                            CharPlacement(
                                char=ch,
                                x=aug_x,
                                y=line_y,
                                font_size=aug_size,
                                page=page_idx,
                                slant=get_slant() if get_slant else 0.0,
                            )
                        )
                    else:
                        char_width = char_advance
                        output.append(
                            CharPlacement(
                                char=ch,
                                x=x,
                                y=y,
                                font_size=char_font_size,
                                page=page_idx,
                            )
                        )

                    prev_halfwidth = cur_halfwidth
                    x += char_width

        return output

    @staticmethod
    def _strip_tag_and_adjacent_spaces(elements: list) -> list:
        """tag 要素を除き、tag に隣接する空白専用 text 要素も除外する。

        本体の中心位置が tag 周辺空白の影響を受けないようにするため、
        tag 直前 text の末尾空白と直後 text の先頭空白も除去する。
        """
        from src.layout.math_layout import MathElement

        result: list = []
        for elem in elements:
            if elem.type == "tag":
                # 直前 text の末尾空白を削る
                while result and result[-1].type == "text":
                    stripped = result[-1].content.rstrip()
                    if stripped == "":
                        result.pop()
                        continue
                    if stripped != result[-1].content:
                        result[-1] = MathElement(type="text", content=stripped)
                    break
                continue
            result.append(elem)
        # 末尾の空白専用 text を削る（tag が末尾にあった場合の trailing space 除去）
        while result and result[-1].type == "text" and result[-1].content.strip() == "":
            result.pop()
        if result and result[-1].type == "text":
            stripped = result[-1].content.rstrip()
            if stripped != result[-1].content:
                result[-1] = MathElement(type="text", content=stripped)
        return result

    @staticmethod
    def _split_by_linebreak(elements: list) -> list[list]:
        """linebreak 要素でグループ分割。空グループは除外。"""
        groups: list[list] = []
        current: list = []
        for elem in elements:
            if elem.type == "linebreak":
                if current:
                    groups.append(current)
                    current = []
                continue
            current.append(elem)
        if current:
            groups.append(current)
        return groups

    def _place_table(
        self,
        rows: list[list[str]],
        line_idx: int,
        line_positions: list[float],
        area: object,
        page_idx: int,
        output: list[CharPlacement],
    ) -> int:
        """パイプ表を罫線(line_segment)＋セル文字として配置する。

        1 表行 = line_positions の 1 行を占有する。列幅は各列の最大文字数から決め、
        合計が本文幅を超える場合は一律縮小して収める。残り行が足りなければ -1
        （呼び出し側で次ページ送り）。

        Returns:
            消費した行数（= 表の行数）。残り行不足なら -1。
        """
        n_rows = len(rows)
        if n_rows == 0:
            return 1
        n_cols = max(len(r) for r in rows)
        remaining = len(line_positions) - line_idx
        if remaining < n_rows:
            return -1

        fs = self.font_size
        pad = fs * 0.3
        # 列幅: 各列の最大文字数 × 文字送り + 左右パディング
        col_w: list[float] = []
        for c in range(n_cols):
            max_chars = max(len(rows[r][c]) for r in range(n_rows)) if n_cols > 0 else 1
            col_w.append(max(max_chars, 1) * fs + 2 * pad)
        total_w = sum(col_w)
        scale = min(1.0, area.width / total_w) if total_w > 0 else 1.0
        col_w = [w * scale for w in col_w]

        row_h = self._config.line_spacing
        # 横罫線は用紙の罫線(line_positions)に一致させる。各行=1罫線バンドを占有。
        # 境界 y: 先頭行の上罫線(L[idx]+row_h) → 各行の下罫線 L[idx+r]。
        ys = [line_positions[line_idx] + row_h]
        ys += [line_positions[line_idx + r] for r in range(n_rows)]
        xs = [area.x]
        for w in col_w:
            xs.append(xs[-1] + w)

        # 横罫線（用紙の罫線に重なる）
        for y in ys:
            output.append(
                CharPlacement(
                    char="",
                    x=area.x,
                    y=y,
                    font_size=fs,
                    page=page_idx,
                    line_segment=(area.x, y, xs[-1], y),
                )
            )
        # 縦罫線（列境界）。表の上端〜下端まで。
        y_top, y_bot = ys[0], ys[-1]
        for x in xs:
            output.append(
                CharPlacement(
                    char="",
                    x=x,
                    y=y_top,
                    font_size=fs,
                    page=page_idx,
                    line_segment=(x, y_top, x, y_bot),
                )
            )
        # セル文字（左寄せ）。ベースライン=下罫線 L[idx+r]。_position_strokes が
        # placement.y を帯下端として line_spacing 帯の中央へ glyph を置くため、
        # これでセル帯（L[idx+r]〜L[idx+r]+row_h）のちょうど中央に収まる。字送りも scale。
        cell_fs = fs * scale
        for r in range(n_rows):
            baseline = line_positions[line_idx + r]
            for c in range(n_cols):
                text = rows[r][c]
                cx = xs[c] + pad * scale
                for ch in text:
                    if ch != " ":
                        output.append(
                            CharPlacement(
                                char=ch, x=cx, y=baseline, font_size=cell_fs, page=page_idx
                            )
                        )
                    cx += self._body_char_advance(ch) * scale
        return n_rows

    def _place_block_math(
        self,
        math_src: str,
        line_idx: int,
        line_positions: list[float],
        area: object,
        page_idx: int,
        output: list[CharPlacement],
    ) -> int:
        """ブロック数式を確保した行範囲の垂直中央に配置する。

        分数等で高さが大きいときは複数行を確保する（最低2行）。\\tag{} は確保した
        行範囲の中央 y で右端に併置する。

        Returns:
            消費した行数（>=2）。残り行数が足りない場合は -1（呼び出し側で次ページ送り）。
        """
        # 層の逆依存を避けるため遅延 import（layout → ui）
        from src.ui.math_skeletonize import formula_draw_width_mm

        elements = MathParser.parse(math_src)
        # 本体中心位置を tag 幅から独立させるため、tag と tag 周辺の空白テキストを除外する
        tag_elem = next((e for e in elements if e.type == "tag"), None)
        body_elements = self._strip_tag_and_adjacent_spaces(elements)
        # 画像レンダラに渡すソースから \tag{} を除去（matplotlib は \tag 非対応）
        body_src = re.sub(r"\\tag\{[^}]*\}", "", math_src).strip()

        # \\ 改行でグループ分割。linebreak が無いときは 1 グループ＝従来挙動。
        groups = self._split_by_linebreak(body_elements)

        line_spacing = self._config.line_spacing

        # 各グループの寸法を仮配置で測定
        group_boxes = [
            MathLayoutEngine.layout(g, x=0.0, y=0.0, font_size=self.font_size) for g in groups
        ]

        if group_boxes:
            # 多段時はグループ間隔として line_spacing を使う（ベースライン間隔）。
            # 全体高さ = 先頭グループ ascent + 末尾グループ descent + (n-1)*line_spacing
            total_height = (
                group_boxes[0].ascent
                + group_boxes[-1].descent
                + line_spacing * (len(group_boxes) - 1)
            )
        else:
            total_height = self.font_size

        # 最低2行、それを超える高さなら必要な行数を ceil で確保
        required_rows = max(2, math.ceil(total_height / line_spacing))

        # ページ末尾チェック: 残り行数が足りなければ次ページ送りシグナル
        remaining = len(line_positions) - line_idx
        if remaining < required_rows:
            return -1

        # 確保した行範囲の垂直中央にグループ列の中心を置く
        top_y = line_positions[line_idx]
        bottom_y = line_positions[line_idx + required_rows - 1]
        center_y = (top_y + bottom_y) / 2

        # 各グループのベースライン y: 中心から等間隔（line_spacing 刻み）に上下分散。
        # 上から下へ並ぶよう、index 0 が最上段（最大の +offset）。
        group_count = len(group_boxes)
        last_baseline_y = center_y
        last_body_right = area.x + area.width  # 本体右端（tag をこの直後に置く）
        for i, (g_elems, g_box) in enumerate(zip(groups, group_boxes)):
            offset = (group_count - 1) / 2 * line_spacing - i * line_spacing
            baseline_y = center_y + offset
            last_baseline_y = baseline_y
            # 中央寄せのため幅測定→本配置の2段構え（既存挙動踏襲）
            center_x = area.x + (area.width - g_box.width) / 2
            placed = MathLayoutEngine.layout(
                g_elems, x=center_x, y=baseline_y, font_size=self.font_size
            )
            # グループ単位の bbox を求めて skeletonize レンダラに渡す
            # 複数グループ時は各グループの math_src を再構築できないため body_src 全体を渡す
            g_h = placed.ascent + placed.descent
            g_bbox = (
                center_x,
                baseline_y - placed.descent,
                placed.width,
                g_h,
            )
            self._convert_math_placements(placed.placements, page_idx, output, body_src, g_bbox)
            # 数式画像は bbox 中心に draw_w(=g_h*aspect)幅で描かれる。実描画右端を式番号配置に使う
            draw_w = formula_draw_width_mm(body_src, g_h)
            last_body_right = (center_x + placed.width / 2) + draw_w / 2

        if tag_elem is not None:
            tag_y = last_baseline_y if group_boxes else center_y
            tag_temp = MathLayoutEngine.layout([tag_elem], x=0, y=tag_y, font_size=self.font_size)
            # 式番号は数式本体の直後（1文字分あけて）に置く。紙右端を超える場合のみ右端へ。
            tag_x = last_body_right + self.font_size
            max_x = area.x + area.width - tag_temp.width
            tag_x = min(tag_x, max_x)
            tag_box = MathLayoutEngine.layout(
                [tag_elem], x=tag_x, y=tag_y, font_size=self.font_size
            )
            self._convert_math_placements(tag_box.placements, page_idx, output)

        return required_rows

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
        # インライン数式中の \tag{} は配置しない（仕様: ブロック数式専用）
        elements = [e for e in elements if e.type != "tag"]
        box = MathLayoutEngine.layout(elements, x=x, y=y, font_size=self.font_size)
        # インライン: bbox[1] に本文ベースライン y を渡し、math_align="baseline" で
        # 数式のベースラインを本文行に揃える（中心配置のズレを根本解消）。
        # 高さ h は ascent+descent（数式画像の縦範囲スケールに使う）。
        math_bbox = (x, y, box.width, box.ascent + box.descent)
        self._convert_math_placements(
            box.placements, page_idx, output, math_src, math_bbox, math_align="baseline"
        )
        return x + box.width

    @staticmethod
    def _convert_math_placements(
        placements: list[MathPlacement],
        page_idx: int,
        output: list[CharPlacement],
        math_source: str | None = None,
        math_bbox: tuple[float, float, float, float] | None = None,
        math_align: str = "center",
    ) -> None:
        """MathPlacement リストを CharPlacement に変換して output に追加。

        math_source/math_bbox が指定されたとき、先頭の文字配置にセットし残りを
        math_skip=True でマークする（skeletonize レンダラが一括描画するため）。
        """
        first_placed = False

        def make_cp(**kwargs: object) -> CharPlacement:
            nonlocal first_placed
            if math_source is not None and not first_placed:
                first_placed = True
                return CharPlacement(
                    math_source=math_source,
                    math_bbox=math_bbox,
                    math_align=math_align,
                    **kwargs,  # type: ignore[arg-type]
                )
            if math_source is not None:
                return CharPlacement(math_skip=True, **kwargs)  # type: ignore[arg-type]
            return CharPlacement(**kwargs)  # type: ignore[arg-type]

        for mp in placements:
            if not mp.text and mp.line_segment is not None:
                output.append(
                    make_cp(
                        char="",
                        x=mp.x,
                        y=mp.y,
                        font_size=mp.font_size,
                        page=page_idx,
                        role=mp.role,
                        line_segment=mp.line_segment,
                    )
                )
            elif len(mp.text) == 1 or mp.role == "operator":
                output.append(
                    make_cp(
                        char=mp.text,
                        x=mp.x,
                        y=mp.y,
                        font_size=mp.font_size,
                        page=page_idx,
                        role=mp.role,
                        line_segment=mp.line_segment,
                    )
                )
            else:
                for i, ch in enumerate(mp.text):
                    role = mp.role if i == 0 else None
                    seg = mp.line_segment if i == 0 else None
                    output.append(
                        make_cp(
                            char=ch,
                            x=mp.x + i * mp.font_size * _CHAR_WIDTH_RATIO,
                            y=mp.y,
                            font_size=mp.font_size,
                            page=page_idx,
                            role=role,
                            line_segment=seg,
                        )
                    )

    @staticmethod
    def _rebuild_lines_with_math(
        original: str,
        broken_lines: list[str],
        math_placeholders: dict[str, str] | None = None,
    ) -> list[str]:
        """break_paragraphの結果を元テキスト（$付き）に復元する。

        break_paragraphには$を除去・スペース→\\x00変換したテキストを渡している。
        broken_linesの各行の文字数を使って、元テキストから対応部分を切り出す。
        """
        if math_placeholders is not None:
            result: list[str] = []
            for bline in broken_lines:
                parts: list[str] = []
                for ch in bline:
                    if ch in math_placeholders:
                        parts.append(f"${math_placeholders[ch]}$")
                    else:
                        parts.append(ch)
                result.append("".join(parts))
            return result if result else [""]

        segments = _split_segments(original)
        # 元テキストをフラットな文字列として、$なしの文字列を作る
        flat_no_dollar: list[str] = []
        # 各位置が元テキストのどのセグメントに属するかマッピング
        char_to_segment: list[tuple[int, int]] = []  # (seg_idx, char_idx_in_seg)
        for seg_idx, (seg_type, seg_content) in enumerate(segments):
            if seg_type == "math":
                content_no_space = seg_content.replace(" ", "\x00")
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
                    math_chars.append(seg_content[ci] if ci < len(seg_content) else "")
                else:
                    if in_math_seg is not None:
                        line_parts.append("$" + "".join(math_chars) + "$")
                        math_chars = []
                        in_math_seg = None
                    line_parts.append(seg_content[ci] if ci < len(seg_content) else "")
            if in_math_seg is not None:
                line_parts.append("$" + "".join(math_chars) + "$")
            result.append("".join(line_parts))
            offset += line_len

        return result if result else [""]
