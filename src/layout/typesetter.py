"""テキスト（Markdown 風の簡易書式）をレポート用紙上の配置要素へ組版する。

処理は 2 パス:

1. :func:`parse_document` — 入力を「行レコード」の列へ分解する。見出し・段落・
   ブロック数式・パイプ表・改ページを判別し、本文は禁則処理付きで折り返す。
2. :meth:`Typesetter.typeset` — 行レコードを罫線位置に割り付け、各文字・数式・
   罫線の :class:`CharPlacement` を生成する（ページ送りもここで行う）。

書式の一覧は ``docs/書式リファレンス.md`` を参照。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from src.layout.char_metrics import effective_char_scale
from src.layout.line_breaking import break_paragraph_by_width, is_halfwidth
from src.layout.math_layout import (
    CHAR_WIDTH_RATIO,
    MathElement,
    MathLayoutEngine,
    MathParser,
    MathPlacement,
)
from src.layout.page_layout import ContentArea, PageConfig, PageLayout
from src.layout.placement import CharPlacement, MathSpec
from src.layout.table_layout import detect_pipe_table

if TYPE_CHECKING:
    from src.handwriting.augmentation import HandwritingAugmenter

# 単純な数式（変数列）を本文と同じ手書き経路で描いてよい文字。描画側が確実に字形を
# 持つものに限る。ここに無い記号を含む式は matplotlib で描く（本文経路だと欠損する）。
_PLAIN_MATH_BODY_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " =+-*/<>%:;!?()[]~.,"
    "αβγδεζηθλμνπρστφχψωΓΔΘΛΠΣΦΨΩ"
)

# 見出しレベル → 見出し行の x / その配下の本文の x / 見出し文字の拡大率
_HEADING_X: dict[int, float] = {1: 15.0, 2: 25.0, 3: 35.0}
_BODY_X: dict[int, float] = {1: 25.0, 2: 35.0, 3: 45.0}
_HEADING_FONT_SCALES: dict[int, float] = {1: 1.15, 2: 1.08, 3: 1.0}
_KANJI_ADVANCE_SCALE = 1.08
# 本文の字間トラッキング（font_size 比）。字種に依らず一律に加える。
_LETTER_SPACING_SCALE = 0.05
# 見出し・インデント付き本文の右端（用紙右端からの距離 mm）
_INDENTED_RIGHT_MARGIN = 10.0

_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)\$")
_BLOCK_MATH_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_MATH_OR_BLOCK_RE = re.compile(r"\$\$.*?\$\$|\$[^$]+?\$", re.DOTALL)
_TAG_RE = re.compile(r"\\tag\{[^}]*\}")
_PAGE_BREAK_RE = re.compile(r"^-{3,}$")
_CAPTION_RE = re.compile(r"^:\s+(.+)$")
_NOINDENT_RE = re.compile(r"^\\noindent[ \t]")
# 折り返し前にインライン数式を 1 文字に畳むための私用領域コードポイント
_INLINE_MATH_PLACEHOLDER_BASE = 0xE000


def _is_kanji(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2CEAF
    )


def normalize_body_punctuation(text: str) -> str:
    """本文の句読点を全角「，」「．」に統一する（数式内の ``.`` ``,`` は変えない）。"""

    def _normalize(seg: str) -> str:
        seg = seg.replace(",", "，").replace("、", "，")
        return seg.replace(".", "．").replace("。", "．")

    result: list[str] = []
    last_end = 0
    for m in _MATH_OR_BLOCK_RE.finditer(text):
        result.append(_normalize(text[last_end : m.start()]))
        result.append(m.group(0))
        last_end = m.end()
    result.append(_normalize(text[last_end:]))
    return "".join(result)


def split_inline_math(text: str) -> list[tuple[str, str]]:
    """``[("text", ...), ("math", ...), ...]`` へ分割する。空の ``$$`` は捨てる。"""
    segments: list[tuple[str, str]] = []
    last_end = 0
    for m in _INLINE_MATH_RE.finditer(text):
        if m.start() > last_end:
            segments.append(("text", text[last_end : m.start()]))
        if m.group(1):
            segments.append(("math", m.group(1)))
        last_end = m.end()
    if last_end < len(text):
        segments.append(("text", text[last_end:]))
    return segments


# =============================================================================
# 第 1 パス: 行レコードへの分解
# =============================================================================

LineKind = Literal["text", "block_math", "table", "page_break"]


@dataclass
class Line:
    """組版の 1 行（ブロック数式・表は複数の罫線行を占める 1 レコード）。

    Attributes:
        kind: 行の種類。
        text: 本文（インライン数式は ``$...$`` のまま）。
        heading_level: 見出しレベル（0=本文）。
        body_level: 直近の見出しレベル（本文のインデント段）。
        para_start: 段落の先頭行か（字下げ判定に使う）。
        no_indent: ``\\noindent`` 指定の段落先頭か。
        math_src: ブロック数式の LaTeX。
        table_rows: 表のセル（先頭がヘッダ）。
        caption: 表キャプション（``""`` はなし）。
        caption_above: キャプションを表の上に置くか。
    """

    kind: LineKind = "text"
    text: str = ""
    heading_level: int = 0
    body_level: int = 0
    para_start: bool = False
    no_indent: bool = False
    math_src: str = ""
    table_rows: list[list[str]] = field(default_factory=list)
    caption: str = ""
    caption_above: bool = False


@dataclass
class _Table:
    rows: list[list[str]]
    caption: str
    caption_above: bool


def _stash_block_math(text: str) -> tuple[list[str | int], list[str]]:
    """``$$...$$``（改行を含んでよい）を段落から切り出し、単独段落の番号へ置き換える。

    Returns:
        ``(paragraphs, maths)``。``paragraphs`` の int 要素は ``maths`` の番号。
    """
    maths: list[str] = []
    paragraphs: list[str | int] = []
    last_end = 0
    for m in _BLOCK_MATH_RE.finditer(text):
        paragraphs.extend(text[last_end : m.start()].split("\n"))
        paragraphs.append(len(maths))
        maths.append(m.group(1).strip())
        last_end = m.end()
    paragraphs.extend(text[last_end:].split("\n"))
    # 数式の前後の改行で生じる空段落は捨てる（元の段落区切りと重複するため）
    cleaned: list[str | int] = []
    for i, p in enumerate(paragraphs):
        if p == "":
            next_is_math = i + 1 < len(paragraphs) and isinstance(paragraphs[i + 1], int)
            prev_is_math = bool(cleaned) and isinstance(cleaned[-1], int)
            if next_is_math or prev_is_math:
                continue
        cleaned.append(p)
    return cleaned, maths


def _collapse_tables(paragraphs: list[str | int]) -> list[str | int | _Table]:
    """パイプ表（と直前/直後の ``: キャプション`` 行）を 1 要素に畳む。"""

    def caption_of(p: str | int) -> str | None:
        if not isinstance(p, str):
            return None
        m = _CAPTION_RE.match(p.strip())
        return m.group(1).strip() if m else None

    def table_at(i: int) -> tuple[list[list[str]], int] | None:
        # 表検出は文字列の連続区間だけを見る（数式番号を挟むと表ではない）
        j = i
        while j < len(paragraphs) and isinstance(paragraphs[j], str):
            j += 1
        return detect_pipe_table([str(p) for p in paragraphs[i:j]], 0)

    out: list[str | int | _Table] = []
    i = 0
    while i < len(paragraphs):
        cap = caption_of(paragraphs[i])
        if cap is not None and i + 1 < len(paragraphs) and (tbl := table_at(i + 1)):
            rows, consumed = tbl
            out.append(_Table(rows, cap, caption_above=True))
            i += 1 + consumed
            continue
        tbl = table_at(i) if isinstance(paragraphs[i], str) else None
        if tbl is None:
            out.append(paragraphs[i])
            i += 1
            continue
        rows, consumed = tbl
        i += consumed
        caption = ""
        if i < len(paragraphs) and (cap_below := caption_of(paragraphs[i])) is not None:
            caption = cap_below
            i += 1
        out.append(_Table(rows, caption, caption_above=False))
    return out


# =============================================================================
# 組版エンジン
# =============================================================================


class Typesetter:
    def __init__(
        self,
        page_config: PageConfig,
        font_size: float | None = None,
        augmenter: HandwritingAugmenter | None = None,
    ) -> None:
        self.config = page_config
        self.layout = PageLayout(page_config)
        self.font_size = font_size if font_size is not None else page_config.line_spacing * 0.9
        self.augmenter = augmenter

    # --- 字送り ---

    def body_char_advance(self, ch: str) -> float:
        """本文 1 文字の字送り(mm)。"""
        letter_spacing = self.font_size * _LETTER_SPACING_SCALE
        if is_halfwidth(ch):
            return self.font_size * 0.55 + letter_spacing
        if _is_kanji(ch):
            return self.font_size * _KANJI_ADVANCE_SCALE + letter_spacing
        return self.font_size * (0.45 + 0.55 * effective_char_scale(ch)) + letter_spacing

    def _char_advance(self, ch: str, is_heading: bool, line_font_size: float) -> float:
        if is_heading:
            return (
                line_font_size * effective_char_scale(ch) + line_font_size * _LETTER_SPACING_SCALE
            )
        return self.body_char_advance(ch)

    def _line_right_x(self, area: ContentArea, is_heading: bool, body_level: int) -> float:
        if is_heading or body_level > 0:
            return self.config.paper_size[0] - _INDENTED_RIGHT_MARGIN
        return area.x + area.width

    # --- インライン数式の寸法 ---

    def _inline_math_draw_size(self, math_src: str) -> tuple[float, float]:
        """インライン数式の実描画 (幅, 高さ) mm。幅予約と描画の単一ソース。

        高さ = インク高の em 比 × font_size（論理高で描くと小文字が大きすぎる）。
        """
        from src.render.math_image import formula_aspect, formula_ink_em

        body_src = _TAG_RE.sub("", math_src)
        h_mm = formula_ink_em(body_src) * self.font_size
        return h_mm * formula_aspect(body_src), h_mm

    def inline_math_width(self, math_src: str) -> float:
        """インライン数式が占める幅(mm)。折り返しとカーソル前進の両方がこれを使う。"""
        elements = [e for e in MathParser.parse(math_src) if e.type != "tag"]
        if _is_plain_math(elements):
            return sum(self.body_char_advance(ch) for ch in _plain_math_text(elements))
        draw_w, _ = self._inline_math_draw_size(math_src)
        if draw_w > 0:
            return draw_w
        return MathLayoutEngine.layout(elements, x=0.0, y=0.0, font_size=self.font_size).width

    # --- 第 1 パス ---

    def parse_document(self, text: str) -> list[Line]:
        """入力を行レコードの列へ分解する（本文は折り返し済み）。"""
        area = self.layout.content_area()
        paragraphs, maths = _stash_block_math(text)
        lines: list[Line] = []
        body_level = 0

        for para in _collapse_tables(paragraphs):
            if isinstance(para, int):
                if maths[para]:
                    lines.append(
                        Line("block_math", math_src=maths[para], para_start=True,
                             body_level=body_level)
                    )  # fmt: skip
                continue
            if isinstance(para, _Table):
                lines.append(
                    Line("table", table_rows=para.rows, caption=para.caption,
                         caption_above=para.caption_above, para_start=True,
                         body_level=body_level)
                )  # fmt: skip
                continue
            if _PAGE_BREAK_RE.match(para.strip()):
                lines.append(Line("page_break"))
                continue

            heading_level, body = _split_heading(para)
            no_indent = False
            if heading_level == 0 and _NOINDENT_RE.match(body):
                no_indent = True
                body = body[len("\\noindent") + 1 :]
            if heading_level > 0:
                # 見出しの前に 1 行空ける（文書先頭の見出しを除く）
                if lines and not (len(lines) == 1 and lines[0].text == ""):
                    lines.append(Line())
                body_level = heading_level
            if not body:
                lines.append(Line(heading_level=heading_level, body_level=body_level,
                                  para_start=True))  # fmt: skip
                continue

            wrapped = self._wrap(body, heading_level, body_level, area)
            for i, line_text in enumerate(wrapped):
                lines.append(
                    Line(
                        text=line_text,
                        heading_level=heading_level,
                        body_level=body_level,
                        para_start=i == 0,
                        no_indent=no_indent and i == 0,
                    )
                )
        return lines

    def _wrap(self, text: str, heading_level: int, body_level: int, area: ContentArea) -> list[str]:
        """禁則処理付きで折り返す。インライン数式は 1 文字に畳んで幅を予約する。"""
        placeholders: dict[str, str] = {}
        parts: list[str] = []
        last_end = 0
        for idx, match in enumerate(_INLINE_MATH_RE.finditer(text)):
            parts.append(text[last_end : match.start()])
            ph = chr(_INLINE_MATH_PLACEHOLDER_BASE + idx)
            placeholders[ph] = match.group(1)
            parts.append(ph)
            last_end = match.end()
        parts.append(text[last_end:])

        is_heading = heading_level > 0
        if is_heading:
            line_x = _HEADING_X.get(heading_level, area.x)
            line_font_size = self.font_size * _HEADING_FONT_SCALES[heading_level]
            char_width: Callable[[str], float] = lambda ch: self._char_advance(  # noqa: E731
                ch, True, line_font_size
            )
        else:
            line_x = _BODY_X.get(body_level, area.x)
            char_width = self.body_char_advance

        def width_fn(ch: str) -> float:
            if ch in placeholders:
                return self.inline_math_width(placeholders[ch])
            return char_width(ch)

        line_width = self._line_right_x(area, is_heading, body_level) - line_x
        broken = break_paragraph_by_width("".join(parts), line_width, width_fn)
        restored = [
            "".join(f"${placeholders[ch]}$" if ch in placeholders else ch for ch in b)
            for b in broken
        ]
        return restored or [""]

    # --- 第 2 パス ---

    def typeset(self, text: str) -> list[list[CharPlacement]]:
        """テキストをページごとの配置要素リストへ組版する。"""
        if not text:
            return [[]]
        text = normalize_body_punctuation(text)
        area = self.layout.content_area()
        rows = self.layout.line_positions()

        pages: list[list[CharPlacement]] = []
        page: list[CharPlacement] = []
        row = 0

        def new_page() -> None:
            nonlocal page, row
            pages.append(page)
            page = []
            row = 0

        for line in self.parse_document(text):
            if row >= len(rows):
                new_page()
            if line.kind == "page_break":
                # 先頭が空ページにならないよう、内容があるときだけ改ページする
                if page or row > 0:
                    new_page()
                continue
            if line.kind in ("block_math", "table"):
                consumed = self._place_block(line, row, rows, area, page)
                if consumed == -1:  # 残り行不足 → 次ページ先頭へ
                    new_page()
                    consumed = self._place_block(line, 0, rows, area, page)
                    if consumed == -1:  # 1 ページに収まらない（無限ループ回避）
                        consumed = 1
                row += consumed
                continue
            page.extend(self._place_line(line, rows[row], area, is_page_first=row == 0))
            row += 1

        pages.append(page)
        return pages

    def _place_block(
        self,
        line: Line,
        row: int,
        rows: list[float],
        area: ContentArea,
        out: list[CharPlacement],
    ) -> int:
        if line.kind == "block_math":
            return self._place_block_math(line.math_src, row, rows, area, out)
        return self._place_table(line, row, rows, area, out)

    def _place_line(
        self, line: Line, y: float, area: ContentArea, *, is_page_first: bool
    ) -> list[CharPlacement]:
        """本文・見出しの 1 行を配置する。"""
        is_heading = line.heading_level > 0
        if is_heading:
            line_font_size = self.font_size * _HEADING_FONT_SCALES[line.heading_level]
            x = _HEADING_X.get(line.heading_level, area.x)
        else:
            line_font_size = self.font_size
            x = _BODY_X.get(line.body_level, area.x) if line.body_level > 0 else area.x
        if line.para_start and not is_page_first and not is_heading and not line.no_indent:
            x += self.font_size  # 段落先頭の字下げ

        aug = self.augmenter
        line_y = y + aug.next_line_baseline() if aug else y
        line_density = aug.line_density_scale() if aug else 1.0

        segments = split_inline_math(line.text)
        neutral_remaining = sum(
            self.inline_math_width(content)
            if kind == "math"
            else sum(self._char_advance(ch, is_heading, line_font_size) for ch in content)
            for kind, content in segments
        )
        line_right_x = self._line_right_x(area, is_heading, line.body_level)

        out: list[CharPlacement] = []
        prev_halfwidth = False
        for kind, content in segments:
            if kind == "math":
                start_x = x
                x = self._place_inline_math(content, x, line_y, out)
                neutral_remaining -= x - start_x
                prev_halfwidth = False
                continue
            for ch in content:
                cur_halfwidth = is_halfwidth(ch)
                # 見出しは見出し用に拡大した line_font_size を基準に字種×密度を掛ける
                char_font_size = line_font_size * effective_char_scale(ch)
                advance = self._char_advance(ch, is_heading, line_font_size)
                neutral_remaining -= advance
                if aug is None:
                    out.append(CharPlacement(ch, x, y, char_font_size))
                    x += advance
                    prev_halfwidth = cur_halfwidth
                    continue

                spacing_jitter = aug.next_char_spacing()
                size = max(char_font_size * aug.next_char_size_scale(), char_font_size * 0.8)
                slant = aug.next_char_slant()
                baseline = aug.next_char_baseline()
                density = line_density * aug.char_density_scale()
                spacing_factor = density * (0.5 if prev_halfwidth and cur_halfwidth else 1.0)
                width = advance * density
                if density > 1.0:
                    # 密度で広げても行末からはみ出さない（残りの字の予約幅は確保する）
                    width = min(width, max(advance, line_right_x - x - neutral_remaining))
                out.append(
                    CharPlacement(
                        ch, x + spacing_jitter * spacing_factor, line_y + baseline, size, slant
                    )
                )
                prev_halfwidth = cur_halfwidth
                x += width
        return out

    def _place_inline_math(
        self, math_src: str, x: float, y: float, out: list[CharPlacement]
    ) -> float:
        """インライン数式を配置し、次のカーソル x を返す。"""
        elements = [e for e in MathParser.parse(math_src) if e.type != "tag"]
        if _is_plain_math(elements):
            # 単純な変数列は本文と同じ手書き経路で描く（書体を本文に揃える）
            for ch in _plain_math_text(elements):
                out.append(CharPlacement(ch, x, y, self.font_size))
                x += self.body_char_advance(ch)
            return x
        box = MathLayoutEngine.layout(elements, x=x, y=y, font_size=self.font_size)
        draw_w, h_mm = self._inline_math_draw_size(math_src)
        # font_size は先頭要素のもの（ベースライン揃えの基準。分数始まりの式では縮小サイズ）
        font_size = box.placements[0].font_size if box.placements else self.font_size
        spec = MathSpec(math_src, (x, y, draw_w, h_mm), "baseline")
        out.append(CharPlacement("", x, y, font_size, math=spec))
        return x + draw_w

    def _place_block_math(
        self,
        math_src: str,
        row: int,
        rows: list[float],
        area: ContentArea,
        out: list[CharPlacement],
    ) -> int:
        """ブロック数式を確保した行範囲（最低 2 行）の縦中央に配置する。

        ``\\\\`` で多段、``\\tag{}`` は数式本体の直後に式番号を置く。

        Returns:
            消費した行数。残り行不足なら -1。
        """
        from src.render.math_image import formula_draw_width_mm

        elements = MathParser.parse(math_src)
        tag_elem = next((e for e in elements if e.type == "tag"), None)
        body_src = _TAG_RE.sub("", math_src).strip()
        groups = _split_by_linebreak(_strip_tag(elements))
        line_spacing = self.config.line_spacing

        boxes = [MathLayoutEngine.layout(g, x=0.0, y=0.0, font_size=self.font_size) for g in groups]
        if boxes:
            total_height = boxes[0].ascent + boxes[-1].descent + line_spacing * (len(boxes) - 1)
        else:
            total_height = self.font_size
        required_rows = max(2, math.ceil(total_height / line_spacing))
        if len(rows) - row < required_rows:
            return -1

        center_y = (rows[row] + rows[row + required_rows - 1]) / 2
        last_baseline_y = center_y
        body_right = area.x + area.width
        for i, (group, box) in enumerate(zip(groups, boxes, strict=True)):
            baseline_y = center_y + (len(boxes) - 1) / 2 * line_spacing - i * line_spacing
            last_baseline_y = baseline_y
            # 中央寄せは実描画幅で行う。本文幅を超える式は縮小して収める。
            h = box.ascent + box.descent
            draw_w = formula_draw_width_mm(body_src, h)
            scale = 1.0
            if draw_w > area.width and draw_w > 0:
                scale = area.width / draw_w
                h *= scale
                draw_w = area.width
            center_x = area.x + (area.width - draw_w) / 2
            placed = MathLayoutEngine.layout(
                group, x=center_x, y=baseline_y, font_size=self.font_size
            )
            bbox = (center_x, baseline_y - placed.descent * scale, draw_w, h)
            if placed.placements:
                first = placed.placements[0]
                out.append(
                    CharPlacement(
                        "", first.x, first.y, first.font_size, math=MathSpec(body_src, bbox)
                    )
                )
            body_right = center_x + draw_w

        if tag_elem is not None:
            tag_width = MathLayoutEngine.layout(
                [tag_elem], x=0, y=0, font_size=self.font_size
            ).width
            # 式番号は本体の直後（1 文字空け）。本文幅を超える場合のみ右端へ寄せる。
            tag_x = min(body_right + self.font_size, area.x + area.width - tag_width)
            tag_box = MathLayoutEngine.layout(
                [tag_elem], x=tag_x, y=last_baseline_y, font_size=self.font_size
            )
            out.extend(_text_placements(tag_box.placements))
        return required_rows

    def _place_table(
        self,
        line: Line,
        row: int,
        rows: list[float],
        area: ContentArea,
        out: list[CharPlacement],
    ) -> int:
        """パイプ表を罫線＋セル文字として本文幅の中央に配置する（1 表行 = 1 罫線行）。

        列幅は各列セルの実字送りの最大＋パディング。本文幅を超える場合は一律縮小。
        キャプションは表幅の中央に置き、+1 行消費する。

        Returns:
            消費した行数。残り行不足なら -1。
        """
        table = line.table_rows
        n_rows = len(table)
        if n_rows == 0:
            return 1
        n_cols = max(len(r) for r in table)
        cap_rows = 1 if line.caption else 0
        if len(rows) - row < n_rows + cap_rows:
            return -1
        first_row = row + (1 if line.caption and line.caption_above else 0)

        fs = self.font_size
        pad = fs * 0.3
        col_w = [
            max(max(sum(self.body_char_advance(ch) for ch in r[c]) for r in table), fs) + 2 * pad
            for c in range(n_cols)
        ]
        total_w = sum(col_w)
        scale = min(1.0, area.width / total_w) if total_w > 0 else 1.0
        col_w = [w * scale for w in col_w]

        # 横罫線は用紙の罫線に一致させる（各表行が 1 罫線帯を占有）
        ys = [rows[first_row] + self.config.line_spacing]
        ys += [rows[first_row + r] for r in range(n_rows)]
        x_start = area.x + max(0.0, (area.width - sum(col_w)) / 2)
        xs = [x_start]
        for w in col_w:
            xs.append(xs[-1] + w)

        for y in ys:
            out.append(CharPlacement("", xs[0], y, fs, line_segment=(xs[0], y, xs[-1], y)))
        for x in xs:
            out.append(CharPlacement("", x, ys[0], fs, line_segment=(x, ys[0], x, ys[-1])))

        cell_fs = fs * scale
        for r in range(n_rows):
            baseline = rows[first_row + r]
            for c in range(n_cols):
                cx = xs[c] + pad * scale
                for ch in table[r][c]:
                    if ch != " ":
                        out.append(CharPlacement(ch, cx, baseline, cell_fs))
                    cx += self.body_char_advance(ch) * scale

        if line.caption:
            cap_row = row if line.caption_above else first_row + n_rows
            cx = (xs[0] + xs[-1]) / 2 - sum(self.body_char_advance(ch) for ch in line.caption) / 2
            for ch in line.caption:
                if ch != " ":
                    out.append(CharPlacement(ch, cx, rows[cap_row], fs))
                cx += self.body_char_advance(ch)
        return n_rows + cap_rows


def _split_heading(para: str) -> tuple[int, str]:
    for level in (3, 2, 1):
        prefix = "#" * level
        if para.startswith(prefix):
            return level, para[level:].strip()
    return 0, para


def _is_plain_math(elements: list[MathElement]) -> bool:
    """添字・分数・演算子語などを含まない単純な変数列か（本文経路で描けるか）。"""
    if not elements or not all(e.type in ("text", "symbol") for e in elements):
        return False
    return all(ch in _PLAIN_MATH_BODY_CHARS for e in elements for ch in e.content)


def _plain_math_text(elements: list[MathElement]) -> str:
    return "".join(e.content for e in elements)


def _strip_tag(elements: list[MathElement]) -> list[MathElement]:
    """``\\tag`` 要素と、それに隣接する空白を除く（本体の中央寄せが式番号に影響しない）。"""
    result: list[MathElement] = []
    for elem in elements:
        if elem.type == "tag":
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
    while result and result[-1].type == "text" and result[-1].content.strip() == "":
        result.pop()
    if result and result[-1].type == "text":
        stripped = result[-1].content.rstrip()
        if stripped != result[-1].content:
            result[-1] = MathElement(type="text", content=stripped)
    return result


def _split_by_linebreak(elements: list[MathElement]) -> list[list[MathElement]]:
    groups: list[list[MathElement]] = []
    current: list[MathElement] = []
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


def _text_placements(placements: list[MathPlacement]) -> list[CharPlacement]:
    """数式レイアウト結果（式番号など文字だけのもの）を 1 文字ずつの配置へ展開する。"""
    out: list[CharPlacement] = []
    for mp in placements:
        for i, ch in enumerate(mp.text):
            out.append(
                CharPlacement(ch, mp.x + i * mp.font_size * CHAR_WIDTH_RATIO, mp.y, mp.font_size)
            )
    return out
