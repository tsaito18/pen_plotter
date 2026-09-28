"""レイアウト自動診断: 字形の無い文字と文字かぶり（重なり）を検出する。

レポート生成前の品質チェック用。組版結果（CharPlacement）を走査し、
- 字形が無く空白化する文字
- 隣接文字・数式の水平方向の重なり（はみ出し）
- 数式が上下の行に食い込む垂直方向の干渉
を洗い出す。実機に流す前にこれで潰す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

from src.layout.placement import CharPlacement
from src.render.math_image import baseline_frac_from_top

if TYPE_CHECKING:
    from src.pipeline import PlotterPipeline


@dataclass
class MissingGlyph:
    char: str
    count: int


@dataclass
class Overlap:
    page: int
    kind: str  # "horizontal" | "vertical"
    a: str  # 手前/上の要素の説明
    b: str  # 後/下の要素の説明
    amount_mm: float


@dataclass
class LayoutReport:
    missing_glyphs: list[MissingGlyph] = field(default_factory=list)
    overlaps: list[Overlap] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_glyphs and not self.overlaps

    def summary(self) -> str:
        lines = []
        if self.missing_glyphs:
            items = "、".join(f"{m.char!r}×{m.count}" for m in self.missing_glyphs)
            lines.append(f"欠損/空白化 {len(self.missing_glyphs)}種: {items}")
        else:
            lines.append("欠損/空白化: なし")
        if self.overlaps:
            lines.append(f"かぶり {len(self.overlaps)}件:")
            for o in self.overlaps[:40]:
                lines.append(f"  p{o.page} [{o.kind}] {o.a} ⇔ {o.b}  ({o.amount_mm:.1f}mm)")
            if len(self.overlaps) > 40:
                lines.append(f"  …他 {len(self.overlaps) - 40} 件")
        else:
            lines.append("かぶり: なし")
        return "\n".join(lines)


def _x_extent(p: CharPlacement, fallback_w: float) -> tuple[float, float]:
    """配置要素の水平範囲 (x_left, x_right)。数式は bbox を使う。"""
    if p.math is not None:
        x, _y, w, _h = p.math.bbox
        return x, x + w
    return p.x, p.x + fallback_w


def _y_extent(p: CharPlacement, line_spacing: float) -> tuple[float, float]:
    """配置要素の垂直範囲 (y_bottom, y_top)。数式は bbox（インラインはベースライン補正）。"""
    if p.math is not None:
        _x, y, _w, h = p.math.bbox
        if p.math.align == "baseline":
            bf = baseline_frac_from_top(p.math.source)
            y = y + (line_spacing - p.font_size) / 2 - (1.0 - bf) * h
        return y, y + h
    return p.y, p.y + line_spacing


def _label(p: CharPlacement) -> str:
    if p.math is not None:
        src = p.math.source
        return f"式「{src if len(src) <= 20 else src[:17] + '…'}」"
    return f"「{p.char}」"


def diagnose_placements(
    pages: list[list[CharPlacement]],
    line_spacing: float,
    *,
    h_tol_mm: float = 0.6,
    v_overhang_ratio: float = 0.6,
) -> list[Overlap]:
    """配置結果から水平・垂直のかぶりを検出する。

    水平: 同一行(同じ y)で隣り合う要素の x 範囲が ``h_tol_mm`` を超えて重なる。
    垂直: 数式 bbox の高さが行間を超え、隣接行の帯へ ``v_overhang_ratio`` 以上
    食い込む（インライン分数・長い式が上下行に干渉するケース）。
    """
    overlaps: list[Overlap] = []
    for page_idx, page in enumerate(pages):
        # 描画対象のみ（罫線は除外）
        items = [p for p in page if p.line_segment is None and (p.char or p.math is not None)]
        # 行ごと（y でグルーピング、近い y は同一行扱い）
        lines: dict[float, list[CharPlacement]] = {}
        for p in items:
            key = round(p.y, 1)
            lines.setdefault(key, []).append(p)

        # 水平かぶり: 各行内で x 昇順に隣接ペアを確認
        for row in lines.values():
            row = sorted(row, key=lambda p: p.x)
            for a, b in pairwise(row):
                a_w = b.x - a.x  # 予約スロット幅（次の文字までの距離）
                _, a_right = _x_extent(a, a_w if a_w > 0 else a.font_size)
                b_left, _ = _x_extent(b, b.font_size)
                ov = a_right - b_left
                if ov > h_tol_mm:
                    overlaps.append(Overlap(page_idx + 1, "horizontal", _label(a), _label(b), ov))

        # 垂直かぶり: インライン数式が行間を超えて隣接行に食い込む。
        # ブロック数式(math_align="center")は複数行を確保済みなので対象外。
        for p in items:
            if p.math is None or p.math.align == "center":
                continue
            y_bottom, y_top = _y_extent(p, line_spacing)
            # この要素の所属行 y に対し、上下に line_spacing を超えてはみ出す量
            over_top = y_top - (p.y + line_spacing)
            over_bottom = p.y - y_bottom
            limit = line_spacing * v_overhang_ratio
            if over_top > limit:
                overlaps.append(Overlap(page_idx + 1, "vertical", _label(p), "上の行", over_top))
            if over_bottom > limit:
                overlaps.append(Overlap(page_idx + 1, "vertical", _label(p), "下の行", over_bottom))
    return overlaps


def diagnose_layout(pipeline: PlotterPipeline, text: str) -> LayoutReport:
    """テキストを組版し、字形の無い文字とかぶりを検出する。"""
    pages = pipeline.typeset(text)
    counts: dict[str, int] = {}
    for page in pages:
        for p in page:
            if p.is_text and p.char:
                counts[p.char] = counts.get(p.char, 0) + 1

    renderer = pipeline.renderer
    missing: list[MissingGlyph] = []
    for ch, count in counts.items():
        renderer.coverage.missing_glyphs.clear()
        renderer.render(CharPlacement(ch, 0.0, 0.0, pipeline.typesetter.font_size))
        if ch in renderer.coverage.missing_glyphs:
            missing.append(MissingGlyph(ch, count))
    missing.sort(key=lambda m: -m.count)
    overlaps = diagnose_placements(pages, pipeline.page_config.line_spacing)
    return LayoutReport(missing_glyphs=missing, overlaps=overlaps)
