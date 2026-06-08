"""レイアウト自動診断: □（字形なし）と文字かぶり（重なり）を検出する。

レポート生成前の品質チェック用。組版結果（CharPlacement）を走査し、
- 字形が無く矩形フォールバック（□）になる文字
- 隣接文字・数式の水平方向の重なり（はみ出し）
- 数式が上下の行に食い込む垂直方向の干渉
を洗い出す。実機に流す前にこれで潰す。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.layout.typesetter import CharPlacement
from src.ui.math_skeletonize import _baseline_frac_from_top


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
            lines.append(f"□(字形なし) {len(self.missing_glyphs)}種: {items}")
        else:
            lines.append("□(字形なし): なし")
        if self.overlaps:
            lines.append(f"かぶり {len(self.overlaps)}件:")
            for o in self.overlaps[:40]:
                lines.append(
                    f"  p{o.page} [{o.kind}] {o.a} ⇔ {o.b}  ({o.amount_mm:.1f}mm)"
                )
            if len(self.overlaps) > 40:
                lines.append(f"  …他 {len(self.overlaps) - 40} 件")
        else:
            lines.append("かぶり: なし")
        return "\n".join(lines)


def _x_extent(p: CharPlacement, fallback_w: float) -> tuple[float, float]:
    """配置要素の水平範囲 (x_left, x_right) を返す。数式は math_bbox を使う。"""
    if p.math_bbox is not None:
        x, _y, w, _h = p.math_bbox
        return x, x + w
    return p.x, p.x + fallback_w


def _y_extent(p: CharPlacement, line_spacing: float) -> tuple[float, float]:
    """配置要素の垂直範囲 (y_bottom, y_top) を返す。数式は math_bbox を使う。"""
    if p.math_bbox is not None:
        _x, y, _w, h = p.math_bbox
        if getattr(p, "math_align", "center") == "baseline":
            baseline_frac = _baseline_frac_from_top(p.math_source or "")
            y = y - (1.0 - baseline_frac) * h
        return y, y + h
    # 通常文字は baseline(y) から line_spacing 帯に収まる想定
    return p.y, p.y + line_spacing


def _label(p: CharPlacement) -> str:
    if p.math_source:
        src = p.math_source if len(p.math_source) <= 20 else p.math_source[:17] + "…"
        return f"式「{src}」"
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
        # 描画対象のみ（罫線 line_segment・math_skip は除外。数式は char="" でも含む）
        items = [
            p
            for p in page
            if p.line_segment is None
            and not p.math_skip
            and (p.char != "" or p.math_source is not None)
        ]
        # 行ごと（y でグルーピング、近い y は同一行扱い）
        lines: dict[float, list[CharPlacement]] = {}
        for p in items:
            key = round(p.y, 1)
            lines.setdefault(key, []).append(p)

        # 水平かぶり: 各行内で x 昇順に隣接ペアを確認
        for _y, row in lines.items():
            row = sorted(row, key=lambda p: p.x)
            for a, b in zip(row, row[1:]):
                a_w = b.x - a.x  # 予約スロット幅（次の文字までの距離）
                _, a_right = _x_extent(a, a_w if a_w > 0 else a.font_size)
                b_left, _ = _x_extent(b, b.font_size)
                ov = a_right - b_left
                if ov > h_tol_mm:
                    overlaps.append(
                        Overlap(page_idx + 1, "horizontal", _label(a), _label(b), ov)
                    )

        # 垂直かぶり: インライン数式が行間を超えて隣接行に食い込む。
        # ブロック数式(math_align="center")は複数行を確保済みなので対象外。
        for p in items:
            if p.math_bbox is None or getattr(p, "math_align", "center") == "center":
                continue
            y_bottom, y_top = _y_extent(p, line_spacing)
            # この要素の所属行 y に対し、上下に line_spacing を超えてはみ出す量
            over_top = y_top - (p.y + line_spacing)
            over_bottom = p.y - y_bottom
            limit = line_spacing * v_overhang_ratio
            if over_top > limit:
                overlaps.append(
                    Overlap(page_idx + 1, "vertical", _label(p), "上の行", over_top)
                )
            if over_bottom > limit:
                overlaps.append(
                    Overlap(page_idx + 1, "vertical", _label(p), "下の行", over_bottom)
                )
    return overlaps


def diagnose_layout(pipeline, text: str) -> LayoutReport:
    """テキストを組版し、□（字形なし）とかぶりを検出して :class:`LayoutReport` を返す。

    Args:
        pipeline: ``text_to_placements`` を持つ :class:`PlotterPipeline`。
        text: 診断対象の本文。

    Returns:
        :class:`LayoutReport`。
    """
    pages = pipeline.text_to_placements(text)

    # --- □（字形なし）検出: ユニーク文字を1回ずつ描画してカバレッジを見る ---
    renderer = pipeline._stroke_renderer
    seen: dict[str, int] = {}
    for page in pages:
        for p in page:
            if p.line_segment is not None or not p.char or p.math_source or p.math_skip:
                continue
            seen[p.char] = seen.get(p.char, 0) + 1

    missing: list[MissingGlyph] = []
    for ch, cnt in seen.items():
        renderer._last_coverage.rect_fallback.clear()
        renderer.generate_char_strokes(
            CharPlacement(char=ch, x=0.0, y=0.0, font_size=pipeline._typesetter.font_size)
        )
        if ch in renderer._last_coverage.rect_fallback:
            missing.append(MissingGlyph(ch, cnt))
    missing.sort(key=lambda m: -m.count)

    line_spacing = pipeline._page_config.line_spacing
    overlaps = diagnose_placements(pages, line_spacing)
    return LayoutReport(missing_glyphs=missing, overlaps=overlaps)
