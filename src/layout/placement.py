"""組版結果（ページ上の描画要素）のデータ型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MathAlign = Literal["center", "baseline"]


@dataclass(frozen=True)
class MathSpec:
    """数式 1 式ぶんの描画指定。

    Attributes:
        source: LaTeX ソース（``$`` なし）。
        bbox: ``(x_left, y, width, height)`` mm。``align="baseline"`` のとき ``y`` は
            本文行ボックスの下端（本文文字の ``placement.y`` と同じ基準）。
        align: ``"center"``=ブロック数式（bbox 中央）/ ``"baseline"``=インライン数式。
    """

    source: str
    bbox: tuple[float, float, float, float]
    align: MathAlign = "center"


@dataclass(frozen=True)
class CharPlacement:
    """ページ上の描画要素 1 つ。文字・罫線（表）・数式のいずれか。

    Attributes:
        char: 描画する文字（罫線・数式では ``""``）。
        x: 文字セル左端(mm)。
        y: 行ボックス下端(mm)。字形は ``y`` から ``line_spacing`` の帯に収まる。
        font_size: 字の目標高(mm)。字種・密度・揺らぎの倍率は焼き込み済み。
        slant: 文字単位の微小傾き(rad)。
        advance: 組版が予約した字送り(mm)。字形はこの幅の中央に置く（None なら字種の既定幅）。
        line_segment: 罫線 ``(x1, y1, x2, y2)``。表の罫線に使う。
        math: 数式の描画指定。
    """

    char: str
    x: float
    y: float
    font_size: float
    slant: float = 0.0
    advance: float | None = None
    line_segment: tuple[float, float, float, float] | None = None
    math: MathSpec | None = None

    @property
    def is_text(self) -> bool:
        return self.line_segment is None and self.math is None
