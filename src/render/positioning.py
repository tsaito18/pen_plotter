"""単位系の字形を配置情報（mm）へスケール・移動する。

縦方向の基準: ``placement.y`` は行ボックスの下端。通常字形は行間 ``line_spacing`` の
帯に縦中央寄せし、英字はベースラインを漢字の下端に揃える。
"""

from __future__ import annotations

import numpy as np

from src.geometry import Stroke, rotate_about
from src.glyphs.geometric import LATIN_CAP_TOP
from src.layout.char_metrics import char_type_scale
from src.layout.line_breaking import is_halfwidth
from src.layout.placement import CharPlacement

# 半角セル幅 = font_size × この係数。組版は字送りを font*0.55 で予約し、字種係数
# (半角=0.8)を font_size に焼くため fs=font*0.8。予約と一致させるには
# fs*(0.55/0.8)≈0.7。これより狭いと数字・英字が横律速で縦に潰れる。
HALFWIDTH_CELL_FACTOR = 0.7
FULLWIDTH_CELL_FACTOR = 0.95

_COMMA_CHARS = ("、", ",", "，")
_PERIOD_CHARS = (".", "。", "．")
_MIDDLE_DOT = "・"


def position_strokes(
    strokes: list[Stroke],
    placement: CharPlacement,
    line_spacing: float,
    *,
    logical_latin: bool = False,
) -> list[Stroke]:
    """字形を実 bbox フィットで配置する（句読点・中黒・英字は専用の配置）。

    Args:
        strokes: 単位系字形。
        placement: 配置情報。
        line_spacing: 行間(mm)。
        logical_latin: True かつ英字なら、実 bbox でなく論理座標（cap height・
            ベースライン）でフィットし、大文字/小文字の高さ差とディセンダを保つ。
    """
    if not strokes:
        return []
    char = placement.char
    if char in _COMMA_CHARS:
        return [_position_comma(strokes, placement, line_spacing)]
    if char in _PERIOD_CHARS:
        return [_position_period(placement, line_spacing)]
    if char == _MIDDLE_DOT:
        return _position_middle_dot(strokes, placement, line_spacing)
    if logical_latin and char.isascii() and char.isalpha():
        return _position_latin_logical(strokes, placement, line_spacing)

    all_pts = np.concatenate(strokes, axis=0)
    mins = all_pts.min(axis=0)
    ranges = all_pts.max(axis=0) - mins

    # サイズ倍率(字種×密度)は組版が font_size に 1 回だけ焼く。ここでは再適用しない。
    fs = placement.font_size
    cell_width = fs * (HALFWIDTH_CELL_FACTOR if is_halfwidth(char) else FULLWIDTH_CELL_FACTOR)

    scale_w = cell_width / ranges[0] if ranges[0] > 1e-6 else float("inf")
    scale_h = fs / ranges[1] if ranges[1] > 1e-6 else float("inf")
    scale = min(scale_w, scale_h)

    rendered_w = ranges[0] * scale
    rendered_h = ranges[1] * scale
    x_offset = placement.x + (cell_width - rendered_w) / 2
    # 小書き仮名・句読点は行ボックス中央だと浮くため下寄せ（字種で判定）
    if char_type_scale(char) < 0.6:
        y_offset = placement.y + 0.1 * line_spacing
    else:
        y_offset = placement.y + (line_spacing - rendered_h) / 2

    offset = np.array([x_offset, y_offset])
    positioned = [(stroke - mins) * scale + offset for stroke in strokes]
    if placement.slant:
        center = (x_offset + rendered_w / 2, y_offset + rendered_h / 2)
        positioned = rotate_about(positioned, placement.slant, center)
    return positioned


def _position_latin_logical(
    strokes: list[Stroke], placement: CharPlacement, line_spacing: float
) -> list[Stroke]:
    """英字を論理座標でフィットする（cap top=font_size 高、ベースライン=CJK 下端）。"""
    all_pts = np.concatenate(strokes, axis=0)
    x_min = float(all_pts[:, 0].min())
    x_range = float(all_pts[:, 0].max()) - x_min

    fs = placement.font_size
    cell_width = fs * HALFWIDTH_CELL_FACTOR
    scale_h = fs / LATIN_CAP_TOP
    scale_w = cell_width / x_range if x_range > 1e-6 else float("inf")
    scale = min(scale_h, scale_w)

    rendered_w = x_range * scale
    x_offset = placement.x + (cell_width - rendered_w) / 2
    baseline_y = placement.y + (line_spacing - fs) / 2

    positioned: list[Stroke] = []
    for stroke in strokes:
        shifted = stroke.copy()
        shifted[:, 0] = (stroke[:, 0] - x_min) * scale + x_offset
        shifted[:, 1] = stroke[:, 1] * scale + baseline_y
        positioned.append(shifted)
    if placement.slant:
        center = (x_offset + rendered_w / 2, baseline_y + fs / 2)
        positioned = rotate_about(positioned, placement.slant, center)
    return positioned


def _position_comma(strokes: list[Stroke], placement: CharPlacement, line_spacing: float) -> Stroke:
    all_pts = np.concatenate(strokes, axis=0)
    mins = all_pts.min(axis=0)
    ranges = all_pts.max(axis=0) - mins

    fs = placement.font_size
    cell_width = fs * 0.55 if is_halfwidth(placement.char) else fs * FULLWIDTH_CELL_FACTOR
    target_h = fs * 0.3675
    target_w = target_h * 0.72
    scale_w = target_w / ranges[0] if ranges[0] > 1e-6 else float("inf")
    scale_h = target_h / ranges[1] if ranges[1] > 1e-6 else float("inf")
    scale = min(scale_w, scale_h)

    rendered_w = ranges[0] * scale
    rendered_h = ranges[1] * scale
    x_offset = placement.x + (cell_width - rendered_w) / 2
    y_offset = placement.y + line_spacing * 0.1
    stroke = (strokes[0] - mins) * scale + np.array([x_offset, y_offset])
    if placement.slant:
        center = (x_offset + rendered_w / 2, y_offset + rendered_h / 2)
        stroke = rotate_about([stroke], placement.slant, center)[0]
    return stroke


def _position_middle_dot(
    strokes: list[Stroke], placement: CharPlacement, line_spacing: float
) -> list[Stroke]:
    cell_width = placement.font_size * FULLWIDTH_CELL_FACTOR
    scale = cell_width * 0.5 / 0.3
    center = np.array([placement.x + cell_width / 2, placement.y + line_spacing / 2])
    raw_center = np.array([0.5, 0.5])
    positioned = [(stroke - raw_center) * scale + center for stroke in strokes]
    if placement.slant:
        positioned = rotate_about(positioned, placement.slant, center)
    return positioned


def _position_period(placement: CharPlacement, line_spacing: float) -> Stroke:
    """句点は丸ではなく短い斜めのダッシュ（ピリオド風の点）として描く。"""
    fs = placement.font_size
    dot_w = min(0.58, max(0.38, fs * 0.08))
    dot_h = dot_w * 0.75
    cx = placement.x + fs * 0.55 * 0.5
    cy = placement.y + line_spacing * 0.14
    return np.array(
        [[cx - dot_w / 2, cy + dot_h / 2], [cx + dot_w / 2, cy - dot_h / 2]],
        dtype=np.float64,
    )
