"""ページのプレビュー画像（matplotlib）。

線幅は G-code の Z 補間と同じ接触率（:mod:`src.handwriting.finishing`）から導くため、
払い・はねの抜けや筆圧の濃淡が実機の見え方と一致する。太さは実寸（mm）で、
完全接触のときにペン幅 :attr:`PlotterConfig.pen_width_mm` になる（図の大きさに依らない）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from src.gcode.config import PlotterConfig
from src.geometry import Stroke
from src.handwriting.finishing import (
    NONE,
    arc_length_from_end,
    contact_profile,
    entry_modulation,
    pressure_modulation,
)

# 終端の抜けでも消えない最小幅（ペン幅比）
WIDTH_MIN_RATIO = 0.17
INK_COLOR = "#1a1a1a"
_PT_PER_INCH = 72.0


def stroke_contact(stroke: Stroke, finish: str, config: PlotterConfig) -> np.ndarray:
    """各セグメント（``N-1`` 本）の接触率 ∈(0,1]。G-code の Z 補間と同じ式（2 点未満は空）。"""
    pts = np.asarray(stroke, dtype=float)
    if len(pts) < 2:
        return np.zeros(0)
    floor = 1.0 - config.finish_strength  # 抜きの強さ 0 なら終端も完全接触
    contact = contact_profile(
        finish, arc_length_from_end(pts), config.finish_lift_length_mm, floor, floor
    )
    contact = contact * pressure_modulation(pts, config.pressure_variation)
    contact = contact * entry_modulation(pts, config.entry_length_mm, config.entry_taper)
    return (contact[:-1] + contact[1:]) / 2.0


def stroke_widths(stroke: Stroke, finish: str, config: PlotterConfig) -> list[float]:
    """各セグメント（``N-1`` 本）の線幅(mm)。接触率に比例する（2 点未満は空）。"""
    seg_contact = stroke_contact(stroke, finish, config)
    ratio = WIDTH_MIN_RATIO + (1.0 - WIDTH_MIN_RATIO) * seg_contact
    return (config.pen_width_mm * ratio).tolist()


def draw_stroke(
    ax: Axes, stroke: Stroke, finish: str, config: PlotterConfig, pt_per_mm: float
) -> None:
    if len(stroke) < 2:
        return
    segments = np.stack([stroke[:-1], stroke[1:]], axis=1)
    widths = np.asarray(stroke_widths(stroke, finish, config)) * pt_per_mm
    ax.add_collection(LineCollection(segments, linewidths=widths, colors=INK_COLOR))


def render_page_preview(
    strokes: list[Stroke],
    finishes: list[str],
    save_path: str | Path,
    *,
    config: PlotterConfig,
    ruled_lines: list[Stroke] | None = None,
    background: Path | None = None,
) -> None:
    """1 ページ分のストロークを用紙画像（無ければ白地）の上に描いて保存する。

    ``ruled_lines`` は背景画像が無いときだけ描く（スキャン画像には罫線がある）。
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches

    w, h = config.paper_width, config.paper_height
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    try:
        if background is not None and background.exists():
            from PIL import Image

            with Image.open(background) as img:
                ax.imshow(img, extent=[0, w, 0, h], aspect="auto", zorder=0)
        else:
            ax.add_patch(patches.Rectangle((0, 0), w, h, facecolor="white", edgecolor="black"))
            for line in ruled_lines or []:
                ax.plot(line[:, 0], line[:, 1], color="#9ab", linewidth=0.3)
        ax.add_patch(
            patches.Rectangle((0, 0), w, h, linewidth=1.0, edgecolor="black", facecolor="none")
        )
        ax.set_xlim(-2, w + 2)
        ax.set_ylim(-2, h + 2)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        # 線幅(mm)を pt に直すため、確定した軸の実寸（インチ）から 1mm あたりの pt を求める
        fig.canvas.draw()
        box = ax.get_window_extent()
        x0, x1 = ax.get_xlim()
        pt_per_mm = box.width / fig.dpi * _PT_PER_INCH / (x1 - x0)
        for i, stroke in enumerate(strokes):
            finish = finishes[i] if i < len(finishes) else NONE
            draw_stroke(ax, stroke, finish, config, pt_per_mm)
        fig.savefig(str(save_path), dpi=300)
    finally:
        plt.close(fig)
