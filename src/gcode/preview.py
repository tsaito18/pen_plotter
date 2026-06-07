from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection

from src.gcode.config import PlotterConfig
from src.model.stroke_finishing import (
    arc_length_from_end,
    contact_profile,
    entry_modulation,
    pressure_modulation,
)

# 入筆ランプ区間長(mm)。PlotterConfig.entry_length_mm と揃える。
PREVIEW_ENTRY_LENGTH_MM = 0.7

Stroke = npt.NDArray[np.float64]

# プレビュー線幅の上限（完全接触）/ 下限（終端の抜け）。
# 上限 0.9 は実機の単線太さに相当。下限は終端で消えない最小幅。
PREVIEW_WIDTH_MAX = 0.9
PREVIEW_WIDTH_MIN = 0.15
# 終端リフト区間長(mm)。PlotterConfig.finish_lift_length_mm と揃える。
PREVIEW_LIFT_LENGTH_MM = 1.5


def compute_stroke_widths(
    stroke: Stroke,
    finish: str = "none",
    lift_length: float = PREVIEW_LIFT_LENGTH_MM,
    pressure_variation: float = 0.0,
    entry_taper: float = 0.0,
) -> list[float]:
    """ストローク各セグメントの太さ(linewidth)を計算する。

    実機の終端Zリフト（接触圧の抜き）と同一の :func:`contact_profile` から線幅を
    導く。終端からの距離(mm)ベースなので、文字サイズが変わっても抜けの見え方が
    実機と一致する（「見た目＝実機」）。``width = w_min + (w_max - w_min) * contact``。
    とめ/none は接触一定＝幅一定。``pressure_variation > 0`` のときは画内の筆圧変調
    （:func:`pressure_modulation`）を contact に掛け、G-code の Z 補間と一致させる。

    Args:
        stroke: ``(N, 2)`` の点列（mm 座標）。
        finish: 筆画タイプ（``"tome"`` / ``"hane"`` / ``"harai"`` / ``"none"``）。
        lift_length: 終端リフト区間長(mm)。
        pressure_variation: 画内の筆圧変調の深さ ∈[0,1]。PlotterConfig と揃える。

    Returns:
        各セグメント（``N-1`` 本）の linewidth リスト。点数 2 未満は空リスト。
    """
    pts = np.asarray(stroke, dtype=float)
    if len(pts) < 2:
        return []
    arc = arc_length_from_end(pts)
    contact = contact_profile(finish, arc, lift_length)
    contact = contact * pressure_modulation(pts, pressure_variation)
    contact = contact * entry_modulation(pts, PREVIEW_ENTRY_LENGTH_MM, entry_taper)
    seg_contact = (contact[:-1] + contact[1:]) / 2.0  # セグメント太さ=両端の平均
    widths = PREVIEW_WIDTH_MIN + (PREVIEW_WIDTH_MAX - PREVIEW_WIDTH_MIN) * seg_contact
    return widths.tolist()


def _draw_stroke_with_width(
    ax: plt.Axes,
    stroke: Stroke,
    color: str = "b",
    finish: str = "none",
    pressure_variation: float = 0.0,
    entry_taper: float = 0.0,
) -> None:
    """LineCollectionを使ってストロークを太さ変調付きで描画

    Args:
        ax: 描画先の matplotlib Axes。
        stroke: ``(N, 2)`` の点列。
        color: 線色。
        finish: 筆画タイプ。:func:`compute_stroke_widths` の太さ分岐に渡す。
    """
    n_points = len(stroke)
    n_segments = n_points - 1
    if n_segments < 1:
        return

    segments = [[stroke[i].tolist(), stroke[i + 1].tolist()] for i in range(n_segments)]
    widths = compute_stroke_widths(
        stroke, finish, pressure_variation=pressure_variation, entry_taper=entry_taper
    )

    lc = LineCollection(segments, linewidths=widths, colors=color)
    ax.add_collection(lc)


def preview_strokes(
    strokes: list[Stroke],
    config: PlotterConfig | None = None,
    show_travel: bool = True,
    show_paper: bool = True,
    save_path: str | Path | None = None,
    vary_width: bool = True,
) -> None:
    """ストロークデータをMatplotlibで描画

    Args:
        strokes: 描画するストロークのリスト
        config: プロッタ設定。Noneの場合デフォルト
        show_travel: ペンアップ移動を赤点線で表示
        show_paper: 用紙の矩形を表示
        save_path: 画像保存先。Noneの場合画面表示
        vary_width: ストローク内の太さを変調する（始筆太→終筆細）
    """
    config = config or PlotterConfig()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if show_paper:
        paper_rect = patches.Rectangle(
            (config.paper_origin_x, config.paper_origin_y),
            config.paper_width,
            config.paper_height,
            linewidth=1,
            edgecolor="gray",
            facecolor="lightyellow",
            linestyle="--",
        )
        ax.add_patch(paper_rect)

    current_pos = np.array([0.0, 0.0])
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        if show_travel:
            travel = np.array([current_pos, stroke[0]])
            ax.plot(travel[:, 0], travel[:, 1], "r--", linewidth=0.5, alpha=0.3)
        if vary_width:
            _draw_stroke_with_width(ax, stroke)
        else:
            ax.plot(stroke[:, 0], stroke[:, 1], "b-", linewidth=1.0)
        current_pos = stroke[-1].copy()

    ax.set_xlim(-5, config.work_area_width + 5)
    ax.set_ylim(-5, config.work_area_height + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Pen Plotter Preview")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
    else:
        plt.show()


def preview_gcode(
    gcode_lines: list[str],
    save_path: str | Path | None = None,
) -> None:
    """G-codeをパースしてMatplotlibで描画

    G0移動を赤点線、G1移動を青実線で表示する。
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    current_x, current_y = 0.0, 0.0
    pen_down = False

    for line in gcode_lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue

        # M3 = ペンダウン、M5 = ペンアップ
        if line.startswith("M3"):
            pen_down = True
            continue
        if line.startswith("M5"):
            pen_down = False
            continue

        if line.startswith("G0") or line.startswith("G1"):
            new_x, new_y = current_x, current_y
            for part in line.split():
                if part.startswith("X"):
                    new_x = float(part[1:])
                elif part.startswith("Y"):
                    new_y = float(part[1:])

            if line.startswith("G0"):
                ax.plot(
                    [current_x, new_x],
                    [current_y, new_y],
                    "r--",
                    linewidth=0.5,
                    alpha=0.3,
                )
            elif pen_down:
                ax.plot(
                    [current_x, new_x],
                    [current_y, new_y],
                    "b-",
                    linewidth=1.0,
                )

            current_x, current_y = new_x, new_y

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("G-code Preview")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
    else:
        plt.show()
