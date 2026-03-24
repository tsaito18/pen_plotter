from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection

from src.gcode.config import PlotterConfig

Stroke = npt.NDArray[np.float64]


def compute_stroke_widths(n_segments: int) -> list[float]:
    """ストローク内の各セグメントの太さを計算する。

    Args:
        n_segments: セグメント数

    Returns:
        各セグメントの linewidth リスト（始点で太く、終点で細い）
    """
    if n_segments <= 0:
        return []
    if n_segments == 1:
        t = np.array([0.5])
    else:
        t = np.linspace(0.0, 1.0, n_segments)
    widths = 0.7 + 0.3 * np.exp(-2.0 * t)
    return widths.tolist()


def _draw_stroke_with_width(
    ax: plt.Axes, stroke: Stroke, color: str = "b"
) -> None:
    """LineCollectionを使ってストロークを太さ変調付きで描画"""
    n_points = len(stroke)
    n_segments = n_points - 1
    if n_segments < 1:
        return

    segments = [
        [stroke[i].tolist(), stroke[i + 1].tolist()] for i in range(n_segments)
    ]
    widths = compute_stroke_widths(n_segments)

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
