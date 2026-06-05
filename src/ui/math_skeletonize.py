"""LaTeX 数式 → matplotlib レンダリング → スケルトン化 → ストローク変換。"""
from __future__ import annotations

import io
import logging

import matplotlib
import numpy as np
from skimage import filters, morphology

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

Stroke = np.ndarray  # (N, 2) float64

_RENDER_DPI = 300
_FONT_SIZE_PT = 28
_PAD_INCHES = 0.05
_MIN_STROKE_PX = 3


def render_latex_to_strokes(
    math_src: str,
    bbox_mm: tuple[float, float, float, float],
) -> list[Stroke]:
    """LaTeX 数式をレンダリング → スケルトン化 → mm 座標ストロークに変換。

    Args:
        math_src: LaTeX ソース（$ なし。例: r'\\frac{F}{A_0}'）
        bbox_mm: (x_left_mm, y_bottom_mm, width_mm, height_mm) — Y-UP

    Returns:
        ストローク列（各要素は (N,2) float64, mm, Y-UP）
    """
    gray = _render_to_gray(math_src)
    if gray is None:
        return []

    binary = _binarize(gray)
    if not binary.any():
        return []

    binary = _crop(binary)
    skeleton = morphology.skeletonize(binary)

    pixel_strokes = _trace_skeleton(skeleton)

    h_px, w_px = skeleton.shape
    x0, y0, w_mm, h_mm = bbox_mm

    result: list[Stroke] = []
    for ps in pixel_strokes:
        if len(ps) < _MIN_STROKE_PX:
            continue
        col = ps[:, 0].astype(float)
        row = ps[:, 1].astype(float)
        x_n = col / max(w_px - 1, 1)
        y_n = 1.0 - row / max(h_px - 1, 1)  # row=0=top → Y=1 in Y-UP
        x_out = x0 + x_n * w_mm
        y_out = y0 + y_n * h_mm
        result.append(np.stack([x_out, y_out], axis=1).astype(np.float64))

    return result


def _render_to_gray(math_src: str) -> np.ndarray | None:
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed; math skeletonize unavailable")
        return None

    try:
        fig = plt.figure(figsize=(8, 2), dpi=_RENDER_DPI)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"${math_src}$",
            fontsize=_FONT_SIZE_PT,
            ha="center",
            va="center",
            color="black",
        )
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=_RENDER_DPI,
            bbox_inches="tight",
            pad_inches=_PAD_INCHES,
        )
        plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf).convert("L"))
    except Exception:
        logger.exception("math render failed: %r", math_src)
        return None


def _binarize(gray: np.ndarray) -> np.ndarray:
    thresh = filters.threshold_otsu(gray)
    return gray < thresh  # True = ink


def _crop(binary: np.ndarray) -> np.ndarray:
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any():
        return binary
    r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return binary[r0 : r1 + 1, c0 : c1 + 1]


def _trace_skeleton(skeleton: np.ndarray) -> list[np.ndarray]:
    """スケルトン画像からストローク列（[(N,2) float64 [col,row], ...]）を抽出。

    DP 簡略化（tolerance=1.5px）でステアケースを除去したあと返す。
    """
    from skimage.measure import approximate_polygon

    coords = np.argwhere(skeleton)  # (K, 2): [row, col]
    if len(coords) == 0:
        return []

    pixel_set: set[tuple[int, int]] = {(int(c), int(r)) for r, c in coords}

    def nbrs(col: int, row: int) -> list[tuple[int, int]]:
        return [
            p
            for p in [
                (col - 1, row - 1), (col, row - 1), (col + 1, row - 1),
                (col - 1, row),                      (col + 1, row),
                (col - 1, row + 1), (col, row + 1), (col + 1, row + 1),
            ]
            if p in pixel_set
        ]

    visited: set[tuple[int, int]] = set()
    raw_strokes: list[list[tuple[int, int]]] = []

    endpoints = [p for p in pixel_set if len(nbrs(*p)) == 1]
    start_order = endpoints + [p for p in pixel_set if p not in set(endpoints)]

    for start in start_order:
        if start in visited:
            continue
        path: list[tuple[int, int]] = [start]
        visited.add(start)
        cur = start

        while True:
            unvisited = [n for n in nbrs(*cur) if n not in visited]
            if not unvisited:
                break
            nxt = unvisited[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
            if len(nbrs(*nxt)) >= 3 and len(path) > 2:
                break

        if len(path) >= 2:
            raw_strokes.append(path)

    # Douglas-Peucker 簡略化でステアケース除去
    result: list[np.ndarray] = []
    for path in raw_strokes:
        arr = np.array(path, dtype=np.float64)  # (N, 2): [col, row]
        # approximate_polygon は (row, col) 順を期待する
        simplified = approximate_polygon(arr[:, ::-1], tolerance=1.5)  # → (M, 2) [row, col]
        if len(simplified) >= 2:
            result.append(simplified[:, ::-1].astype(np.float64))  # [col, row]

    return result
