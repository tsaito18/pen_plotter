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
_MIN_STROKE_PX = 2
_MIN_STROKE_MM = 0.5  # 0.5mm 未満のストロークは除去


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
        stroke = np.stack([x_out, y_out], axis=1).astype(np.float64)
        # 全長が短すぎるストロークは除去（スケルトン化の孤立ピクセル雑音）
        diffs = np.diff(stroke, axis=0)
        length = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        if length >= _MIN_STROKE_MM:
            result.append(stroke)

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

    # 各ピクセルの近傍数を事前計算
    deg: dict[tuple[int, int], int] = {p: len(nbrs(*p)) for p in pixel_set}

    # エッジ単位で訪問管理（同じ分岐点を複数の枝で通過できるよう pixel ではなく edge で管理）
    visited_edges: set[frozenset[tuple[int, int]]] = set()
    raw_strokes: list[list[tuple[int, int]]] = []

    endpoints = [p for p in pixel_set if deg[p] == 1]
    # 端点がなければ任意の点（孤立ループ）から開始
    start_order = endpoints if endpoints else list(pixel_set)[:1]
    # 未訪問エッジが残れば分岐点からも開始
    all_starts = start_order + [p for p in pixel_set if deg[p] >= 3]

    def _best_next(
        cur: tuple[int, int],
        prev: tuple[int, int] | None,
        candidates: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """進行方向に最も近い（角度変化最小）の隣接点を返す。"""
        if prev is None or len(candidates) == 1:
            return candidates[0]
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        best, best_dot = candidates[0], float("-inf")
        for c in candidates:
            cdx, cdy = c[0] - cur[0], c[1] - cur[1]
            dot = dx * cdx + dy * cdy
            if dot > best_dot:
                best_dot, best = dot, c
        return best

    for start in all_starts:
        for init_nb in nbrs(*start):
            edge = frozenset((start, init_nb))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            path: list[tuple[int, int]] = [start, init_nb]
            prev, cur = start, init_nb

            while True:
                unvisited = [
                    n for n in nbrs(*cur)
                    if frozenset((cur, n)) not in visited_edges
                ]
                if not unvisited:
                    break
                nxt = _best_next(cur, prev, unvisited)
                edge = frozenset((cur, nxt))
                visited_edges.add(edge)
                path.append(nxt)
                prev, cur = cur, nxt

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
