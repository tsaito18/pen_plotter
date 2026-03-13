"""ストローク描画順序の最適化モジュール

nearest-neighbor法によりペンアップ移動距離を最小化する。
"""

import numpy as np
import numpy.typing as npt

Stroke = npt.NDArray[np.float64]


def _distance(p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]) -> float:
    """2点間のユークリッド距離"""
    return float(np.linalg.norm(p1 - p2))


def calculate_travel_distance(
    strokes: list[Stroke],
    start_pos: tuple[float, float] = (0.0, 0.0),
) -> float:
    """ペンアップ移動の総距離を計算

    Args:
        strokes: ストロークのリスト
        start_pos: 開始位置

    Returns:
        ペンアップ移動の総距離 (mm)
    """
    total = 0.0
    current = np.array(start_pos)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        total += _distance(current, stroke[0])
        current = stroke[-1].copy()
    total += _distance(current, np.array(start_pos))
    return total


def optimize_stroke_order(
    strokes: list[Stroke],
    start_pos: tuple[float, float] = (0.0, 0.0),
) -> list[Stroke]:
    """nearest-neighbor法でストローク描画順序を最適化

    各ステップで現在位置から最も近い未描画ストロークを選択。
    ストロークの向きも反転を考慮し、始点・終点のどちらが近いかで決定。

    Args:
        strokes: 最適化前のストロークリスト
        start_pos: 開始位置

    Returns:
        最適化後のストロークリスト
    """
    if len(strokes) <= 1:
        return list(strokes)

    remaining = list(range(len(strokes)))
    ordered: list[Stroke] = []
    current = np.array(start_pos)

    while remaining:
        best_idx = -1
        best_dist = float("inf")
        best_reversed = False

        for idx in remaining:
            stroke = strokes[idx]
            if len(stroke) < 2:
                continue

            d_start = _distance(current, stroke[0])
            d_end = _distance(current, stroke[-1])

            if d_start <= d_end:
                if d_start < best_dist:
                    best_dist = d_start
                    best_idx = idx
                    best_reversed = False
            else:
                if d_end < best_dist:
                    best_dist = d_end
                    best_idx = idx
                    best_reversed = True

        if best_idx == -1:
            break

        remaining.remove(best_idx)
        stroke = strokes[best_idx]
        if best_reversed:
            stroke = stroke[::-1].copy()
        ordered.append(stroke)
        current = stroke[-1].copy()

    return ordered
