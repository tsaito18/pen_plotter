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


def _optimize_with_indices(
    strokes: list[Stroke],
    start_pos: tuple[float, float] = (0.0, 0.0),
) -> tuple[list[Stroke], list[int]]:
    """nearest-neighbor法で最適化し、選ばれた元indexも返す内部関数

    Args:
        strokes: 最適化前のストロークリスト
        start_pos: 開始位置

    Returns:
        (最適化後のストロークリスト, 各ストロークの元indexリスト)。
        len(stroke) < 2 のストロークは描画対象から除外されるため、
        index リストの長さは入力より短くなることがある。
    """
    if len(strokes) <= 1:
        return list(strokes), list(range(len(strokes)))

    remaining = list(range(len(strokes)))
    ordered: list[Stroke] = []
    order_indices: list[int] = []
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
        order_indices.append(best_idx)
        current = stroke[-1].copy()

    return ordered, order_indices


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
    ordered, _ = _optimize_with_indices(strokes, start_pos)
    return ordered


def optimize_stroke_order_with_finishes(
    strokes: list[Stroke],
    finishes: list[str] | None,
    start_pos: tuple[float, float] = (0.0, 0.0),
) -> tuple[list[Stroke], list[str]]:
    """ストローク順序を最適化し、finishes も同じ順序に並べ替える

    ``optimize_stroke_order`` と同一の最適化ロジックを再利用し、選ばれた
    ストロークの元 index を追跡して finishes を同順に並べ替える。長さが
    一致しない場合も安全に扱い、対応する finish が無いストロークは
    "none" で補う。

    Args:
        strokes: 最適化前のストロークリスト
        finishes: 各ストロークの筆画タイプ
            ("tome"/"hane"/"harai"/"none")。None の場合は全て "none" 扱い。
        start_pos: 開始位置

    Returns:
        (最適化後のストロークリスト, 同順に並べ替えた finishes)。
        2要素のタプルで、両リストの長さは一致する。
    """
    ordered, order_indices = _optimize_with_indices(strokes, start_pos)
    src_finishes = finishes if finishes is not None else []
    ordered_finishes = [
        src_finishes[idx] if 0 <= idx < len(src_finishes) else "none" for idx in order_indices
    ]
    return ordered, ordered_finishes
