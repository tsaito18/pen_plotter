"""ストローク（点列）を扱う共通の型と幾何ユーティリティ。

座標系は常に mm・Y-UP（上が +Y）。ストロークは ``(N, 2)`` の float64 配列。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Stroke = npt.NDArray[np.float64]


def rotation_matrix(angle: float) -> npt.NDArray[np.float64]:
    """反時計回りに ``angle`` [rad] 回す 2x2 回転行列を返す。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def rotate_about(strokes: list[Stroke], angle: float, center: npt.ArrayLike) -> list[Stroke]:
    """全ストロークを ``center`` まわりに ``angle`` [rad] 反時計回りに回転する。"""
    rot_t = rotation_matrix(angle).T
    c = np.asarray(center, dtype=np.float64)
    return [(s - c) @ rot_t + c for s in strokes]


def bbox_span(strokes: list[Stroke]) -> float:
    """全ストロークを内包する bbox の長辺（x/y の大きい方）を返す。"""
    pts = np.concatenate(strokes, axis=0)
    return float((pts.max(axis=0) - pts.min(axis=0)).max())


def resample_stroke(points: np.ndarray, num_points: int = 32) -> np.ndarray:
    """弧長で等間隔に ``num_points`` 点へ再サンプリングする（float32 を返す）。"""
    if len(points) < 2:
        return np.tile(points[0] if len(points) == 1 else np.zeros(2), (num_points, 1))

    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length < 1e-12:
        return np.tile(points[0], (num_points, 1))

    target_lengths = np.linspace(0.0, total_length, num_points)
    x_resampled = np.interp(target_lengths, cum_lengths, points[:, 0])
    y_resampled = np.interp(target_lengths, cum_lengths, points[:, 1])
    return np.stack([x_resampled, y_resampled], axis=1).astype(np.float32)
