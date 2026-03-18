"""Stroke data preprocessing utilities for LSTM+MDN handwriting generation.

Implements the Graves (2013) approach: absolute stroke coordinates are converted
to delta sequences with pen-up/pen-down state for autoregressive generation.
"""

from __future__ import annotations

import numpy as np
import torch


def _build_delta_tensor(
    abs_points: list[tuple[float, float]], pen_states: list[int]
) -> torch.Tensor:
    """Convert absolute points + pen states to (N, 3) delta tensor.

    Args:
        abs_points: List of (x, y) absolute coordinates.
        pen_states: List of pen state flags (0=down, 1=up) per point.

    Returns:
        Tensor of shape (N, 3) with columns [dx, dy, pen_state].
    """
    n = len(abs_points)
    result = torch.zeros(n, 3, dtype=torch.float32)

    for i in range(n):
        if i == 0:
            result[i, 0] = 0.0
            result[i, 1] = 0.0
        else:
            result[i, 0] = abs_points[i][0] - abs_points[i - 1][0]
            result[i, 1] = abs_points[i][1] - abs_points[i - 1][1]
        result[i, 2] = pen_states[i]

    return result


def strokes_to_deltas(
    strokes: list[list[dict[str, float]]],
) -> torch.Tensor:
    """Convert dict-based strokes to a Graves-style delta tensor.

    Args:
        strokes: List of strokes, each a list of dicts with "x", "y" keys.

    Returns:
        Tensor of shape (N, 3) with columns [dx, dy, pen_state].
        pen_state=1 marks the last point of each stroke (pen lifts after it).
    """
    abs_points: list[tuple[float, float]] = []
    pen_states: list[int] = []

    for stroke in strokes:
        for j, pt in enumerate(stroke):
            abs_points.append((pt["x"], pt["y"]))
            is_last = j == len(stroke) - 1
            pen_states.append(1 if is_last else 0)

    return _build_delta_tensor(abs_points, pen_states)


def strokes_to_deltas_from_arrays(
    strokes: list[np.ndarray],
) -> torch.Tensor:
    """Convert numpy-array-based strokes to a Graves-style delta tensor.

    Args:
        strokes: List of arrays, each of shape (N, 2) with columns [x, y].

    Returns:
        Tensor of shape (N, 3) with columns [dx, dy, pen_state].
    """
    abs_points: list[tuple[float, float]] = []
    pen_states: list[int] = []

    for stroke in strokes:
        n = stroke.shape[0]
        for j in range(n):
            abs_points.append((float(stroke[j, 0]), float(stroke[j, 1])))
            pen_states.append(1 if j == n - 1 else 0)

    return _build_delta_tensor(abs_points, pen_states)


def compute_normalization_stats(
    tensors: list[torch.Tensor],
) -> dict[str, float]:
    """Compute mean and std of dx/dy columns across multiple delta tensors.

    Args:
        tensors: List of (N, 3) tensors from strokes_to_deltas.

    Returns:
        Dict with keys "mean_x", "mean_y", "std_x", "std_y".
    """
    all_dx = torch.cat([t[:, 0] for t in tensors])
    all_dy = torch.cat([t[:, 1] for t in tensors])

    std_x = all_dx.std().item() if len(all_dx) > 1 else 0.0
    std_y = all_dy.std().item() if len(all_dy) > 1 else 0.0

    return {
        "mean_x": all_dx.mean().item(),
        "mean_y": all_dy.mean().item(),
        "std_x": max(std_x, 1e-6),
        "std_y": max(std_y, 1e-6),
    }


def normalize_deltas(
    tensor: torch.Tensor, stats: dict[str, float]
) -> torch.Tensor:
    """Normalize dx/dy columns, leaving pen_state unchanged.

    Args:
        tensor: Shape (N, 3) or (batch, N, 3).
        stats: Dict from compute_normalization_stats.

    Returns:
        Normalized tensor of same shape.
    """
    result = tensor.clone()
    result[..., 0] = (tensor[..., 0] - stats["mean_x"]) / stats["std_x"]
    result[..., 1] = (tensor[..., 1] - stats["mean_y"]) / stats["std_y"]
    return result


def denormalize_point(
    dx: float, dy: float, stats: dict[str, float]
) -> tuple[float, float]:
    """Reverse normalization for a single point.

    Args:
        dx: Normalized delta-x value.
        dy: Normalized delta-y value.
        stats: Dict from compute_normalization_stats.

    Returns:
        Tuple of (original_dx, original_dy).
    """
    return (
        dx * stats["std_x"] + stats["mean_x"],
        dy * stats["std_y"] + stats["mean_y"],
    )
