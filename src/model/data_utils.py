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


def reference_to_sequence(strokes: list[list[dict[str, float]]]) -> torch.Tensor:
    """Convert dict-based reference strokes to separator-delimited sequence.

    Same format as CharEncoder.strokes_to_sequence but from dict input.
    Inserts (-1, -1) between strokes.

    Returns:
        (N, 2) tensor
    """
    parts: list[list[tuple[float, float]]] = []
    for i, stroke in enumerate(strokes):
        if len(stroke) >= 2:
            parts.append([(pt["x"], pt["y"]) for pt in stroke])

    if not parts:
        return torch.zeros(1, 2)

    segments: list[torch.Tensor] = []
    for i, pts in enumerate(parts):
        segments.append(torch.tensor(pts, dtype=torch.float32))
        if i < len(parts) - 1:
            segments.append(torch.tensor([[-1.0, -1.0]]))

    return torch.cat(segments, dim=0)


def reference_to_sequence_from_arrays(strokes: list[np.ndarray]) -> torch.Tensor:
    """Convert numpy array strokes to separator-delimited sequence.

    Returns:
        (N, 2) tensor
    """
    valid = [s for s in strokes if s.shape[0] >= 2]

    if not valid:
        return torch.zeros(1, 2)

    segments: list[torch.Tensor] = []
    for i, stroke in enumerate(valid):
        segments.append(torch.from_numpy(stroke.astype(np.float32)))
        if i < len(valid) - 1:
            segments.append(torch.tensor([[-1.0, -1.0]]))

    return torch.cat(segments, dim=0)


def compute_reference_stats(tensors: list[torch.Tensor]) -> dict[str, float]:
    """Compute mean/std of reference coordinates, excluding separator values (-1).

    Returns:
        Dict with mean_x, mean_y, std_x, std_y.
    """
    all_x = []
    all_y = []
    for t in tensors:
        mask = t[:, 0] != -1.0
        all_x.append(t[mask, 0])
        all_y.append(t[mask, 1])
    all_x_cat = torch.cat(all_x)
    all_y_cat = torch.cat(all_y)

    std_x = all_x_cat.std().item() if len(all_x_cat) > 1 else 0.0
    std_y = all_y_cat.std().item() if len(all_y_cat) > 1 else 0.0

    return {
        "mean_x": all_x_cat.mean().item(),
        "mean_y": all_y_cat.mean().item(),
        "std_x": max(std_x, 1e-6),
        "std_y": max(std_y, 1e-6),
    }


def normalize_reference(
    tensor: torch.Tensor, stats: dict[str, float]
) -> torch.Tensor:
    """Normalize reference coordinates. Separator points (-1, -1) become (0, 0).

    Works with (N, 2) or (batch, N, 2).
    """
    result = tensor.clone()
    sep_mask = tensor[..., 0] == -1.0
    result[..., 0] = (tensor[..., 0] - stats["mean_x"]) / stats["std_x"]
    result[..., 1] = (tensor[..., 1] - stats["mean_y"]) / stats["std_y"]
    result[sep_mask] = 0.0
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


def stroke_to_deltas_2d(points: list | np.ndarray) -> torch.Tensor:
    """Convert a single stroke's absolute points to (N, 2) delta tensor [dx, dy].

    Args:
        points: Array-like of shape (N, 2) with [x, y] columns.

    Returns:
        Tensor of shape (N, 2) with [dx, dy] columns.
    """
    if isinstance(points, np.ndarray):
        pts = points
    else:
        pts = np.array(points, dtype=np.float32)

    n = len(pts)
    result = torch.zeros(n, 2, dtype=torch.float32)
    for i in range(1, n):
        result[i, 0] = float(pts[i, 0] - pts[i - 1, 0])
        result[i, 1] = float(pts[i, 1] - pts[i - 1, 1])
    return result


def normalize_deltas_2d(
    tensor: torch.Tensor, stats: dict[str, float]
) -> torch.Tensor:
    """Normalize 2D deltas (no pen_state column).

    Args:
        tensor: Shape (N, 2) or (batch, N, 2).
        stats: Dict from compute_normalization_stats.

    Returns:
        Normalized tensor of same shape.
    """
    result = tensor.clone()
    result[..., 0] = (tensor[..., 0] - stats["mean_x"]) / stats["std_x"]
    result[..., 1] = (tensor[..., 1] - stats["mean_y"]) / stats["std_y"]
    return result


def compute_normalization_stats_2d(
    tensors: list[torch.Tensor],
) -> dict[str, float]:
    """Compute mean and std of dx/dy from 2-column delta tensors.

    Args:
        tensors: List of (N, 2) tensors.

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
