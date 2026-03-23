"""MLP-based stroke deformer for V3 style transfer.

Predicts per-point offsets to deform KanjiVG reference strokes
toward a target handwriting style, replacing the autoregressive LSTM+MDN approach.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StrokeDeformer(nn.Module):
    """Predicts per-point offsets to deform reference strokes to user style."""

    MAX_STROKE_INDEX = 16

    def __init__(
        self,
        style_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        stroke_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.stroke_embed_dim = stroke_embed_dim

        self.stroke_embedding = nn.Embedding(self.MAX_STROKE_INDEX, stroke_embed_dim)

        # ref_x, ref_y, normalized_t, style_vector, stroke_embed
        input_dim = 2 + 1 + style_dim + stroke_embed_dim

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        reference_points: torch.Tensor,
        style: torch.Tensor,
        stroke_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict per-point offsets.

        Args:
            reference_points: (batch, N, 2) reference xy coordinates.
            style: (batch, style_dim) style vector from StyleEncoder.
            stroke_index: (batch,) integer stroke index, clamped to MAX_STROKE_INDEX-1.

        Returns:
            offsets: (batch, N, 2) predicted offset per point.
        """
        batch_size, n_points, _ = reference_points.shape

        # normalized_t: position along the stroke [0, 1]
        t = torch.linspace(0, 1, n_points, device=reference_points.device)
        t = t.unsqueeze(0).unsqueeze(-1).expand(batch_size, n_points, 1)

        # style broadcast to each point
        style_expanded = style.unsqueeze(1).expand(batch_size, n_points, self.style_dim)

        # stroke embedding
        if stroke_index is not None:
            idx = stroke_index.clamp(0, self.MAX_STROKE_INDEX - 1)
            stroke_emb = self.stroke_embedding(idx)
            stroke_emb = stroke_emb.unsqueeze(1).expand(batch_size, n_points, self.stroke_embed_dim)
        else:
            stroke_emb = torch.zeros(
                batch_size, n_points, self.stroke_embed_dim,
                device=reference_points.device,
            )

        features = torch.cat([reference_points, t, style_expanded, stroke_emb], dim=-1)
        return self.mlp(features)


def deformation_loss(
    predicted_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss between predicted and target offsets.

    Args:
        predicted_offsets: (batch, N, 2)
        target_offsets: (batch, N, 2)
        mask: optional (batch, N) boolean mask, True for valid points.

    Returns:
        Scalar loss.
    """
    diff_sq = (predicted_offsets - target_offsets) ** 2

    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).float()
        diff_sq = diff_sq * mask_expanded
        return diff_sq.sum() / mask_expanded.sum().clamp(min=1.0)

    return diff_sq.mean()
