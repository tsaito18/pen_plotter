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
        dropout: float = 0.0,
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
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
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
                batch_size,
                n_points,
                self.stroke_embed_dim,
                device=reference_points.device,
            )

        features = torch.cat([reference_points, t, style_expanded, stroke_emb], dim=-1)
        return self.mlp(features)


class AffineStrokeDeformer(nn.Module):
    """Predicts per-stroke affine transformation (rotation, scale, shear, translation)."""

    MAX_STROKE_INDEX = 16

    def __init__(
        self,
        style_dim: int = 128,
        hidden_dim: int = 64,
        stroke_embed_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.stroke_embedding = nn.Embedding(self.MAX_STROKE_INDEX, stroke_embed_dim)
        input_dim = style_dim + stroke_embed_dim + 4  # +4 for stroke stats (cx, cy, w, h)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),  # theta, sx, sy, shear, tx, ty
        )
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()

    def forward(
        self,
        reference_points: torch.Tensor,
        style: torch.Tensor,
        stroke_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict affine transformation and apply to reference points.

        Args:
            reference_points: (batch, N, 2) reference xy coordinates.
            style: (batch, style_dim) style vector from StyleEncoder.
            stroke_index: (batch,) integer stroke index, clamped to MAX_STROKE_INDEX-1.

        Returns:
            transformed: (batch, N, 2) transformed points.
            params: (batch, 6) raw affine parameters.
        """
        batch = reference_points.shape[0]
        center = reference_points.mean(dim=1)
        mins = reference_points.min(dim=1).values
        maxs = reference_points.max(dim=1).values
        size = maxs - mins
        stroke_stats = torch.cat([center, size], dim=-1)

        if stroke_index is not None:
            idx = stroke_index.clamp(0, self.MAX_STROKE_INDEX - 1)
            s_emb = self.stroke_embedding(idx)
        else:
            s_emb = torch.zeros(batch, self.stroke_embedding.embedding_dim, device=style.device)

        features = torch.cat([style, s_emb, stroke_stats], dim=-1)
        params = self.mlp(features)

        theta = params[:, 0] * 0.1
        sx = 1.0 + params[:, 1] * 0.2
        sy = 1.0 + params[:, 2] * 0.2
        shear = params[:, 3] * 0.1
        tx = params[:, 4] * 0.5
        ty = params[:, 5] * 0.5

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        a11 = sx * cos_t
        a12 = -sy * sin_t + shear
        a21 = sx * sin_t
        a22 = sy * cos_t

        centered = reference_points - center.unsqueeze(1)
        x_new = centered[:, :, 0] * a11.unsqueeze(1) + centered[:, :, 1] * a12.unsqueeze(1)
        y_new = centered[:, :, 0] * a21.unsqueeze(1) + centered[:, :, 1] * a22.unsqueeze(1)
        transformed = torch.stack([x_new, y_new], dim=-1)
        translation = torch.stack([tx, ty], dim=-1).unsqueeze(1)
        transformed = transformed + center.unsqueeze(1) + translation

        return transformed, params


def affine_deformation_loss(
    transformed: torch.Tensor,
    target_points: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between affine-transformed points and target points.

    Args:
        transformed: (batch, N, 2) transformed points.
        target_points: (batch, N, 2) target points.

    Returns:
        Scalar loss.
    """
    return ((transformed - target_points) ** 2).mean()


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


def smoothness_loss(offsets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Penalize large differences between offsets of adjacent points.

    Args:
        offsets: (batch, N, 2) predicted offsets.
        mask: (batch, N) optional mask.
    """
    diff = offsets[:, 1:] - offsets[:, :-1]  # (batch, N-1, 2)
    sq_diff = (diff**2).sum(dim=-1)  # (batch, N-1)

    if mask is not None:
        valid = mask[:, 1:] * mask[:, :-1]  # (batch, N-1)
        return (sq_diff * valid).sum() / valid.sum().clamp(min=1)
    return sq_diff.mean()
