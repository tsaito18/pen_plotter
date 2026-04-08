from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning. Output is L2-normalized."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.mlp(x)
        return torch.nn.functional.normalize(projected, dim=-1)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.3,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020).

    Args:
        embeddings: (batch, dim), assumed L2-normalized
        labels: (batch,) integer labels
        temperature: scaling temperature

    Returns:
        Scalar loss tensor.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # (batch, batch) similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # Mask: same label pairs (excluding self)
    labels_col = labels.unsqueeze(0)  # (1, batch)
    labels_row = labels.unsqueeze(1)  # (batch, 1)
    positive_mask = (labels_row == labels_col).float()
    # Remove diagonal (self-similarity)
    self_mask = torch.eye(batch_size, device=device)
    positive_mask = positive_mask - self_mask

    # Number of positives per anchor
    num_positives = positive_mask.sum(dim=1)  # (batch,)

    # If no positive pairs exist in the batch, return 0
    if (num_positives == 0).all():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Numerical stability: subtract max from logits
    logits_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - logits_max

    # Mask out self from denominator
    exp_sim = torch.exp(sim) * (1 - self_mask)
    log_denominator = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # Log-prob of positives
    log_prob = sim - log_denominator  # (batch, batch)

    # Mean log-prob over positive pairs per anchor
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-12)

    # Only average over anchors that have at least one positive
    valid_mask = (num_positives > 0).float()
    loss = -(mean_log_prob * valid_mask).sum() / (valid_mask.sum() + 1e-12)

    return loss


class StyleEncoder(nn.Module):
    """Bi-LSTM ベースのスタイルエンコーダ

    ユーザーの筆跡サンプルから128次元スタイルベクトルを抽出する。
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        style_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Bi-LSTM の出力は hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, style_dim)
        self.norm = nn.LayerNorm(style_dim)
        self.projection_head: ProjectionHead | None = None
        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                hidden_dim = param.shape[0] // 4
                param.data[hidden_dim : 2 * hidden_dim].fill_(1.0)

    def enable_projection_head(
        self, hidden_dim: int = 128, output_dim: int = 64
    ) -> None:
        self.projection_head = ProjectionHead(
            self.fc.out_features, hidden_dim, output_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_projection: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: optional (batch,) — actual sequence lengths for packed sequences
            return_projection: if True, return (style, projection) tuple
        Returns:
            style: (batch, style_dim) when return_projection is False
            (style, projection): when return_projection is True
        """
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence

            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden_dim)
        h_forward = h_n[-2]  # 最後のレイヤーの forward
        h_backward = h_n[-1]  # 最後のレイヤーの backward
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        style = self.norm(self.fc(h_cat))

        if return_projection:
            if self.projection_head is not None:
                return style, self.projection_head(style)
            return style, None
        return style
