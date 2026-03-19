"""LSTM + Mixture Density Network によるストローク生成モデル。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StrokeGenerator(nn.Module):
    """スタイル条件付き LSTM-MDN ストロークジェネレータ。

    入力ストロークとスタイルベクトルから、次ステップの座標分布を
    混合ガウス分布（MDN）のパラメータとして出力する。
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        style_dim: int = 128,
        char_dim: int = 0,
        num_mixtures: int = 20,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_mixtures = num_mixtures
        self.char_dim = char_dim

        self.lstm = nn.LSTM(
            input_size=input_dim + style_dim + char_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # MDN パラメータ: pi, mu_x, mu_y, sigma_x, sigma_y, rho + pen_logit
        n_mdn = num_mixtures * 6 + 1
        self.mdn_head = nn.Linear(hidden_dim, n_mdn)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        char_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) ストロークシーケンス
            style: (batch, style_dim) スタイルベクトル
            char_embedding: (batch, char_dim) 文字埋め込み（char_dim > 0 時は必須）
        Returns:
            MDN パラメータの辞書
        """
        batch, seq_len, _ = x.shape
        style_expanded = style.unsqueeze(1).expand(-1, seq_len, -1)

        if self.char_dim > 0:
            if char_embedding is None:
                raise ValueError("char_embedding required when char_dim > 0")
            char_expanded = char_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([x, style_expanded, char_expanded], dim=-1)
        else:
            lstm_input = torch.cat([x, style_expanded], dim=-1)

        h, _ = self.lstm(lstm_input)
        params = self.mdn_head(h)

        k = self.num_mixtures
        pi = torch.softmax(params[:, :, :k], dim=-1)
        mu_x = params[:, :, k : 2 * k]
        mu_y = params[:, :, 2 * k : 3 * k]
        sigma_x = torch.exp(params[:, :, 3 * k : 4 * k]).clamp(min=1e-4)
        sigma_y = torch.exp(params[:, :, 4 * k : 5 * k]).clamp(min=1e-4)
        rho = torch.tanh(params[:, :, 5 * k : 6 * k]).clamp(-0.95, 0.95)
        pen_logit = params[:, :, 6 * k : 6 * k + 1]

        return {
            "pi": pi,
            "pi_logits": params[:, :, :k],
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho,
            "pen_logit": pen_logit,
        }


def mdn_loss(output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """MDN の負の対数尤度損失 + ペン状態のBCE損失。

    Args:
        output: StrokeGenerator の出力辞書
        target: (batch, seq_len, 3) — dx, dy, pen_state
    """
    dx = target[:, :, 0].unsqueeze(-1)
    dy = target[:, :, 1].unsqueeze(-1)
    pen_target = target[:, :, 2:]

    pi = output["pi"]
    mu_x = output["mu_x"]
    mu_y = output["mu_y"]
    sigma_x = output["sigma_x"]
    sigma_y = output["sigma_y"]
    rho = output["rho"]
    pen_logit = output["pen_logit"]

    z_x = (dx - mu_x) / sigma_x
    z_y = (dy - mu_y) / sigma_y
    z = z_x**2 + z_y**2 - 2 * rho * z_x * z_y

    denom = 1.0 - rho**2 + 1e-6
    log_norm = (
        -torch.log(2 * torch.tensor(torch.pi))
        - torch.log(sigma_x)
        - torch.log(sigma_y)
        - 0.5 * torch.log(denom)
    )
    log_exp = -z / (2 * denom)
    log_gauss = log_norm + log_exp

    log_pi = F.log_softmax(output["pi_logits"], dim=-1)
    log_prob = torch.logsumexp(log_pi + log_gauss, dim=-1)

    stroke_loss = -log_prob.mean()

    pen_loss = nn.functional.binary_cross_entropy_with_logits(
        pen_logit,
        pen_target,
        reduction="mean",
        pos_weight=torch.tensor([10.0], device=pen_logit.device),
    )

    return stroke_loss + pen_loss


def embedding_variance_loss(
    embeddings: torch.Tensor, min_variance: float = 1.0
) -> torch.Tensor:
    """Penalize low variance in batch embeddings to prevent encoder collapse.

    Args:
        embeddings: (batch, dim)
        min_variance: target minimum variance per dimension

    Returns:
        Scalar loss: mean of max(0, min_variance - var) across dimensions
    """
    if embeddings.shape[0] < 2:
        return torch.tensor(0.0, device=embeddings.device)
    var = embeddings.var(dim=0)
    return torch.clamp(min_variance - var, min=0).mean()
