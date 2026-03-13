"""LSTM + Mixture Density Network によるストローク生成モデル。"""

from __future__ import annotations

import torch
import torch.nn as nn


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
        num_mixtures: int = 5,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_mixtures = num_mixtures

        self.lstm = nn.LSTM(
            input_size=input_dim + style_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # MDN パラメータ: pi, mu_x, mu_y, sigma_x, sigma_y, rho + pen_logit
        n_mdn = num_mixtures * 6 + 1
        self.mdn_head = nn.Linear(hidden_dim, n_mdn)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) ストロークシーケンス
            style: (batch, style_dim) スタイルベクトル
        Returns:
            MDN パラメータの辞書
        """
        batch, seq_len, _ = x.shape
        style_expanded = style.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([x, style_expanded], dim=-1)

        h, _ = self.lstm(lstm_input)
        params = self.mdn_head(h)

        k = self.num_mixtures
        pi = torch.softmax(params[:, :, :k], dim=-1)
        mu_x = params[:, :, k : 2 * k]
        mu_y = params[:, :, 2 * k : 3 * k]
        sigma_x = torch.exp(params[:, :, 3 * k : 4 * k]).clamp(min=1e-4)
        sigma_y = torch.exp(params[:, :, 4 * k : 5 * k]).clamp(min=1e-4)
        rho = torch.tanh(params[:, :, 5 * k : 6 * k])
        pen_logit = params[:, :, 6 * k : 6 * k + 1]

        return {
            "pi": pi,
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

    log_pi = torch.log(pi + 1e-8)
    log_prob = torch.logsumexp(log_pi + log_gauss, dim=-1)

    stroke_loss = -log_prob.mean()

    pen_loss = nn.functional.binary_cross_entropy_with_logits(
        pen_logit, pen_target, reduction="mean"
    )

    return stroke_loss + pen_loss
