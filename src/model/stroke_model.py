"""LSTM + Mixture Density Network によるストローク生成モデル。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StrokeGenerator(nn.Module):
    """スタイル条件付き LSTM-MDN ストロークジェネレータ。

    入力ストロークとスタイルベクトルから、次ステップの座標分布を
    混合ガウス分布（MDN）のパラメータとして出力する。
    ストローク単位で生成し、EOS（End-of-Stroke）を予測する。
    """

    MAX_STROKE_INDEX = 16

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        style_dim: int = 128,
        char_dim: int = 0,
        num_mixtures: int = 20,
        num_layers: int = 2,
        stroke_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_mixtures = num_mixtures
        self.char_dim = char_dim
        self.stroke_embed_dim = stroke_embed_dim

        self.stroke_embed = nn.Embedding(self.MAX_STROKE_INDEX, stroke_embed_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim + style_dim + stroke_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        if char_dim > 0:
            self.char_to_h0 = nn.Linear(char_dim, hidden_dim * num_layers)
            self.char_to_c0 = nn.Linear(char_dim, hidden_dim * num_layers)

        n_mdn = num_mixtures * 6
        self.mdn_head = nn.Linear(hidden_dim, n_mdn)

        self.eos_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        char_embedding: torch.Tensor | None = None,
        stroke_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, 2) ストロークデルタシーケンス [dx, dy]
            style: (batch, style_dim) スタイルベクトル
            char_embedding: (batch, char_dim) 文字埋め込み（char_dim > 0 時は必須）
            stroke_index: (batch,) 整数ストロークインデックス（0-based）
        Returns:
            MDN パラメータの辞書
        """
        batch, seq_len, _ = x.shape
        style_expanded = style.unsqueeze(1).expand(-1, seq_len, -1)

        if stroke_index is not None:
            clamped = stroke_index.clamp(0, self.MAX_STROKE_INDEX - 1)
            stroke_emb = self.stroke_embed(clamped)
        else:
            stroke_emb = torch.zeros(
                batch, self.stroke_embed_dim, device=x.device
            )
        stroke_expanded = stroke_emb.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([x, style_expanded, stroke_expanded], dim=-1)

        if self.char_dim > 0 and char_embedding is not None:
            h0 = torch.tanh(self.char_to_h0(char_embedding))
            h0 = h0.view(batch, self.lstm.num_layers, -1).permute(1, 0, 2).contiguous()
            c0 = torch.tanh(self.char_to_c0(char_embedding))
            c0 = c0.view(batch, self.lstm.num_layers, -1).permute(1, 0, 2).contiguous()
            h, _ = self.lstm(lstm_input, (h0, c0))
        else:
            if self.char_dim > 0 and char_embedding is None:
                raise ValueError("char_embedding required when char_dim > 0")
            h, _ = self.lstm(lstm_input)
        params = self.mdn_head(h)

        k = self.num_mixtures
        pi = torch.softmax(params[:, :, :k], dim=-1)
        mu_x = params[:, :, k : 2 * k]
        mu_y = params[:, :, 2 * k : 3 * k]
        sigma_x = torch.exp(params[:, :, 3 * k : 4 * k]).clamp(min=1e-4)
        sigma_y = torch.exp(params[:, :, 4 * k : 5 * k]).clamp(min=1e-4)
        rho = torch.tanh(params[:, :, 5 * k : 6 * k]).clamp(-0.95, 0.95)
        eos_logit = self.eos_head(h)

        return {
            "pi": pi,
            "pi_logits": params[:, :, :k],
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho,
            "eos_logit": eos_logit,
        }


def mdn_loss(
    output: dict[str, torch.Tensor],
    target_xy: torch.Tensor,
    target_eos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MDN の負の対数尤度損失とEOS のBCE損失を分離して返す。

    Args:
        output: StrokeGenerator の出力辞書
        target_xy: (batch, seq_len, 2) — dx, dy
        target_eos: (batch, seq_len, 1) — end-of-stroke flag

    Returns:
        (stroke_loss, eos_loss) のタプル
    """
    dx = target_xy[:, :, 0].unsqueeze(-1)
    dy = target_xy[:, :, 1].unsqueeze(-1)

    mu_x = output["mu_x"]
    mu_y = output["mu_y"]
    sigma_x = output["sigma_x"]
    sigma_y = output["sigma_y"]
    rho = output["rho"]
    eos_logit = output["eos_logit"]

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

    eos_loss = nn.functional.binary_cross_entropy_with_logits(
        eos_logit,
        target_eos,
        reduction="mean",
        pos_weight=torch.tensor([5.0], device=eos_logit.device),
    )

    return stroke_loss, eos_loss


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
