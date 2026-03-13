import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            style: (batch, style_dim)
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden_dim)
        h_forward = h_n[-2]  # 最後のレイヤーの forward
        h_backward = h_n[-1]  # 最後のレイヤーの backward
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        style = self.fc(h_cat)
        return style
