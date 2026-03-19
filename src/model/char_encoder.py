"""KanjiVGスケルトンストロークから文字構造の埋め込みベクトルを生成するエンコーダ。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


class CharEncoder(nn.Module):
    """KanjiVGスケルトンストロークから文字構造の埋め込みベクトルを生成する。

    入力: 正規化された参照ストローク座標 (batch, seq_len, 2) — (x, y) のみ
    出力: 文字埋め込みベクトル (batch, char_dim)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        char_dim: int = 128,
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
        self.fc = nn.Linear(hidden_dim * 2, char_dim)
        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                hidden_dim = param.shape[0] // 4
                param.data[hidden_dim : 2 * hidden_dim].fill_(1.0)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 2) — reference stroke coordinates
            lengths: optional (batch,) — actual sequence lengths for packed sequences

        Returns:
            (batch, char_dim) — character embedding
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
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        return self.fc(h_cat)

    @staticmethod
    def strokes_to_sequence(strokes: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """ストロークリストをセパレータ付き単一シーケンスに変換する。

        Args:
            strokes: 各要素が (N, 2) の座標配列であるストロークのリスト

        Returns:
            (total_points, 2) の連結配列。ストローク間に (-1, -1) セパレータを挿入。
        """
        if not strokes:
            return np.zeros((1, 2))

        separator = np.array([[-1.0, -1.0]])
        parts: list[NDArray[np.float64]] = []
        for i, stroke in enumerate(strokes):
            if len(stroke) >= 2:
                parts.append(stroke)
            if i < len(strokes) - 1:
                parts.append(separator)

        if not parts:
            return np.zeros((1, 2))
        return np.concatenate(parts, axis=0)
