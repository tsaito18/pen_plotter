"""訓練済みモデルを使ったストローク生成推論。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.model.stroke_model import StrokeGenerator
from src.model.style_encoder import StyleEncoder
from src.model.train import TrainConfig


class StrokeInference:
    def __init__(
        self,
        checkpoint_path: Path | str,
        generator_kwargs: dict | None = None,
        style_encoder_kwargs: dict | None = None,
    ) -> None:
        torch.serialization.add_safe_globals([TrainConfig])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        is_v2 = "char_encoder_state_dict" in checkpoint
        gen_kwargs = dict(generator_kwargs or {})

        self.char_encoder = None
        if is_v2:
            from src.model.char_encoder import CharEncoder

            char_dim = checkpoint.get("char_dim", 128)
            gen_kwargs["char_dim"] = char_dim

            char_enc_sd = checkpoint["char_encoder_state_dict"]
            lstm_weight = char_enc_sd["lstm.weight_ih_l0"]
            input_dim = lstm_weight.shape[1]
            hidden_dim = lstm_weight.shape[0] // 4
            fc_weight = char_enc_sd["fc.weight"]
            out_dim = fc_weight.shape[0]
            num_layers = sum(
                1 for k in char_enc_sd if k.startswith("lstm.weight_ih_l") and "_reverse" not in k
            )

            self.char_encoder = CharEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                char_dim=out_dim,
                num_layers=num_layers,
            )
            self.char_encoder.load_state_dict(char_enc_sd)
            self.char_encoder.eval()

        self.generator = StrokeGenerator(**gen_kwargs)
        self.style_encoder = StyleEncoder(**(style_encoder_kwargs or {}))

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.style_encoder.load_state_dict(checkpoint["style_encoder_state_dict"])

        self.generator.eval()
        self.style_encoder.eval()

    @torch.no_grad()
    def generate(
        self,
        style_sample: torch.Tensor,
        num_steps: int = 100,
        temperature: float = 1.0,
        reference_strokes: list[NDArray[np.float64]] | None = None,
    ) -> list[np.ndarray]:
        """スタイルサンプルからストロークを自己回帰生成する。"""
        style = self.style_encoder(style_sample)

        char_embedding: torch.Tensor | None = None
        if self.char_encoder is not None:
            if reference_strokes is not None:
                from src.model.char_encoder import CharEncoder

                seq = CharEncoder.strokes_to_sequence(reference_strokes)
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                char_embedding = self.char_encoder(seq_tensor)
            else:
                char_embedding = torch.zeros(1, self.generator.char_dim)

        current = torch.zeros(1, 1, 3)
        points: list[list[float]] = []
        pen_states: list[float] = []

        for _ in range(num_steps):
            output = self.generator(current, style, char_embedding=char_embedding)

            pi = output["pi"][:, -1] / temperature
            pi = torch.softmax(pi, dim=-1)

            k = torch.multinomial(pi, 1).squeeze(-1)

            mu_x = output["mu_x"][:, -1].gather(1, k.unsqueeze(1)).squeeze(1)
            mu_y = output["mu_y"][:, -1].gather(1, k.unsqueeze(1)).squeeze(1)
            sigma_x = output["sigma_x"][:, -1].gather(1, k.unsqueeze(1)).squeeze(1) * temperature
            sigma_y = output["sigma_y"][:, -1].gather(1, k.unsqueeze(1)).squeeze(1) * temperature
            rho = output["rho"][:, -1].gather(1, k.unsqueeze(1)).squeeze(1)

            z1 = torch.randn_like(mu_x)
            z2 = torch.randn_like(mu_y)
            dx = mu_x + sigma_x * z1
            dy = mu_y + sigma_y * (rho * z1 + torch.sqrt(1 - rho**2 + 1e-6) * z2)

            pen_prob = torch.sigmoid(output["pen_logit"][:, -1, 0])
            pen_state = (pen_prob > 0.5).float()

            points.append([dx.item(), dy.item()])
            pen_states.append(pen_state.item())

            next_input = torch.tensor([[[dx.item(), dy.item(), pen_state.item()]]])
            current = torch.cat([current, next_input], dim=1)

        strokes: list[np.ndarray] = []
        current_stroke: list[list[float]] = []
        cumulative_x, cumulative_y = 0.0, 0.0

        for (dx, dy), pen in zip(points, pen_states):
            cumulative_x += dx
            cumulative_y += dy
            if pen < 0.5:
                current_stroke.append([cumulative_x, cumulative_y])
            else:
                if len(current_stroke) >= 2:
                    strokes.append(np.array(current_stroke))
                current_stroke = []

        if len(current_stroke) >= 2:
            strokes.append(np.array(current_stroke))

        if not strokes:
            strokes = [np.array(points)]

        return strokes
