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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        is_v3 = "deformer_state_dict" in checkpoint
        is_v2 = "char_encoder_state_dict" in checkpoint

        config = checkpoint.get("config", {})
        self.norm_stats = checkpoint.get("norm_stats", None)

        if is_v3:
            from src.model.stroke_deformer import StrokeDeformer

            style_dim = config.get("style_dim", 128)
            hidden_dim = config.get("hidden_dim", 256)

            self.deformer = StrokeDeformer(
                style_dim=style_dim,
                hidden_dim=hidden_dim,
            )
            self.deformer.load_state_dict(checkpoint["deformer_state_dict"])
            self.deformer.eval()

            style_enc_kwargs = dict(style_encoder_kwargs or {})
            if "style_dim" not in style_enc_kwargs:
                style_enc_kwargs["style_dim"] = style_dim
            self.style_encoder = StyleEncoder(**style_enc_kwargs)
            self.style_encoder.load_state_dict(checkpoint["style_encoder_state_dict"])
            self.style_encoder.eval()

            self.version = 3
            self.num_points = config.get("num_points", 32)
            self.char_encoder = None
            self.ref_norm_stats = None
            self.generator = None
            return

        gen_kwargs = dict(generator_kwargs or {})
        style_enc_kwargs = dict(style_encoder_kwargs or {})

        config_to_gen = {"hidden_dim", "style_dim", "num_mixtures"}
        for key in config_to_gen:
            if key in config and key not in gen_kwargs:
                gen_kwargs[key] = config[key]
        if "style_dim" in config and "style_dim" not in style_enc_kwargs:
            style_enc_kwargs["style_dim"] = config["style_dim"]

        gen_kwargs.setdefault("input_dim", 2)

        self.char_encoder = None
        self.ref_norm_stats = checkpoint.get("ref_norm_stats", None)
        if is_v2:
            from src.model.char_encoder import CharEncoder

            char_dim = config.get("char_dim", 128)
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
            self.version = 2
        else:
            self.version = 1

        self.generator = StrokeGenerator(**gen_kwargs)
        self.style_encoder = StyleEncoder(**style_enc_kwargs)

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
        noise_scale: float = 0.02,
    ) -> list[np.ndarray]:
        """スタイルサンプルからストロークを生成する。V3はデフォーメーション方式。"""
        if self.version == 3:
            return self._generate_v3(style_sample, reference_strokes, noise_scale)

        if self.norm_stats is not None:
            from src.model.data_utils import normalize_deltas

            style_sample = normalize_deltas(style_sample, self.norm_stats)

        style = self.style_encoder(style_sample)

        char_embedding: torch.Tensor | None = None
        if self.char_encoder is not None:
            if reference_strokes is not None:
                from src.model.char_encoder import CharEncoder

                seq = CharEncoder.strokes_to_sequence(reference_strokes)
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                if self.ref_norm_stats is not None:
                    from src.model.data_utils import normalize_reference

                    seq_tensor = normalize_reference(seq_tensor, self.ref_norm_stats)
                char_embedding = self.char_encoder(seq_tensor)
            else:
                char_embedding = torch.zeros(1, self.generator.char_dim)

        num_ref_strokes = len(reference_strokes) if reference_strokes is not None else 1
        all_strokes: list[np.ndarray] = []

        for stroke_idx in range(num_ref_strokes):
            stroke_index_tensor = torch.tensor([stroke_idx])

            ref_stroke_len = len(reference_strokes[stroke_idx]) if reference_strokes else num_steps
            max_stroke_steps = min(int(ref_stroke_len * 1.5), num_steps)

            if self.norm_stats is not None:
                init_dx = -self.norm_stats["mean_x"] / self.norm_stats["std_x"]
                init_dy = -self.norm_stats["mean_y"] / self.norm_stats["std_y"]
                current = torch.tensor([[[init_dx, init_dy]]])
            else:
                current = torch.zeros(1, 1, 2)

            points: list[list[float]] = []

            for _ in range(max_stroke_steps):
                output = self.generator(
                    current, style,
                    char_embedding=char_embedding,
                    stroke_index=stroke_index_tensor,
                )

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

                eos_prob = torch.sigmoid(output["eos_logit"][:, -1, 0])

                if self.norm_stats is not None:
                    from src.model.data_utils import denormalize_point

                    dx_raw, dy_raw = denormalize_point(
                        dx.item(), dy.item(), self.norm_stats
                    )
                else:
                    dx_raw, dy_raw = dx.item(), dy.item()

                points.append([dx_raw, dy_raw])

                if eos_prob > 0.5:
                    break

                next_input = torch.tensor([[[dx.item(), dy.item()]]])
                current = torch.cat([current, next_input], dim=1)

            if len(points) >= 2:
                stroke_points = []
                cx, cy = 0.0, 0.0
                for dx_val, dy_val in points:
                    cx += dx_val
                    cy += dy_val
                    stroke_points.append([cx, cy])

                if reference_strokes is not None and stroke_idx < len(reference_strokes):
                    ref_start = reference_strokes[stroke_idx][0]
                    for pt in stroke_points:
                        pt[0] += ref_start[0]
                        pt[1] += ref_start[1]

                all_strokes.append(np.array(stroke_points))

        if not all_strokes:
            all_strokes = [np.array([[0.0, 0.0], [1.0, 1.0]])]

        return all_strokes

    def _generate_v3(
        self,
        style_sample: torch.Tensor,
        reference_strokes: list[NDArray[np.float64]] | None,
        noise_scale: float = 0.02,
    ) -> list[np.ndarray]:
        """V3: Deform reference strokes using predicted offsets."""
        from src.model.data_utils import normalize_deltas, resample_stroke

        if reference_strokes is None or len(reference_strokes) == 0:
            return [np.array([[0.0, 0.0], [1.0, 1.0]])]

        if self.norm_stats is not None:
            style_sample = normalize_deltas(style_sample, self.norm_stats)

        style = self.style_encoder(style_sample)

        all_strokes: list[np.ndarray] = []
        for i, ref_stroke in enumerate(reference_strokes):
            if len(ref_stroke) < 2:
                continue
            ref_resampled = resample_stroke(
                np.asarray(ref_stroke, dtype=np.float32), self.num_points
            )
            ref_tensor = torch.tensor(
                ref_resampled, dtype=torch.float32
            ).unsqueeze(0)
            stroke_idx = torch.tensor([i])

            offsets = self.deformer(ref_tensor, style, stroke_idx)
            noise = torch.randn_like(offsets) * noise_scale
            deformed = ref_tensor + offsets + noise

            all_strokes.append(deformed.squeeze(0).detach().numpy())

        if not all_strokes:
            all_strokes = [np.array([[0.0, 0.0], [1.0, 1.0]])]

        return all_strokes
