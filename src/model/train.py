"""ストローク生成モデルの訓練パイプライン。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model.data_utils import compute_normalization_stats, normalize_deltas
from src.model.dataset import StrokeDataset, collate_strokes
from src.model.stroke_model import StrokeGenerator, mdn_loss
from src.model.style_encoder import StyleEncoder


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    grad_clip_norm: float = 5.0
    style_dim: int = 128
    hidden_dim: int = 128
    num_mixtures: int = 5


class Trainer:
    def __init__(
        self, config: TrainConfig, data_dir: Path | list[Path], output_dir: Path
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = StrokeDataset(data_dir)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_strokes,
        )

        sample_tensors = [self.dataset[i]["strokes"] for i in range(len(self.dataset))]
        self.norm_stats = compute_normalization_stats(sample_tensors)

        self.generator = StrokeGenerator(
            hidden_dim=config.hidden_dim,
            style_dim=config.style_dim,
            num_mixtures=config.num_mixtures,
        )
        self.style_encoder = StyleEncoder(style_dim=config.style_dim)

        self.optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.style_encoder.parameters()),
            lr=config.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

    def train(self) -> dict:
        history: dict[str, list[float]] = {"losses": []}

        for _epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in self.dataloader:
                strokes = batch["strokes"]
                strokes = normalize_deltas(strokes, self.norm_stats)
                style = self.style_encoder(strokes)

                x = strokes[:, :-1]
                target = strokes[:, 1:]

                output = self.generator(x, style)

                # output と target のシーケンス長を揃える
                min_len = min(output["pi"].shape[1], target.shape[1])
                trimmed_output = {k: v[:, :min_len] for k, v in output.items()}
                trimmed_target = target[:, :min_len]

                loss = mdn_loss(trimmed_output, trimmed_target)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.generator.parameters())
                    + list(self.style_encoder.parameters()),
                    self.config.grad_clip_norm,
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["losses"].append(avg_loss)
            self.scheduler.step()

        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "style_encoder_state_dict": self.style_encoder.state_dict(),
                "config": self.config,
                "norm_stats": self.norm_stats,
            },
            self.output_dir / "checkpoint.pt",
        )

        return history
