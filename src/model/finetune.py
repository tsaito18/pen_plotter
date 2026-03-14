"""事前学習済みモデルのFine-tuningパイプライン。

StyleEncoderのみを訓練し、CharEncoderとStrokeGeneratorは凍結する。
少数のユーザーサンプル（20-30文字）で過学習を防ぎつつスタイル適応を行う。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.model.char_encoder import CharEncoder
from src.model.stroke_model import StrokeGenerator, mdn_loss
from src.model.style_encoder import StyleEncoder


@dataclass
class FinetuneConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 5e-4
    grad_clip_norm: float = 5.0


class FinetuneDataset(Dataset):
    """ユーザーサンプルと参照ストロークのペアデータセット。

    user_dirとref_dirの両方に存在する文字のみをペアリングする。
    """

    def __init__(self, user_dir: Path, ref_dir: Path) -> None:
        self.samples: list[tuple[str, Path, Path]] = []
        user_dir = Path(user_dir)
        ref_dir = Path(ref_dir)

        ref_chars: dict[str, Path] = {}
        if ref_dir.is_dir():
            for char_dir in ref_dir.iterdir():
                if char_dir.is_dir():
                    ref_files = list(char_dir.glob("*.json"))
                    if ref_files:
                        ref_chars[char_dir.name] = ref_files[0]

        if user_dir.is_dir():
            for char_dir in sorted(user_dir.iterdir()):
                if char_dir.is_dir() and char_dir.name in ref_chars:
                    for f in sorted(char_dir.glob("*.json")):
                        self.samples.append((char_dir.name, f, ref_chars[char_dir.name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        character, user_path, ref_path = self.samples[idx]

        user_data = json.loads(user_path.read_text(encoding="utf-8"))
        points = []
        for stroke in user_data["strokes"]:
            for pt in stroke:
                points.append([pt["x"], pt["y"], pt.get("pressure", 1.0)])
        strokes_tensor = torch.tensor(points, dtype=torch.float32)

        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_points = []
        for stroke in ref_data["strokes"]:
            for pt in stroke:
                ref_points.append([pt["x"], pt["y"]])
        ref_tensor = torch.tensor(ref_points, dtype=torch.float32)

        return {
            "strokes": strokes_tensor,
            "ref_strokes": ref_tensor,
            "character": character,
        }


def collate_finetune(batch: list[dict]) -> dict:
    """ユーザーストロークと参照ストロークをそれぞれパディングしてバッチ化。"""
    strokes = [item["strokes"] for item in batch]
    refs = [item["ref_strokes"] for item in batch]
    padded_strokes = pad_sequence(strokes, batch_first=True, padding_value=0.0)
    padded_refs = pad_sequence(refs, batch_first=True, padding_value=0.0)
    characters = [item["character"] for item in batch]
    return {
        "strokes": padded_strokes,
        "ref_strokes": padded_refs,
        "characters": characters,
    }


class Finetuner:
    """事前学習済みチェックポイントからStyleEncoderのみをFine-tuningする。"""

    def __init__(
        self,
        config: FinetuneConfig,
        pretrain_checkpoint: Path,
        user_data_dir: Path,
        ref_dir: Path,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = torch.load(pretrain_checkpoint, weights_only=False, map_location="cpu")
        self.ckpt_config = checkpoint["config"]

        cfg = self.ckpt_config
        self.generator = StrokeGenerator(
            char_dim=cfg["char_dim"],
            hidden_dim=cfg["hidden_dim"],
            style_dim=cfg["style_dim"],
            num_mixtures=cfg["num_mixtures"],
        )
        self.generator.load_state_dict(checkpoint["generator_state_dict"])

        self.style_encoder = StyleEncoder(style_dim=cfg["style_dim"])
        self.style_encoder.load_state_dict(checkpoint["style_encoder_state_dict"])

        self.char_encoder = CharEncoder(char_dim=cfg["char_dim"])
        self.char_encoder.load_state_dict(checkpoint["char_encoder_state_dict"])

        for p in self.generator.parameters():
            p.requires_grad = False
        for p in self.char_encoder.parameters():
            p.requires_grad = False

        self.dataset = FinetuneDataset(user_data_dir, ref_dir)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_finetune,
        )

        self.optimizer = torch.optim.Adam(self.style_encoder.parameters(), lr=config.learning_rate)

    def train(self) -> dict:
        history: dict[str, list[float]] = {"losses": []}

        self.generator.eval()
        self.char_encoder.eval()
        self.style_encoder.train()

        for _epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in self.dataloader:
                strokes = batch["strokes"]
                ref_strokes = batch["ref_strokes"]

                style = self.style_encoder(strokes)

                with torch.no_grad():
                    char_emb = self.char_encoder(ref_strokes)

                x = strokes[:, :-1]
                target = strokes[:, 1:]

                output = self.generator(x, style, char_embedding=char_emb)

                min_len = min(output["pi"].shape[1], target.shape[1])
                trimmed_output = {k: v[:, :min_len] for k, v in output.items()}
                trimmed_target = target[:, :min_len]

                loss = mdn_loss(trimmed_output, trimmed_target)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.style_encoder.parameters(),
                    self.config.grad_clip_norm,
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["losses"].append(avg_loss)

        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "style_encoder_state_dict": self.style_encoder.state_dict(),
                "char_encoder_state_dict": self.char_encoder.state_dict(),
                "config": self.ckpt_config,
            },
            self.output_dir / "finetuned.pt",
        )

        return history
