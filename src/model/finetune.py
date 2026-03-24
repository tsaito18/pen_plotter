"""事前学習済みモデルのFine-tuningパイプライン。

StyleEncoderのみを訓練し、CharEncoderとStrokeGeneratorは凍結する。
少数のユーザーサンプル（20-30文字）で過学習を防ぎつつスタイル適応を行う。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.model.char_encoder import CharEncoder
from src.model.data_utils import (
    normalize_deltas,
    normalize_deltas_2d,
    normalize_reference,
    reference_to_sequence,
    resample_stroke,
    strokes_to_deltas,
)
from src.model.pretrain import _detect_device
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

    各ストロークが個別のサンプルとなる（ストローク単位生成）。
    user_dirとref_dirの両方に存在する文字のみをペアリングする。
    """

    def __init__(self, user_dir: Path, ref_dir: Path) -> None:
        self.char_samples: list[tuple[str, Path, Path]] = []
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
                        self.char_samples.append((char_dir.name, f, ref_chars[char_dir.name]))

        self.samples: list[tuple[int, int, int]] = []
        for char_idx, (ch, user_path, _) in enumerate(self.char_samples):
            user_data = json.loads(user_path.read_text(encoding="utf-8"))
            n_strokes = len(user_data["strokes"])
            for stroke_idx in range(n_strokes):
                stroke = user_data["strokes"][stroke_idx]
                if len(stroke) >= 2:
                    self.samples.append((char_idx, stroke_idx, n_strokes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        char_idx, stroke_idx, num_strokes = self.samples[idx]
        character, user_path, ref_path = self.char_samples[char_idx]

        user_data = json.loads(user_path.read_text(encoding="utf-8"))
        stroke_points = user_data["strokes"][stroke_idx]

        pts = [(pt["x"], pt["y"]) for pt in stroke_points]
        deltas = torch.zeros(len(pts), 2, dtype=torch.float32)
        for i in range(1, len(pts)):
            deltas[i, 0] = pts[i][0] - pts[i - 1][0]
            deltas[i, 1] = pts[i][1] - pts[i - 1][1]

        eos = torch.zeros(len(pts), 1, dtype=torch.float32)
        eos[-1, 0] = 1.0

        style_tensor = strokes_to_deltas(user_data["strokes"])

        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_tensor = reference_to_sequence(ref_data["strokes"])

        return {
            "stroke_deltas": deltas,
            "eos": eos,
            "stroke_index": stroke_idx,
            "num_strokes": num_strokes,
            "ref_strokes": ref_tensor,
            "character": character,
            "style_strokes": style_tensor,
        }


def collate_finetune(batch: list[dict]) -> dict:
    """ユーザーストロークと参照ストロークをそれぞれパディングしてバッチ化。"""
    stroke_deltas = [item["stroke_deltas"] for item in batch]
    eos_list = [item["eos"] for item in batch]
    style_strokes = [item["style_strokes"] for item in batch]
    refs = [item["ref_strokes"] for item in batch]

    stroke_lengths = torch.tensor([s.shape[0] for s in stroke_deltas])
    style_lengths = torch.tensor([s.shape[0] for s in style_strokes])
    ref_lengths = torch.tensor([r.shape[0] for r in refs])
    stroke_indices = torch.tensor([item["stroke_index"] for item in batch])

    padded_deltas = pad_sequence(stroke_deltas, batch_first=True, padding_value=0.0)
    padded_eos = pad_sequence(eos_list, batch_first=True, padding_value=0.0)
    padded_style = pad_sequence(style_strokes, batch_first=True, padding_value=0.0)
    padded_refs = pad_sequence(refs, batch_first=True, padding_value=0.0)

    characters = [item["character"] for item in batch]
    return {
        "stroke_deltas": padded_deltas,
        "eos": padded_eos,
        "stroke_indices": stroke_indices,
        "style_strokes": padded_style,
        "ref_strokes": padded_refs,
        "stroke_lengths": stroke_lengths,
        "style_lengths": style_lengths,
        "ref_lengths": ref_lengths,
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
        device: str | None = None,
        num_workers: int = 0,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _detect_device(device)

        checkpoint = torch.load(pretrain_checkpoint, weights_only=False, map_location="cpu")
        self.ckpt_config = checkpoint["config"]
        self.norm_stats = checkpoint.get("norm_stats", None)
        self.ref_norm_stats = checkpoint.get("ref_norm_stats", None)

        cfg = self.ckpt_config
        self.generator = StrokeGenerator(
            input_dim=2,
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

        self.generator.to(self.device)
        self.style_encoder.to(self.device)
        self.char_encoder.to(self.device)

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
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        self.optimizer = torch.optim.Adam(self.style_encoder.parameters(), lr=config.learning_rate)

    def train(self) -> dict:
        print(f"Device: {self.device}")
        history: dict[str, list[float]] = {"losses": []}

        self.generator.train()
        self.char_encoder.train()
        self.style_encoder.train()

        for _epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in self.dataloader:
                stroke_deltas = batch["stroke_deltas"].to(self.device)
                eos = batch["eos"].to(self.device)
                stroke_indices = batch["stroke_indices"].to(self.device)

                if self.norm_stats is not None:
                    stroke_deltas_norm = normalize_deltas_2d(stroke_deltas, self.norm_stats)
                else:
                    stroke_deltas_norm = stroke_deltas

                style_strokes = batch["style_strokes"].to(self.device)
                if self.norm_stats is not None:
                    style_strokes = normalize_deltas(style_strokes, self.norm_stats)

                ref_strokes = batch["ref_strokes"].to(self.device)
                if self.ref_norm_stats is not None:
                    ref_strokes = normalize_reference(ref_strokes, self.ref_norm_stats)

                style_lengths = batch["style_lengths"]
                ref_lengths = batch["ref_lengths"]

                style = self.style_encoder(style_strokes, lengths=style_lengths)

                with torch.no_grad():
                    char_emb = self.char_encoder(ref_strokes, lengths=ref_lengths)

                x = stroke_deltas_norm[:, :-1]
                target_xy = stroke_deltas_norm[:, 1:]
                target_eos = eos[:, 1:]

                output = self.generator(
                    x, style,
                    char_embedding=char_emb,
                    stroke_index=stroke_indices,
                )

                min_len = min(output["pi"].shape[1], target_xy.shape[1])
                trimmed_output = {k: v[:, :min_len] for k, v in output.items()}
                trimmed_xy = target_xy[:, :min_len]
                trimmed_eos = target_eos[:, :min_len]

                stroke_loss, eos_loss = mdn_loss(trimmed_output, trimmed_xy, trimmed_eos)
                loss = stroke_loss + 0.5 * eos_loss

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

        cpu = torch.device("cpu")
        torch.save(
            {
                "generator_state_dict": {
                    k: v.to(cpu) for k, v in self.generator.state_dict().items()
                },
                "style_encoder_state_dict": {
                    k: v.to(cpu) for k, v in self.style_encoder.state_dict().items()
                },
                "char_encoder_state_dict": {
                    k: v.to(cpu) for k, v in self.char_encoder.state_dict().items()
                },
                "config": self.ckpt_config,
                "norm_stats": self.norm_stats,
                "ref_norm_stats": self.ref_norm_stats,
            },
            self.output_dir / "finetuned.pt",
        )

        return history


class FinetuneDeformationDataset(Dataset):
    """User samples + KanjiVG for deformation fine-tuning."""

    def __init__(
        self,
        user_dir: Path,
        ref_dir: Path,
        num_points: int = 32,
    ) -> None:
        self.num_points = num_points
        self.char_samples: list[tuple[str, Path, Path]] = []
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
                        self.char_samples.append(
                            (char_dir.name, f, ref_chars[char_dir.name])
                        )

        self.samples: list[tuple[int, int, int]] = []
        for char_idx, (ch, user_path, ref_path) in enumerate(self.char_samples):
            user_data = json.loads(user_path.read_text(encoding="utf-8"))
            ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
            n_common = min(len(user_data["strokes"]), len(ref_data["strokes"]))
            for stroke_idx in range(n_common):
                user_stroke = user_data["strokes"][stroke_idx]
                ref_stroke = ref_data["strokes"][stroke_idx]
                if len(user_stroke) >= 2 and len(ref_stroke) >= 2:
                    self.samples.append(
                        (char_idx, stroke_idx, len(user_data["strokes"]))
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        char_idx, stroke_idx, num_strokes = self.samples[idx]
        character, user_path, ref_path = self.char_samples[char_idx]

        user_data = json.loads(user_path.read_text(encoding="utf-8"))
        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))

        user_stroke_pts = np.array(
            [[pt["x"], pt["y"]] for pt in user_data["strokes"][stroke_idx]],
            dtype=np.float32,
        )
        ref_stroke_pts = np.array(
            [[pt["x"], pt["y"]] for pt in ref_data["strokes"][stroke_idx]],
            dtype=np.float32,
        )

        # Normalize user strokes to same [0, target_size] range as KanjiVG reference
        all_user_pts = np.concatenate(
            [np.array([[pt["x"], pt["y"]] for pt in s]) for s in user_data["strokes"]],
            axis=0,
        )
        u_min = all_user_pts.min(axis=0)
        u_max = all_user_pts.max(axis=0)
        u_range = (u_max - u_min).max()
        all_ref_pts = np.concatenate(
            [np.array([[pt["x"], pt["y"]] for pt in s]) for s in ref_data["strokes"]],
            axis=0,
        )
        r_range = (all_ref_pts.max(axis=0) - all_ref_pts.min(axis=0)).max()
        if u_range > 0:
            scale = r_range / u_range
            user_stroke_pts = (user_stroke_pts - u_min) * scale + all_ref_pts.min(axis=0)

        ref_resampled = resample_stroke(ref_stroke_pts, self.num_points)
        hand_resampled = resample_stroke(user_stroke_pts, self.num_points)
        offset = hand_resampled - ref_resampled

        # Style tensor also needs normalization
        normalized_user_strokes = []
        for s in user_data["strokes"]:
            pts = np.array([[pt["x"], pt["y"]] for pt in s], dtype=np.float32)
            if u_range > 0:
                pts = (pts - u_min) * scale + all_ref_pts.min(axis=0)
            normalized_user_strokes.append(
                [{"x": float(p[0]), "y": float(p[1])} for p in pts]
            )
        style_tensor = strokes_to_deltas(normalized_user_strokes)

        return {
            "reference_points": torch.tensor(ref_resampled, dtype=torch.float32),
            "target_offsets": torch.tensor(offset, dtype=torch.float32),
            "stroke_index": stroke_idx,
            "style_strokes": style_tensor,
            "character": character,
        }


def collate_deformation_finetune(batch: list[dict]) -> dict:
    """Collate for FinetuneDeformationDataset."""
    reference_points = torch.stack([item["reference_points"] for item in batch])
    target_offsets = torch.stack([item["target_offsets"] for item in batch])
    stroke_indices = torch.tensor([item["stroke_index"] for item in batch])
    style_strokes = [item["style_strokes"] for item in batch]
    style_lengths = torch.tensor([s.shape[0] for s in style_strokes])
    padded_style = pad_sequence(style_strokes, batch_first=True, padding_value=0.0)
    return {
        "reference_points": reference_points,
        "target_offsets": target_offsets,
        "stroke_indices": stroke_indices,
        "style_strokes": padded_style,
        "style_lengths": style_lengths,
        "characters": [item["character"] for item in batch],
    }


class DeformationFinetuner:
    """Fine-tune StyleEncoder only, freeze StrokeDeformer."""

    def __init__(
        self,
        config: FinetuneConfig,
        pretrain_checkpoint: Path,
        user_data_dir: Path,
        ref_dir: Path,
        output_dir: Path,
        device: str | None = None,
        num_workers: int = 0,
    ) -> None:
        from src.model.stroke_deformer import StrokeDeformer

        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _detect_device(device)

        checkpoint = torch.load(
            pretrain_checkpoint, weights_only=False, map_location="cpu"
        )
        self.ckpt_config = checkpoint["config"]
        self.norm_stats = checkpoint.get("norm_stats", None)

        cfg = self.ckpt_config
        num_points = cfg.get("num_points", 32)

        self.deformer = StrokeDeformer(
            style_dim=cfg.get("style_dim", 128),
            hidden_dim=cfg.get("hidden_dim", 256),
        )
        self.deformer.load_state_dict(checkpoint["deformer_state_dict"])

        self.style_encoder = StyleEncoder(style_dim=cfg.get("style_dim", 128))
        self.style_encoder.load_state_dict(checkpoint["style_encoder_state_dict"])

        self.deformer.to(self.device)
        self.style_encoder.to(self.device)

        for p in self.deformer.parameters():
            p.requires_grad = False

        self.dataset = FinetuneDeformationDataset(
            user_data_dir, ref_dir, num_points=num_points
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_deformation_finetune,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        self.optimizer = torch.optim.Adam(
            self.style_encoder.parameters(), lr=config.learning_rate
        )

    def train(self) -> dict:
        from src.model.stroke_deformer import deformation_loss

        print(f"[V3] Device: {self.device}")
        history: dict[str, list[float]] = {"losses": []}

        self.deformer.eval()
        self.style_encoder.train()

        for _epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in self.dataloader:
                ref_points = batch["reference_points"].to(self.device)
                target_offsets = batch["target_offsets"].to(self.device)
                stroke_indices = batch["stroke_indices"].to(self.device)

                style_strokes = batch["style_strokes"].to(self.device)
                if self.norm_stats is not None:
                    style_strokes = normalize_deltas(style_strokes, self.norm_stats)
                style_lengths = batch["style_lengths"]

                style = self.style_encoder(style_strokes, lengths=style_lengths)

                predicted = self.deformer(ref_points, style, stroke_indices)

                loss = deformation_loss(predicted, target_offsets)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.style_encoder.parameters(), self.config.grad_clip_norm
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["losses"].append(avg_loss)

        cpu = torch.device("cpu")
        torch.save(
            {
                "deformer_state_dict": {
                    k: v.to(cpu) for k, v in self.deformer.state_dict().items()
                },
                "style_encoder_state_dict": {
                    k: v.to(cpu) for k, v in self.style_encoder.state_dict().items()
                },
                "config": self.ckpt_config,
                "norm_stats": self.norm_stats,
                "version": 3,
            },
            self.output_dir / "finetuned.pt",
        )

        return history
