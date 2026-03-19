"""CharEncoder + StrokeGenerator の事前学習パイプライン。"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.collector.casia_parser import CASIAParser
from src.model.char_encoder import CharEncoder
from src.model.data_utils import (
    compute_normalization_stats,
    compute_reference_stats,
    normalize_deltas,
    normalize_reference,
    reference_to_sequence,
    strokes_to_deltas,
    strokes_to_deltas_from_arrays,
)
from src.model.stroke_model import StrokeGenerator, embedding_variance_loss, mdn_loss
from src.model.style_encoder import StyleEncoder


@dataclass
class PretrainConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    grad_clip_norm: float = 5.0
    style_dim: int = 128
    char_dim: int = 128
    hidden_dim: int = 128
    num_mixtures: int = 5


class PairedStrokeDataset(Dataset):
    """手書きストロークと対応するKanjiVG参照ストロークのペアを提供する。

    hand_dir 内の文字サブディレクトリと ref_dir 内の文字サブディレクトリを
    照合し、両方に存在する文字のみをデータセットに含める。
    """

    def __init__(self, hand_dir: Path | list[Path], ref_dir: Path) -> None:
        self.samples: list[tuple[str, Path, Path]] = []  # (char, hand_path, ref_file_path)
        hand_dirs = hand_dir if isinstance(hand_dir, list) else [hand_dir]

        ref_dir = Path(ref_dir)
        ref_files: dict[str, Path] = {}
        if ref_dir.is_dir():
            for d in ref_dir.iterdir():
                if d.is_dir():
                    files = sorted(d.glob("*.json"))
                    if files:
                        ref_files[d.name] = files[0]

        for hd in hand_dirs:
            hd = Path(hd)
            if not hd.is_dir():
                continue
            for char_dir in sorted(hd.iterdir()):
                if not char_dir.is_dir():
                    continue
                ch = char_dir.name
                if ch not in ref_files:
                    continue
                for f in sorted(char_dir.glob("*.json")):
                    self.samples.append((ch, f, ref_files[ch]))

        self.char_to_indices: dict[str, list[int]] = {}
        for i, (ch, _, _) in enumerate(self.samples):
            self.char_to_indices.setdefault(ch, []).append(i)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        character, hand_path, ref_path = self.samples[idx]

        hand_data = json.loads(hand_path.read_text(encoding="utf-8"))
        strokes_tensor = strokes_to_deltas(hand_data["strokes"])

        same_char = self.char_to_indices[character]
        candidates = [i for i in same_char if i != idx]
        if candidates:
            style_idx = random.choice(candidates)
            _, style_path, _ = self.samples[style_idx]
            style_data = json.loads(style_path.read_text(encoding="utf-8"))
            style_tensor = strokes_to_deltas(style_data["strokes"])
        else:
            style_tensor = strokes_tensor.clone()
            style_tensor[:, :2] += torch.randn_like(style_tensor[:, :2]) * 0.1

        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        reference_tensor = reference_to_sequence(ref_data["strokes"])

        return {
            "strokes": strokes_tensor,
            "style_strokes": style_tensor,
            "reference": reference_tensor,
            "character": character,
        }


class CASIAPairedDataset(Dataset):
    """CASIA .pot バイナリから直接読み込み、KanjiVG参照とペアリングする。

    大量のJSONファイル生成を回避し、.potファイルをメモリ上で展開して使用する。
    """

    def __init__(
        self,
        pot_dir: Path,
        ref_dir: Path,
        target_size: float = 10.0,
        num_points: int = 0,
        max_samples: int = 0,
    ) -> None:
        self.target_size = target_size
        self.num_points = num_points
        parser = CASIAParser()

        ref_dir = Path(ref_dir)
        ref_files: dict[str, Path] = {}
        if ref_dir.is_dir():
            for d in ref_dir.iterdir():
                if d.is_dir():
                    files = sorted(d.glob("*.json"))
                    if files:
                        ref_files[d.name] = files[0]

        pot_dir = Path(pot_dir)
        pot_files = sorted(pot_dir.glob("*.pot"))
        n_files = len(pot_files)
        print(f"Loading .pot files from {pot_dir}...")

        self.samples: list[tuple[str, list[np.ndarray], Path]] = []
        char_set: set[str] = set()

        for pot_file in pot_files:
            casia_samples = parser.parse_pot_file(pot_file)
            for sample in casia_samples:
                ch = sample.character
                if ch not in ref_files:
                    continue
                normalized = parser.normalize(sample.strokes, target_size=target_size)
                if not normalized:
                    continue
                self.samples.append((ch, normalized, ref_files[ch]))
                char_set.add(ch)
                if 0 < max_samples <= len(self.samples):
                    break
            if 0 < max_samples <= len(self.samples):
                break

        print(
            f"Loaded {len(self.samples)} samples "
            f"({len(char_set)} characters) from {n_files} .pot files"
        )

        self.char_to_indices: dict[str, list[int]] = {}
        for i, (ch, _, _) in enumerate(self.samples):
            self.char_to_indices.setdefault(ch, []).append(i)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        character, normalized_strokes, ref_path = self.samples[idx]

        strokes_tensor = strokes_to_deltas_from_arrays(normalized_strokes)

        same_char = self.char_to_indices[character]
        candidates = [i for i in same_char if i != idx]
        if candidates:
            style_idx = random.choice(candidates)
            _, style_strokes_raw, _ = self.samples[style_idx]
            style_tensor = strokes_to_deltas_from_arrays(style_strokes_raw)
        else:
            style_tensor = strokes_tensor.clone()
            style_tensor[:, :2] += torch.randn_like(style_tensor[:, :2]) * 0.1

        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        reference_tensor = reference_to_sequence(ref_data["strokes"])

        return {
            "strokes": strokes_tensor,
            "style_strokes": style_tensor,
            "reference": reference_tensor,
            "character": character,
        }


def collate_paired(batch: list[dict]) -> dict:
    """PairedStrokeDataset の可変長シーケンスをパディングしてバッチ化する。"""
    strokes = [item["strokes"] for item in batch]
    style_strokes = [item["style_strokes"] for item in batch]
    references = [item["reference"] for item in batch]
    lengths = torch.tensor([s.shape[0] for s in strokes])
    style_lengths = torch.tensor([s.shape[0] for s in style_strokes])
    ref_lengths = torch.tensor([r.shape[0] for r in references])
    padded_strokes = pad_sequence(strokes, batch_first=True, padding_value=0.0)
    padded_style = pad_sequence(style_strokes, batch_first=True, padding_value=0.0)
    padded_refs = pad_sequence(references, batch_first=True, padding_value=0.0)
    characters = [item["character"] for item in batch]
    return {
        "strokes": padded_strokes,
        "style_strokes": padded_style,
        "reference": padded_refs,
        "lengths": lengths,
        "style_lengths": style_lengths,
        "ref_lengths": ref_lengths,
        "characters": characters,
    }


def _detect_device(device: str | None = None) -> torch.device:
    """利用可能なアクセラレータを自動検出する。明示指定時はそれを使用。"""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class Pretrainer:
    """CharEncoder + StyleEncoder + StrokeGenerator を同時に事前学習する。"""

    def __init__(
        self,
        config: PretrainConfig,
        hand_dir: Path | list[Path],
        ref_dir: Path,
        output_dir: Path,
        device: str | None = None,
        pot_dir: Path | None = None,
        max_samples: int = 0,
        num_workers: int = 0,
        amp: bool = False,
        norm_sample_size: int = 5000,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _detect_device(device)

        if pot_dir is not None:
            self.dataset = CASIAPairedDataset(pot_dir, ref_dir, max_samples=max_samples)
        else:
            self.dataset = PairedStrokeDataset(hand_dir, ref_dir)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_paired,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        indices = list(range(len(self.dataset)))
        if len(indices) > norm_sample_size:
            indices = random.sample(indices, norm_sample_size)
        sample_tensors = [self.dataset[i]["strokes"] for i in indices]
        self.norm_stats = compute_normalization_stats(sample_tensors)

        ref_tensors = [self.dataset[i]["reference"] for i in indices]
        self.ref_norm_stats = compute_reference_stats(ref_tensors)

        self.char_encoder = CharEncoder(char_dim=config.char_dim).to(self.device)
        self.style_encoder = StyleEncoder(style_dim=config.style_dim).to(self.device)
        self.generator = StrokeGenerator(
            style_dim=config.style_dim,
            char_dim=config.char_dim,
            hidden_dim=config.hidden_dim,
            num_mixtures=config.num_mixtures,
        ).to(self.device)

        self.optimizer = torch.optim.Adam([
            {"params": self.generator.parameters(), "lr": config.learning_rate},
            {"params": self.char_encoder.parameters(), "lr": config.learning_rate * 3},
            {"params": self.style_encoder.parameters(), "lr": config.learning_rate * 3},
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.amp = amp
        if amp:
            self.scaler = torch.amp.GradScaler(self.device.type)
        else:
            self.scaler = None

    def train(self) -> dict:
        print(f"Device: {self.device}" + (" (AMP)" if self.amp else ""))
        history: dict[str, list[float]] = {"losses": []}
        total_batches = len(self.dataloader)

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(self.dataloader):
                strokes = batch["strokes"].to(self.device)
                strokes = normalize_deltas(strokes, self.norm_stats)

                style_strokes = batch["style_strokes"].to(self.device)
                style_strokes = normalize_deltas(style_strokes, self.norm_stats)

                reference = batch["reference"].to(self.device)
                reference = normalize_reference(reference, self.ref_norm_stats)

                lengths = batch["lengths"]
                style_lengths = batch["style_lengths"]
                ref_lengths = batch["ref_lengths"]

                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    char_embedding = self.char_encoder(reference, lengths=ref_lengths)
                    style_vector = self.style_encoder(style_strokes, lengths=style_lengths)

                    x = strokes[:, :-1]
                    target = strokes[:, 1:]

                    output = self.generator(x, style_vector, char_embedding)

                    min_len = min(output["pi"].shape[1], target.shape[1])
                    trimmed_output = {k: v[:, :min_len] for k, v in output.items()}
                    trimmed_target = target[:, :min_len]

                    loss_mdn = mdn_loss(trimmed_output, trimmed_target)
                    loss_char_var = embedding_variance_loss(char_embedding)
                    loss_style_var = embedding_variance_loss(style_vector)
                    loss = loss_mdn + 0.1 * (loss_char_var + loss_style_var)

                self.optimizer.zero_grad()
                all_params = [
                    p for group in self.optimizer.param_groups for p in group["params"]
                ]
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, self.config.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, self.config.grad_clip_norm)
                    self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                    print(
                        f"\r  Epoch {epoch + 1}/{self.config.epochs} "
                        f"[{batch_idx + 1}/{total_batches}] "
                        f"loss: {loss.item():.4f}",
                        end="", flush=True,
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            history["losses"].append(avg_loss)
            self.scheduler.step()
            print(f"\r  Epoch {epoch + 1}/{self.config.epochs} — avg loss: {avg_loss:.4f}")

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
                "config": asdict(self.config),
                "norm_stats": self.norm_stats,
                "ref_norm_stats": self.ref_norm_stats,
            },
            self.output_dir / "pretrain_checkpoint.pt",
        )

        return history
