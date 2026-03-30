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
    normalize_deltas_2d,
    normalize_reference,
    reference_to_sequence,
    resample_stroke,
    stroke_to_deltas_2d,
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

    各ストロークが個別のサンプルとなる（ストローク単位生成）。
    hand_dir 内の文字サブディレクトリと ref_dir 内の文字サブディレクトリを
    照合し、両方に存在する文字のみをデータセットに含める。
    """

    def __init__(self, hand_dir: Path | list[Path], ref_dir: Path) -> None:
        self.char_samples: list[tuple[str, Path, Path]] = []
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
                    self.char_samples.append((ch, f, ref_files[ch]))

        self.char_to_indices: dict[str, list[int]] = {}
        for i, (ch, _, _) in enumerate(self.char_samples):
            self.char_to_indices.setdefault(ch, []).append(i)

        self.samples: list[tuple[int, int, int]] = []
        for char_idx, (ch, hand_path, _) in enumerate(self.char_samples):
            hand_data = json.loads(hand_path.read_text(encoding="utf-8"))
            n_strokes = len(hand_data["strokes"])
            for stroke_idx in range(n_strokes):
                stroke = hand_data["strokes"][stroke_idx]
                if len(stroke) >= 2:
                    self.samples.append((char_idx, stroke_idx, n_strokes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        char_idx, stroke_idx, num_strokes = self.samples[idx]
        character, hand_path, ref_path = self.char_samples[char_idx]

        hand_data = json.loads(hand_path.read_text(encoding="utf-8"))
        stroke_points = hand_data["strokes"][stroke_idx]

        pts = [(pt["x"], pt["y"]) for pt in stroke_points]
        deltas = torch.zeros(len(pts), 2, dtype=torch.float32)
        for i in range(1, len(pts)):
            deltas[i, 0] = pts[i][0] - pts[i - 1][0]
            deltas[i, 1] = pts[i][1] - pts[i - 1][1]

        eos = torch.zeros(len(pts), 1, dtype=torch.float32)
        eos[-1, 0] = 1.0

        style_tensor = strokes_to_deltas(hand_data["strokes"])
        same_char = self.char_to_indices[character]
        candidates = [i for i in same_char if i != char_idx]
        if candidates:
            style_char_idx = random.choice(candidates)
            _, style_path, _ = self.char_samples[style_char_idx]
            style_data = json.loads(style_path.read_text(encoding="utf-8"))
            style_tensor = strokes_to_deltas(style_data["strokes"])

        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        reference_tensor = reference_to_sequence(ref_data["strokes"])

        return {
            "stroke_deltas": deltas,
            "eos": eos,
            "stroke_index": stroke_idx,
            "num_strokes": num_strokes,
            "reference": reference_tensor,
            "character": character,
            "style_strokes": style_tensor,
        }


class CASIAPairedDataset(Dataset):
    """CASIA .pot バイナリから直接読み込み、KanjiVG参照とペアリングする。

    各ストロークが個別のサンプルとなる（ストローク単位生成）。
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

        self.char_samples: list[tuple[str, list[np.ndarray], Path]] = []
        char_set: set[str] = set()
        total_stroke_samples = 0

        for pot_file in pot_files:
            casia_samples = parser.parse_pot_file(pot_file)
            for sample in casia_samples:
                ch = sample.character
                if ch not in ref_files:
                    continue
                normalized = parser.normalize(sample.strokes, target_size=target_size)
                if not normalized:
                    continue
                valid_strokes = [s for s in normalized if len(s) >= 2]
                if not valid_strokes:
                    continue
                self.char_samples.append((ch, normalized, ref_files[ch]))
                char_set.add(ch)
                total_stroke_samples += len(valid_strokes)
                if 0 < max_samples <= total_stroke_samples:
                    break
            if 0 < max_samples <= total_stroke_samples:
                break

        self.char_to_indices: dict[str, list[int]] = {}
        for i, (ch, _, _) in enumerate(self.char_samples):
            self.char_to_indices.setdefault(ch, []).append(i)

        self.samples: list[tuple[int, int, int]] = []
        for char_idx, (ch, stroke_arrays, _) in enumerate(self.char_samples):
            for stroke_idx, stroke_arr in enumerate(stroke_arrays):
                if len(stroke_arr) >= 2:
                    self.samples.append((char_idx, stroke_idx, len(stroke_arrays)))

        print(
            f"Loaded {len(self.samples)} stroke samples "
            f"({len(self.char_samples)} characters, {len(char_set)} unique) "
            f"from {n_files} .pot files"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        char_idx, stroke_idx, num_strokes = self.samples[idx]
        character, normalized_strokes, ref_path = self.char_samples[char_idx]

        stroke_arr = normalized_strokes[stroke_idx]
        deltas = stroke_to_deltas_2d(stroke_arr)

        eos = torch.zeros(len(stroke_arr), 1, dtype=torch.float32)
        eos[-1, 0] = 1.0

        style_tensor = strokes_to_deltas_from_arrays(normalized_strokes)
        same_char = self.char_to_indices[character]
        candidates = [i for i in same_char if i != char_idx]
        if candidates:
            style_char_idx = random.choice(candidates)
            _, style_strokes_raw, _ = self.char_samples[style_char_idx]
            style_tensor = strokes_to_deltas_from_arrays(style_strokes_raw)

        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        reference_tensor = reference_to_sequence(ref_data["strokes"])

        return {
            "stroke_deltas": deltas,
            "eos": eos,
            "stroke_index": stroke_idx,
            "num_strokes": num_strokes,
            "reference": reference_tensor,
            "character": character,
            "style_strokes": style_tensor,
        }


def collate_paired(batch: list[dict]) -> dict:
    """PairedStrokeDataset の可変長シーケンスをパディングしてバッチ化する。"""
    stroke_deltas = [item["stroke_deltas"] for item in batch]
    eos_list = [item["eos"] for item in batch]
    style_strokes = [item["style_strokes"] for item in batch]
    references = [item["reference"] for item in batch]

    stroke_lengths = torch.tensor([s.shape[0] for s in stroke_deltas])
    style_lengths = torch.tensor([s.shape[0] for s in style_strokes])
    ref_lengths = torch.tensor([r.shape[0] for r in references])
    stroke_indices = torch.tensor([item["stroke_index"] for item in batch])

    padded_deltas = pad_sequence(stroke_deltas, batch_first=True, padding_value=0.0)
    padded_eos = pad_sequence(eos_list, batch_first=True, padding_value=0.0)
    padded_style = pad_sequence(style_strokes, batch_first=True, padding_value=0.0)
    padded_refs = pad_sequence(references, batch_first=True, padding_value=0.0)

    characters = [item["character"] for item in batch]
    return {
        "stroke_deltas": padded_deltas,
        "eos": padded_eos,
        "stroke_indices": stroke_indices,
        "style_strokes": padded_style,
        "reference": padded_refs,
        "stroke_lengths": stroke_lengths,
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
        sample_tensors = [self.dataset[i]["stroke_deltas"] for i in indices]
        from src.model.data_utils import compute_normalization_stats_2d

        self.norm_stats = compute_normalization_stats_2d(sample_tensors)

        ref_tensors = [self.dataset[i]["reference"] for i in indices]
        self.ref_norm_stats = compute_reference_stats(ref_tensors)

        self.char_encoder = CharEncoder(char_dim=config.char_dim).to(self.device)
        self.style_encoder = StyleEncoder(style_dim=config.style_dim).to(self.device)
        self.generator = StrokeGenerator(
            input_dim=2,
            style_dim=config.style_dim,
            char_dim=config.char_dim,
            hidden_dim=config.hidden_dim,
            num_mixtures=config.num_mixtures,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.generator.parameters(), "lr": config.learning_rate},
                {"params": self.char_encoder.parameters(), "lr": config.learning_rate * 3},
                {"params": self.style_encoder.parameters(), "lr": config.learning_rate * 3},
            ]
        )
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
                stroke_deltas = batch["stroke_deltas"].to(self.device)
                stroke_deltas_norm = normalize_deltas_2d(stroke_deltas, self.norm_stats)

                eos = batch["eos"].to(self.device)
                stroke_indices = batch["stroke_indices"].to(self.device)

                style_strokes = batch["style_strokes"].to(self.device)
                style_strokes = normalize_deltas(style_strokes, self.norm_stats)

                reference = batch["reference"].to(self.device)
                reference = normalize_reference(reference, self.ref_norm_stats)

                style_lengths = batch["style_lengths"]
                ref_lengths = batch["ref_lengths"]

                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    char_embedding = self.char_encoder(reference, lengths=ref_lengths)
                    style_vector = self.style_encoder(style_strokes, lengths=style_lengths)

                    x = stroke_deltas_norm[:, :-1]
                    target_xy = stroke_deltas_norm[:, 1:]
                    target_eos = eos[:, 1:]

                    output = self.generator(
                        x,
                        style_vector,
                        char_embedding=char_embedding,
                        stroke_index=stroke_indices,
                    )

                    min_len = min(output["pi"].shape[1], target_xy.shape[1])
                    trimmed_output = {k: v[:, :min_len] for k, v in output.items()}
                    trimmed_xy = target_xy[:, :min_len]
                    trimmed_eos = target_eos[:, :min_len]

                    stroke_loss, eos_loss = mdn_loss(trimmed_output, trimmed_xy, trimmed_eos)
                    loss_char_var = embedding_variance_loss(char_embedding)
                    loss_style_var = embedding_variance_loss(style_vector)
                    loss = stroke_loss + 1.0 * eos_loss + 1.0 * (loss_char_var + loss_style_var)

                self.optimizer.zero_grad()
                all_params = [p for group in self.optimizer.param_groups for p in group["params"]]
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
                        end="",
                        flush=True,
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


@dataclass
class DeformationConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    grad_clip_norm: float = 5.0
    style_dim: int = 128
    hidden_dim: int = 256
    num_points: int = 32


class CASIADeformationDataset(Dataset):
    """CASIA + KanjiVG pairs for deformation learning. Each stroke is a sample."""

    def __init__(
        self,
        pot_dir: Path,
        ref_dir: Path,
        target_size: float = 10.0,
        max_samples: int = 0,
        num_points: int = 32,
        use_aligner: bool = False,
    ) -> None:
        self.num_points = num_points
        self.target_size = target_size
        parser = CASIAParser()

        if use_aligner:
            from src.model.stroke_aligner import StrokeAligner

            aligner = StrokeAligner(num_points=num_points)

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
        print(f"[V3] Loading .pot files from {pot_dir}...")

        self.char_samples: list[tuple[str, list[np.ndarray], list[np.ndarray], Path]] = []
        char_set: set[str] = set()
        total_stroke_samples = 0

        for pot_file in pot_files:
            casia_samples = parser.parse_pot_file(pot_file)
            for sample in casia_samples:
                ch = sample.character
                if ch not in ref_files:
                    continue
                normalized = parser.normalize(sample.strokes, target_size=target_size)
                if not normalized:
                    continue

                ref_path = ref_files[ch]
                ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
                kanjivg_strokes_np = [
                    np.array([[pt["x"], pt["y"]] for pt in stroke], dtype=np.float32)
                    for stroke in ref_data["strokes"]
                ]

                n_common = min(len(normalized), len(kanjivg_strokes_np))
                valid_count = 0
                for si in range(n_common):
                    if len(normalized[si]) >= 2 and len(kanjivg_strokes_np[si]) >= 2:
                        valid_count += 1
                if valid_count == 0:
                    continue

                self.char_samples.append((ch, normalized, kanjivg_strokes_np, ref_path))
                char_set.add(ch)
                total_stroke_samples += valid_count

                if 0 < max_samples <= total_stroke_samples:
                    break
            if 0 < max_samples <= total_stroke_samples:
                break

        self.char_to_indices: dict[str, list[int]] = {}
        for i, (ch, _, _, _) in enumerate(self.char_samples):
            self.char_to_indices.setdefault(ch, []).append(i)

        self.samples: list[tuple[int, int, int, int]] = []
        self.sample_reversed: list[bool] = []
        for char_idx, (ch, casia_strokes, kvg_strokes, _) in enumerate(self.char_samples):
            if use_aligner:
                casia_valid = [(i, s) for i, s in enumerate(casia_strokes) if len(s) >= 2]
                kvg_valid = [(i, s) for i, s in enumerate(kvg_strokes) if len(s) >= 2]
                if casia_valid and kvg_valid:
                    casia_arrs = [s for _, s in casia_valid]
                    kvg_arrs = [s for _, s in kvg_valid]
                    # Normalize CASIA to KanjiVG scale for alignment
                    all_c = np.concatenate(casia_arrs)
                    all_k = np.concatenate(kvg_arrs)
                    c_min, c_range = (
                        all_c.min(axis=0),
                        (all_c.max(axis=0) - all_c.min(axis=0)).max(),
                    )
                    k_min, k_range = (
                        all_k.min(axis=0),
                        (all_k.max(axis=0) - all_k.min(axis=0)).max(),
                    )
                    if c_range > 0:
                        sc = k_range / c_range
                        casia_norm = [(a - c_min) * sc + k_min for a in casia_arrs]
                    else:
                        casia_norm = casia_arrs
                    aresult = aligner.align(casia_norm, kvg_arrs)
                    for u_i, r_i, rev in zip(
                        aresult.user_indices, aresult.ref_indices, aresult.reversed_flags
                    ):
                        c_orig = casia_valid[u_i][0]
                        k_orig = kvg_valid[r_i][0]
                        self.samples.append((char_idx, c_orig, k_orig, len(casia_strokes)))
                        self.sample_reversed.append(rev)
            else:
                n_common = min(len(casia_strokes), len(kvg_strokes))
                for stroke_idx in range(n_common):
                    if len(casia_strokes[stroke_idx]) >= 2 and len(kvg_strokes[stroke_idx]) >= 2:
                        self.samples.append((char_idx, stroke_idx, stroke_idx, len(casia_strokes)))
                        self.sample_reversed.append(False)

        print(
            f"[V3] Loaded {len(self.samples)} stroke samples "
            f"({len(self.char_samples)} characters, {len(char_set)} unique) "
            f"from {n_files} .pot files"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        char_idx, casia_stroke_idx, kvg_stroke_idx, num_strokes = self.samples[idx]
        character, casia_strokes, kvg_strokes, _ = self.char_samples[char_idx]

        hand = casia_strokes[casia_stroke_idx]
        if self.sample_reversed[idx]:
            hand = hand[::-1].copy()

        ref_resampled = resample_stroke(kvg_strokes[kvg_stroke_idx], self.num_points)
        hand_resampled = resample_stroke(hand, self.num_points)

        # Style: use a different writer's sample of same character if available
        style_tensor = strokes_to_deltas_from_arrays(casia_strokes)
        same_char = self.char_to_indices[character]
        candidates = [i for i in same_char if i != char_idx]
        if candidates:
            style_char_idx = random.choice(candidates)
            _, style_strokes_raw, _, _ = self.char_samples[style_char_idx]
            style_tensor = strokes_to_deltas_from_arrays(style_strokes_raw)

        return {
            "reference_points": torch.tensor(ref_resampled, dtype=torch.float32),
            "target_points": torch.tensor(hand_resampled, dtype=torch.float32),
            "stroke_index": kvg_stroke_idx,
            "style_strokes": style_tensor,
            "character": character,
        }


def collate_deformation(batch: list[dict]) -> dict:
    """Collate function for CASIADeformationDataset."""
    reference_points = torch.stack([item["reference_points"] for item in batch])
    target_points = torch.stack([item["target_points"] for item in batch])
    stroke_indices = torch.tensor([item["stroke_index"] for item in batch])
    style_strokes = [item["style_strokes"] for item in batch]
    style_lengths = torch.tensor([s.shape[0] for s in style_strokes])
    padded_style = pad_sequence(style_strokes, batch_first=True, padding_value=0.0)
    return {
        "reference_points": reference_points,
        "target_points": target_points,
        "stroke_indices": stroke_indices,
        "style_strokes": padded_style,
        "style_lengths": style_lengths,
        "characters": [item["character"] for item in batch],
    }


class DeformationPretrainer:
    """AffineStrokeDeformer + StyleEncoder の事前学習。"""

    def __init__(
        self,
        config: DeformationConfig,
        ref_dir: Path,
        output_dir: Path,
        device: str | None = None,
        pot_dir: Path | None = None,
        max_samples: int = 0,
        num_workers: int = 0,
        amp: bool = False,
        norm_sample_size: int = 5000,
    ) -> None:
        from src.model.stroke_deformer import AffineStrokeDeformer

        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _detect_device(device)

        if pot_dir is None:
            raise ValueError("pot_dir is required for DeformationPretrainer")

        self.dataset = CASIADeformationDataset(
            pot_dir,
            ref_dir,
            max_samples=max_samples,
            num_points=config.num_points,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_deformation,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        indices = list(range(len(self.dataset)))
        if len(indices) > norm_sample_size:
            indices = random.sample(indices, norm_sample_size)
        style_tensors = [self.dataset[i]["style_strokes"] for i in indices]
        self.norm_stats = compute_normalization_stats(style_tensors)

        self.style_encoder = StyleEncoder(style_dim=config.style_dim).to(self.device)
        self.deformer = AffineStrokeDeformer(
            style_dim=config.style_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.deformer.parameters(), "lr": config.learning_rate},
                {"params": self.style_encoder.parameters(), "lr": config.learning_rate * 3},
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,
            gamma=0.5,
        )

        self.amp = amp
        if amp:
            self.scaler = torch.amp.GradScaler(self.device.type)
        else:
            self.scaler = None

    def train(self) -> dict:
        from src.model.stroke_deformer import affine_deformation_loss
        from src.model.stroke_model import embedding_variance_loss

        print(f"[V3] Device: {self.device}" + (" (AMP)" if self.amp else ""))
        history: dict[str, list[float]] = {"losses": []}
        total_batches = len(self.dataloader)

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(self.dataloader):
                ref_points = batch["reference_points"].to(self.device)
                target_points = batch["target_points"].to(self.device)
                stroke_indices = batch["stroke_indices"].to(self.device)

                style_strokes = batch["style_strokes"].to(self.device)
                style_strokes = normalize_deltas(style_strokes, self.norm_stats)
                style_lengths = batch["style_lengths"]

                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    style = self.style_encoder(style_strokes, lengths=style_lengths)
                    transformed, _params = self.deformer(ref_points, style, stroke_indices)

                    loss_deform = affine_deformation_loss(transformed, target_points)
                    loss_style_var = embedding_variance_loss(style)
                    loss = loss_deform + 0.1 * loss_style_var

                self.optimizer.zero_grad()
                all_params = [p for group in self.optimizer.param_groups for p in group["params"]]
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
                        f"\r  [V3] Epoch {epoch + 1}/{self.config.epochs} "
                        f"[{batch_idx + 1}/{total_batches}] "
                        f"loss: {loss.item():.4f}",
                        end="",
                        flush=True,
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            history["losses"].append(avg_loss)
            self.scheduler.step()
            print(f"\r  [V3] Epoch {epoch + 1}/{self.config.epochs} — avg loss: {avg_loss:.4f}")

        cpu = torch.device("cpu")
        config_dict = asdict(self.config)
        config_dict["deformer_type"] = "affine"
        torch.save(
            {
                "deformer_state_dict": {
                    k: v.to(cpu) for k, v in self.deformer.state_dict().items()
                },
                "style_encoder_state_dict": {
                    k: v.to(cpu) for k, v in self.style_encoder.state_dict().items()
                },
                "config": config_dict,
                "norm_stats": self.norm_stats,
                "version": 3,
            },
            self.output_dir / "pretrain_checkpoint.pt",
        )

        return history
