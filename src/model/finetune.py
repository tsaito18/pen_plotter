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
from src.model.data_utils import compute_normalization_stats
from src.model.pretrain import _detect_device
from src.model.stroke_model import StrokeGenerator, mdn_loss
from src.model.style_encoder import StyleEncoder

OFFSET_CLAMP = 1.0
SMOOTHING_KERNEL_SIZE = 15


def _scan_char_pairs(user_dir: Path, ref_dir: Path) -> list[tuple[str, Path, Path]]:
    """Scan user_dir and ref_dir, return matched (char, user_json, ref_json) triples."""
    user_dir = Path(user_dir)
    ref_dir = Path(ref_dir)

    ref_chars: dict[str, Path] = {}
    if ref_dir.is_dir():
        for char_dir in ref_dir.iterdir():
            if char_dir.is_dir():
                ref_files = list(char_dir.glob("*.json"))
                if ref_files:
                    ref_chars[char_dir.name] = ref_files[0]

    pairs: list[tuple[str, Path, Path]] = []
    if user_dir.is_dir():
        for char_dir in sorted(user_dir.iterdir()):
            if char_dir.is_dir() and char_dir.name in ref_chars:
                for f in sorted(char_dir.glob("*.json")):
                    pairs.append((char_dir.name, f, ref_chars[char_dir.name]))
    return pairs


def _state_dict_cpu(module: torch.nn.Module) -> dict:
    """Move all state_dict tensors to CPU for checkpoint saving."""
    cpu = torch.device("cpu")
    return {k: v.to(cpu) for k, v in module.state_dict().items()}


def augment_style_strokes(style_tensor: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Create augmented view of style strokes for contrastive pairs.

    Applies point jitter and small rotation to delta sequences.
    Input: (seq_len, 3) delta strokes [dx, dy, pen_state].
    Output: (seq_len, 3) augmented delta strokes.
    """
    result = style_tensor.clone()
    seq_len = result.shape[0]
    if seq_len == 0:
        return result

    bbox = result[:, :2].abs().max().item() + 1e-6
    jitter = torch.from_numpy(
        rng.normal(0, 0.02 * bbox, size=(seq_len, 2)).astype(np.float32)
    )
    result[:, :2] += jitter

    angle = rng.normal(0, np.radians(5))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = result[:, 0].clone()
    dy = result[:, 1].clone()
    result[:, 0] = dx * cos_a - dy * sin_a
    result[:, 1] = dx * sin_a + dy * cos_a

    return result


def smooth_offsets(offsets: torch.Tensor, kernel_size: int = SMOOTHING_KERNEL_SIZE) -> torch.Tensor:
    """Apply 1D moving average to smooth per-point offsets.

    Args:
        offsets: (batch, N, 2)
        kernel_size: smoothing window size (odd number)
    """
    if offsets.shape[1] <= kernel_size:
        return offsets
    B, N, _ = offsets.shape
    x = offsets.permute(0, 2, 1).reshape(B * 2, 1, N)
    pad = kernel_size // 2
    x_padded = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
    smoothed = torch.nn.functional.conv1d(x_padded, kernel)
    return smoothed.reshape(B, 2, N).permute(0, 2, 1)


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
        self.char_samples = _scan_char_pairs(user_dir, ref_dir)

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


class BaseFinetuner:
    """Template base for all fine-tuning/training classes.

    Subclasses implement the hooks; the training loop structure is shared.
    """

    config: FinetuneConfig | UserTrainConfig
    dataloader: DataLoader
    optimizer: torch.optim.Optimizer

    def __init__(self, config: FinetuneConfig | UserTrainConfig, output_dir: Path, device: str | None = None) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _detect_device(device)

    def _log_start(self) -> None:
        print(f"Device: {self.device}")

    def _set_train_mode(self) -> None:
        raise NotImplementedError

    def _compute_batch_loss(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def _get_clip_params(self) -> list | torch.nn.Parameter:
        raise NotImplementedError

    def _pre_epoch(self, epoch: int) -> None:
        pass

    def _post_epoch(self, epoch: int, avg_loss: float) -> None:
        pass

    def _build_checkpoint(self) -> dict:
        raise NotImplementedError

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "finetuned.pt"

    def train(self) -> dict:
        self._log_start()
        history: dict[str, list[float]] = {"losses": []}
        self._set_train_mode()

        for epoch in range(self.config.epochs):
            self._pre_epoch(epoch)
            epoch_loss = 0.0
            n_batches = 0

            for batch in self.dataloader:
                loss = self._compute_batch_loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._get_clip_params(), self.config.grad_clip_norm
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["losses"].append(avg_loss)
            self._post_epoch(epoch, avg_loss)

        torch.save(self._build_checkpoint(), self._checkpoint_path())
        return history


class Finetuner(BaseFinetuner):
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
        super().__init__(config, output_dir, device)

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

    def _set_train_mode(self) -> None:
        self.generator.train()
        self.char_encoder.train()
        self.style_encoder.train()

    def _get_clip_params(self):
        return self.style_encoder.parameters()

    def _compute_batch_loss(self, batch: dict) -> torch.Tensor:
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
            x,
            style,
            char_embedding=char_emb,
            stroke_index=stroke_indices,
        )

        min_len = min(output["pi"].shape[1], target_xy.shape[1])
        trimmed_output = {k: v[:, :min_len] for k, v in output.items()}
        trimmed_xy = target_xy[:, :min_len]
        trimmed_eos = target_eos[:, :min_len]

        stroke_loss, eos_loss = mdn_loss(trimmed_output, trimmed_xy, trimmed_eos)
        return stroke_loss + 0.5 * eos_loss

    def _build_checkpoint(self) -> dict:
        return {
            "generator_state_dict": _state_dict_cpu(self.generator),
            "style_encoder_state_dict": _state_dict_cpu(self.style_encoder),
            "char_encoder_state_dict": _state_dict_cpu(self.char_encoder),
            "config": self.ckpt_config,
            "norm_stats": self.norm_stats,
            "ref_norm_stats": self.ref_norm_stats,
        }


class FinetuneDeformationDataset(Dataset):
    """User samples + KanjiVG for deformation fine-tuning."""

    def __init__(
        self,
        user_dir: Path,
        ref_dir: Path,
        num_points: int = 32,
        augment: bool = False,
        use_aligner: bool = True,
    ) -> None:
        self.num_points = num_points
        self.augment = augment
        self.char_samples = _scan_char_pairs(user_dir, ref_dir)

        if use_aligner:
            from src.model.stroke_aligner import StrokeAligner

            aligner = StrokeAligner(num_points=num_points)

        self.samples: list[tuple[int, int, int, int]] = []
        self.sample_reversed: list[bool] = []
        for char_idx, (ch, user_path, ref_path) in enumerate(self.char_samples):
            user_data = json.loads(user_path.read_text(encoding="utf-8"))
            ref_data = json.loads(ref_path.read_text(encoding="utf-8"))

            if use_aligner:
                self._align_strokes(char_idx, user_data, ref_data, aligner)
            else:
                n_common = min(len(user_data["strokes"]), len(ref_data["strokes"]))
                for stroke_idx in range(n_common):
                    if (
                        len(user_data["strokes"][stroke_idx]) >= 2
                        and len(ref_data["strokes"][stroke_idx]) >= 2
                    ):
                        self.samples.append(
                            (char_idx, stroke_idx, stroke_idx, len(user_data["strokes"]))
                        )
                        self.sample_reversed.append(False)

    def _align_strokes(
        self,
        char_idx: int,
        user_data: dict,
        ref_data: dict,
        aligner: object,
    ) -> None:
        """Run aligner on one character's strokes and populate self.samples."""
        user_valid = [(i, s) for i, s in enumerate(user_data["strokes"]) if len(s) >= 2]
        ref_valid = [(i, s) for i, s in enumerate(ref_data["strokes"]) if len(s) >= 2]
        if not user_valid or not ref_valid:
            return

        user_arrays = [
            np.array([[pt["x"], pt["y"]] for pt in s], dtype=np.float32) for _, s in user_valid
        ]
        ref_arrays = [
            np.array([[pt["x"], pt["y"]] for pt in s], dtype=np.float32) for _, s in ref_valid
        ]

        # Flip user Y-axis to match KanjiVG (Y-up) convention
        all_u = np.concatenate(user_arrays)
        uy_min, uy_max = all_u[:, 1].min(), all_u[:, 1].max()
        for arr in user_arrays:
            arr[:, 1] = uy_min + uy_max - arr[:, 1]

        all_u = np.concatenate(user_arrays)
        all_r = np.concatenate(ref_arrays)
        u_min, u_range = all_u.min(axis=0), (all_u.max(axis=0) - all_u.min(axis=0)).max()
        r_min, r_range = all_r.min(axis=0), (all_r.max(axis=0) - all_r.min(axis=0)).max()
        if u_range > 0:
            scale = r_range / u_range
            user_norm = [(a - u_min) * scale + r_min for a in user_arrays]
        else:
            user_norm = user_arrays

        from src.model.stroke_aligner import StrokeAligner

        assert isinstance(aligner, StrokeAligner)
        result = aligner.align(user_norm, ref_arrays)
        for u_arr_idx, r_arr_idx, rev in zip(
            result.user_indices, result.ref_indices, result.reversed_flags
        ):
            u_orig = user_valid[u_arr_idx][0]
            r_orig = ref_valid[r_arr_idx][0]
            self.samples.append((char_idx, u_orig, r_orig, len(user_data["strokes"])))
            self.sample_reversed.append(rev)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        char_idx, user_stroke_idx, ref_stroke_idx, num_strokes = self.samples[idx]
        character, user_path, ref_path = self.char_samples[char_idx]

        user_data = json.loads(user_path.read_text(encoding="utf-8"))
        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))

        user_stroke_pts = np.array(
            [[pt["x"], pt["y"]] for pt in user_data["strokes"][user_stroke_idx]],
            dtype=np.float32,
        )
        ref_stroke_pts = np.array(
            [[pt["x"], pt["y"]] for pt in ref_data["strokes"][ref_stroke_idx]],
            dtype=np.float32,
        )

        if self.sample_reversed[idx]:
            user_stroke_pts = user_stroke_pts[::-1].copy()

        # Flip user Y-axis: iPad strokes are Y-down, KanjiVG reference is Y-up
        all_user_pts = np.concatenate(
            [np.array([[pt["x"], pt["y"]] for pt in s]) for s in user_data["strokes"]],
            axis=0,
        )
        y_min, y_max = all_user_pts[:, 1].min(), all_user_pts[:, 1].max()
        user_stroke_pts[:, 1] = y_min + y_max - user_stroke_pts[:, 1]
        all_user_pts[:, 1] = y_min + y_max - all_user_pts[:, 1]

        # Normalize user strokes to same [0, target_size] range as KanjiVG reference
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

        # Normalize all user strokes to KanjiVG scale (Y already flipped via all_user_pts)
        normalized_user_strokes = []
        for s in user_data["strokes"]:
            pts = np.array([[pt["x"], pt["y"]] for pt in s], dtype=np.float32)
            pts[:, 1] = y_min + y_max - pts[:, 1]  # Y-flip to match KanjiVG
            if u_range > 0:
                pts = (pts - u_min) * scale + all_ref_pts.min(axis=0)
            normalized_user_strokes.append(pts)

        # Apply augmentation: same transform to all strokes of this character
        if self.augment:
            angle = np.random.uniform(-5, 5) * np.pi / 180
            scale_aug = np.random.uniform(0.9, 1.1)
            ref_center = all_ref_pts.mean(axis=0)
            bbox_size = (all_ref_pts.max(axis=0) - all_ref_pts.min(axis=0)).max()
            tx = np.random.uniform(-0.05, 0.05) * bbox_size
            ty = np.random.uniform(-0.05, 0.05) * bbox_size
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

            user_stroke_pts = (
                (user_stroke_pts - ref_center) * scale_aug @ rot + ref_center + np.array([tx, ty])
            )
            for i in range(len(normalized_user_strokes)):
                normalized_user_strokes[i] = (
                    (normalized_user_strokes[i] - ref_center) * scale_aug @ rot
                    + ref_center
                    + np.array([tx, ty])
                )

        ref_resampled = resample_stroke(ref_stroke_pts, self.num_points)
        hand_resampled = resample_stroke(user_stroke_pts, self.num_points)

        # Build style tensor from normalized (and possibly augmented) user strokes
        normalized_user_strokes_dicts = [
            [{"x": float(p[0]), "y": float(p[1])} for p in pts] for pts in normalized_user_strokes
        ]
        style_tensor = strokes_to_deltas(normalized_user_strokes_dicts)

        return {
            "reference_points": torch.tensor(ref_resampled, dtype=torch.float32),
            "target_points": torch.tensor(hand_resampled, dtype=torch.float32),
            "stroke_index": ref_stroke_idx,
            "style_strokes": style_tensor,
            "character": character,
        }


def collate_deformation_finetune(batch: list[dict]) -> dict:
    """Collate for FinetuneDeformationDataset.

    Each sample produces 2 augmented style views for contrastive learning.
    The batch is doubled: original items + augmented copies share the same labels.
    """
    rng = np.random.default_rng()

    reference_points = torch.stack([item["reference_points"] for item in batch])
    target_points = torch.stack([item["target_points"] for item in batch])
    stroke_indices = torch.tensor([item["stroke_index"] for item in batch])

    # Duplicate: original + augmented view (both get same character label)
    style_view1 = [item["style_strokes"] for item in batch]
    style_view2 = [augment_style_strokes(s.clone(), rng) for s in style_view1]
    all_styles = style_view1 + style_view2
    style_lengths = torch.tensor([s.shape[0] for s in all_styles])
    padded_style = pad_sequence(all_styles, batch_first=True, padding_value=0.0)

    # Duplicate reference/target/stroke_indices for augmented view
    reference_points = reference_points.repeat(2, 1, 1)
    target_points = target_points.repeat(2, 1, 1)
    stroke_indices = stroke_indices.repeat(2)

    chars = [item["character"] for item in batch]
    doubled_chars = chars + chars
    unique_chars = sorted(set(chars))
    char_to_idx = {c: i for i, c in enumerate(unique_chars)}
    character_labels = torch.tensor(
        [char_to_idx[c] for c in doubled_chars], dtype=torch.long
    )

    return {
        "reference_points": reference_points,
        "target_points": target_points,
        "stroke_indices": stroke_indices,
        "style_strokes": padded_style,
        "style_lengths": style_lengths,
        "characters": doubled_chars,
        "character_labels": character_labels,
    }


class DeformationFinetuner(BaseFinetuner):
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
        use_aligner: bool = False,
    ) -> None:
        from src.model.stroke_deformer import AffineStrokeDeformer, StrokeDeformer

        super().__init__(config, output_dir, device)

        checkpoint = torch.load(pretrain_checkpoint, weights_only=False, map_location="cpu")
        self.ckpt_config = checkpoint["config"]
        self.norm_stats = checkpoint.get("norm_stats", None)

        cfg = self.ckpt_config
        num_points = cfg.get("num_points", 32)

        self.deformer_type = cfg.get("deformer_type", "offset")
        if self.deformer_type == "affine":
            self.deformer = AffineStrokeDeformer(
                style_dim=cfg.get("style_dim", 128),
                hidden_dim=cfg.get("hidden_dim", 64),
                dropout=cfg.get("dropout", 0.0),
            )
        else:
            self.deformer = StrokeDeformer(
                style_dim=cfg.get("style_dim", 128),
                hidden_dim=cfg.get("hidden_dim", 256),
                dropout=cfg.get("dropout", 0.0),
            )
        self.deformer.load_state_dict(checkpoint["deformer_state_dict"])

        self.style_encoder = StyleEncoder(style_dim=cfg.get("style_dim", 128))
        self.style_encoder.load_state_dict(checkpoint["style_encoder_state_dict"])

        self.deformer.to(self.device)
        self.style_encoder.to(self.device)

        for p in self.deformer.parameters():
            p.requires_grad = False

        self.dataset = FinetuneDeformationDataset(
            user_data_dir, ref_dir, num_points=num_points, use_aligner=use_aligner
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_deformation_finetune,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        self.optimizer = torch.optim.Adam(self.style_encoder.parameters(), lr=config.learning_rate)

    def _log_start(self) -> None:
        print(f"[V3] Device: {self.device}")

    def _set_train_mode(self) -> None:
        self.deformer.eval()
        self.style_encoder.train()

    def _get_clip_params(self):
        return self.style_encoder.parameters()

    def _compute_batch_loss(self, batch: dict) -> torch.Tensor:
        from src.model.stroke_deformer import affine_deformation_loss, deformation_loss

        ref_points = batch["reference_points"].to(self.device)
        target_points = batch["target_points"].to(self.device)
        stroke_indices = batch["stroke_indices"].to(self.device)

        style_strokes = batch["style_strokes"].to(self.device)
        if self.norm_stats is not None:
            style_strokes = normalize_deltas(style_strokes, self.norm_stats)
        style_lengths = batch["style_lengths"]

        style = self.style_encoder(style_strokes, lengths=style_lengths)

        if self.deformer_type == "affine":
            transformed, _params = self.deformer(ref_points, style, stroke_indices)
            return affine_deformation_loss(transformed, target_points)
        else:
            predicted = self.deformer(ref_points, style, stroke_indices)
            predicted = smooth_offsets(predicted)
            predicted = predicted.clamp(-OFFSET_CLAMP, OFFSET_CLAMP)
            target_offsets = target_points - ref_points
            return deformation_loss(predicted, target_offsets)

    def _build_checkpoint(self) -> dict:
        return {
            "deformer_state_dict": _state_dict_cpu(self.deformer),
            "style_encoder_state_dict": _state_dict_cpu(self.style_encoder),
            "config": self.ckpt_config,
            "norm_stats": self.norm_stats,
            "version": 3,
        }


@dataclass
class UserTrainConfig:
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 5e-4
    grad_clip_norm: float = 5.0
    style_dim: int = 128
    hidden_dim: int = 64
    num_points: int = 32
    dropout: float = 0.2
    weight_decay: float = 0.01
    deformer_type: str = "offset"
    contrastive_weight: float = 0.1
    contrastive_warmup_frac: float = 0.1
    contrastive_temperature: float = 0.07
    d_model: int = 64
    nhead: int = 4
    num_self_attn_layers: int = 2
    ff_dim: int = 128


class UserDeformationTrainer(BaseFinetuner):
    """Train StrokeDeformer + StyleEncoder directly on user data (no CASIA)."""

    def __init__(
        self,
        config: UserTrainConfig,
        user_data_dir: Path,
        ref_dir: Path,
        output_dir: Path,
        device: str | None = None,
        num_workers: int = 0,
        use_aligner: bool = False,
    ) -> None:
        from src.model.stroke_deformer import StrokeDeformer

        super().__init__(config, output_dir, device)

        self.dataset = FinetuneDeformationDataset(
            user_data_dir,
            ref_dir,
            num_points=config.num_points,
            augment=True,
            use_aligner=use_aligner,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_deformation_finetune,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        style_tensors = [self.dataset[i]["style_strokes"] for i in range(len(self.dataset))]
        self.norm_stats = compute_normalization_stats(style_tensors) if style_tensors else None

        self.style_encoder = StyleEncoder(style_dim=config.style_dim)
        self.style_encoder.enable_projection_head()
        self.style_encoder.to(self.device)

        self.deformer_type = config.deformer_type
        if config.deformer_type == "transformer":
            from src.model.stroke_deformer import TransformerDeformer

            self.deformer = TransformerDeformer(
                style_dim=config.style_dim,
                d_model=config.d_model,
                nhead=config.nhead,
                num_self_attn_layers=config.num_self_attn_layers,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
            ).to(self.device)
        else:
            self.deformer = StrokeDeformer(
                style_dim=config.style_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            ).to(self.device)

        self._current_epoch = 0

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.deformer.parameters(), "lr": config.learning_rate},
                {"params": self.style_encoder.parameters(), "lr": config.learning_rate * 3},
            ],
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )

    def _log_start(self) -> None:
        print(f"[V3-user] Device: {self.device}")
        print(f"[V3-user] Dataset: {len(self.dataset)} stroke pairs")

    def _set_train_mode(self) -> None:
        self.deformer.train()
        self.style_encoder.train()

    def _get_clip_params(self):
        return [p for group in self.optimizer.param_groups for p in group["params"]]

    def _compute_batch_loss(self, batch: dict) -> torch.Tensor:
        from src.model.stroke_deformer import deformation_loss, smoothness_loss
        from src.model.style_encoder import supervised_contrastive_loss

        ref_points = batch["reference_points"].to(self.device)
        target_points = batch["target_points"].to(self.device)
        stroke_indices = batch["stroke_indices"].to(self.device)
        target_offsets = target_points - ref_points

        style_strokes = batch["style_strokes"].to(self.device)
        if self.norm_stats is not None:
            style_strokes = normalize_deltas(style_strokes, self.norm_stats)
        style_lengths = batch["style_lengths"]

        style, z_proj = self.style_encoder(
            style_strokes, lengths=style_lengths, return_projection=True
        )
        predicted = self.deformer(ref_points, style, stroke_indices)
        if self.deformer_type != "transformer":
            predicted = smooth_offsets(predicted)
        predicted = predicted.clamp(-OFFSET_CLAMP, OFFSET_CLAMP)

        loss_deform = deformation_loss(predicted, target_offsets)
        loss_smooth = (
            smoothness_loss(predicted)
            if self.deformer_type != "transformer"
            else torch.tensor(0.0, device=self.device)
        )

        beta = self._get_contrastive_beta()
        if beta > 0 and z_proj is not None and "character_labels" in batch:
            labels = batch["character_labels"].to(self.device)
            loss_con = supervised_contrastive_loss(
                z_proj, labels, self.config.contrastive_temperature
            )
        else:
            loss_con = torch.tensor(0.0, device=self.device)

        return loss_deform + 0.1 * loss_smooth + beta * loss_con

    def _get_contrastive_beta(self) -> float:
        warmup_epochs = int(self.config.epochs * self.config.contrastive_warmup_frac)
        if self._current_epoch < warmup_epochs:
            return self.config.contrastive_weight * (
                self._current_epoch / max(warmup_epochs, 1)
            )
        return self.config.contrastive_weight

    def _pre_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    def _post_epoch(self, epoch: int, avg_loss: float) -> None:
        self.scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [V3-user] Epoch {epoch + 1}/{self.config.epochs} — loss: {avg_loss:.4f}")

    def _build_checkpoint(self) -> dict:
        return {
            "deformer_state_dict": _state_dict_cpu(self.deformer),
            "style_encoder_state_dict": {
                k: v
                for k, v in _state_dict_cpu(self.style_encoder).items()
                if not k.startswith("projection_head.")
            },
            "config": {
                "style_dim": self.config.style_dim,
                "hidden_dim": self.config.hidden_dim,
                "num_points": self.config.num_points,
                "dropout": self.config.dropout,
                "deformer_type": self.config.deformer_type,
                "d_model": self.config.d_model,
                "nhead": self.config.nhead,
                "num_self_attn_layers": self.config.num_self_attn_layers,
                "ff_dim": self.config.ff_dim,
            },
            "norm_stats": self.norm_stats,
            "version": 3,
        }

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "pretrain_checkpoint.pt"
