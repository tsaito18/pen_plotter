"""V3 変形モデルの訓練。

- :class:`UserDeformationTrainer`: ユーザー筆跡だけで変形器＋StyleEncoder を
  スクラッチ訓練する（``pretrain_checkpoint.pt``）。StyleEncoder には SupCon
  対照学習を併用してスタイル空間を識別的にする。
- :class:`DeformationFinetuner`: 変形器を凍結し StyleEncoder だけを微調整する
  （``finetuned.pt``）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model.data import (
    DeformationDataset,
    collate_deformation,
    compute_normalization_stats,
    normalize_deltas,
)
from src.model.deformers import (
    affine_deformation_loss,
    build_deformer,
    deformation_loss,
    postprocess_offsets,
    smoothness_loss,
)
from src.model.device import detect_device
from src.model.style_encoder import StyleEncoder, supervised_contrastive_loss

CHECKPOINT_VERSION = 3


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

    def model_config(self) -> dict:
        """チェックポイントに保存するモデル構造の設定。"""
        keys = (
            "style_dim", "hidden_dim", "num_points", "dropout", "deformer_type",
            "d_model", "nhead", "num_self_attn_layers", "ff_dim",
        )  # fmt: skip
        d = asdict(self)
        return {k: d[k] for k in keys}


@dataclass
class FinetuneConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 5e-4
    grad_clip_norm: float = 5.0


def _state_dict_cpu(module: torch.nn.Module) -> dict:
    return {k: v.to("cpu") for k, v in module.state_dict().items()}


class _Trainer:
    """訓練ループの共通部分。サブクラスが損失・保存内容を実装する。

    ``on_epoch_start(epoch)`` / ``on_epoch_end(epoch, avg_loss)`` を設定すると
    進捗通知・中断に使える（コールバックから例外を投げれば訓練を中断できる）。
    """

    checkpoint_name = "checkpoint.pt"

    def __init__(
        self,
        epochs: int,
        grad_clip_norm: float,
        output_dir: Path,
        device: str | None,
    ) -> None:
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = detect_device(device)
        self.on_epoch_start: Callable[[int], None] | None = None
        self.on_epoch_end: Callable[[int, float], None] | None = None
        self.dataloader: DataLoader
        self.optimizer: torch.optim.Optimizer

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_name

    def train(self) -> list[float]:
        """全エポック訓練してチェックポイントを保存し、エポックごとの平均損失を返す。"""
        print(f"Device: {self.device}")
        losses: list[float] = []
        self._set_train_mode()
        for epoch in range(self.epochs):
            if self.on_epoch_start:
                self.on_epoch_start(epoch)
            self._epoch_start(epoch)
            total, n = 0.0, 0
            for batch in self.dataloader:
                loss = self._batch_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._trainable_params(), self.grad_clip_norm)
                self.optimizer.step()
                total += loss.item()
                n += 1
            avg = total / max(n, 1)
            losses.append(avg)
            self._epoch_end(epoch, avg)
            if self.on_epoch_end:
                self.on_epoch_end(epoch, avg)
        torch.save(self._checkpoint(), self.checkpoint_path)
        return losses

    def _style(self, batch: dict, norm_stats: dict | None) -> torch.Tensor:
        style = batch["style_strokes"].to(self.device)
        if norm_stats is not None:
            style = normalize_deltas(style, norm_stats)
        return style

    # --- サブクラスが実装 ---

    def _set_train_mode(self) -> None:
        raise NotImplementedError

    def _batch_loss(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError

    def _checkpoint(self) -> dict:
        raise NotImplementedError

    def _epoch_start(self, epoch: int) -> None:
        pass

    def _epoch_end(self, epoch: int, avg_loss: float) -> None:
        pass


def _make_loader(dataset: DeformationDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_deformation,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


class UserDeformationTrainer(_Trainer):
    """ユーザー筆跡だけで変形器＋StyleEncoder を訓練する。"""

    checkpoint_name = "pretrain_checkpoint.pt"

    def __init__(
        self,
        config: UserTrainConfig,
        user_dirs: list[Path],
        ref_dir: Path,
        output_dir: Path,
        device: str | None = None,
        num_workers: int = 0,
        use_aligner: bool = False,
    ) -> None:
        if config.deformer_type == "affine":
            raise ValueError("affine deformer is fine-tune only; use offset/transformer/twostage")
        super().__init__(config.epochs, config.grad_clip_norm, output_dir, device)
        self.config = config
        self.dataset = DeformationDataset(
            user_dirs, ref_dir, num_points=config.num_points, augment=True, use_aligner=use_aligner
        )
        self.dataloader = _make_loader(self.dataset, config.batch_size, num_workers)

        styles = [self.dataset[i]["style_strokes"] for i in range(len(self.dataset))]
        self.norm_stats = compute_normalization_stats(styles) if styles else None

        self.style_encoder = StyleEncoder(style_dim=config.style_dim)
        self.style_encoder.enable_projection_head()
        self.style_encoder.to(self.device)
        self.deformer = build_deformer(config.model_config()).to(self.device)

        self._epoch = 0
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.deformer.parameters(), "lr": config.learning_rate},
                {"params": self.style_encoder.parameters(), "lr": config.learning_rate * 3},
            ],
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        print(f"Dataset: {len(self.dataset)} stroke pairs")

    def _set_train_mode(self) -> None:
        self.deformer.train()
        self.style_encoder.train()

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for group in self.optimizer.param_groups for p in group["params"]]

    def contrastive_beta(self) -> float:
        """対照学習損失の重み（最初の warmup 割合のエポックで 0 から線形に上げる）。"""
        cfg = self.config
        warmup = int(cfg.epochs * cfg.contrastive_warmup_frac)
        if self._epoch < warmup:
            return cfg.contrastive_weight * (self._epoch / max(warmup, 1))
        return cfg.contrastive_weight

    def _batch_loss(self, batch: dict) -> torch.Tensor:
        ref = batch["reference_points"].to(self.device)
        target = batch["target_points"].to(self.device)
        style, z = self.style_encoder(
            self._style(batch, self.norm_stats),
            lengths=batch["style_lengths"],
            return_projection=True,
        )
        predicted = postprocess_offsets(
            self.deformer(ref, style, batch["stroke_indices"].to(self.device))
        )
        loss = deformation_loss(predicted, target - ref) + 0.1 * smoothness_loss(predicted)
        beta = self.contrastive_beta()
        if beta > 0 and z is not None:
            labels = batch["character_labels"].to(self.device)
            loss = loss + beta * supervised_contrastive_loss(
                z, labels, self.config.contrastive_temperature
            )
        return loss

    def _epoch_start(self, epoch: int) -> None:
        self._epoch = epoch

    def _epoch_end(self, epoch: int, avg_loss: float) -> None:
        self.scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{self.epochs} — loss: {avg_loss:.4f}")

    def _checkpoint(self) -> dict:
        encoder_state = {
            k: v
            for k, v in _state_dict_cpu(self.style_encoder).items()
            if not k.startswith("projection_head.")
        }
        return {
            "deformer_state_dict": _state_dict_cpu(self.deformer),
            "style_encoder_state_dict": encoder_state,
            "config": self.config.model_config(),
            "norm_stats": self.norm_stats,
            "version": CHECKPOINT_VERSION,
        }


class DeformationFinetuner(_Trainer):
    """変形器を凍結し、StyleEncoder だけを新しい筆跡へ微調整する。"""

    checkpoint_name = "finetuned.pt"

    def __init__(
        self,
        config: FinetuneConfig,
        pretrain_checkpoint: Path,
        user_dirs: list[Path],
        ref_dir: Path,
        output_dir: Path,
        device: str | None = None,
        num_workers: int = 0,
        use_aligner: bool = False,
    ) -> None:
        super().__init__(config.epochs, config.grad_clip_norm, output_dir, device)
        checkpoint = torch.load(pretrain_checkpoint, weights_only=False, map_location="cpu")
        if "deformer_state_dict" not in checkpoint:
            raise ValueError(f"not a V3 deformation checkpoint: {pretrain_checkpoint}")
        self.model_config = dict(checkpoint["config"])
        self.norm_stats = checkpoint.get("norm_stats")

        self.deformer = build_deformer(self.model_config)
        self.deformer.load_state_dict(checkpoint["deformer_state_dict"])
        self.deformer.to(self.device)
        for p in self.deformer.parameters():
            p.requires_grad = False
        self.style_encoder = StyleEncoder(style_dim=self.model_config.get("style_dim", 128))
        self.style_encoder.load_state_dict(checkpoint["style_encoder_state_dict"])
        self.style_encoder.to(self.device)

        self.dataset = DeformationDataset(
            user_dirs,
            ref_dir,
            num_points=self.model_config.get("num_points", 32),
            use_aligner=use_aligner,
        )
        self.dataloader = _make_loader(self.dataset, config.batch_size, num_workers)
        self.optimizer = torch.optim.Adam(self.style_encoder.parameters(), lr=config.learning_rate)

    def _set_train_mode(self) -> None:
        self.deformer.eval()
        self.style_encoder.train()

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        return list(self.style_encoder.parameters())

    def _batch_loss(self, batch: dict) -> torch.Tensor:
        ref = batch["reference_points"].to(self.device)
        target = batch["target_points"].to(self.device)
        style = self.style_encoder(
            self._style(batch, self.norm_stats), lengths=batch["style_lengths"]
        )
        indices = batch["stroke_indices"].to(self.device)
        if self.model_config.get("deformer_type") == "affine":
            transformed, _params = self.deformer(ref, style, indices)
            return affine_deformation_loss(transformed, target)
        predicted = postprocess_offsets(self.deformer(ref, style, indices))
        return deformation_loss(predicted, target - ref)

    def _checkpoint(self) -> dict:
        return {
            "deformer_state_dict": _state_dict_cpu(self.deformer),
            "style_encoder_state_dict": _state_dict_cpu(self.style_encoder),
            "config": self.model_config,
            "norm_stats": self.norm_stats,
            "version": CHECKPOINT_VERSION,
        }
