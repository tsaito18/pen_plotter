"""収集 UI から起動する訓練ジョブ（バックグラウンドスレッドで 1 件ずつ実行）。"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.collector.profiles import resolve_training_dirs


class TrainingCancelled(RuntimeError):
    """ユーザーの中断要求（エポック開始時に投げて訓練ループを抜ける）。"""


@dataclass
class TrainingJobStatus:
    state: str = "idle"  # idle / running / succeeded / failed / cancelled
    kind: str | None = None
    epoch: int = 0
    total_epochs: int = 0
    loss: float | None = None
    checkpoint_path: str | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["logs"] = self.logs[-80:]
        return d


def _defaults(kind: str) -> tuple[int, int, float]:
    """``(epochs, batch_size, learning_rate)`` の既定値。"""
    return (20, 8, 5e-4) if kind == "finetune" else (80, 256, 1e-3)


class TrainingJobManager:
    def __init__(self, *, root_dir: Path, ref_dir: Path | None) -> None:
        self.root_dir = Path(root_dir)
        self.ref_dir = Path(ref_dir) if ref_dir else Path("data/strokes")
        self._lock = threading.Lock()
        self._status = TrainingJobStatus()
        self._thread: threading.Thread | None = None
        self._cancel_requested = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status.to_dict()

    def start(self, config: dict[str, Any], current_profile: str) -> dict[str, Any]:
        with self._lock:
            if self._status.state == "running":
                raise RuntimeError("training job is already running")
            kind = str(config.get("kind") or "user_train")
            self._status = TrainingJobStatus(
                state="running",
                kind=kind,
                total_epochs=int(config.get("epochs") or _defaults(kind)[0]),
                started_at=time.time(),
            )
            self._cancel_requested = False
            self._thread = threading.Thread(
                target=self._run, args=(dict(config), current_profile), daemon=True
            )
            self._thread.start()
            return self._status.to_dict()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._status.state == "running":
                self._cancel_requested = True
                self._append_log_locked("Cancel requested; current epoch will finish before stop.")
            return self._status.to_dict()

    def _append_log(self, message: str) -> None:
        with self._lock:
            self._append_log_locked(message)

    def _append_log_locked(self, message: str) -> None:
        self._status.logs.append(message)
        del self._status.logs[:-200]

    def _on_epoch(self, epoch: int, total_epochs: int, loss: float) -> None:
        with self._lock:
            self._status.epoch = epoch
            self._status.total_epochs = total_epochs
            self._status.loss = loss
            self._append_log_locked(f"Epoch {epoch}/{total_epochs}: loss={loss:.4f}")

    def _check_cancel(self, _epoch: int) -> None:
        if self._cancel_requested:
            raise TrainingCancelled("Training cancelled")

    def _run(self, config: dict[str, Any], current_profile: str) -> None:
        try:
            checkpoint_path = self._run_training(config, current_profile)
        except TrainingCancelled as exc:
            self._finish("cancelled", log=str(exc), error=str(exc))
        except Exception as exc:  # noqa: BLE001 — ジョブの失敗は状態として UI へ返す
            self._finish("failed", log=traceback.format_exc(limit=8), error=str(exc))
        else:
            self._finish(
                "succeeded",
                log=f"Checkpoint saved: {checkpoint_path}",
                checkpoint=str(checkpoint_path),
            )

    def _finish(
        self, state: str, *, log: str, error: str | None = None, checkpoint: str | None = None
    ) -> None:
        with self._lock:
            self._status.state = state
            self._status.error = error
            self._status.checkpoint_path = checkpoint
            self._status.finished_at = time.time()
            self._append_log_locked(log)

    def _run_training(self, config: dict[str, Any], current_profile: str) -> Path:
        from src.model.training import (
            DeformationFinetuner,
            FinetuneConfig,
            UserDeformationTrainer,
            UserTrainConfig,
        )

        kind = str(config.get("kind") or "user_train")
        dataset = config.get("dataset") or {"mode": "current", "profile": current_profile}
        if dataset.get("mode") == "current":
            dataset["profile"] = current_profile
        data_dirs = resolve_training_dirs(self.root_dir, dataset)

        default_epochs, default_batch, default_lr = _defaults(kind)
        epochs = int(config.get("epochs") or default_epochs)
        batch_size = int(config.get("batch_size") or default_batch)
        learning_rate = float(config.get("learning_rate") or default_lr)
        output_dir = Path(config.get("output_dir") or f"data/models/{kind}_{int(time.time())}")
        device = config.get("device") or None
        use_aligner = bool(config.get("use_aligner", True))

        self._append_log(f"Kind: {kind}")
        self._append_log(f"Dataset: {', '.join(str(p) for p in data_dirs)}")
        self._append_log(f"Output: {output_dir}")

        trainer: UserDeformationTrainer | DeformationFinetuner
        if kind == "finetune":
            trainer = DeformationFinetuner(
                FinetuneConfig(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate),
                pretrain_checkpoint=Path(
                    config.get("checkpoint") or "data/models/pretrain_checkpoint.pt"
                ),
                user_dirs=data_dirs,
                ref_dir=self.ref_dir,
                output_dir=output_dir,
                device=device,
                use_aligner=use_aligner,
            )
        else:
            trainer = UserDeformationTrainer(
                UserTrainConfig(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    deformer_type=str(config.get("deformer_type") or "twostage"),
                ),
                user_dirs=data_dirs,
                ref_dir=self.ref_dir,
                output_dir=output_dir,
                device=device,
                use_aligner=use_aligner,
            )
        trainer.on_epoch_start = self._check_cancel
        trainer.on_epoch_end = lambda epoch, loss: self._on_epoch(epoch + 1, epochs, loss)
        trainer.train()
        return trainer.checkpoint_path
