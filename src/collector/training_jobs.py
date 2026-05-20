from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.collector.profiles import resolve_training_dirs


class TrainingCancelled(RuntimeError):
    pass


@dataclass
class TrainingJobStatus:
    state: str = "idle"
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
        return {
            "state": self.state,
            "kind": self.kind,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "loss": self.loss,
            "checkpoint_path": self.checkpoint_path,
            "error": self.error,
            "logs": self.logs[-80:],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


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
            epochs = int(config.get("epochs") or (20 if kind == "finetune" else 80))
            self._status = TrainingJobStatus(
                state="running",
                kind=kind,
                total_epochs=epochs,
                started_at=time.time(),
                logs=[],
            )
            self._cancel_requested = False
            self._thread = threading.Thread(
                target=self._run,
                args=(dict(config), current_profile),
                daemon=True,
            )
            self._thread.start()
            return self._status.to_dict()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._status.state != "running":
                return self._status.to_dict()
            self._cancel_requested = True
            self._append_log_locked("Cancel requested; current epoch will finish before stop.")
            return self._status.to_dict()

    def _append_log(self, message: str) -> None:
        with self._lock:
            self._append_log_locked(message)

    def _append_log_locked(self, message: str) -> None:
        self._status.logs.append(message)
        if len(self._status.logs) > 200:
            self._status.logs = self._status.logs[-200:]

    def _on_epoch(self, epoch: int, total_epochs: int, loss: float) -> None:
        with self._lock:
            self._status.epoch = epoch
            self._status.total_epochs = total_epochs
            self._status.loss = loss
            self._append_log_locked(f"Epoch {epoch}/{total_epochs}: loss={loss:.4f}")

    def _run(self, config: dict[str, Any], current_profile: str) -> None:
        try:
            checkpoint_path = self._run_training(config, current_profile)
            with self._lock:
                self._status.state = "succeeded"
                self._status.checkpoint_path = str(checkpoint_path)
                self._status.finished_at = time.time()
                self._append_log_locked(f"Checkpoint saved: {checkpoint_path}")
        except TrainingCancelled as exc:
            with self._lock:
                self._status.state = "cancelled"
                self._status.error = str(exc)
                self._status.finished_at = time.time()
                self._append_log_locked(str(exc))
        except RuntimeError as exc:
            if str(exc) == "Training cancelled":
                with self._lock:
                    self._status.state = "cancelled"
                    self._status.error = str(exc)
                    self._status.finished_at = time.time()
                    self._append_log_locked(str(exc))
                return
            with self._lock:
                self._status.state = "failed"
                self._status.error = str(exc)
                self._status.finished_at = time.time()
                self._append_log_locked(traceback.format_exc(limit=8))
        except Exception as exc:
            with self._lock:
                self._status.state = "failed"
                self._status.error = str(exc)
                self._status.finished_at = time.time()
                self._append_log_locked(traceback.format_exc(limit=8))

    def _run_training(self, config: dict[str, Any], current_profile: str) -> Path:
        kind = str(config.get("kind") or "user_train")
        dataset = config.get("dataset") or {"mode": "current", "profile": current_profile}
        if dataset.get("mode") == "current":
            dataset["profile"] = current_profile
        data_dirs = resolve_training_dirs(self.root_dir, dataset)
        data_arg: Path | list[Path] = data_dirs[0] if len(data_dirs) == 1 else data_dirs

        epochs = int(config.get("epochs") or (20 if kind == "finetune" else 80))
        batch_size = int(config.get("batch_size") or (8 if kind == "finetune" else 256))
        learning_rate = float(config.get("learning_rate") or (5e-4 if kind == "finetune" else 1e-3))
        output_dir = Path(config.get("output_dir") or f"data/models/{kind}_{int(time.time())}")
        device = config.get("device") or None
        use_aligner = bool(config.get("use_aligner", True))

        self._append_log(f"Kind: {kind}")
        self._append_log(f"Dataset: {', '.join(str(p) for p in data_dirs)}")
        self._append_log(f"Output: {output_dir}")

        if kind == "finetune":
            checkpoint = Path(config.get("checkpoint") or "data/models/pretrain_checkpoint.pt")
            return self._run_finetune(
                checkpoint=checkpoint,
                data_arg=data_arg,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                use_aligner=use_aligner,
            )
        if kind == "pretrain":
            pot_dir = Path(config.get("pot_dir") or "data/casia_raw/train")
            return self._run_pretrain(
                pot_dir=pot_dir,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
            )
        return self._run_user_train(
            data_arg=data_arg,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            use_aligner=use_aligner,
            deformer_type=str(config.get("deformer_type") or "twostage"),
        )

    def _attach_epoch_callback(self, trainer: object, epochs: int) -> None:
        original = getattr(trainer, "_post_epoch", None)
        original_pre = getattr(trainer, "_pre_epoch", None)

        def _pre_epoch(epoch: int) -> None:
            if self._cancel_requested:
                raise TrainingCancelled("Training cancelled")
            if original_pre is not None:
                original_pre(epoch)

        def _post_epoch(epoch: int, avg_loss: float) -> None:
            if original is not None:
                original(epoch, avg_loss)
            self._on_epoch(epoch + 1, epochs, float(avg_loss))

        setattr(trainer, "_pre_epoch", _pre_epoch)
        setattr(trainer, "_post_epoch", _post_epoch)

    def _run_user_train(
        self,
        *,
        data_arg: Path | list[Path],
        output_dir: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str | None,
        use_aligner: bool,
        deformer_type: str,
    ) -> Path:
        from src.model.finetune import UserDeformationTrainer, UserTrainConfig

        cfg = UserTrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            deformer_type=deformer_type,
        )
        trainer = UserDeformationTrainer(
            config=cfg,
            user_data_dir=data_arg,
            ref_dir=self.ref_dir,
            output_dir=output_dir,
            device=device,
            use_aligner=use_aligner,
        )
        self._attach_epoch_callback(trainer, epochs)
        trainer.train()
        return output_dir / "pretrain_checkpoint.pt"

    def _run_finetune(
        self,
        *,
        checkpoint: Path,
        data_arg: Path | list[Path],
        output_dir: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str | None,
        use_aligner: bool,
    ) -> Path:
        from src.model.finetune import DeformationFinetuner, FinetuneConfig, Finetuner
        import torch

        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        is_v3 = "deformer_state_dict" in checkpoint_data
        del checkpoint_data

        cfg = FinetuneConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        if is_v3:
            trainer = DeformationFinetuner(
                config=cfg,
                pretrain_checkpoint=checkpoint,
                user_data_dir=data_arg,
                ref_dir=self.ref_dir,
                output_dir=output_dir,
                device=device,
                use_aligner=use_aligner,
            )
        else:
            trainer = Finetuner(
                config=cfg,
                pretrain_checkpoint=checkpoint,
                user_data_dir=data_arg,
                ref_dir=self.ref_dir,
                output_dir=output_dir,
                device=device,
            )
        self._attach_epoch_callback(trainer, epochs)
        trainer.train()
        return output_dir / "finetuned.pt"

    def _run_pretrain(
        self,
        *,
        pot_dir: Path,
        output_dir: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str | None,
    ) -> Path:
        from src.model.pretrain import DeformationConfig, DeformationPretrainer

        cfg = DeformationConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        trainer = DeformationPretrainer(
            config=cfg,
            ref_dir=self.ref_dir,
            output_dir=output_dir,
            device=device,
            pot_dir=pot_dir,
        )
        setattr(
            trainer,
            "progress_callback",
            lambda epoch, total, loss: self._on_epoch(epoch, total, loss),
        )
        setattr(
            trainer,
            "cancel_callback",
            lambda: self._cancel_requested,
        )
        trainer.train()
        return output_dir / "pretrain_checkpoint.pt"
