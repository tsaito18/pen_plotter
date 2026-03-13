import json
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")

from src.model.train import Trainer, TrainConfig


def _make_samples(tmp_path: Path, n_chars: int = 3, n_samples: int = 5):
    """テスト用のサンプルデータを作成"""
    chars = ["あ", "い", "う", "え", "お"][:n_chars]
    for ch in chars:
        for i in range(n_samples):
            stroke = [{"x": float(j), "y": float(j) * 0.5, "pressure": 1.0, "timestamp": float(j * 10)}
                      for j in range(15)]
            data = {"character": ch, "strokes": [stroke], "metadata": {}}
            d = tmp_path / ch
            d.mkdir(exist_ok=True)
            (d / f"{ch}_{i}.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.epochs > 0
        assert cfg.batch_size > 0
        assert cfg.learning_rate > 0
        assert cfg.grad_clip_norm == 5.0

    def test_custom(self):
        cfg = TrainConfig(epochs=5, batch_size=8)
        assert cfg.epochs == 5
        assert cfg.batch_size == 8


class TestTrainer:
    @pytest.fixture
    def data_dir(self, tmp_path):
        return _make_samples(tmp_path)

    def test_trainer_creation(self, data_dir, tmp_path):
        cfg = TrainConfig(epochs=1, batch_size=2)
        trainer = Trainer(cfg, data_dir=data_dir, output_dir=tmp_path / "output")
        assert trainer is not None

    @pytest.mark.slow
    def test_train_one_epoch(self, data_dir, tmp_path):
        cfg = TrainConfig(epochs=1, batch_size=4)
        trainer = Trainer(cfg, data_dir=data_dir, output_dir=tmp_path / "output")
        history = trainer.train()
        assert "losses" in history
        assert len(history["losses"]) == 1
        assert history["losses"][0] > 0

    @pytest.mark.slow
    def test_checkpoint_saved(self, data_dir, tmp_path):
        output_dir = tmp_path / "output"
        cfg = TrainConfig(epochs=1, batch_size=4)
        trainer = Trainer(cfg, data_dir=data_dir, output_dir=output_dir)
        trainer.train()
        checkpoints = list(output_dir.glob("*.pt"))
        assert len(checkpoints) >= 1
