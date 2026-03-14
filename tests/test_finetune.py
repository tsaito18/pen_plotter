"""Fine-tuning パイプラインのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.model.char_encoder import CharEncoder
from src.model.finetune import FinetuneConfig, FinetuneDataset, Finetuner
from src.model.stroke_model import StrokeGenerator
from src.model.style_encoder import StyleEncoder


def _make_stroke_json(char: str, n_points: int = 15) -> str:
    stroke = [
        {"x": float(j), "y": float(j) * 0.5, "pressure": 1.0, "timestamp": float(j * 10)}
        for j in range(n_points)
    ]
    return json.dumps({"character": char, "strokes": [stroke], "metadata": {}})


def _make_data_dir(base: Path, chars: list[str], n_samples: int = 3) -> Path:
    for ch in chars:
        d = base / ch
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_samples):
            (d / f"{ch}_{i}.json").write_text(_make_stroke_json(ch), encoding="utf-8")
    return base


def _make_ref_dir(base: Path, chars: list[str]) -> Path:
    """参照ストローク用ディレクトリ（2次元座標のみ）。"""
    for ch in chars:
        d = base / ch
        d.mkdir(parents=True, exist_ok=True)
        stroke = [{"x": float(j) * 0.3, "y": float(j) * 0.7} for j in range(10)]
        data = {"character": ch, "strokes": [stroke], "metadata": {}}
        (d / f"{ch}_ref.json").write_text(json.dumps(data), encoding="utf-8")
    return base


def _make_pretrain_checkpoint(path: Path, config: dict | None = None) -> Path:
    """テスト用の事前学習済みチェックポイントを作成。"""
    cfg = config or {
        "style_dim": 128,
        "char_dim": 128,
        "hidden_dim": 128,
        "num_mixtures": 5,
    }
    checkpoint = {
        "generator_state_dict": StrokeGenerator(
            char_dim=cfg["char_dim"],
            hidden_dim=cfg["hidden_dim"],
            style_dim=cfg["style_dim"],
            num_mixtures=cfg["num_mixtures"],
        ).state_dict(),
        "style_encoder_state_dict": StyleEncoder(style_dim=cfg["style_dim"]).state_dict(),
        "char_encoder_state_dict": CharEncoder(char_dim=cfg["char_dim"]).state_dict(),
        "config": cfg,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


class TestFinetuneConfig:
    def test_defaults(self):
        cfg = FinetuneConfig()
        assert cfg.epochs == 20
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 5e-4
        assert cfg.grad_clip_norm == 5.0

    def test_custom(self):
        cfg = FinetuneConfig(epochs=5, batch_size=4, learning_rate=1e-4)
        assert cfg.epochs == 5
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 1e-4


class TestFinetuneDataset:
    def test_dataset_finds_paired_chars(self, tmp_path):
        chars = ["あ", "い", "う"]
        user_dir = _make_data_dir(tmp_path / "user", chars)
        ref_dir = _make_ref_dir(tmp_path / "ref", ["あ", "い"])
        ds = FinetuneDataset(user_dir, ref_dir)
        assert len(ds) > 0
        # "う" has no ref, so only "あ" and "い" pairs
        for i in range(len(ds)):
            item = ds[i]
            assert "strokes" in item
            assert "ref_strokes" in item
            assert "character" in item
            assert item["character"] in ["あ", "い"]

    def test_dataset_item_shapes(self, tmp_path):
        chars = ["あ"]
        user_dir = _make_data_dir(tmp_path / "user", chars, n_samples=1)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ds = FinetuneDataset(user_dir, ref_dir)
        item = ds[0]
        assert item["strokes"].ndim == 2
        assert item["strokes"].shape[1] == 3  # x, y, pressure
        assert item["ref_strokes"].ndim == 2
        assert item["ref_strokes"].shape[1] == 2  # x, y


class TestFinetuner:
    @pytest.fixture
    def setup(self, tmp_path):
        chars = ["あ", "い", "う"]
        user_dir = _make_data_dir(tmp_path / "user", chars)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ckpt_path = _make_pretrain_checkpoint(tmp_path / "ckpt" / "checkpoint.pt")
        output_dir = tmp_path / "output"
        return {
            "user_dir": user_dir,
            "ref_dir": ref_dir,
            "ckpt_path": ckpt_path,
            "output_dir": output_dir,
        }

    def test_finetuner_creation(self, setup):
        cfg = FinetuneConfig(epochs=1)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        assert finetuner is not None

    def test_finetuner_freezes_generator(self, setup):
        cfg = FinetuneConfig(epochs=1)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        for p in finetuner.generator.parameters():
            assert not p.requires_grad

    def test_finetuner_freezes_char_encoder(self, setup):
        cfg = FinetuneConfig(epochs=1)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        for p in finetuner.char_encoder.parameters():
            assert not p.requires_grad

    def test_finetuner_style_encoder_unfrozen(self, setup):
        cfg = FinetuneConfig(epochs=1)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        for p in finetuner.style_encoder.parameters():
            assert p.requires_grad

    @pytest.mark.slow
    def test_finetune_one_epoch(self, setup):
        cfg = FinetuneConfig(epochs=1, batch_size=4)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        history = finetuner.train()
        assert "losses" in history
        assert len(history["losses"]) == 1
        assert history["losses"][0] > 0
        assert (setup["output_dir"] / "finetuned.pt").exists()

    @pytest.mark.slow
    def test_finetune_checkpoint_compatible(self, setup):
        """出力チェックポイントが入力と同じキーを持つことを確認。"""
        cfg = FinetuneConfig(epochs=1, batch_size=4)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        finetuner.train()

        input_ckpt = torch.load(setup["ckpt_path"], weights_only=False)
        output_ckpt = torch.load(setup["output_dir"] / "finetuned.pt", weights_only=False)

        for key in [
            "generator_state_dict",
            "style_encoder_state_dict",
            "char_encoder_state_dict",
            "config",
        ]:
            assert key in output_ckpt, f"Missing key: {key}"

        # Generator weights should be unchanged (frozen)
        for k in input_ckpt["generator_state_dict"]:
            assert torch.equal(
                input_ckpt["generator_state_dict"][k],
                output_ckpt["generator_state_dict"][k],
            )
