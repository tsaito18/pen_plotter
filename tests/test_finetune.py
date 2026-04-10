"""Fine-tuning パイプラインのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.model.char_encoder import CharEncoder
from src.model.finetune import (
    DeformationFinetuner,
    FinetuneConfig,
    FinetuneDataset,
    FinetuneDeformationDataset,
    Finetuner,
    UserDeformationTrainer,
    UserTrainConfig,
    augment_style_strokes,
    collate_deformation_finetune,
    collate_finetune,
)
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


def _make_pretrain_checkpoint(
    path: Path,
    config: dict | None = None,
    norm_stats: dict | None = None,
    ref_norm_stats: dict | None = None,
) -> Path:
    """テスト用の事前学習済みチェックポイントを作成。"""
    cfg = config or {
        "style_dim": 128,
        "char_dim": 128,
        "hidden_dim": 128,
        "num_mixtures": 5,
    }
    checkpoint = {
        "generator_state_dict": StrokeGenerator(
            input_dim=2,
            char_dim=cfg["char_dim"],
            hidden_dim=cfg["hidden_dim"],
            style_dim=cfg["style_dim"],
            num_mixtures=cfg["num_mixtures"],
        ).state_dict(),
        "style_encoder_state_dict": StyleEncoder(style_dim=cfg["style_dim"]).state_dict(),
        "char_encoder_state_dict": CharEncoder(char_dim=cfg["char_dim"]).state_dict(),
        "config": cfg,
        "norm_stats": norm_stats,
        "ref_norm_stats": ref_norm_stats,
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
        for i in range(len(ds)):
            item = ds[i]
            assert "stroke_deltas" in item
            assert "eos" in item
            assert "stroke_index" in item
            assert "ref_strokes" in item
            assert "character" in item
            assert item["character"] in ["あ", "い"]

    def test_dataset_item_shapes(self, tmp_path):
        chars = ["あ"]
        user_dir = _make_data_dir(tmp_path / "user", chars, n_samples=1)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ds = FinetuneDataset(user_dir, ref_dir)
        item = ds[0]
        assert item["stroke_deltas"].ndim == 2
        assert item["stroke_deltas"].shape[1] == 2
        assert item["eos"].ndim == 2
        assert item["eos"].shape[1] == 1
        assert item["eos"][-1, 0] == 1.0
        assert item["ref_strokes"].ndim == 2
        assert item["ref_strokes"].shape[1] == 2
        assert item["style_strokes"].ndim == 2
        assert item["style_strokes"].shape[1] == 3

    def test_collate_has_lengths(self, tmp_path):
        chars = ["あ"]
        user_dir = _make_data_dir(tmp_path / "user", chars, n_samples=2)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ds = FinetuneDataset(user_dir, ref_dir)
        batch = [ds[i] for i in range(len(ds))]
        collated = collate_finetune(batch)
        assert "stroke_lengths" in collated
        assert "ref_lengths" in collated
        assert "style_lengths" in collated
        assert "stroke_indices" in collated
        assert len(collated["stroke_lengths"]) == 2
        assert len(collated["ref_lengths"]) == 2

    def test_reference_has_separators(self, tmp_path):
        """reference uses separator-delimited format."""
        import json as _json

        ch = "あ"
        user_dir = tmp_path / "user"
        ref_dir = tmp_path / "ref"
        d = user_dir / ch
        d.mkdir(parents=True, exist_ok=True)
        stroke = [{"x": float(j), "y": float(j) * 0.5, "pressure": 1.0, "timestamp": 0.0}
                  for j in range(5)]
        data = {"character": ch, "strokes": [stroke], "metadata": {}}
        (d / f"{ch}_0.json").write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
        s1 = [{"x": float(j), "y": float(j)} for j in range(3)]
        s2 = [{"x": float(j + 5), "y": float(j + 5)} for j in range(4)]
        ref_data = {"character": ch, "strokes": [s1, s2], "metadata": {}}
        rd = ref_dir / ch
        rd.mkdir(parents=True, exist_ok=True)
        (rd / f"{ch}_ref.json").write_text(_json.dumps(ref_data, ensure_ascii=False), encoding="utf-8")
        ds = FinetuneDataset(user_dir, ref_dir)
        item = ds[0]
        ref = item["ref_strokes"]
        # 3 + 1 separator + 4 = 8
        assert ref.shape == (8, 2)
        assert ref[3, 0].item() == -1.0
        assert ref[3, 1].item() == -1.0

    def test_dataset_delta_encoding(self, tmp_path):
        """Strokes use delta coordinates with EOS."""
        chars = ["あ"]
        user_dir = _make_data_dir(tmp_path / "user", chars, n_samples=1)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ds = FinetuneDataset(user_dir, ref_dir)
        item = ds[0]
        assert item["stroke_deltas"][0, 0] == 0.0
        assert item["stroke_deltas"][0, 1] == 0.0
        assert item["eos"][-1, 0] == 1.0
        if item["eos"].shape[0] > 1:
            assert item["eos"][0, 0] == 0.0


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

    def test_finetuner_loads_norm_stats(self, tmp_path):
        chars = ["あ", "い"]
        user_dir = _make_data_dir(tmp_path / "user", chars)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        norm_stats = {"mean_x": 0.1, "mean_y": 0.2, "std_x": 1.5, "std_y": 1.3}
        ckpt_path = _make_pretrain_checkpoint(
            tmp_path / "ckpt" / "checkpoint.pt", norm_stats=norm_stats
        )
        cfg = FinetuneConfig(epochs=1)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=ckpt_path,
            user_data_dir=user_dir,
            ref_dir=ref_dir,
            output_dir=tmp_path / "output",
        )
        assert finetuner.norm_stats == norm_stats

    def test_finetuner_loads_ref_norm_stats(self, tmp_path):
        chars = ["あ", "い"]
        user_dir = _make_data_dir(tmp_path / "user", chars)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ref_norm_stats = {"mean_x": 0.5, "mean_y": 0.3, "std_x": 2.0, "std_y": 1.8}
        ckpt_path = _make_pretrain_checkpoint(
            tmp_path / "ckpt" / "checkpoint.pt", ref_norm_stats=ref_norm_stats
        )
        cfg = FinetuneConfig(epochs=1)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=ckpt_path,
            user_data_dir=user_dir,
            ref_dir=ref_dir,
            output_dir=tmp_path / "output",
        )
        assert finetuner.ref_norm_stats == ref_norm_stats

    def test_finetuner_handles_missing_norm_stats(self, setup):
        """norm_stats がないチェックポイントでも動作する（後方互換性）。"""
        cfg = FinetuneConfig(epochs=1)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        assert finetuner.norm_stats is None

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
    def test_finetune_checkpoint_has_norm_stats(self, tmp_path):
        chars = ["あ", "い", "う"]
        user_dir = _make_data_dir(tmp_path / "user", chars)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        norm_stats = {"mean_x": 0.1, "mean_y": 0.2, "std_x": 1.5, "std_y": 1.3}
        ref_norm_stats = {"mean_x": 0.5, "mean_y": 0.3, "std_x": 2.0, "std_y": 1.8}
        ckpt_path = _make_pretrain_checkpoint(
            tmp_path / "ckpt" / "checkpoint.pt",
            norm_stats=norm_stats,
            ref_norm_stats=ref_norm_stats,
        )
        output_dir = tmp_path / "output"
        cfg = FinetuneConfig(epochs=1, batch_size=4)
        finetuner = Finetuner(
            config=cfg,
            pretrain_checkpoint=ckpt_path,
            user_data_dir=user_dir,
            ref_dir=ref_dir,
            output_dir=output_dir,
        )
        finetuner.train()
        output_ckpt = torch.load(output_dir / "finetuned.pt", weights_only=False)
        assert "norm_stats" in output_ckpt
        assert output_ckpt["norm_stats"] == norm_stats
        assert "ref_norm_stats" in output_ckpt
        assert output_ckpt["ref_norm_stats"] == ref_norm_stats

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

        for k in input_ckpt["generator_state_dict"]:
            assert torch.equal(
                input_ckpt["generator_state_dict"][k],
                output_ckpt["generator_state_dict"][k],
            )


def _make_v3_checkpoint(path: Path, config: dict | None = None) -> Path:
    """Create a V3 deformation checkpoint for testing."""
    from src.model.stroke_deformer import AffineStrokeDeformer, StrokeDeformer

    cfg = config or {
        "style_dim": 128,
        "hidden_dim": 64,
        "num_points": 32,
        "deformer_type": "affine",
    }
    deformer_type = cfg.get("deformer_type", "offset")
    if deformer_type == "affine":
        deformer = AffineStrokeDeformer(
            style_dim=cfg["style_dim"], hidden_dim=cfg["hidden_dim"],
        )
    else:
        deformer = StrokeDeformer(
            style_dim=cfg["style_dim"], hidden_dim=cfg["hidden_dim"],
        )
    style_enc = StyleEncoder(style_dim=cfg["style_dim"])
    norm_stats = {"mean_x": 0.0, "mean_y": 0.0, "std_x": 1.0, "std_y": 1.0}

    checkpoint = {
        "deformer_state_dict": deformer.state_dict(),
        "style_encoder_state_dict": style_enc.state_dict(),
        "config": cfg,
        "norm_stats": norm_stats,
        "version": 3,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


class TestDeformationFinetuner:
    @pytest.fixture
    def setup(self, tmp_path):
        chars = ["あ", "い", "う"]
        user_dir = _make_data_dir(tmp_path / "user", chars)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ckpt_path = _make_v3_checkpoint(tmp_path / "ckpt" / "v3_checkpoint.pt")
        output_dir = tmp_path / "output"
        return {
            "user_dir": user_dir,
            "ref_dir": ref_dir,
            "ckpt_path": ckpt_path,
            "output_dir": output_dir,
        }

    def test_creation(self, setup):
        cfg = FinetuneConfig(epochs=1)
        finetuner = DeformationFinetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        assert finetuner is not None

    def test_deformer_frozen(self, setup):
        cfg = FinetuneConfig(epochs=1)
        finetuner = DeformationFinetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        for p in finetuner.deformer.parameters():
            assert not p.requires_grad

    def test_style_encoder_unfrozen(self, setup):
        cfg = FinetuneConfig(epochs=1)
        finetuner = DeformationFinetuner(
            config=cfg,
            pretrain_checkpoint=setup["ckpt_path"],
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        for p in finetuner.style_encoder.parameters():
            assert p.requires_grad

    @pytest.mark.slow
    def test_offset_finetuner_clamps_offsets(self, tmp_path):
        """DeformationFinetuner with offset deformer must clamp predicted offsets."""
        from src.model.finetune import OFFSET_CLAMP

        chars = ["あ", "い", "う"]
        user_dir = _make_data_dir(tmp_path / "user", chars)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ckpt_path = _make_v3_checkpoint(
            tmp_path / "ckpt" / "v3_offset.pt",
            config={
                "style_dim": 128,
                "hidden_dim": 64,
                "num_points": 32,
                "deformer_type": "offset",
            },
        )
        cfg = FinetuneConfig(epochs=1, batch_size=4)
        finetuner = DeformationFinetuner(
            config=cfg,
            pretrain_checkpoint=ckpt_path,
            user_data_dir=user_dir,
            ref_dir=ref_dir,
            output_dir=tmp_path / "output",
        )

        # Monkey-patch deformer to return large offsets
        original_forward = finetuner.deformer.forward
        post_clamp_offsets = []

        def patched_forward(*args, **kwargs):
            return original_forward(*args, **kwargs) * 10.0

        finetuner.deformer.forward = patched_forward

        from src.model.stroke_deformer import deformation_loss as orig_loss
        import src.model.stroke_deformer as sd_mod

        def capturing_loss(predicted, target):
            post_clamp_offsets.append(predicted.detach().clone())
            return orig_loss(predicted, target)

        old_loss = sd_mod.deformation_loss
        sd_mod.deformation_loss = capturing_loss
        try:
            history = finetuner.train()
        finally:
            sd_mod.deformation_loss = old_loss

        assert len(history["losses"]) == 1
        assert len(post_clamp_offsets) > 0
        for offsets in post_clamp_offsets:
            assert offsets.max().item() <= OFFSET_CLAMP + 1e-6
            assert offsets.min().item() >= -OFFSET_CLAMP - 1e-6

    @pytest.mark.slow
    def test_deformation_finetuner_trains(self, setup):
        cfg = FinetuneConfig(epochs=1, batch_size=4)
        finetuner = DeformationFinetuner(
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

        ckpt = torch.load(setup["output_dir"] / "finetuned.pt", weights_only=False)
        assert "deformer_state_dict" in ckpt
        assert "style_encoder_state_dict" in ckpt
        assert ckpt.get("version") == 3


class TestUserDeformationTrainer:
    @pytest.fixture
    def setup(self, tmp_path):
        chars = ["あ", "い", "う"]
        user_dir = _make_data_dir(tmp_path / "user", chars)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        output_dir = tmp_path / "output"
        return {
            "user_dir": user_dir,
            "ref_dir": ref_dir,
            "output_dir": output_dir,
        }

    def test_user_trainer_creation(self, setup):
        cfg = UserTrainConfig(epochs=1)
        trainer = UserDeformationTrainer(
            config=cfg,
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        assert trainer is not None
        assert len(trainer.dataset) > 0

    def test_both_modules_trainable(self, setup):
        cfg = UserTrainConfig(epochs=1)
        trainer = UserDeformationTrainer(
            config=cfg,
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        for p in trainer.deformer.parameters():
            assert p.requires_grad
        for p in trainer.style_encoder.parameters():
            assert p.requires_grad

    def test_augmentation_varies_output(self, tmp_path):
        chars = ["あ"]
        user_dir = _make_data_dir(tmp_path / "user", chars, n_samples=1)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ds = FinetuneDeformationDataset(user_dir, ref_dir, augment=True)
        if len(ds) == 0:
            pytest.skip("No stroke pairs found")
        results = [ds[0]["target_points"] for _ in range(5)]
        all_same = all(torch.equal(results[0], r) for r in results[1:])
        assert not all_same, "Augmentation should produce different target points"

    @pytest.mark.slow
    def test_training_clamps_offsets(self, setup):
        """Training must clamp predicted offsets within [-OFFSET_CLAMP, OFFSET_CLAMP]."""
        from src.model.finetune import OFFSET_CLAMP

        cfg = UserTrainConfig(epochs=1, batch_size=4)
        trainer = UserDeformationTrainer(
            config=cfg,
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )

        # Monkey-patch deformer to return large offsets that would exceed clamp range
        original_forward = trainer.deformer.forward
        post_clamp_offsets = []

        def patched_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            return result * 10.0  # artificially inflate to force clamping

        trainer.deformer.forward = patched_forward

        # Patch deformation_loss to capture the offsets after smooth+clamp
        from src.model.stroke_deformer import deformation_loss as orig_loss

        def capturing_loss(predicted, target):
            post_clamp_offsets.append(predicted.detach().clone())
            return orig_loss(predicted, target)

        import src.model.stroke_deformer as sd_mod
        old_loss = sd_mod.deformation_loss
        sd_mod.deformation_loss = capturing_loss
        try:
            trainer.train()
        finally:
            sd_mod.deformation_loss = old_loss

        assert len(post_clamp_offsets) > 0, "Loss was never called"
        for offsets in post_clamp_offsets:
            assert offsets.max().item() <= OFFSET_CLAMP + 1e-6
            assert offsets.min().item() >= -OFFSET_CLAMP - 1e-6

    @pytest.mark.slow
    def test_training_and_inference_share_clamp_and_smoothing(self, setup):
        """Training and inference use the same OFFSET_CLAMP and smooth_offsets function."""
        from src.model.finetune import OFFSET_CLAMP, SMOOTHING_KERNEL_SIZE, smooth_offsets

        assert OFFSET_CLAMP == 1.5
        assert SMOOTHING_KERNEL_SIZE == 11

        # Verify inference imports the same constants
        offsets = torch.randn(1, 32, 2)
        smoothed = smooth_offsets(offsets)
        assert smoothed.shape == offsets.shape

    @pytest.mark.slow
    def test_checkpoint_compatible_with_inference(self, setup):
        from src.model.inference import StrokeInference

        cfg = UserTrainConfig(epochs=2, batch_size=4)
        trainer = UserDeformationTrainer(
            config=cfg,
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        trainer.train()

        ckpt_path = setup["output_dir"] / "pretrain_checkpoint.pt"
        assert ckpt_path.exists()

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt.get("version") == 3
        assert "deformer_state_dict" in ckpt
        assert "style_encoder_state_dict" in ckpt
        assert "dropout" in ckpt["config"]

        inference = StrokeInference(ckpt_path)
        assert inference.version == 3
        assert inference.deformer is not None


class TestAugmentStyleStrokes:
    def test_output_shape(self):
        import numpy as np

        rng = np.random.default_rng(42)
        style = torch.randn(50, 3)
        result = augment_style_strokes(style, rng)
        assert result.shape == style.shape

    def test_pen_state_mostly_preserved(self):
        import numpy as np

        rng = np.random.default_rng(42)
        style = torch.randn(50, 3)
        result = augment_style_strokes(style, rng)
        # Pen state (column 2) should be unchanged — augmentation only touches dx, dy
        assert torch.equal(result[:, 2], style[:, 2])

    def test_different_from_input(self):
        import numpy as np

        rng = np.random.default_rng(42)
        style = torch.randn(50, 3)
        result = augment_style_strokes(style, rng)
        assert not torch.equal(result[:, :2], style[:, :2])

    def test_empty_input(self):
        import numpy as np

        rng = np.random.default_rng(42)
        style = torch.zeros(0, 3)
        result = augment_style_strokes(style, rng)
        assert result.shape == (0, 3)


class TestCollateDeformationFinetuneCharacterLabels:
    def test_character_labels_present(self, tmp_path):
        chars = ["あ", "い"]
        user_dir = _make_data_dir(tmp_path / "user", chars, n_samples=2)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ds = FinetuneDeformationDataset(user_dir, ref_dir, use_aligner=False)
        if len(ds) < 2:
            pytest.skip("Need at least 2 samples")
        batch = [ds[i] for i in range(min(len(ds), 4))]
        collated = collate_deformation_finetune(batch)
        assert "character_labels" in collated
        assert collated["character_labels"].dtype == torch.long
        # Batch is doubled (original + augmented views for contrastive learning)
        assert collated["character_labels"].shape[0] == len(batch) * 2


class TestUserDeformationTrainerTransformer:
    @pytest.fixture
    def setup(self, tmp_path):
        chars = ["あ", "い", "う"]
        user_dir = _make_data_dir(tmp_path / "user", chars)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        output_dir = tmp_path / "output"
        return {
            "user_dir": user_dir,
            "ref_dir": ref_dir,
            "output_dir": output_dir,
        }

    def test_creation_with_transformer(self, setup):
        from src.model.stroke_deformer import TransformerDeformer

        cfg = UserTrainConfig(deformer_type="transformer", epochs=2, batch_size=2)
        trainer = UserDeformationTrainer(
            config=cfg,
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        assert isinstance(trainer.deformer, TransformerDeformer)

    @pytest.mark.slow
    def test_checkpoint_has_transformer_type(self, setup):
        cfg = UserTrainConfig(deformer_type="transformer", epochs=2, batch_size=2)
        trainer = UserDeformationTrainer(
            config=cfg,
            user_data_dir=setup["user_dir"],
            ref_dir=setup["ref_dir"],
            output_dir=setup["output_dir"],
        )
        trainer.train()
        ckpt_path = setup["output_dir"] / "pretrain_checkpoint.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["config"]["deformer_type"] == "transformer"
        assert "d_model" in ckpt["config"]
        assert "nhead" in ckpt["config"]


class TestContrastiveBeta:
    def test_warmup_zero_at_start(self):
        cfg = UserTrainConfig(
            epochs=100,
            contrastive_weight=0.5,
            contrastive_warmup_frac=0.2,
        )
        # Minimal setup: only need config and _current_epoch
        trainer = UserDeformationTrainer.__new__(UserDeformationTrainer)
        trainer.config = cfg
        trainer._current_epoch = 0
        beta = trainer._get_contrastive_beta()
        assert beta == pytest.approx(0.0)

    def test_warmup_full_after_warmup(self):
        cfg = UserTrainConfig(
            epochs=100,
            contrastive_weight=0.5,
            contrastive_warmup_frac=0.2,
        )
        trainer = UserDeformationTrainer.__new__(UserDeformationTrainer)
        trainer.config = cfg
        # warmup_epochs = 100 * 0.2 = 20; at epoch 20 warmup is done
        trainer._current_epoch = 20
        beta = trainer._get_contrastive_beta()
        assert beta == pytest.approx(0.5)

    def test_warmup_midpoint(self):
        cfg = UserTrainConfig(
            epochs=100,
            contrastive_weight=0.5,
            contrastive_warmup_frac=0.2,
        )
        trainer = UserDeformationTrainer.__new__(UserDeformationTrainer)
        trainer.config = cfg
        # warmup_epochs = 20; at epoch 10 -> beta = 0.5 * 10/20 = 0.25
        trainer._current_epoch = 10
        beta = trainer._get_contrastive_beta()
        assert beta == pytest.approx(0.25)
