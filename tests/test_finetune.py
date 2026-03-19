"""Fine-tuning パイプラインのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.model.char_encoder import CharEncoder
from src.model.finetune import FinetuneConfig, FinetuneDataset, Finetuner, collate_finetune
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
        assert item["strokes"].shape[1] == 3  # dx, dy, pen_state
        assert item["ref_strokes"].ndim == 2
        assert item["ref_strokes"].shape[1] == 2  # x, y

    def test_collate_has_lengths(self, tmp_path):
        chars = ["あ"]
        user_dir = _make_data_dir(tmp_path / "user", chars, n_samples=2)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ds = FinetuneDataset(user_dir, ref_dir)
        batch = [ds[i] for i in range(len(ds))]
        collated = collate_finetune(batch)
        assert "lengths" in collated
        assert "ref_lengths" in collated
        assert len(collated["lengths"]) == 2
        assert len(collated["ref_lengths"]) == 2

    def test_reference_has_separators(self, tmp_path):
        """reference uses separator-delimited format."""
        import json as _json

        ch = "あ"
        user_dir = tmp_path / "user"
        ref_dir = tmp_path / "ref"
        # user data
        d = user_dir / ch
        d.mkdir(parents=True, exist_ok=True)
        stroke = [{"x": float(j), "y": float(j) * 0.5, "pressure": 1.0, "timestamp": 0.0}
                  for j in range(5)]
        data = {"character": ch, "strokes": [stroke], "metadata": {}}
        (d / f"{ch}_0.json").write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
        # ref data with 2 strokes
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
        """Strokes use delta coordinates and proper pen_state (0/1)."""
        chars = ["あ"]
        user_dir = _make_data_dir(tmp_path / "user", chars, n_samples=1)
        ref_dir = _make_ref_dir(tmp_path / "ref", chars)
        ds = FinetuneDataset(user_dir, ref_dir)
        item = ds[0]
        pen_states = item["strokes"][:, 2]
        assert ((pen_states == 0) | (pen_states == 1)).all()
        assert pen_states[-1] == 1.0  # last point is pen-up
        # first point delta is (0, 0)
        assert item["strokes"][0, 0] == 0.0
        assert item["strokes"][0, 1] == 0.0


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

        # Generator weights should be unchanged (frozen)
        for k in input_ckpt["generator_state_dict"]:
            assert torch.equal(
                input_ckpt["generator_state_dict"][k],
                output_ckpt["generator_state_dict"][k],
            )
