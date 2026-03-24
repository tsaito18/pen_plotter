import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.char_encoder import CharEncoder
from src.model.inference import StrokeInference
from src.model.stroke_model import StrokeGenerator
from src.model.style_encoder import StyleEncoder


class TestStrokeInference:
    @pytest.fixture
    def inference_engine(self, tmp_path):
        generator = StrokeGenerator(input_dim=2, hidden_dim=64, style_dim=128, num_mixtures=3)
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=128)

        checkpoint = {
            "generator_state_dict": generator.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
        }
        ckpt_path = tmp_path / "model.pt"
        torch.save(checkpoint, ckpt_path)

        return StrokeInference(
            checkpoint_path=ckpt_path,
            generator_kwargs={
                "input_dim": 2,
                "hidden_dim": 64,
                "style_dim": 128,
                "num_mixtures": 3,
            },
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 128},
        )

    def test_generate_stroke(self, inference_engine):
        style_sample = torch.randn(1, 20, 3)
        strokes = inference_engine.generate(
            style_sample=style_sample,
            num_steps=10,
        )
        assert isinstance(strokes, list)
        assert len(strokes) > 0

    def test_temperature_affects_output(self, inference_engine):
        style_sample = torch.randn(1, 20, 3)
        torch.manual_seed(42)
        s1 = inference_engine.generate(style_sample=style_sample, num_steps=10, temperature=0.1)
        torch.manual_seed(42)
        s2 = inference_engine.generate(style_sample=style_sample, num_steps=10, temperature=2.0)
        assert len(s1) > 0
        assert len(s2) > 0

    def test_generate_returns_numpy(self, inference_engine):
        import numpy as np

        style_sample = torch.randn(1, 15, 3)
        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
            np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64),
        ]
        strokes = inference_engine.generate(
            style_sample=style_sample, num_steps=20,
            reference_strokes=reference,
        )
        for stroke in strokes:
            assert isinstance(stroke, np.ndarray)
            assert stroke.ndim == 2
            assert stroke.shape[1] == 2  # (x, y)

    def test_no_norm_stats_for_v1(self, inference_engine):
        """V1 checkpoint has no norm_stats."""
        assert inference_engine.norm_stats is None

    def test_stroke_count_matches_reference(self, inference_engine):
        """Number of generated strokes should match reference stroke count."""
        style_sample = torch.randn(1, 20, 3)
        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
            np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]], dtype=np.float64),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        ]
        strokes = inference_engine.generate(
            style_sample=style_sample,
            num_steps=20,
            reference_strokes=reference,
        )
        assert len(strokes) <= 3


class TestStrokeInferenceV2:
    """V2チェックポイント（CharEncoder付き）のテスト。"""

    @pytest.fixture
    def v2_checkpoint_path(self, tmp_path):
        char_dim = 64
        generator = StrokeGenerator(
            input_dim=2, hidden_dim=64, style_dim=128, char_dim=char_dim, num_mixtures=3
        )
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=128)
        char_enc = CharEncoder(input_dim=2, hidden_dim=32, char_dim=char_dim, num_layers=1)
        checkpoint = {
            "generator_state_dict": generator.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
            "char_encoder_state_dict": char_enc.state_dict(),
            "config": {"char_dim": char_dim, "hidden_dim": 64, "style_dim": 128, "num_mixtures": 3},
        }
        ckpt_path = tmp_path / "v2_model.pt"
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    @pytest.fixture
    def v2_engine(self, v2_checkpoint_path):
        return StrokeInference(
            checkpoint_path=v2_checkpoint_path,
            generator_kwargs={
                "input_dim": 2,
                "hidden_dim": 64,
                "style_dim": 128,
                "num_mixtures": 3,
            },
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 128},
        )

    def test_v2_checkpoint_loads(self, v2_engine):
        """V2チェックポイント（char_encoder付き）が正しく読み込まれる。"""
        assert v2_engine.char_encoder is not None
        assert v2_engine.generator.char_dim > 0

    def test_v2_generate_with_reference(self, v2_engine):
        """reference_strokesを渡すとchar_embeddingが使われる。"""
        style_sample = torch.randn(1, 10, 3)
        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
            np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64),
        ]
        strokes = v2_engine.generate(
            style_sample=style_sample,
            num_steps=10,
            reference_strokes=reference,
        )
        assert isinstance(strokes, list)
        assert len(strokes) > 0
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            assert s.shape[1] == 2

    def test_v1_checkpoint_still_works(self, tmp_path):
        """V1チェックポイント（char_encoder無し）が従来通り動作する。"""
        generator = StrokeGenerator(input_dim=2, hidden_dim=64, style_dim=128, num_mixtures=3)
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=128)
        checkpoint = {
            "generator_state_dict": generator.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
        }
        ckpt_path = tmp_path / "v1_model.pt"
        torch.save(checkpoint, ckpt_path)

        engine = StrokeInference(
            checkpoint_path=ckpt_path,
            generator_kwargs={
                "input_dim": 2,
                "hidden_dim": 64,
                "style_dim": 128,
                "num_mixtures": 3,
            },
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 128},
        )
        assert engine.char_encoder is None
        assert engine.generator.char_dim == 0

        style_sample = torch.randn(1, 10, 3)
        strokes = engine.generate(style_sample=style_sample, num_steps=5)
        assert len(strokes) > 0

    def test_generate_without_reference_on_v2(self, v2_engine):
        """V2モデルでreference_strokes無しでも動作する（char_embeddingゼロ）。"""
        style_sample = torch.randn(1, 10, 3)
        strokes = v2_engine.generate(
            style_sample=style_sample,
            num_steps=10,
        )
        assert isinstance(strokes, list)
        assert len(strokes) > 0


class TestStrokeInferenceNormStats:
    """norm_stats付きチェックポイントのテスト。"""

    @pytest.fixture
    def norm_stats(self):
        return {
            "mean_x": 0.5,
            "mean_y": -0.3,
            "std_x": 2.0,
            "std_y": 1.5,
        }

    @pytest.fixture
    def engine_with_norm(self, tmp_path, norm_stats):
        generator = StrokeGenerator(input_dim=2, hidden_dim=64, style_dim=128, num_mixtures=3)
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=128)
        checkpoint = {
            "generator_state_dict": generator.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
            "norm_stats": norm_stats,
        }
        ckpt_path = tmp_path / "model_norm.pt"
        torch.save(checkpoint, ckpt_path)
        return StrokeInference(
            checkpoint_path=ckpt_path,
            generator_kwargs={
                "input_dim": 2,
                "hidden_dim": 64,
                "style_dim": 128,
                "num_mixtures": 3,
            },
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 128},
        )

    def test_norm_stats_loaded(self, engine_with_norm, norm_stats):
        """norm_stats がチェックポイントから正しく読み込まれる。"""
        assert engine_with_norm.norm_stats is not None
        assert engine_with_norm.norm_stats["mean_x"] == norm_stats["mean_x"]
        assert engine_with_norm.norm_stats["std_x"] == norm_stats["std_x"]

    def test_generate_with_norm_stats(self, engine_with_norm):
        """norm_stats付きでも生成が正常に動作する。"""
        style_sample = torch.randn(1, 15, 3)
        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
        ]
        strokes = engine_with_norm.generate(
            style_sample=style_sample, num_steps=20,
            reference_strokes=reference,
        )
        assert isinstance(strokes, list)
        assert len(strokes) > 0
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            assert s.shape[1] == 2

    def test_output_coordinates_reasonable_range(self, engine_with_norm):
        """出力座標が発散していないことを確認。"""
        style_sample = torch.randn(1, 15, 3)
        torch.manual_seed(123)
        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
        ]
        strokes = engine_with_norm.generate(
            style_sample=style_sample, num_steps=50, temperature=0.5,
            reference_strokes=reference,
        )
        for s in strokes:
            assert np.all(np.isfinite(s)), "Output contains non-finite values"
            assert np.all(np.abs(s) < 1000), (
                f"Output coordinates too large: max={np.abs(s).max()}"
            )

    def test_ref_norm_stats_loaded(self, tmp_path, norm_stats):
        """ref_norm_stats is loaded from checkpoint."""
        ref_norm_stats = {"mean_x": 1.0, "mean_y": 2.0, "std_x": 3.0, "std_y": 4.0}
        char_dim = 64
        generator = StrokeGenerator(
            input_dim=2, hidden_dim=64, style_dim=128, char_dim=char_dim, num_mixtures=3
        )
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=128)
        char_enc = CharEncoder(input_dim=2, hidden_dim=32, char_dim=char_dim, num_layers=1)
        checkpoint = {
            "generator_state_dict": generator.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
            "char_encoder_state_dict": char_enc.state_dict(),
            "config": {"char_dim": char_dim, "hidden_dim": 64, "style_dim": 128, "num_mixtures": 3},
            "norm_stats": norm_stats,
            "ref_norm_stats": ref_norm_stats,
        }
        ckpt_path = tmp_path / "v2_refnorm.pt"
        torch.save(checkpoint, ckpt_path)

        engine = StrokeInference(
            checkpoint_path=ckpt_path,
            generator_kwargs={"input_dim": 2, "hidden_dim": 64, "style_dim": 128, "num_mixtures": 3},
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 128},
        )
        assert engine.ref_norm_stats == ref_norm_stats

        reference = [np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)]
        strokes = engine.generate(
            style_sample=torch.randn(1, 10, 3),
            num_steps=10,
            reference_strokes=reference,
        )
        assert len(strokes) > 0

    def test_v2_with_norm_stats(self, tmp_path, norm_stats):
        """V2チェックポイント + norm_stats の組み合わせが動作する。"""
        char_dim = 64
        generator = StrokeGenerator(
            input_dim=2, hidden_dim=64, style_dim=128, char_dim=char_dim, num_mixtures=3
        )
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=128)
        char_enc = CharEncoder(input_dim=2, hidden_dim=32, char_dim=char_dim, num_layers=1)
        checkpoint = {
            "generator_state_dict": generator.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
            "char_encoder_state_dict": char_enc.state_dict(),
            "config": {"char_dim": char_dim, "hidden_dim": 64, "style_dim": 128, "num_mixtures": 3},
            "norm_stats": norm_stats,
        }
        ckpt_path = tmp_path / "v2_norm.pt"
        torch.save(checkpoint, ckpt_path)

        engine = StrokeInference(
            checkpoint_path=ckpt_path,
            generator_kwargs={
                "input_dim": 2,
                "hidden_dim": 64,
                "style_dim": 128,
                "num_mixtures": 3,
            },
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 128},
        )
        assert engine.norm_stats is not None
        assert engine.char_encoder is not None

        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
        ]
        strokes = engine.generate(
            style_sample=torch.randn(1, 10, 3),
            num_steps=10,
            reference_strokes=reference,
        )
        assert len(strokes) > 0


class TestStrokeInferenceV3:
    """V3 deformation checkpoint tests."""

    @pytest.fixture
    def v3_checkpoint_path(self, tmp_path):
        from src.model.stroke_deformer import AffineStrokeDeformer

        style_dim = 64
        deformer = AffineStrokeDeformer(style_dim=style_dim, hidden_dim=128)
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=style_dim)
        checkpoint = {
            "deformer_state_dict": deformer.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
            "config": {
                "style_dim": style_dim,
                "hidden_dim": 128,
                "num_points": 16,
                "deformer_type": "affine",
            },
            "norm_stats": {"mean_x": 0.0, "mean_y": 0.0, "std_x": 1.0, "std_y": 1.0},
            "version": 3,
        }
        ckpt_path = tmp_path / "v3_model.pt"
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    @pytest.fixture
    def v3_engine(self, v3_checkpoint_path):
        return StrokeInference(
            checkpoint_path=v3_checkpoint_path,
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 64},
        )

    def test_v3_detection(self, v3_engine):
        assert v3_engine.version == 3
        assert v3_engine.deformer is not None
        assert v3_engine.generator is None
        assert v3_engine.char_encoder is None

    def test_v3_generation_stroke_count(self, v3_engine):
        style_sample = torch.randn(1, 20, 3)
        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
            np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64),
            np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64),
        ]
        strokes = v3_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
        )
        assert len(strokes) == 3
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            assert s.shape == (16, 2)

    def test_v3_no_reference_returns_fallback(self, v3_engine):
        style_sample = torch.randn(1, 10, 3)
        strokes = v3_engine.generate(style_sample=style_sample, reference_strokes=None)
        assert len(strokes) == 1

    def test_v3_output_finite(self, v3_engine):
        style_sample = torch.randn(1, 15, 3)
        reference = [
            np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64),
        ]
        strokes = v3_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.01,
        )
        for s in strokes:
            assert np.all(np.isfinite(s))

    def test_v3_smooth_offsets(self, v3_engine):
        """Smoothing should reduce variation between adjacent offset differences."""
        from src.model.finetune import smooth_offsets

        offsets = torch.randn(1, 32, 2)
        smoothed = smooth_offsets(offsets, kernel_size=5)
        assert smoothed.shape == offsets.shape

        raw_diff = (offsets[:, 1:] - offsets[:, :-1]).abs().mean().item()
        smooth_diff = (smoothed[:, 1:] - smoothed[:, :-1]).abs().mean().item()
        assert smooth_diff < raw_diff

    def test_v3_smooth_offsets_short_sequence(self, v3_engine):
        """Sequences shorter than kernel_size are returned unchanged."""
        from src.model.finetune import smooth_offsets

        offsets = torch.randn(1, 3, 2)
        result = smooth_offsets(offsets, kernel_size=5)
        assert torch.equal(result, offsets)
