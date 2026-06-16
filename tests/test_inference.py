import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.char_encoder import CharEncoder  # noqa: E402
from src.model.inference import (  # noqa: E402
    TEMP_NOISE_AMP,
    StrokeInference,
    _detect_device,
    _limit_style_sample,
    _temperature_noise,
)
from src.model.stroke_model import StrokeGenerator  # noqa: E402
from src.model.style_encoder import StyleEncoder  # noqa: E402


def test_limit_style_sample_keeps_short_sequence_contiguous():
    style_sample = torch.randn(1, 20, 3).transpose(1, 2).transpose(1, 2)

    result = _limit_style_sample(style_sample, max_points=32)

    assert result.shape == (1, 20, 3)
    assert result.is_contiguous()


def test_limit_style_sample_downsamples_long_sequence():
    style_sample = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)

    result = _limit_style_sample(style_sample, max_points=4)

    assert result.shape == (1, 4, 3)
    assert result.is_contiguous()
    assert torch.equal(result[0, :, 0], torch.tensor([0.0, 9.0, 18.0, 27.0]))


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
            style_sample=style_sample,
            num_steps=20,
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
            style_sample=style_sample,
            num_steps=20,
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
            style_sample=style_sample,
            num_steps=50,
            temperature=0.5,
            reference_strokes=reference,
        )
        for s in strokes:
            assert np.all(np.isfinite(s)), "Output contains non-finite values"
            assert np.all(np.abs(s) < 1000), f"Output coordinates too large: max={np.abs(s).max()}"

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
            generator_kwargs={
                "input_dim": 2,
                "hidden_dim": 64,
                "style_dim": 128,
                "num_mixtures": 3,
            },
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

    def test_auto_device_prefers_usable_cuda(self, monkeypatch):
        monkeypatch.setattr("src.model.inference._cuda_is_usable", lambda: True)

        assert _detect_device().type == "cuda"

    def test_v3_explicit_cpu_device(self, v3_checkpoint_path):
        engine = StrokeInference(
            checkpoint_path=v3_checkpoint_path,
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 64},
            device="cpu",
        )

        assert engine.device.type == "cpu"
        assert next(engine.style_encoder.parameters()).device.type == "cpu"
        assert next(engine.deformer.parameters()).device.type == "cpu"

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
            assert s.shape[1] == 2 and s.shape[0] >= 3

    def test_v3_no_reference_raises_value_error(self, v3_engine):
        style_sample = torch.randn(1, 10, 3)
        with pytest.raises(ValueError, match="requires reference_strokes"):
            v3_engine.generate(style_sample=style_sample, reference_strokes=None)

    def test_v3_empty_reference_raises_value_error(self, v3_engine):
        style_sample = torch.randn(1, 10, 3)
        with pytest.raises(ValueError, match="requires reference_strokes"):
            v3_engine.generate(style_sample=style_sample, reference_strokes=[])

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

    def test_v3_cuda_generation_returns_numpy(self, v3_checkpoint_path):
        if _detect_device().type != "cuda":
            pytest.skip("usable CUDA is unavailable")

        engine = StrokeInference(
            checkpoint_path=v3_checkpoint_path,
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 64},
            device="cuda",
        )

        strokes = engine.generate(
            style_sample=torch.randn(1, 15, 3),
            reference_strokes=[np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)],
            noise_scale=0.01,
        )

        assert engine.device.type == "cuda"
        assert next(engine.style_encoder.parameters()).device.type == "cuda"
        assert next(engine.deformer.parameters()).device.type == "cuda"
        assert isinstance(strokes[0], np.ndarray)

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

    def test_v3_batch_skips_short_strokes(self, v3_engine):
        """Strokes with < 2 points should be filtered out in batched inference."""
        style_sample = torch.randn(1, 20, 3)
        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
            np.array([[0.5, 0.5]], dtype=np.float64),  # < 2 points
            np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64),
        ]
        strokes = v3_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
        )
        assert len(strokes) == 2
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.shape[1] == 2 and s.shape[0] >= 3

    def test_v3_all_short_strokes_raises_value_error(self, v3_engine):
        """If all strokes are too short, inference reports missing usable references."""
        style_sample = torch.randn(1, 10, 3)
        reference = [
            np.array([[0.5, 0.5]], dtype=np.float64),
        ]
        with pytest.raises(ValueError, match="valid reference stroke"):
            v3_engine.generate(
                style_sample=style_sample,
                reference_strokes=reference,
            )


class TestStrokeInferenceV3Offset:
    """V3 offset deformer batch tests."""

    @pytest.fixture
    def v3_offset_checkpoint_path(self, tmp_path):
        from src.model.stroke_deformer import StrokeDeformer

        style_dim = 64
        deformer = StrokeDeformer(style_dim=style_dim, hidden_dim=128)
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=style_dim)
        checkpoint = {
            "deformer_state_dict": deformer.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
            "config": {
                "style_dim": style_dim,
                "hidden_dim": 128,
                "num_points": 16,
                "deformer_type": "offset",
            },
            "norm_stats": {"mean_x": 0.0, "mean_y": 0.0, "std_x": 1.0, "std_y": 1.0},
        }
        ckpt_path = tmp_path / "v3_offset_model.pt"
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    @pytest.fixture
    def v3_offset_engine(self, v3_offset_checkpoint_path):
        return StrokeInference(
            checkpoint_path=v3_offset_checkpoint_path,
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 64},
        )

    def test_v3_offset_batch_stroke_count(self, v3_offset_engine):
        """Batched offset deformer produces correct stroke count and shape."""
        style_sample = torch.randn(1, 20, 3)
        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
            np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64),
            np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64),
        ]
        strokes = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
        )
        assert len(strokes) == 3
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            assert s.shape[1] == 2 and s.shape[0] >= 3
            assert s.dtype == np.float32

    def test_v3_offset_batch_output_finite(self, v3_offset_engine):
        """Batched offset deformer output is finite."""
        style_sample = torch.randn(1, 15, 3)
        reference = [
            np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64),
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64),
        ]
        strokes = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.01,
        )
        for s in strokes:
            assert np.all(np.isfinite(s))

    def test_v3_offset_batch_skips_short_strokes(self, v3_offset_engine):
        """Offset deformer also skips strokes with < 2 points."""
        style_sample = torch.randn(1, 20, 3)
        reference = [
            np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            np.array([[0.5, 0.5]], dtype=np.float64),  # skipped
        ]
        strokes = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
        )
        assert len(strokes) == 1
        assert strokes[0].shape[1] == 2 and strokes[0].shape[0] >= 3


class TestStrokeInferenceV3Transformer:
    """V3 transformer deformer tests."""

    @pytest.fixture
    def v3_transformer_checkpoint_path(self, tmp_path):
        from src.model.stroke_deformer import TransformerDeformer

        style_dim = 64
        deformer = TransformerDeformer(
            style_dim=style_dim,
            d_model=32,
            nhead=2,
            num_self_attn_layers=1,
            ff_dim=64,
        )
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=style_dim)
        checkpoint = {
            "deformer_state_dict": deformer.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
            "config": {
                "style_dim": style_dim,
                "num_points": 16,
                "deformer_type": "transformer",
                "d_model": 32,
                "nhead": 2,
                "num_self_attn_layers": 1,
                "ff_dim": 64,
            },
            "norm_stats": {"mean_x": 0.0, "mean_y": 0.0, "std_x": 1.0, "std_y": 1.0},
        }
        ckpt_path = tmp_path / "v3_transformer_model.pt"
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    @pytest.fixture
    def v3_transformer_engine(self, v3_transformer_checkpoint_path):
        return StrokeInference(
            checkpoint_path=v3_transformer_checkpoint_path,
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 64},
        )

    def test_v3_transformer_detection(self, v3_transformer_engine):
        assert v3_transformer_engine.version == 3
        assert v3_transformer_engine.deformer_type == "transformer"

    def test_v3_transformer_generation(self, v3_transformer_engine):
        style_sample = torch.randn(1, 20, 3)
        reference = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
            np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64),
        ]
        strokes = v3_transformer_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
        )
        assert len(strokes) == 2
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.shape[1] == 2 and s.shape[0] >= 3
            assert s.dtype == np.float32

    def test_v3_transformer_output_finite(self, v3_transformer_engine):
        style_sample = torch.randn(1, 15, 3)
        reference = [
            np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64),
        ]
        strokes = v3_transformer_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.01,
        )
        for s in strokes:
            assert np.all(np.isfinite(s))


class TestTemperatureNoise:
    """低周波温度ノイズ生成関数の単体テスト（モデル不要）。"""

    def test_amp_constant_within_clamp_third(self):
        """振幅定数は offset clamp の約1/3 以内（字形を破綻させない上限ガード）。"""
        from src.model.finetune import OFFSET_CLAMP

        assert 0.0 < TEMP_NOISE_AMP <= OFFSET_CLAMP / 2.0

    def test_shape_and_dtype(self):
        out = _temperature_noise(num_strokes=3, num_points=32, amp=0.1)
        assert out.shape == (3, 32, 2)
        assert out.dtype == np.float32

    def test_zero_amp_is_identity_zero(self):
        out = _temperature_noise(num_strokes=2, num_points=32, amp=0.0)
        assert np.all(out == 0.0)

    def test_reproducible_under_same_seed(self):
        np.random.seed(7)
        a = _temperature_noise(num_strokes=2, num_points=32, amp=0.1)
        np.random.seed(7)
        b = _temperature_noise(num_strokes=2, num_points=32, amp=0.1)
        assert np.array_equal(a, b)

    def test_differs_under_different_seed(self):
        np.random.seed(1)
        a = _temperature_noise(num_strokes=2, num_points=32, amp=0.1)
        np.random.seed(2)
        b = _temperature_noise(num_strokes=2, num_points=32, amp=0.1)
        assert not np.allclose(a, b)

    def test_low_frequency_smoother_than_white_noise(self):
        """低周波性: 制御点補間ノイズの隣接点差分RMSが、同振幅の白色ガウスより明確に小さい。"""
        amp = 0.1
        num_strokes, num_points = 8, 32
        np.random.seed(0)
        lf = _temperature_noise(num_strokes, num_points, amp=amp)

        np.random.seed(0)
        white = np.random.normal(0.0, amp, size=(num_strokes, num_points, 2)).astype(np.float32)

        lf_step_rms = np.sqrt(np.mean(np.diff(lf, axis=1) ** 2))
        white_step_rms = np.sqrt(np.mean(np.diff(white, axis=1) ** 2))
        # 補間で滑らかにしたぶん隣接差分は白色ノイズの半分未満になるはず
        assert lf_step_rms < white_step_rms * 0.5


class TestStrokeInferenceV3Temperature:
    """温度連動の低周波字形ゆらぎ（V3 per-point offset 経路）。"""

    @pytest.fixture
    def v3_offset_engine(self, tmp_path):
        from src.model.stroke_deformer import StrokeDeformer

        style_dim = 64
        deformer = StrokeDeformer(style_dim=style_dim, hidden_dim=128)
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=style_dim)
        checkpoint = {
            "deformer_state_dict": deformer.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
            "config": {
                "style_dim": style_dim,
                "hidden_dim": 128,
                "num_points": 32,
                "deformer_type": "offset",
            },
            "norm_stats": {"mean_x": 0.0, "mean_y": 0.0, "std_x": 1.0, "std_y": 1.0},
        }
        ckpt_path = tmp_path / "v3_temp_model.pt"
        torch.save(checkpoint, ckpt_path)
        return StrokeInference(
            checkpoint_path=ckpt_path,
            style_encoder_kwargs={"input_dim": 3, "hidden_dim": 32, "style_dim": 64},
            device="cpu",
        )

    @pytest.fixture
    def style_and_ref(self):
        style_sample = torch.randn(1, 20, 3)
        reference = [
            np.array([[0.0, 0.0], [2.0, 3.0], [5.0, 5.0]], dtype=np.float64),
            np.array([[5.0, 0.0], [3.0, 2.0], [0.0, 5.0]], dtype=np.float64),
        ]
        return style_sample, reference

    def test_temperature_zero_matches_no_noise(self, v3_offset_engine, style_and_ref):
        """後方互換: temperature=0 はノイズ非加算パスと完全一致（noise_scale=0 で幾何揺らぎも排除）。"""
        style_sample, reference = style_and_ref
        np.random.seed(99)
        a = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.0,
            temperature=0.0,
        )
        np.random.seed(123)  # 異なる seed でも温度0なら不変
        b = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.0,
            temperature=0.0,
        )
        assert len(a) == len(b)
        for sa, sb in zip(a, b):
            assert np.array_equal(sa, sb)

    def test_temperature_produces_diversity(self, v3_offset_engine, style_and_ref):
        """多様性: 同一(style, ref)でも seed が違えば temperature>0 で出力が有意に変わる。"""
        style_sample, reference = style_and_ref
        np.random.seed(1)
        a = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.0,
            temperature=1.0,
        )
        np.random.seed(2)
        b = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.0,
            temperature=1.0,
        )
        # _smooth_stroke は弧長比例で点数を割り当てるため、形が変わると点数も変わりうる。
        # 点数非依存に重心と端点で「有意に異なる」を判定する。
        def _signature(stroke):
            return np.concatenate([stroke.mean(axis=0), stroke[0], stroke[-1]])

        sig_diffs = [np.abs(_signature(sa) - _signature(sb)).max() for sa, sb in zip(a, b)]
        assert max(sig_diffs) > 1e-3

    def test_temperature_reproducible(self, v3_offset_engine, style_and_ref):
        """再現性: 同一 seed・同一 temperature なら2回生成は完全一致。"""
        style_sample, reference = style_and_ref
        np.random.seed(55)
        a = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.0,
            temperature=1.0,
        )
        np.random.seed(55)
        b = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.0,
            temperature=1.0,
        )
        for sa, sb in zip(a, b):
            assert np.array_equal(sa, sb)

    def test_temperature_offsets_stay_clamped(self, v3_offset_engine, style_and_ref):
        """clamp維持: 大きな temperature でも変形は参照±OFFSET_CLAMP の範囲に収まる。

        ノイズは clamp の前に加算されるため、振幅を上げても最終 offset は
        ±OFFSET_CLAMP で頭打ちになり字形は破綻しない。
        """
        from src.model.data_utils import resample_stroke
        from src.model.finetune import OFFSET_CLAMP

        style_sample, _ = style_and_ref
        # smooth_offsets を効かせるため num_points(=32) より長い直線参照を使う
        reference = [np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)]

        np.random.seed(3)
        strokes = v3_offset_engine.generate(
            style_sample=style_sample,
            reference_strokes=reference,
            noise_scale=0.0,
            temperature=5.0,
        )

        ref_rs = resample_stroke(reference[0].astype(np.float32), v3_offset_engine.num_points)
        ref_min = ref_rs.min(axis=0)
        ref_max = ref_rs.max(axis=0)
        out = strokes[0]
        # _smooth_stroke の補間で生じる僅かなオーバーシュートを許容するマージン
        margin = 0.2
        assert np.all(out >= ref_min - OFFSET_CLAMP - margin)
        assert np.all(out <= ref_max + OFFSET_CLAMP + margin)
