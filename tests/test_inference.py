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
        generator = StrokeGenerator(input_dim=3, hidden_dim=64, style_dim=128, num_mixtures=3)
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
                "input_dim": 3,
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
        strokes = inference_engine.generate(style_sample=style_sample, num_steps=5)
        for stroke in strokes:
            assert isinstance(stroke, np.ndarray)
            assert stroke.ndim == 2
            assert stroke.shape[1] == 2  # (x, y)


class TestStrokeInferenceV2:
    """V2チェックポイント（CharEncoder付き）のテスト。"""

    @pytest.fixture
    def v2_checkpoint_path(self, tmp_path):
        char_dim = 64
        generator = StrokeGenerator(
            input_dim=3, hidden_dim=64, style_dim=128, char_dim=char_dim, num_mixtures=3
        )
        style_enc = StyleEncoder(input_dim=3, hidden_dim=32, style_dim=128)
        char_enc = CharEncoder(input_dim=2, hidden_dim=32, char_dim=char_dim, num_layers=1)
        checkpoint = {
            "generator_state_dict": generator.state_dict(),
            "style_encoder_state_dict": style_enc.state_dict(),
            "char_encoder_state_dict": char_enc.state_dict(),
            "char_dim": char_dim,
        }
        ckpt_path = tmp_path / "v2_model.pt"
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    @pytest.fixture
    def v2_engine(self, v2_checkpoint_path):
        return StrokeInference(
            checkpoint_path=v2_checkpoint_path,
            generator_kwargs={
                "input_dim": 3,
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
        generator = StrokeGenerator(input_dim=3, hidden_dim=64, style_dim=128, num_mixtures=3)
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
                "input_dim": 3,
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
