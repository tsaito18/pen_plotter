import json
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")

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
            generator_kwargs={"input_dim": 3, "hidden_dim": 64, "style_dim": 128, "num_mixtures": 3},
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
