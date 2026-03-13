import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from src.model.style_encoder import StyleEncoder


class TestStyleEncoder:
    @pytest.fixture
    def encoder(self):
        return StyleEncoder(input_dim=3, hidden_dim=64, style_dim=128)

    def test_output_shape(self, encoder):
        # batch=4, seq_len=20, features=3 (x, y, pressure)
        x = torch.randn(4, 20, 3)
        style = encoder(x)
        assert style.shape == (4, 128)

    def test_different_seq_lengths(self, encoder):
        x1 = torch.randn(2, 10, 3)
        x2 = torch.randn(2, 50, 3)
        s1 = encoder(x1)
        s2 = encoder(x2)
        assert s1.shape == s2.shape == (2, 128)

    def test_single_sample(self, encoder):
        x = torch.randn(1, 5, 3)
        style = encoder(x)
        assert style.shape == (1, 128)

    def test_output_is_different_for_different_input(self, encoder):
        encoder.eval()
        x1 = torch.randn(1, 20, 3)
        x2 = torch.randn(1, 20, 3)
        s1 = encoder(x1)
        s2 = encoder(x2)
        assert not torch.allclose(s1, s2)

    def test_default_dims(self):
        enc = StyleEncoder()
        x = torch.randn(2, 15, 3)
        style = enc(x)
        assert style.shape == (2, 128)

    def test_is_nn_module(self, encoder):
        assert isinstance(encoder, nn.Module)

    def test_has_parameters(self, encoder):
        params = list(encoder.parameters())
        assert len(params) > 0
