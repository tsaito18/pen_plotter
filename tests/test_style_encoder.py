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


class TestStyleEncoderForgetGateBias:
    def test_forget_gate_bias_initialized_to_one(self):
        enc = StyleEncoder(input_dim=3, hidden_dim=64, style_dim=128)
        for name, param in enc.lstm.named_parameters():
            if "bias" in name:
                hidden_dim = param.shape[0] // 4
                forget_bias = param.data[hidden_dim : 2 * hidden_dim]
                assert torch.allclose(forget_bias, torch.ones_like(forget_bias))


class TestStyleEncoderPackedSequence:
    @pytest.fixture
    def encoder(self):
        return StyleEncoder(input_dim=3, hidden_dim=64, style_dim=128)

    def test_forward_without_lengths_backward_compatible(self, encoder):
        x = torch.randn(3, 20, 3)
        out = encoder(x)
        assert out.shape == (3, 128)

    def test_forward_with_lengths_shape(self, encoder):
        x = torch.randn(3, 20, 3)
        lengths = torch.tensor([20, 15, 10])
        out = encoder(x, lengths=lengths)
        assert out.shape == (3, 128)

    def test_packed_differs_from_padded(self, encoder):
        """Packed sequence output differs when padding is present."""
        encoder.eval()
        x = torch.randn(2, 20, 3)
        x[1, 10:, :] = 0.0
        lengths = torch.tensor([20, 10])

        out_no_pack = encoder(x)
        out_packed = encoder(x, lengths=lengths)
        assert torch.allclose(out_no_pack[0], out_packed[0], atol=1e-6)
        assert not torch.allclose(out_no_pack[1], out_packed[1], atol=1e-5)
