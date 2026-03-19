import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from src.model.char_encoder import CharEncoder  # noqa: E402


class TestCharEncoder:
    @pytest.fixture
    def encoder(self):
        return CharEncoder(input_dim=2, hidden_dim=128, char_dim=128)

    def test_output_shape(self, encoder):
        x = torch.randn(2, 20, 2)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_output_shape_different_seq_lengths(self, encoder):
        x1 = torch.randn(2, 10, 2)
        x2 = torch.randn(2, 50, 2)
        o1 = encoder(x1)
        o2 = encoder(x2)
        assert o1.shape == o2.shape == (2, 128)

    def test_char_dim_configurable(self):
        enc = CharEncoder(char_dim=64)
        x = torch.randn(1, 15, 2)
        out = enc(x)
        assert out.shape == (1, 64)

    def test_is_nn_module(self, encoder):
        assert isinstance(encoder, nn.Module)

    def test_has_parameters(self, encoder):
        params = list(encoder.parameters())
        assert len(params) > 0

    def test_different_chars_different_embeddings(self, encoder):
        encoder.eval()
        x1 = torch.randn(1, 20, 2)
        x2 = torch.randn(1, 20, 2)
        e1 = encoder(x1)
        e2 = encoder(x2)
        assert not torch.allclose(e1, e2)


class TestCharEncoderForgetGateBias:
    def test_forget_gate_bias_initialized_to_one(self):
        enc = CharEncoder(input_dim=2, hidden_dim=64, char_dim=64)
        for name, param in enc.lstm.named_parameters():
            if "bias" in name:
                hidden_dim = param.shape[0] // 4
                forget_bias = param.data[hidden_dim : 2 * hidden_dim]
                assert torch.allclose(forget_bias, torch.ones_like(forget_bias))


class TestCharEncoderPackedSequence:
    @pytest.fixture
    def encoder(self):
        return CharEncoder(input_dim=2, hidden_dim=64, char_dim=64)

    def test_forward_without_lengths_backward_compatible(self, encoder):
        x = torch.randn(3, 20, 2)
        out = encoder(x)
        assert out.shape == (3, 64)

    def test_forward_with_lengths_shape(self, encoder):
        x = torch.randn(3, 20, 2)
        lengths = torch.tensor([20, 15, 10])
        out = encoder(x, lengths=lengths)
        assert out.shape == (3, 64)

    def test_packed_differs_from_padded(self, encoder):
        """Packed sequence output differs when padding is present."""
        encoder.eval()
        x = torch.randn(2, 20, 2)
        x[1, 10:, :] = 0.0  # simulate padding after length 10
        lengths = torch.tensor([20, 10])

        out_no_pack = encoder(x)
        out_packed = encoder(x, lengths=lengths)
        # First sample (full length) should be identical
        assert torch.allclose(out_no_pack[0], out_packed[0], atol=1e-6)
        # Second sample (has padding) should differ
        assert not torch.allclose(out_no_pack[1], out_packed[1], atol=1e-5)


class TestStrokesToSequence:
    def test_strokes_to_sequence_basic(self):
        s1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        s2 = np.array([[2.0, 2.0], [3.0, 3.0]])
        result = CharEncoder.strokes_to_sequence([s1, s2])
        assert result.shape == (5, 2)

    def test_strokes_to_sequence_separator(self):
        s1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        s2 = np.array([[2.0, 2.0], [3.0, 3.0]])
        result = CharEncoder.strokes_to_sequence([s1, s2])
        separator = result[2]
        np.testing.assert_array_equal(separator, [-1.0, -1.0])

    def test_strokes_to_sequence_empty(self):
        result = CharEncoder.strokes_to_sequence([])
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, [[0.0, 0.0]])
