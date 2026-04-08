import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import torch.nn.functional as F

from src.model.style_encoder import ProjectionHead, StyleEncoder, supervised_contrastive_loss


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


class TestProjectionHead:
    @pytest.fixture
    def head(self):
        return ProjectionHead(128, 128, 64)

    def test_output_shape(self, head):
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, 64)

    def test_l2_normalized(self, head):
        x = torch.randn(4, 128)
        out = head(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_gradient_flows(self, head):
        x = torch.randn(4, 128)
        out = head(x)
        out.sum().backward()
        assert any(p.grad is not None for p in head.parameters())


class TestSupConLoss:
    def test_positive_scalar(self):
        embeddings = F.normalize(torch.randn(4, 64), dim=1)
        labels = torch.tensor([0, 0, 1, 1])
        loss = supervised_contrastive_loss(embeddings, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_same_class_lower(self):
        v0 = F.normalize(torch.randn(1, 64), dim=1)
        v1 = F.normalize(torch.randn(1, 64), dim=1)
        embeddings_same = torch.cat([v0.expand(2, -1), v1.expand(2, -1)], dim=0)
        labels = torch.tensor([0, 0, 1, 1])
        loss_same = supervised_contrastive_loss(embeddings_same, labels)

        embeddings_rand = F.normalize(torch.randn(4, 64), dim=1)
        loss_rand = supervised_contrastive_loss(embeddings_rand, labels)

        assert loss_same.item() < loss_rand.item()

    def test_temperature_effect(self):
        torch.manual_seed(42)
        embeddings = F.normalize(torch.randn(8, 64), dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss_low_t = supervised_contrastive_loss(embeddings, labels, temperature=0.05)
        loss_high_t = supervised_contrastive_loss(embeddings, labels, temperature=2.0)
        assert loss_low_t.item() > loss_high_t.item()

    def test_single_class_no_crash(self):
        embeddings = F.normalize(torch.randn(4, 64), dim=1)
        labels = torch.tensor([0, 0, 0, 0])
        loss = supervised_contrastive_loss(embeddings, labels)
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_gradient_flows(self):
        raw = torch.randn(4, 64, requires_grad=True)
        embeddings = F.normalize(raw, dim=1)
        labels = torch.tensor([0, 0, 1, 1])
        loss = supervised_contrastive_loss(embeddings, labels)
        loss.backward()
        assert raw.grad is not None


class TestStyleEncoderWithProjection:
    @pytest.fixture
    def encoder(self):
        return StyleEncoder(input_dim=3, hidden_dim=64, style_dim=128)

    def test_forward_default_unchanged(self, encoder):
        x = torch.randn(4, 20, 3)
        out = encoder(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 128)

    def test_forward_with_projection(self, encoder):
        encoder.enable_projection_head(128, 64)
        x = torch.randn(4, 20, 3)
        style, z = encoder(x, return_projection=True)
        assert style.shape == (4, 128)
        assert z.shape == (4, 64)

    def test_projection_head_none_returns_style_only(self, encoder):
        x = torch.randn(4, 20, 3)
        style, z = encoder(x, return_projection=True)
        assert style.shape == (4, 128)
        assert z is None
