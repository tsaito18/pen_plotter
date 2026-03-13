import pytest

torch = pytest.importorskip("torch")

from src.model.stroke_model import StrokeGenerator, mdn_loss


class TestStrokeGenerator:
    @pytest.fixture
    def model(self):
        return StrokeGenerator(
            input_dim=3,
            hidden_dim=128,
            style_dim=128,
            num_mixtures=5,
        )

    def test_forward_shape(self, model):
        batch, seq_len = 4, 20
        x = torch.randn(batch, seq_len, 3)
        style = torch.randn(batch, 128)
        output = model(x, style)
        # output は MDN パラメータ: pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen_logit
        # pi: (batch, seq_len, num_mixtures)
        # mu: (batch, seq_len, num_mixtures) x2
        # sigma: (batch, seq_len, num_mixtures) x2
        # rho: (batch, seq_len, num_mixtures)
        # pen_logit: (batch, seq_len, 1)
        assert "pi" in output
        assert "mu_x" in output
        assert "mu_y" in output
        assert "sigma_x" in output
        assert "sigma_y" in output
        assert "rho" in output
        assert "pen_logit" in output
        assert output["pi"].shape == (batch, seq_len, 5)
        assert output["pen_logit"].shape == (batch, seq_len, 1)

    def test_pi_sums_to_one(self, model):
        model.eval()
        x = torch.randn(2, 10, 3)
        style = torch.randn(2, 128)
        output = model(x, style)
        pi_sum = output["pi"].sum(dim=-1)
        assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5)

    def test_sigma_positive(self, model):
        model.eval()
        x = torch.randn(2, 10, 3)
        style = torch.randn(2, 128)
        output = model(x, style)
        assert (output["sigma_x"] > 0).all()
        assert (output["sigma_y"] > 0).all()

    def test_rho_bounded(self, model):
        model.eval()
        x = torch.randn(2, 10, 3)
        style = torch.randn(2, 128)
        output = model(x, style)
        assert (output["rho"] > -1).all()
        assert (output["rho"] < 1).all()

    def test_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_default_construction(self):
        model = StrokeGenerator()
        x = torch.randn(1, 5, 3)
        style = torch.randn(1, 128)
        output = model(x, style)
        assert output["pi"].shape[0] == 1


class TestMDNLoss:
    def test_loss_is_scalar(self):
        batch, seq_len, n_mix = 4, 10, 5
        output = {
            "pi": torch.softmax(torch.randn(batch, seq_len, n_mix), dim=-1),
            "mu_x": torch.randn(batch, seq_len, n_mix),
            "mu_y": torch.randn(batch, seq_len, n_mix),
            "sigma_x": torch.exp(torch.randn(batch, seq_len, n_mix)),
            "sigma_y": torch.exp(torch.randn(batch, seq_len, n_mix)),
            "rho": torch.tanh(torch.randn(batch, seq_len, n_mix)),
            "pen_logit": torch.randn(batch, seq_len, 1),
        }
        target = torch.randn(batch, seq_len, 3)  # dx, dy, pen_state
        loss = mdn_loss(output, target)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_loss_is_finite(self):
        batch, seq_len, n_mix = 2, 5, 3
        output = {
            "pi": torch.softmax(torch.randn(batch, seq_len, n_mix), dim=-1),
            "mu_x": torch.randn(batch, seq_len, n_mix),
            "mu_y": torch.randn(batch, seq_len, n_mix),
            "sigma_x": torch.exp(torch.randn(batch, seq_len, n_mix)).clamp(min=1e-4),
            "sigma_y": torch.exp(torch.randn(batch, seq_len, n_mix)).clamp(min=1e-4),
            "rho": torch.tanh(torch.randn(batch, seq_len, n_mix)) * 0.9,
            "pen_logit": torch.randn(batch, seq_len, 1),
        }
        target = torch.randn(batch, seq_len, 3)
        loss = mdn_loss(output, target)
        assert torch.isfinite(loss)
