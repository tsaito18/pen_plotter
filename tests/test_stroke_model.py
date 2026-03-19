import pytest

torch = pytest.importorskip("torch")

from src.model.stroke_model import StrokeGenerator, embedding_variance_loss, mdn_loss


class TestStrokeGenerator:
    @pytest.fixture
    def model(self):
        return StrokeGenerator(
            input_dim=2,
            hidden_dim=128,
            style_dim=128,
            num_mixtures=5,
        )

    def test_forward_shape(self, model):
        batch, seq_len = 4, 20
        x = torch.randn(batch, seq_len, 2)
        style = torch.randn(batch, 128)
        stroke_index = torch.zeros(batch, dtype=torch.long)
        output = model(x, style, stroke_index=stroke_index)
        assert "pi" in output
        assert "mu_x" in output
        assert "mu_y" in output
        assert "sigma_x" in output
        assert "sigma_y" in output
        assert "rho" in output
        assert "eos_logit" in output
        assert output["pi"].shape == (batch, seq_len, 5)
        assert output["eos_logit"].shape == (batch, seq_len, 1)

    def test_pi_sums_to_one(self, model):
        model.eval()
        x = torch.randn(2, 10, 2)
        style = torch.randn(2, 128)
        output = model(x, style)
        pi_sum = output["pi"].sum(dim=-1)
        assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5)

    def test_sigma_positive(self, model):
        model.eval()
        x = torch.randn(2, 10, 2)
        style = torch.randn(2, 128)
        output = model(x, style)
        assert (output["sigma_x"] > 0).all()
        assert (output["sigma_y"] > 0).all()

    def test_rho_bounded(self, model):
        model.eval()
        x = torch.randn(2, 10, 2)
        style = torch.randn(2, 128)
        output = model(x, style)
        assert (output["rho"] > -1).all()
        assert (output["rho"] < 1).all()

    def test_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_default_construction(self):
        model = StrokeGenerator()
        x = torch.randn(1, 5, 2)
        style = torch.randn(1, 128)
        output = model(x, style)
        assert output["pi"].shape == (1, 5, 20)

    def test_pi_logits_in_output(self, model):
        x = torch.randn(2, 10, 2)
        style = torch.randn(2, 128)
        output = model(x, style)
        assert "pi_logits" in output
        assert output["pi_logits"].shape == (2, 10, 5)

    def test_rho_clamped(self, model):
        model.eval()
        x = torch.randn(2, 10, 2)
        style = torch.randn(2, 128)
        output = model(x, style)
        assert (output["rho"] >= -0.95).all()
        assert (output["rho"] <= 0.95).all()

    def test_stroke_index_affects_output(self, model):
        model.eval()
        x = torch.randn(1, 10, 2)
        style = torch.randn(1, 128)
        out_a = model(x, style, stroke_index=torch.tensor([0]))
        out_b = model(x, style, stroke_index=torch.tensor([5]))
        assert not torch.allclose(out_a["mu_x"], out_b["mu_x"], atol=1e-6)


class TestMDNLoss:
    def test_loss_returns_tuple(self):
        batch, seq_len, n_mix = 4, 10, 5
        pi_logits = torch.randn(batch, seq_len, n_mix)
        output = {
            "pi": torch.softmax(pi_logits, dim=-1),
            "pi_logits": pi_logits,
            "mu_x": torch.randn(batch, seq_len, n_mix),
            "mu_y": torch.randn(batch, seq_len, n_mix),
            "sigma_x": torch.exp(torch.randn(batch, seq_len, n_mix)),
            "sigma_y": torch.exp(torch.randn(batch, seq_len, n_mix)),
            "rho": torch.tanh(torch.randn(batch, seq_len, n_mix)),
            "eos_logit": torch.randn(batch, seq_len, 1),
        }
        target_xy = torch.randn(batch, seq_len, 2)
        target_eos = torch.zeros(batch, seq_len, 1)
        stroke_loss, eos_loss = mdn_loss(output, target_xy, target_eos)
        assert stroke_loss.ndim == 0
        assert eos_loss.ndim == 0
        assert torch.isfinite(stroke_loss)
        assert torch.isfinite(eos_loss)

    def test_loss_is_finite(self):
        batch, seq_len, n_mix = 2, 5, 3
        pi_logits = torch.randn(batch, seq_len, n_mix)
        output = {
            "pi": torch.softmax(pi_logits, dim=-1),
            "pi_logits": pi_logits,
            "mu_x": torch.randn(batch, seq_len, n_mix),
            "mu_y": torch.randn(batch, seq_len, n_mix),
            "sigma_x": torch.exp(torch.randn(batch, seq_len, n_mix)).clamp(min=1e-4),
            "sigma_y": torch.exp(torch.randn(batch, seq_len, n_mix)).clamp(min=1e-4),
            "rho": torch.tanh(torch.randn(batch, seq_len, n_mix)) * 0.9,
            "eos_logit": torch.randn(batch, seq_len, 1),
        }
        target_xy = torch.randn(batch, seq_len, 2)
        target_eos = torch.zeros(batch, seq_len, 1)
        stroke_loss, eos_loss = mdn_loss(output, target_xy, target_eos)
        assert torch.isfinite(stroke_loss)
        assert torch.isfinite(eos_loss)

    def test_eos_loss_weighted(self):
        """EOS（正例）の誤分類がnon-EOSより高い損失を生むことを確認。"""
        batch, seq_len, n_mix = 2, 5, 3
        pi_logits = torch.randn(batch, seq_len, n_mix)
        base_output = {
            "pi": torch.softmax(pi_logits, dim=-1),
            "pi_logits": pi_logits,
            "mu_x": torch.zeros(batch, seq_len, n_mix),
            "mu_y": torch.zeros(batch, seq_len, n_mix),
            "sigma_x": torch.ones(batch, seq_len, n_mix),
            "sigma_y": torch.ones(batch, seq_len, n_mix),
            "rho": torch.zeros(batch, seq_len, n_mix),
            "eos_logit": torch.full((batch, seq_len, 1), -5.0),
        }
        target_xy = torch.zeros(batch, seq_len, 2)
        target_eos_positive = torch.ones(batch, seq_len, 1)
        target_eos_negative = torch.zeros(batch, seq_len, 1)

        _, eos_loss_positive = mdn_loss(base_output, target_xy, target_eos_positive)
        _, eos_loss_negative = mdn_loss(base_output, target_xy, target_eos_negative)
        assert eos_loss_positive > eos_loss_negative


class TestStrokeGeneratorCharEmbedding:
    """char_embedding 対応のテスト。"""

    def test_char_dim_zero_backward_compatible(self):
        model = StrokeGenerator(
            input_dim=2, hidden_dim=128, style_dim=128, char_dim=0, num_mixtures=5
        )
        x = torch.randn(4, 20, 2)
        style = torch.randn(4, 128)
        output = model(x, style)
        assert output["pi"].shape == (4, 20, 5)
        assert output["eos_logit"].shape == (4, 20, 1)

    def test_char_dim_nonzero_output_shape(self):
        model = StrokeGenerator(
            input_dim=2, hidden_dim=128, style_dim=128, char_dim=128, num_mixtures=5
        )
        x = torch.randn(4, 20, 2)
        style = torch.randn(4, 128)
        char_emb = torch.randn(4, 128)
        output = model(x, style, char_embedding=char_emb)
        assert output["pi"].shape == (4, 20, 5)
        assert output["mu_x"].shape == (4, 20, 5)
        assert output["eos_logit"].shape == (4, 20, 1)

    def test_char_dim_nonzero_requires_embedding(self):
        model = StrokeGenerator(
            input_dim=2, hidden_dim=128, style_dim=128, char_dim=128, num_mixtures=5
        )
        x = torch.randn(2, 10, 2)
        style = torch.randn(2, 128)
        with pytest.raises(ValueError, match="char_embedding required"):
            model(x, style)

    def test_char_dim_zero_ignores_embedding(self):
        model = StrokeGenerator(
            input_dim=2, hidden_dim=128, style_dim=128, char_dim=0, num_mixtures=5
        )
        x = torch.randn(2, 10, 2)
        style = torch.randn(2, 128)
        char_emb = torch.randn(2, 128)
        output = model(x, style, char_embedding=char_emb)
        assert output["pi"].shape == (2, 10, 5)

    def test_loss_with_char_embedding(self):
        model = StrokeGenerator(
            input_dim=2, hidden_dim=128, style_dim=128, char_dim=128, num_mixtures=5
        )
        x = torch.randn(4, 20, 2)
        style = torch.randn(4, 128)
        char_emb = torch.randn(4, 128)
        target_xy = torch.randn(4, 20, 2)
        target_eos = torch.zeros(4, 20, 1)
        output = model(x, style, char_embedding=char_emb)
        stroke_loss, eos_loss = mdn_loss(output, target_xy, target_eos)
        assert stroke_loss.ndim == 0
        assert eos_loss.ndim == 0
        assert torch.isfinite(stroke_loss)
        assert torch.isfinite(eos_loss)

    def test_char_embedding_affects_output(self):
        """Different char_embeddings should produce different outputs."""
        model = StrokeGenerator(
            input_dim=2, hidden_dim=128, style_dim=128, char_dim=128, num_mixtures=5
        )
        model.eval()
        x = torch.randn(1, 10, 2)
        style = torch.randn(1, 128)
        char_emb_a = torch.randn(1, 128)
        char_emb_b = torch.randn(1, 128)
        out_a = model(x, style, char_embedding=char_emb_a)
        out_b = model(x, style, char_embedding=char_emb_b)
        assert not torch.allclose(out_a["mu_x"], out_b["mu_x"], atol=1e-6)


class TestEmbeddingVarianceLoss:
    def test_constant_embeddings(self):
        embeddings = torch.ones(8, 64)
        loss = embedding_variance_loss(embeddings)
        assert loss.item() > 0

    def test_diverse_embeddings(self):
        embeddings = torch.randn(8, 64) * 2.0
        loss = embedding_variance_loss(embeddings)
        assert loss.item() == pytest.approx(0.0, abs=0.5)

    def test_single_sample(self):
        embeddings = torch.randn(1, 64)
        loss = embedding_variance_loss(embeddings)
        assert loss.item() == 0.0
