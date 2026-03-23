"""Tests for StrokeDeformer module."""

import pytest

torch = pytest.importorskip("torch")

from src.model.stroke_deformer import StrokeDeformer, deformation_loss


class TestStrokeDeformer:
    def test_output_shape(self) -> None:
        model = StrokeDeformer(style_dim=64, hidden_dim=128)
        ref = torch.randn(4, 32, 2)
        style = torch.randn(4, 64)
        out = model(ref, style)
        assert out.shape == (4, 32, 2)

    def test_with_stroke_index(self) -> None:
        model = StrokeDeformer(style_dim=64, hidden_dim=128)
        ref = torch.randn(4, 32, 2)
        style = torch.randn(4, 64)
        stroke_idx = torch.tensor([0, 1, 2, 3])
        out = model(ref, style, stroke_idx)
        assert out.shape == (4, 32, 2)

    def test_without_stroke_index(self) -> None:
        model = StrokeDeformer(style_dim=64, hidden_dim=128)
        ref = torch.randn(2, 16, 2)
        style = torch.randn(2, 64)
        out = model(ref, style, stroke_index=None)
        assert out.shape == (2, 16, 2)

    def test_stroke_index_clamped(self) -> None:
        model = StrokeDeformer(style_dim=64, hidden_dim=128)
        ref = torch.randn(2, 8, 2)
        style = torch.randn(2, 64)
        stroke_idx = torch.tensor([100, 200])
        out = model(ref, style, stroke_idx)
        assert out.shape == (2, 8, 2)


class TestDeformationLoss:
    def test_basic(self) -> None:
        pred = torch.randn(4, 32, 2)
        target = torch.randn(4, 32, 2)
        loss = deformation_loss(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_with_mask(self) -> None:
        pred = torch.randn(4, 32, 2)
        target = torch.randn(4, 32, 2)
        mask = torch.ones(4, 32, dtype=torch.bool)
        mask[:, 16:] = False
        loss = deformation_loss(pred, target, mask=mask)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_perfect_prediction(self) -> None:
        target = torch.randn(4, 32, 2)
        loss = deformation_loss(target.clone(), target)
        assert loss.item() < 1e-6
