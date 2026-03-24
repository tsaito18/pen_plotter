"""Tests for StrokeDeformer module."""

import pytest

torch = pytest.importorskip("torch")

import math

from src.model.stroke_deformer import (
    AffineStrokeDeformer,
    StrokeDeformer,
    affine_deformation_loss,
    compute_local_curvature,
    deformation_loss,
    smoothness_loss,
)


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


class TestComputeLocalCurvature:
    """Tests for the compute_local_curvature helper function."""

    def test_output_shape(self) -> None:
        points = torch.randn(4, 32, 2)
        curv = compute_local_curvature(points)
        assert curv.shape == (4, 32, 1)

    def test_straight_line_low_curvature(self) -> None:
        """Collinear points should have minimal curvature (sigmoid baseline ~0.12)."""
        # batch=1, 5 points along y=x
        pts = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
        curv = compute_local_curvature(pts)
        assert curv.shape == (1, 5, 1)
        # sigmoid(0*4 - 2) ≈ 0.119: straight lines produce the sigmoid floor value
        baseline = torch.sigmoid(torch.tensor(-2.0)).item()
        assert abs(curv[0, 1, 0].item() - baseline) < 0.01
        assert abs(curv[0, 2, 0].item() - baseline) < 0.01
        assert abs(curv[0, 3, 0].item() - baseline) < 0.01

    def test_right_angle_high_curvature(self) -> None:
        """A sharp 90-degree bend should produce high curvature."""
        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
        curv = compute_local_curvature(pts)
        assert curv[0, 1, 0].item() > 0.3

    def test_boundary_copies_neighbor(self) -> None:
        """First and last points should copy curvature from their neighbor."""
        pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]])
        curv = compute_local_curvature(pts)
        assert torch.allclose(curv[0, 0], curv[0, 1], atol=1e-6)
        assert torch.allclose(curv[0, -1], curv[0, -2], atol=1e-6)

    def test_values_in_zero_one(self) -> None:
        """Curvature values should be in [0, 1] after sigmoid normalization."""
        pts = torch.randn(8, 32, 2)
        curv = compute_local_curvature(pts)
        assert (curv >= 0.0).all()
        assert (curv <= 1.0).all()

    def test_differentiable(self) -> None:
        """Curvature computation must be differentiable for backprop."""
        pts = torch.randn(2, 10, 2, requires_grad=True)
        curv = compute_local_curvature(pts)
        curv.sum().backward()
        assert pts.grad is not None
        assert pts.grad.shape == pts.shape

    def test_two_points(self) -> None:
        """Minimum viable input: 2 points should not crash."""
        pts = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        curv = compute_local_curvature(pts)
        assert curv.shape == (1, 2, 1)


class TestStrokeDeformerWithCurvature:
    """Tests that StrokeDeformer uses curvature in its input features."""

    def test_input_dim_includes_curvature(self) -> None:
        """input_dim should be 2 (xy) + 1 (t) + 1 (curvature) + style_dim + stroke_embed_dim."""
        model = StrokeDeformer(style_dim=64, hidden_dim=128, stroke_embed_dim=16)
        first_layer = model.mlp[0]
        expected_input_dim = 2 + 1 + 1 + 64 + 16  # 84
        assert first_layer.in_features == expected_input_dim

    def test_forward_still_works(self) -> None:
        """Forward pass with curvature feature should produce correct output shape."""
        model = StrokeDeformer(style_dim=64, hidden_dim=128)
        ref = torch.randn(4, 32, 2)
        style = torch.randn(4, 64)
        out = model(ref, style)
        assert out.shape == (4, 32, 2)

    def test_curvature_affects_output(self) -> None:
        """Different curvature patterns (straight vs curved) should yield different outputs."""
        model = StrokeDeformer(style_dim=64, hidden_dim=128)
        style = torch.randn(1, 64)

        # Straight line
        t = torch.linspace(0, 1, 32).unsqueeze(0)
        straight = torch.stack([t, t], dim=-1).squeeze(0).unsqueeze(0)

        # Curved (sine wave)
        curved = torch.stack([t, torch.sin(t * 2 * math.pi)], dim=-1).squeeze(0).unsqueeze(0)

        out_straight = model(straight, style)
        out_curved = model(curved, style)

        # Outputs should differ since curvature features differ
        assert not torch.allclose(out_straight, out_curved, atol=1e-5)


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


class TestSmoothnessLoss:
    def test_smoothness_loss_constant_offsets(self) -> None:
        """All points have the same offset -> loss = 0."""
        offsets = torch.ones(4, 32, 2) * 0.5
        loss = smoothness_loss(offsets)
        assert loss.item() < 1e-6

    def test_smoothness_loss_varying_offsets(self) -> None:
        """Different offsets per point -> loss > 0."""
        offsets = torch.randn(4, 32, 2)
        loss = smoothness_loss(offsets)
        assert loss.item() > 0

    def test_smoothness_loss_with_mask(self) -> None:
        """Masked points should be excluded from loss computation."""
        offsets = torch.randn(4, 32, 2)
        mask = torch.ones(4, 32)
        mask[:, 16:] = 0.0
        loss_masked = smoothness_loss(offsets, mask=mask)
        assert loss_masked.shape == ()
        assert loss_masked.item() > 0


class TestAffineStrokeDeformer:
    def test_output_shape(self) -> None:
        model = AffineStrokeDeformer(style_dim=64, hidden_dim=32)
        ref = torch.randn(4, 32, 2)
        style = torch.randn(4, 64)
        transformed, params = model(ref, style)
        assert transformed.shape == (4, 32, 2)
        assert params.shape == (4, 6)

    def test_identity_init(self) -> None:
        """With zero-initialized output layer, output should approximate input."""
        model = AffineStrokeDeformer(style_dim=64, hidden_dim=32)
        ref = torch.randn(4, 16, 2)
        style = torch.zeros(4, 64)
        transformed, params = model(ref, style)
        assert torch.allclose(transformed, ref, atol=1e-4)

    def test_with_stroke_index(self) -> None:
        model = AffineStrokeDeformer(style_dim=64, hidden_dim=32)
        ref = torch.randn(4, 32, 2)
        style = torch.randn(4, 64)
        stroke_idx = torch.tensor([0, 1, 2, 3])
        transformed, params = model(ref, style, stroke_idx)
        assert transformed.shape == (4, 32, 2)
        assert params.shape == (4, 6)

    def test_without_stroke_index(self) -> None:
        model = AffineStrokeDeformer(style_dim=64, hidden_dim=32)
        ref = torch.randn(2, 16, 2)
        style = torch.randn(2, 64)
        transformed, params = model(ref, style, stroke_index=None)
        assert transformed.shape == (2, 16, 2)

    def test_stroke_index_clamped(self) -> None:
        model = AffineStrokeDeformer(style_dim=64, hidden_dim=32)
        ref = torch.randn(2, 8, 2)
        style = torch.randn(2, 64)
        stroke_idx = torch.tensor([100, 200])
        transformed, params = model(ref, style, stroke_idx)
        assert transformed.shape == (2, 8, 2)


class TestAffineDeformationLoss:
    def test_zero_loss(self) -> None:
        target = torch.randn(4, 32, 2)
        loss = affine_deformation_loss(target.clone(), target)
        assert loss.item() < 1e-6

    def test_nonzero(self) -> None:
        transformed = torch.randn(4, 32, 2)
        target = torch.randn(4, 32, 2)
        loss = affine_deformation_loss(transformed, target)
        assert loss.item() > 0
