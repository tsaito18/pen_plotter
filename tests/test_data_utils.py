"""Tests for stroke data preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model.data_utils import (
    compute_normalization_stats,
    compute_reference_stats,
    denormalize_point,
    normalize_deltas,
    normalize_reference,
    reference_to_sequence,
    reference_to_sequence_from_arrays,
    strokes_to_deltas,
    strokes_to_deltas_from_arrays,
)


class TestStrokesToDeltas:
    def test_basic_two_strokes(self) -> None:
        strokes = [
            [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}],
            [{"x": 10, "y": 10}, {"x": 12, "y": 11}],
        ]
        result = strokes_to_deltas(strokes)

        assert result.shape == (5, 3)
        assert result.dtype == torch.float32

        expected = torch.tensor(
            [
                [0, 0, 0],  # first point
                [2, 2, 0],  # (3-1, 4-2)
                [2, 2, 1],  # (5-3, 6-4), end of stroke 0
                [5, 4, 0],  # (10-5, 10-6), start of stroke 1
                [2, 1, 1],  # (12-10, 11-10), end of stroke 1
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result, expected)

    def test_single_stroke(self) -> None:
        strokes = [
            [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 2, "y": 3}],
        ]
        result = strokes_to_deltas(strokes)

        assert result.shape == (3, 3)
        assert result[-1, 2] == 1  # last point pen_state=1
        assert result[0, 2] == 0
        assert result[1, 2] == 0

    def test_pen_state_at_stroke_boundaries(self) -> None:
        strokes = [
            [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
            [{"x": 5, "y": 5}, {"x": 6, "y": 6}],
            [{"x": 10, "y": 10}, {"x": 11, "y": 11}],
        ]
        result = strokes_to_deltas(strokes)

        # pen_state should be 1 at indices 1, 3, 5 (last point of each stroke)
        assert result.shape == (6, 3)
        pen_states = result[:, 2].tolist()
        assert pen_states == [0, 1, 0, 1, 0, 1]

    def test_single_point_stroke(self) -> None:
        strokes = [
            [{"x": 5, "y": 5}],
        ]
        result = strokes_to_deltas(strokes)

        assert result.shape == (1, 3)
        torch.testing.assert_close(
            result, torch.tensor([[0, 0, 1]], dtype=torch.float32)
        )

    def test_with_pressure_ignored(self) -> None:
        strokes = [
            [{"x": 0, "y": 0, "pressure": 0.8}, {"x": 1, "y": 1, "pressure": 0.5}],
        ]
        result = strokes_to_deltas(strokes)
        assert result.shape == (2, 3)


class TestStrokesToDeltasFromArrays:
    def test_basic(self) -> None:
        strokes = [
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
            np.array([[10, 10], [12, 11]], dtype=np.float32),
        ]
        result = strokes_to_deltas_from_arrays(strokes)

        expected = torch.tensor(
            [
                [0, 0, 0],
                [2, 2, 0],
                [2, 2, 1],
                [5, 4, 0],
                [2, 1, 1],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result, expected)

    def test_single_stroke(self) -> None:
        strokes = [np.array([[0, 0], [3, 4]], dtype=np.float32)]
        result = strokes_to_deltas_from_arrays(strokes)

        assert result.shape == (2, 3)
        assert result[-1, 2] == 1
        torch.testing.assert_close(result[1, :2], torch.tensor([3.0, 4.0]))

    def test_pen_state_matches_dict_version(self) -> None:
        dict_strokes = [
            [{"x": 0, "y": 0}, {"x": 1, "y": 2}],
            [{"x": 5, "y": 5}, {"x": 7, "y": 8}],
        ]
        array_strokes = [
            np.array([[0, 0], [1, 2]], dtype=np.float32),
            np.array([[5, 5], [7, 8]], dtype=np.float32),
        ]
        result_dict = strokes_to_deltas(dict_strokes)
        result_array = strokes_to_deltas_from_arrays(array_strokes)
        torch.testing.assert_close(result_dict, result_array)


class TestNormalization:
    def test_compute_stats(self) -> None:
        t1 = torch.tensor([[1.0, 2.0, 0], [3.0, 4.0, 1]])
        t2 = torch.tensor([[5.0, 6.0, 0], [7.0, 8.0, 1]])
        stats = compute_normalization_stats([t1, t2])

        all_dx = torch.tensor([1.0, 3.0, 5.0, 7.0])
        all_dy = torch.tensor([2.0, 4.0, 6.0, 8.0])
        assert stats["mean_x"] == pytest.approx(all_dx.mean().item())
        assert stats["mean_y"] == pytest.approx(all_dy.mean().item())
        assert stats["std_x"] == pytest.approx(all_dx.std().item())
        assert stats["std_y"] == pytest.approx(all_dy.std().item())

    def test_std_clamped(self) -> None:
        t = torch.tensor([[5.0, 5.0, 0]])
        stats = compute_normalization_stats([t])
        assert stats["std_x"] >= 1e-6
        assert stats["std_y"] >= 1e-6

    def test_normalize_denormalize_roundtrip(self) -> None:
        t = torch.tensor(
            [[1.0, 2.0, 0], [3.0, 4.0, 1], [-1.0, 5.0, 0]], dtype=torch.float32
        )
        stats = compute_normalization_stats([t])

        normalized = normalize_deltas(t, stats)
        for i in range(t.shape[0]):
            dx, dy = denormalize_point(
                normalized[i, 0].item(), normalized[i, 1].item(), stats
            )
            assert dx == pytest.approx(t[i, 0].item(), abs=1e-5)
            assert dy == pytest.approx(t[i, 1].item(), abs=1e-5)

    def test_normalize_preserves_pen_state(self) -> None:
        t = torch.tensor(
            [[1.0, 2.0, 0], [3.0, 4.0, 1], [5.0, 6.0, 0]], dtype=torch.float32
        )
        stats = compute_normalization_stats([t])
        normalized = normalize_deltas(t, stats)

        torch.testing.assert_close(normalized[:, 2], t[:, 2])

    def test_normalize_batched(self) -> None:
        t = torch.randn(4, 10, 3)
        t[:, :, 2] = (torch.rand(4, 10) > 0.5).float()
        stats = compute_normalization_stats([t[i] for i in range(4)])

        normalized = normalize_deltas(t, stats)
        assert normalized.shape == t.shape
        torch.testing.assert_close(normalized[:, :, 2], t[:, :, 2])


class TestReferenceToSequence:
    def test_reference_to_sequence(self) -> None:
        strokes = [
            [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
            [{"x": 2.0, "y": 2.0}, {"x": 3.0, "y": 3.0}],
        ]
        result = reference_to_sequence(strokes)
        # 2 points + separator + 2 points = 5
        assert result.shape == (5, 2)
        assert result.dtype == torch.float32
        # separator at index 2
        torch.testing.assert_close(result[2], torch.tensor([-1.0, -1.0]))
        torch.testing.assert_close(result[0], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(result[4], torch.tensor([3.0, 3.0]))

    def test_reference_to_sequence_from_arrays(self) -> None:
        strokes = [
            np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        ]
        result = reference_to_sequence_from_arrays(strokes)
        assert result.shape == (5, 2)
        torch.testing.assert_close(result[2], torch.tensor([-1.0, -1.0]))

    def test_reference_to_sequence_matches_char_encoder(self) -> None:
        from src.model.char_encoder import CharEncoder

        arr_strokes = [
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
            np.array([[5.0, 5.0], [6.0, 6.0]], dtype=np.float64),
        ]
        expected = CharEncoder.strokes_to_sequence(arr_strokes)
        expected_tensor = torch.from_numpy(expected.astype(np.float32))

        dict_strokes = [
            [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 2.0}],
            [{"x": 5.0, "y": 5.0}, {"x": 6.0, "y": 6.0}],
        ]
        result = reference_to_sequence(dict_strokes)
        torch.testing.assert_close(result, expected_tensor)

    def test_reference_to_sequence_single_point_stroke_skipped(self) -> None:
        strokes = [
            [{"x": 0.0, "y": 0.0}],  # single point, skipped
            [{"x": 2.0, "y": 2.0}, {"x": 3.0, "y": 3.0}],
        ]
        result = reference_to_sequence(strokes)
        assert result.shape == (2, 2)


class TestReferenceNormalization:
    def test_compute_reference_stats(self) -> None:
        t1 = torch.tensor([[1.0, 2.0], [-1.0, -1.0], [3.0, 4.0]])
        t2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        stats = compute_reference_stats([t1, t2])

        valid_x = torch.tensor([1.0, 3.0, 5.0, 7.0])
        valid_y = torch.tensor([2.0, 4.0, 6.0, 8.0])
        assert stats["mean_x"] == pytest.approx(valid_x.mean().item())
        assert stats["mean_y"] == pytest.approx(valid_y.mean().item())
        assert stats["std_x"] == pytest.approx(valid_x.std().item())
        assert stats["std_y"] == pytest.approx(valid_y.std().item())

    def test_normalize_reference(self) -> None:
        t = torch.tensor([[1.0, 2.0], [-1.0, -1.0], [3.0, 4.0]])
        stats = compute_reference_stats([t])
        normalized = normalize_reference(t, stats)

        assert normalized.shape == t.shape
        # separator becomes (0, 0)
        torch.testing.assert_close(normalized[1], torch.tensor([0.0, 0.0]))
        # non-separator points are normalized
        assert normalized[0, 0].item() != t[0, 0].item()

    def test_normalize_reference_batch(self) -> None:
        batch = torch.tensor([
            [[1.0, 2.0], [-1.0, -1.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0], [-1.0, -1.0]],
        ])
        stats = compute_reference_stats([batch[0], batch[1]])
        normalized = normalize_reference(batch, stats)

        assert normalized.shape == batch.shape
        # separators become (0, 0)
        torch.testing.assert_close(normalized[0, 1], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(normalized[1, 2], torch.tensor([0.0, 0.0]))
