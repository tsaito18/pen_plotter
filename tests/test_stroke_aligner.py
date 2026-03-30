"""StrokeAligner テスト — ストロークアライメント（MHD + Hungarian）"""

import numpy as np
import pytest

from src.model.stroke_aligner import AlignmentResult, StrokeAligner


# ---------------------------------------------------------------------------
# テストデータヘルパー
# ---------------------------------------------------------------------------

def _horizontal() -> np.ndarray:
    """水平線: (0,5) → (10,5)"""
    return np.array([[i, 5.0] for i in range(11)], dtype=np.float32)


def _vertical() -> np.ndarray:
    """垂直線: (5,0) → (5,10)"""
    return np.array([[5.0, i] for i in range(11)], dtype=np.float32)


def _diagonal() -> np.ndarray:
    """対角線: (0,0) → (10,10)"""
    return np.array([[i, float(i)] for i in range(11)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Modified Hausdorff Distance
# ---------------------------------------------------------------------------

class TestModifiedHausdorffDistance:
    def setup_method(self):
        self.aligner = StrokeAligner()

    def test_identical_strokes(self):
        s = _horizontal()
        assert self.aligner._compute_mhd(s, s) == pytest.approx(0.0, abs=1e-6)

    def test_translated_stroke(self):
        a = np.array([[i, float(i)] for i in range(11)], dtype=np.float32)
        b = a + 5.0
        mhd = self.aligner._compute_mhd(a, b)
        expected = np.sqrt(5.0**2 + 5.0**2)
        assert mhd == pytest.approx(expected, rel=0.05)

    def test_symmetry(self):
        a = _horizontal()
        b = _diagonal()
        assert self.aligner._compute_mhd(a, b) == pytest.approx(
            self.aligner._compute_mhd(b, a), abs=1e-6
        )


# ---------------------------------------------------------------------------
# コスト行列
# ---------------------------------------------------------------------------

class TestCostMatrix:
    def setup_method(self):
        self.aligner = StrokeAligner()

    def test_square_perfect_match(self):
        strokes = [_horizontal(), _vertical(), _diagonal()]
        cost_matrix, _reversed = self.aligner._build_cost_matrix(strokes, strokes)
        assert cost_matrix.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert cost_matrix[i, j] < cost_matrix[i, (j + 1) % 3]

    def test_reversed_stroke_detected(self):
        ref = [_horizontal()]
        user = [_horizontal()[::-1].copy()]
        _cost_matrix, reversed_flags = self.aligner._build_cost_matrix(user, ref)
        assert reversed_flags.shape == (1, 1)
        assert reversed_flags[0, 0] is True or reversed_flags[0, 0] == True  # noqa: E712


# ---------------------------------------------------------------------------
# Hungarian 割り当て
# ---------------------------------------------------------------------------

class TestHungarianAssignment:
    def setup_method(self):
        self.aligner = StrokeAligner()

    def test_identity_assignment(self):
        strokes = [_horizontal(), _vertical(), _diagonal()]
        result = self.aligner.align(strokes, strokes)
        assert sorted(zip(result.user_indices, result.ref_indices)) == [(0, 0), (1, 1), (2, 2)]

    def test_reordered_strokes(self):
        ref = [_horizontal(), _vertical(), _diagonal()]
        user = [ref[2].copy(), ref[0].copy(), ref[1].copy()]  # [diagonal, horizontal, vertical]
        result = self.aligner.align(user, ref)
        pairs = dict(zip(result.user_indices, result.ref_indices))
        assert pairs[0] == 2  # user[0]=diagonal → ref[2]=diagonal
        assert pairs[1] == 0  # user[1]=horizontal → ref[0]=horizontal
        assert pairs[2] == 1  # user[2]=vertical → ref[1]=vertical

    def test_fewer_user_strokes(self):
        ref = [_horizontal(), _vertical(), _diagonal()]
        user = [_horizontal(), _vertical()]
        result = self.aligner.align(user, ref)
        assert len(result.user_indices) == 2
        assert len(result.ref_indices) == 2

    def test_more_user_strokes(self):
        ref = [_horizontal(), _vertical()]
        user = [_horizontal(), _vertical(), _diagonal()]
        result = self.aligner.align(user, ref)
        assert len(result.user_indices) == 2
        assert len(result.ref_indices) == 2


# ---------------------------------------------------------------------------
# 品質フィルタリング
# ---------------------------------------------------------------------------

class TestQualityFiltering:
    def test_high_cost_rejected(self):
        aligner = StrokeAligner(quality_threshold=0.1)
        user = [np.array([[0, 0], [1, 0]], dtype=np.float32)]
        ref = [np.array([[100, 100], [200, 200]], dtype=np.float32)]
        result = aligner.align(user, ref)
        assert 0 in result.rejected_indices

    def test_low_cost_accepted(self):
        aligner = StrokeAligner(quality_threshold=10.0)
        strokes = [_horizontal(), _vertical()]
        result = aligner.align(strokes, strokes)
        assert result.rejected_indices == []


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestFullAlignment:
    def setup_method(self):
        self.aligner = StrokeAligner()

    def test_end_to_end_simple(self):
        strokes = [_horizontal(), _vertical(), _diagonal()]
        result = self.aligner.align(strokes, strokes)
        assert isinstance(result, AlignmentResult)
        assert len(result.user_indices) == 3
        assert len(result.ref_indices) == 3
        assert result.total_cost == pytest.approx(0.0, abs=1e-3)

    def test_end_to_end_simple_with_merge_split_noop(self):
        """Merge/split phases should be no-op when counts match."""
        strokes = [_horizontal(), _vertical(), _diagonal()]
        result = self.aligner.align(strokes, strokes)
        assert isinstance(result, AlignmentResult)
        assert result.total_cost == pytest.approx(0.0, abs=1e-3)

    def test_end_to_end_reordered(self):
        ref = [_horizontal(), _vertical(), _diagonal()]
        user = [ref[1].copy(), ref[2].copy(), ref[0].copy()]  # [vertical, diagonal, horizontal]
        result = self.aligner.align(user, ref)
        assert isinstance(result, AlignmentResult)
        pairs = dict(zip(result.user_indices, result.ref_indices))
        assert pairs[0] == 1  # user[0]=vertical → ref[1]=vertical
        assert pairs[1] == 2  # user[1]=diagonal → ref[2]=diagonal
        assert pairs[2] == 0  # user[2]=horizontal → ref[0]=horizontal


# ---------------------------------------------------------------------------
# マージ検出（ユーザーが複数参照ストロークを1本で書いた場合）
# ---------------------------------------------------------------------------

class TestMergeDetection:
    def setup_method(self):
        self.aligner = StrokeAligner()

    def test_two_ref_merged_in_user(self):
        """1 user stroke (L-shape) should be split to match 2 ref strokes."""
        h = np.array([[i, 5.0] for i in range(11)], dtype=np.float32)
        v = np.array([[10.0, 5.0 + i] for i in range(11)], dtype=np.float32)
        merged = np.concatenate([h, v[1:]])

        result = self.aligner.align([merged], [h, v])
        assert len(result.ref_indices) == 2
        assert 0 in result.ref_indices
        assert 1 in result.ref_indices

    def test_no_merge_when_counts_match(self):
        """No merge detection when stroke counts already match."""
        strokes = [_horizontal(), _vertical()]
        result = self.aligner.align(strokes, strokes)
        assert len(result.user_indices) == 2
        assert len(result.ref_indices) == 2

    def test_split_point_from_speed(self):
        """Speed dip in timestamps identifies split candidate."""
        stroke = np.array([[i, 5.0] for i in range(21)], dtype=np.float32)
        timestamps = np.zeros(21)
        for i in range(1, 21):
            if 9 <= i <= 12:
                timestamps[i] = timestamps[i - 1] + 1.0
            else:
                timestamps[i] = timestamps[i - 1] + 0.1
        points = self.aligner._find_split_points(stroke, None, timestamps)
        assert len(points) >= 1
        assert any(9 <= p <= 12 for p in points)

    def test_merge_rejected_if_no_improvement(self):
        """Splitting rejected when neither part matches the distant ref."""
        aligner = StrokeAligner(quality_threshold=10.0)
        user = [np.array([[i, 0.0] for i in range(11)], dtype=np.float32)]
        ref = [
            np.array([[i, 0.0] for i in range(11)], dtype=np.float32),
            np.array([[i, 50.0] for i in range(11)], dtype=np.float32),
        ]
        result = aligner.align(user, ref)
        assert len(result.ref_indices) == 1
        assert 0 in result.ref_indices


# ---------------------------------------------------------------------------
# スプリット検出（ユーザーが1参照ストロークを複数に分けて書いた場合）
# ---------------------------------------------------------------------------

class TestSplitDetection:
    def test_user_split_one_stroke(self):
        """3 user strokes vs 2 ref — two halves should join to match one ref."""
        h = _horizontal()
        v = _vertical()
        h1 = h[:6].copy()
        h2 = h[5:].copy()

        aligner = StrokeAligner()
        result = aligner.align([h1, h2, v], [h, v])
        assert len(result.ref_indices) == 2
        assert 0 in result.ref_indices
        assert 1 in result.ref_indices


# ---------------------------------------------------------------------------
# 既存コード統合
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_compute_stroke_offsets_with_aligner(self):
        """Aligner-based pairing corrects reordered strokes."""
        from src.model.data_utils import compute_stroke_offsets

        h = _horizontal()
        v = _vertical()

        result = compute_stroke_offsets(
            hand_strokes=[v, h],
            ref_strokes=[h, v],
            aligner=StrokeAligner(),
        )
        assert len(result) == 2
        for _ref_r, offset in result:
            assert np.abs(offset).max() < 1.0

    def test_backward_compatible(self):
        """aligner=None preserves original index-based behavior."""
        from src.model.data_utils import compute_stroke_offsets

        h = _horizontal()
        v = _vertical()

        result_old = compute_stroke_offsets([h, v], [h, v])
        result_new = compute_stroke_offsets([h, v], [h, v], aligner=None)

        assert len(result_old) == len(result_new)
        for (r1, o1), (r2, o2) in zip(result_old, result_new):
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(o1, o2)
