import numpy as np
import pytest

from src.model.augmentation import AugmentConfig, HandwritingAugmenter


@pytest.fixture
def augmenter() -> HandwritingAugmenter:
    return HandwritingAugmenter(seed=42)


@pytest.fixture
def sample_stroke() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 0.0]],
        dtype=np.float64,
    )


class TestAugmentConfig:
    def test_augment_config_defaults(self):
        cfg = AugmentConfig()
        assert cfg.baseline_drift == 0.3
        assert cfg.size_variation == 0.05
        assert cfg.slant_variation == 0.02
        assert cfg.jitter_amplitude == 0.03
        assert cfg.spacing_variation == 0.2
        assert cfg.char_density_variation == 0.02
        assert cfg.enabled is True


class TestDisabled:
    def test_disabled_returns_unchanged(self, sample_stroke: np.ndarray):
        cfg = AugmentConfig(enabled=False)
        aug = HandwritingAugmenter(config=cfg, seed=42)

        result = aug.augment_page([sample_stroke])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], sample_stroke)

    def test_disabled_char_placement_unchanged(self):
        cfg = AugmentConfig(enabled=False)
        aug = HandwritingAugmenter(config=cfg, seed=42)

        x, y, fs = aug.augment_char_placement(10.0, 20.0, 5.0)
        assert x == 10.0
        assert y == 20.0
        assert fs == 5.0

    def test_disabled_slant_unchanged(self, sample_stroke: np.ndarray):
        cfg = AugmentConfig(enabled=False)
        aug = HandwritingAugmenter(config=cfg, seed=42)

        result = aug.apply_slant(sample_stroke, 2.0, 0.5)
        np.testing.assert_array_equal(result, sample_stroke)


class TestJitter:
    def test_jitter_changes_coordinates(
        self, augmenter: HandwritingAugmenter, sample_stroke: np.ndarray
    ):
        result = augmenter.augment_page([sample_stroke])
        assert not np.array_equal(result[0], sample_stroke)

    def test_jitter_preserves_shape(
        self, augmenter: HandwritingAugmenter, sample_stroke: np.ndarray
    ):
        result = augmenter.augment_page([sample_stroke])
        assert result[0].shape == sample_stroke.shape

    def test_jitter_small_magnitude(
        self, augmenter: HandwritingAugmenter, sample_stroke: np.ndarray
    ):
        result = augmenter.augment_page([sample_stroke])
        diff = np.abs(result[0] - sample_stroke)
        assert np.all(diff < 1.0), f"Jitter exceeded 1mm: max={diff.max():.4f}"


class TestCharPlacement:
    def test_augment_char_placement_varies(self, augmenter: HandwritingAugmenter):
        results = [augmenter.augment_char_placement(10.0, 20.0, 5.0) for _ in range(10)]
        xs = [r[0] for r in results]
        assert len(set(xs)) > 1, "All x values identical"

    def test_augment_char_placement_clamps_size(self):
        cfg = AugmentConfig(size_variation=1.0)
        aug = HandwritingAugmenter(config=cfg, seed=0)
        for _ in range(100):
            _, _, fs = aug.augment_char_placement(10.0, 20.0, 5.0)
            assert fs >= 5.0 * 0.8, f"Font size {fs} below 80% minimum"


class TestSlant:
    def test_apply_slant_preserves_shape(
        self, augmenter: HandwritingAugmenter, sample_stroke: np.ndarray
    ):
        result = augmenter.apply_slant(sample_stroke, 2.0, 0.5)
        assert result.shape == sample_stroke.shape

    def test_apply_slant_rotates(self, sample_stroke: np.ndarray):
        cfg = AugmentConfig(slant_variation=0.5)
        aug = HandwritingAugmenter(config=cfg, seed=42)
        result = aug.apply_slant(sample_stroke, 2.0, 0.5)
        assert not np.array_equal(result, sample_stroke)

    def test_get_char_slant_nonzero_when_enabled(self):
        aug = HandwritingAugmenter(AugmentConfig(slant_variation=0.05), seed=1)
        angles = [aug.get_char_slant() for _ in range(20)]
        assert any(a != 0.0 for a in angles)
        assert all(abs(a) < 0.5 for a in angles)  # 微小角

    def test_get_char_slant_zero_when_disabled(self):
        aug = HandwritingAugmenter(AugmentConfig(enabled=False), seed=1)
        assert aug.get_char_slant() == 0.0


class TestReproducibility:
    def test_seed_reproducibility(self, sample_stroke: np.ndarray):
        aug1 = HandwritingAugmenter(seed=123)
        aug2 = HandwritingAugmenter(seed=123)

        r1 = aug1.augment_page([sample_stroke])
        r2 = aug2.augment_page([sample_stroke])
        np.testing.assert_array_equal(r1[0], r2[0])


class TestAugmentPage:
    def test_augment_page_processes_all(self, augmenter: HandwritingAugmenter):
        strokes = [
            np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float64),
            np.array([[4.0, 4.0], [5.0, 5.0]], dtype=np.float64),
        ]
        result = augmenter.augment_page(strokes)
        assert len(result) == 3
        for original, augmented in zip(strokes, result):
            assert not np.array_equal(original, augmented)

    def test_empty_strokes(self, augmenter: HandwritingAugmenter):
        result = augmenter.augment_page([])
        assert result == []


class TestAugmentConfigLineDensity:
    def test_augment_config_line_density_variation_default(self):
        cfg = AugmentConfig()
        assert cfg.line_density_variation == 0.05


class TestLineDensityScale:
    def test_get_line_density_scale_returns_float(self):
        aug = HandwritingAugmenter(seed=42)
        result = aug.get_line_density_scale()
        assert isinstance(result, float)

    def test_get_line_density_scale_varies(self):
        aug = HandwritingAugmenter(seed=42)
        values = [aug.get_line_density_scale() for _ in range(20)]
        assert len(set(values)) >= 2, "Expected at least 2 distinct values in 20 calls"

    def test_get_line_density_scale_range(self):
        aug = HandwritingAugmenter(seed=42)
        for _ in range(100):
            scale = aug.get_line_density_scale()
            assert 0.85 <= scale <= 1.15, f"Scale {scale} outside expected range [0.85, 1.15]"

    def test_get_line_density_scale_disabled(self):
        cfg = AugmentConfig(enabled=False)
        aug = HandwritingAugmenter(config=cfg, seed=42)
        result = aug.get_line_density_scale()
        assert result == 1.0


class TestCharDensityScale:
    def test_get_char_density_scale_returns_float(self):
        aug = HandwritingAugmenter(seed=42)
        result = aug.get_char_density_scale()
        assert isinstance(result, float)

    def test_get_char_density_scale_varies(self):
        aug = HandwritingAugmenter(seed=42)
        values = [aug.get_char_density_scale() for _ in range(20)]
        assert len(set(values)) >= 2, "Expected at least 2 distinct values in 20 calls"

    def test_get_char_density_scale_range(self):
        aug = HandwritingAugmenter(seed=42)
        for _ in range(100):
            scale = aug.get_char_density_scale()
            assert 0.98 <= scale <= 1.02, f"Scale {scale} outside expected range [0.98, 1.02]"

    def test_get_char_density_scale_uses_char_variation(self):
        cfg = AugmentConfig(line_density_variation=0.0, char_density_variation=0.02)
        aug = HandwritingAugmenter(config=cfg, seed=42)
        values = [aug.get_char_density_scale() for _ in range(20)]
        assert len(set(values)) >= 2, "Char density should not use line variation"

    def test_get_char_density_scale_disabled(self):
        cfg = AugmentConfig(enabled=False)
        aug = HandwritingAugmenter(config=cfg, seed=42)
        result = aug.get_char_density_scale()
        assert result == 1.0


class TestElasticDistort:
    @pytest.fixture
    def stroke(self) -> np.ndarray:
        return np.array(
            [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 0.5], [4.0, 0.0]],
            dtype=np.float64,
        )

    def test_preserves_shape(self, stroke: np.ndarray):
        aug = HandwritingAugmenter(seed=42)
        result = aug.elastic_distort(stroke)
        assert result.shape == stroke.shape

    def test_changes_coordinates(self, stroke: np.ndarray):
        aug = HandwritingAugmenter(seed=42)
        result = aug.elastic_distort(stroke)
        assert not np.array_equal(result, stroke)

    def test_disabled_returns_unchanged(self, stroke: np.ndarray):
        cfg = AugmentConfig(enabled=False)
        aug = HandwritingAugmenter(config=cfg, seed=42)
        result = aug.elastic_distort(stroke)
        np.testing.assert_array_equal(result, stroke)

    def test_short_stroke_safe(self):
        aug = HandwritingAugmenter(seed=42)
        short = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        result = aug.elastic_distort(short)
        np.testing.assert_array_equal(result, short)

    def test_single_point_safe(self):
        aug = HandwritingAugmenter(seed=42)
        single = np.array([[0.0, 0.0]], dtype=np.float64)
        result = aug.elastic_distort(single)
        np.testing.assert_array_equal(result, single)

    def test_displacement_proportional_to_amplitude(self, stroke: np.ndarray):
        aug = HandwritingAugmenter(seed=42)
        result = aug.elastic_distort(stroke, amplitude=0.02)
        bbox_size = max(stroke.max(axis=0) - stroke.min(axis=0))
        max_disp = np.abs(result - stroke).max()
        assert max_disp < 3 * 0.02 * bbox_size, (
            f"Displacement {max_disp:.4f} exceeds 3*amplitude*bbox ({3 * 0.02 * bbox_size:.4f})"
        )

    def test_zero_size_stroke_safe(self):
        aug = HandwritingAugmenter(seed=42)
        zero = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        result = aug.elastic_distort(zero)
        np.testing.assert_array_equal(result, zero)

    def test_reproducible_with_same_seed(self, stroke: np.ndarray):
        r1 = HandwritingAugmenter(seed=99).elastic_distort(stroke)
        r2 = HandwritingAugmenter(seed=99).elastic_distort(stroke)
        np.testing.assert_array_equal(r1, r2)


class TestApplyTremor:
    @pytest.fixture
    def stroke(self) -> np.ndarray:
        return np.array(
            [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 0.5], [4.0, 0.0]],
            dtype=np.float64,
        )

    def test_preserves_shape(self, stroke: np.ndarray):
        aug = HandwritingAugmenter(seed=42)
        result = aug.apply_tremor(stroke)
        assert result.shape == stroke.shape

    def test_changes_coordinates(self, stroke: np.ndarray):
        aug = HandwritingAugmenter(seed=42)
        result = aug.apply_tremor(stroke)
        assert not np.array_equal(result, stroke)

    def test_disabled_returns_unchanged(self, stroke: np.ndarray):
        cfg = AugmentConfig(enabled=False)
        aug = HandwritingAugmenter(config=cfg, seed=42)
        result = aug.apply_tremor(stroke)
        np.testing.assert_array_equal(result, stroke)

    def test_short_stroke_safe(self):
        aug = HandwritingAugmenter(seed=42)
        single = np.array([[0.0, 0.0]], dtype=np.float64)
        result = aug.apply_tremor(single)
        np.testing.assert_array_equal(result, single)

    def test_amplitude_bounds(self, stroke: np.ndarray):
        amp = 0.05
        aug = HandwritingAugmenter(seed=42)
        result = aug.apply_tremor(stroke, amplitude=amp)
        max_disp = np.abs(result - stroke).max()
        assert max_disp <= amp + 1e-10, (
            f"Tremor displacement {max_disp:.6f} exceeds amplitude {amp}"
        )

    def test_reproducible_with_same_seed(self, stroke: np.ndarray):
        r1 = HandwritingAugmenter(seed=99).apply_tremor(stroke)
        r2 = HandwritingAugmenter(seed=99).apply_tremor(stroke)
        np.testing.assert_array_equal(r1, r2)

    def test_two_point_stroke_safe(self):
        """2点ストローク（幾何バー）でも例外なく点数2を維持。"""
        aug = HandwritingAugmenter(seed=42)
        bar = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        result = aug.apply_tremor(bar)
        assert result.shape == bar.shape

    def test_zero_length_stroke_unchanged(self):
        """総弧長0（全点同一座標）なら素通り。"""
        aug = HandwritingAugmenter(seed=42)
        degenerate = np.array([[1.0, 1.0]] * 5, dtype=np.float64)
        result = aug.apply_tremor(degenerate)
        np.testing.assert_array_equal(result, degenerate)

    def test_amplitude_bounds_spatial(self):
        """変位の絶対値は amplitude 以下（弧長基準でも保証）。"""
        amp = 0.05
        aug = HandwritingAugmenter(seed=7)
        line = np.column_stack(
            [np.linspace(0.0, 10.0, 100), np.zeros(100)]
        ).astype(np.float64)
        result = aug.apply_tremor(line, amplitude=amp)
        max_disp = np.abs(result - line).max()
        assert max_disp <= amp + 1e-10

    def test_wavelength_constant_across_lengths(self):
        """波長が画の長さによらず弧長(mm)で一定。

        全長正規化(旧仕様)だと波長は長さに反比例して短い画ほど高周波の
        さざ波になる。弧長基準では同一 spatial_freq でゼロ交差間隔(=半波長,
        mm)が長い画と短い画でほぼ一致することを確認する（横棒のガタガタ抑制）。
        """
        spatial_freq = (0.4, 0.4)  # cycles/mm 固定（決定的に比較）
        long_line = np.column_stack(
            [np.linspace(0.0, 10.0, 100), np.zeros(100)]
        ).astype(np.float64)
        short_line = np.column_stack(
            [np.linspace(0.0, 2.0, 20), np.zeros(20)]
        ).astype(np.float64)

        long_disp = (
            HandwritingAugmenter(seed=3).apply_tremor(
                long_line, spatial_freq_range=spatial_freq
            )
            - long_line
        )[:, 0]
        short_disp = (
            HandwritingAugmenter(seed=3).apply_tremor(
                short_line, spatial_freq_range=spatial_freq
            )
            - short_line
        )[:, 0]

        long_x = long_line[:, 0]
        short_x = short_line[:, 0]
        long_hw = _mean_zero_crossing_spacing(long_x, long_disp)
        short_hw = _mean_zero_crossing_spacing(short_x, short_disp)

        # 半波長 = 1/(2*spatial_freq) = 1.25mm を両者とも近似するはず
        expected = 1.0 / (2.0 * spatial_freq[0])
        assert abs(long_hw - expected) < 0.25
        assert abs(short_hw - expected) < 0.25
        assert abs(long_hw - short_hw) < 0.25


def _mean_zero_crossing_spacing(x: np.ndarray, disp: np.ndarray) -> float:
    """変位系列のゼロ交差間隔の平均（x軸=mm単位）を返す。

    隣接ゼロ交差間の x 距離 ≒ 半波長。波長が弧長(mm)で一定かを検証するため、
    点インデックスではなく物理座標 x の差分で間隔を測る。
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(disp, dtype=np.float64)
    signs = np.sign(d)
    crossings: list[float] = []
    for i in range(len(d) - 1):
        if signs[i] == 0 or signs[i + 1] == 0 or signs[i] == signs[i + 1]:
            continue
        # 線形補間で d=0 となる x を求める
        frac = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]))
        crossings.append(x[i] + frac * (x[i + 1] - x[i]))
    if len(crossings) < 2:
        return float("nan")
    return float(np.mean(np.diff(crossings)))


def _lag1_autocorr(series: np.ndarray) -> float:
    """系列の lag-1 自己相関を返す。"""
    s = np.asarray(series, dtype=np.float64)
    s = s - s.mean()
    denom = float(np.dot(s, s))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(s[:-1], s[1:]) / denom)


class TestPinkNoiseConfig:
    def test_pink_noise_defaults(self):
        cfg = AugmentConfig()
        assert cfg.use_pink_noise is True
        assert cfg.pink_octaves == 16


class TestPinkSequenceCorrelation:
    def test_char_baseline_positive_autocorrelation(self):
        """1/fノイズは隣接サンプルが正相関する(白色との違い)。"""
        aug = HandwritingAugmenter(AugmentConfig(use_pink_noise=True), seed=7)
        series = np.array([aug.next_char_baseline() for _ in range(2000)])
        assert _lag1_autocorr(series) > 0.1

    def test_white_fallback_near_zero_autocorrelation(self):
        """白色ノイズは隣接サンプルが無相関(lag-1 ≈ 0)。"""
        aug = HandwritingAugmenter(AugmentConfig(use_pink_noise=False), seed=7)
        series = np.array([aug.next_char_baseline() for _ in range(2000)])
        assert abs(_lag1_autocorr(series)) < 0.1


class TestPinkSequenceReproducibility:
    def test_same_seed_identical_series(self):
        cfg = AugmentConfig(use_pink_noise=True)
        a = HandwritingAugmenter(cfg, seed=321)
        b = HandwritingAugmenter(cfg, seed=321)
        for method in (
            "next_line_baseline",
            "next_char_baseline",
            "next_char_spacing",
            "next_char_size_scale",
            "next_char_slant",
        ):
            sa = [getattr(a, method)() for _ in range(50)]
            sb = [getattr(b, method)() for _ in range(50)]
            assert sa == sb, f"{method} not reproducible"

    def test_streams_independent(self):
        """各ストリームは異なるseed派生で独立(同一系列にならない)。"""
        cfg = AugmentConfig(use_pink_noise=True)
        aug = HandwritingAugmenter(cfg, seed=5)
        baseline = [aug.next_char_baseline() for _ in range(50)]
        spacing = [aug.next_char_spacing() for _ in range(50)]
        assert baseline != spacing


class TestPinkNeutralValues:
    def test_disabled_returns_neutral(self):
        cfg = AugmentConfig(enabled=False, use_pink_noise=True)
        aug = HandwritingAugmenter(cfg, seed=1)
        for _ in range(10):
            assert aug.next_line_baseline() == 0.0
            assert aug.next_char_baseline() == 0.0
            assert aug.next_char_spacing() == 0.0
            assert aug.next_char_slant() == 0.0
            assert aug.next_char_size_scale() == 1.0

    def test_disabled_white_returns_neutral(self):
        cfg = AugmentConfig(enabled=False, use_pink_noise=False)
        aug = HandwritingAugmenter(cfg, seed=1)
        for _ in range(10):
            assert aug.next_line_baseline() == 0.0
            assert aug.next_char_baseline() == 0.0
            assert aug.next_char_spacing() == 0.0
            assert aug.next_char_slant() == 0.0
            assert aug.next_char_size_scale() == 1.0


class TestPinkScaling:
    def test_amplitude_proportional_to_config(self):
        """振幅が config の σ に比例する。"""
        small = HandwritingAugmenter(AugmentConfig(baseline_drift=0.1), seed=3)
        large = HandwritingAugmenter(AugmentConfig(baseline_drift=1.0), seed=3)
        s_std = np.std([small.next_char_baseline() for _ in range(1000)])
        l_std = np.std([large.next_char_baseline() for _ in range(1000)])
        assert l_std > s_std * 5

    def test_zero_sigma_returns_neutral(self):
        cfg = AugmentConfig(
            baseline_drift=0.0,
            spacing_variation=0.0,
            size_variation=0.0,
            slant_variation=0.0,
        )
        aug = HandwritingAugmenter(cfg, seed=2)
        for _ in range(20):
            assert aug.next_line_baseline() == 0.0
            assert aug.next_char_baseline() == 0.0
            assert aug.next_char_spacing() == 0.0
            assert aug.next_char_slant() == 0.0
            assert aug.next_char_size_scale() == 1.0

    def test_white_mode_amplitude_scales(self):
        small = HandwritingAugmenter(
            AugmentConfig(use_pink_noise=False, baseline_drift=0.1), seed=3
        )
        large = HandwritingAugmenter(
            AugmentConfig(use_pink_noise=False, baseline_drift=1.0), seed=3
        )
        s_std = np.std([small.next_char_baseline() for _ in range(1000)])
        l_std = np.std([large.next_char_baseline() for _ in range(1000)])
        assert l_std > s_std * 5
