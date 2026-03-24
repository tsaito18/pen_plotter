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
        assert cfg.slant_variation == 0.04
        assert cfg.jitter_amplitude == 0.08
        assert cfg.spacing_variation == 0.2
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
        assert cfg.line_density_variation == 0.1


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
            assert 0.85 <= scale <= 1.15, (
                f"Scale {scale} outside expected range [0.85, 1.15]"
            )

    def test_get_line_density_scale_disabled(self):
        cfg = AugmentConfig(enabled=False)
        aug = HandwritingAugmenter(config=cfg, seed=42)
        result = aug.get_line_density_scale()
        assert result == 1.0
