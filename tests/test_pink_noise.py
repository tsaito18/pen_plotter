import numpy as np
import pytest

from src.model.pink_noise import PinkNoise1D


def _psd_slope(series: np.ndarray) -> float:
    """系列のPSDをlog-log直線フィットし傾きを返す（pink≈-1）。"""
    n = len(series)
    spectrum = np.fft.rfft(series - series.mean())
    psd = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n)

    # 低周波0（DC）を除外。極端な高周波端も除いて安定化
    mask = (freqs > 0) & (freqs < 0.5)
    freqs = freqs[mask]
    psd = psd[mask]

    # 対数等間隔ビニングでlog-log上の点を均す（低周波の少数点に過剰加重しない）
    log_f = np.log10(freqs)
    bins = np.linspace(log_f.min(), log_f.max(), 40)
    idx = np.digitize(log_f, bins)
    binned_f = []
    binned_p = []
    for b in range(1, len(bins)):
        sel = idx == b
        if np.any(sel):
            binned_f.append(np.log10(freqs[sel]).mean())
            binned_p.append(np.log10(psd[sel]).mean())

    slope, _ = np.polyfit(binned_f, binned_p, 1)
    return float(slope)


class TestSpectrum:
    def test_psd_slope_is_pink(self):
        noise = PinkNoise1D(octaves=16, seed=42)
        series = noise.samples(2**14)

        slope = _psd_slope(series)
        # pinkノイズはPSD ∝ 1/f → log-log傾き≈-1
        assert -1.5 < slope < -0.5, f"slope={slope}"

    def test_white_noise_slope_is_flat(self):
        # 対比: 白色ノイズはほぼ平坦（傾き≈0）でpink帯域に入らない
        rng = np.random.default_rng(42)
        series = rng.standard_normal(2**14)

        slope = _psd_slope(series)
        assert slope > -0.5, f"white slope={slope}"


class TestAutocorrelation:
    def _lag1_autocorr(self, series: np.ndarray) -> float:
        s = series - series.mean()
        return float(np.sum(s[:-1] * s[1:]) / np.sum(s * s))

    def test_lag1_autocorr_positive(self):
        noise = PinkNoise1D(octaves=16, seed=7)
        series = noise.samples(2**14)

        ac = self._lag1_autocorr(series)
        # 低周波優位 → 隣接サンプルが明確に正相関
        assert ac > 0.1, f"autocorr={ac}"

    def test_white_noise_autocorr_near_zero(self):
        rng = np.random.default_rng(7)
        series = rng.standard_normal(2**14)

        ac = self._lag1_autocorr(series)
        assert abs(ac) < 0.05, f"white autocorr={ac}"


class TestReproducibility:
    def test_same_seed_same_series(self):
        a = PinkNoise1D(octaves=16, seed=123)
        b = PinkNoise1D(octaves=16, seed=123)

        sa = a.samples(2000)
        sb = b.samples(2000)
        np.testing.assert_array_equal(sa, sb)

    def test_different_seed_different_series(self):
        a = PinkNoise1D(octaves=16, seed=1)
        b = PinkNoise1D(octaves=16, seed=2)

        assert not np.array_equal(a.samples(1000), b.samples(1000))


class TestReset:
    def test_reset_reproduces_series(self):
        noise = PinkNoise1D(octaves=16, seed=99)
        first = noise.samples(1000)

        noise.reset()
        second = noise.samples(1000)
        np.testing.assert_array_equal(first, second)

    def test_sample_advances_state(self):
        noise = PinkNoise1D(octaves=16, seed=99)
        v1 = noise.sample()
        v2 = noise.sample()
        # 状態前進: 連続samplが同値で固定されていない
        assert v1 != v2


class TestUnitVariance:
    def test_std_near_one(self):
        noise = PinkNoise1D(octaves=16, seed=42)
        series = noise.samples(2**14)

        std = float(series.std())
        assert 0.5 < std < 2.0, f"std={std}"


class TestOctavesParam:
    @pytest.mark.parametrize("octaves", [8, 16])
    def test_various_octaves_work(self, octaves: int):
        noise = PinkNoise1D(octaves=octaves, seed=5)
        series = noise.samples(2**13)

        assert len(series) == 2**13
        assert np.all(np.isfinite(series))
        # 正規化が効いて常識的な分散
        assert 0.3 < float(series.std()) < 3.0

    def test_sample_returns_float(self):
        noise = PinkNoise1D(octaves=8, seed=0)
        v = noise.sample()
        assert isinstance(v, float)
