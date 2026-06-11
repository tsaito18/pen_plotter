from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.model.pink_noise import PinkNoise1D

Stroke = NDArray[np.float64]


@dataclass
class AugmentConfig:
    """リアルさパラメータ。値は標準偏差（mm単位）。"""

    baseline_drift: float = 0.3
    size_variation: float = 0.05
    slant_variation: float = 0.02
    jitter_amplitude: float = 0.03
    spacing_variation: float = 0.2
    line_density_variation: float = 0.05
    char_density_variation: float = 0.02
    enabled: bool = True
    # 手書き揺らぎを白色から1/f(ピンク)ノイズへ。隣接文字/行で揺らぎが相関し
    # 自然な低周波のうねり(行のうねり・字間のばらつき)を生む
    use_pink_noise: bool = True
    pink_octaves: int = 16


class HandwritingAugmenter:
    # 各1/fストリームの seed 派生オフセット。同一masterから互いに独立な系列を得る
    _PINK_SEED_OFFSETS = {
        "line_baseline": 0,
        "char_baseline": 1,
        "spacing": 2,
        "size": 3,
        "slant": 4,
    }

    def __init__(self, config: AugmentConfig | None = None, seed: int | None = None) -> None:
        self._config = config or AugmentConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # 用途別の1/fストリーム(行baseline/文字baseline/字間/サイズ/傾き)。
        # master seed が None なら各ストリームも None(非決定論)、指定時は
        # seed + 固定オフセットで互いに独立かつ再現可能にする
        if self._config.use_pink_noise:
            octaves = self._config.pink_octaves
            self._pink_line_baseline = self._make_pink("line_baseline", octaves)
            self._pink_char_baseline = self._make_pink("char_baseline", octaves)
            self._pink_spacing = self._make_pink("spacing", octaves)
            self._pink_size = self._make_pink("size", octaves)
            self._pink_slant = self._make_pink("slant", octaves)

    def _make_pink(self, stream: str, octaves: int) -> PinkNoise1D:
        """ストリーム名から seed を派生し PinkNoise1D を生成する。"""
        if self._seed is None:
            stream_seed: int | None = None
        else:
            stream_seed = self._seed + self._PINK_SEED_OFFSETS[stream]
        return PinkNoise1D(octaves=octaves, seed=stream_seed)

    def next_line_baseline(self) -> float:
        """次の行のベースラインオフセット(mm)を返し系列を1つ前進させる。

        行ごとの上下のうねり用。1/f化により隣接行で緩やかに相関する。

        Returns:
            ベースラインオフセット(mm)。enabled=False のとき 0.0。
        """
        if not self._config.enabled:
            return 0.0
        amp = self._config.baseline_drift * 0.7
        if self._config.use_pink_noise:
            return amp * self._pink_line_baseline.sample()
        return float(self._rng.normal(0, amp))

    def next_char_baseline(self) -> float:
        """行内の次の文字のベースラインオフセット(mm)を返す。

        行内の中心線蛇行用。1/f化で隣接文字が相関し滑らかにうねる。

        Returns:
            ベースラインオフセット(mm)。enabled=False のとき 0.0。
        """
        if not self._config.enabled:
            return 0.0
        amp = self._config.baseline_drift * 0.5
        if self._config.use_pink_noise:
            return amp * self._pink_char_baseline.sample()
        return float(self._rng.normal(0, amp))

    def next_char_spacing(self) -> float:
        """次の文字の字間オフセット(mm)を返し系列を1つ前進させる。

        Returns:
            字間オフセット(mm)。enabled=False のとき 0.0。
        """
        if not self._config.enabled:
            return 0.0
        amp = self._config.spacing_variation
        if self._config.use_pink_noise:
            return amp * self._pink_spacing.sample()
        return float(self._rng.normal(0, amp))

    def next_char_size_scale(self) -> float:
        """次の文字のサイズ倍率を返し系列を1つ前進させる。

        Returns:
            サイズ倍率(1.0中心)。enabled=False のとき 1.0。
        """
        if not self._config.enabled:
            return 1.0
        amp = self._config.size_variation
        if self._config.use_pink_noise:
            return 1.0 + amp * self._pink_size.sample()
        return 1.0 + float(self._rng.normal(0, amp))

    def next_char_slant(self) -> float:
        """次の文字の傾き角(rad)を返し系列を1つ前進させる。

        Returns:
            傾き角(rad)。enabled=False のとき 0.0。
        """
        if not self._config.enabled:
            return 0.0
        amp = self._config.slant_variation
        if self._config.use_pink_noise:
            return amp * self._pink_slant.sample()
        return float(self._rng.normal(0, amp))

    def augment_page(self, strokes: list[Stroke]) -> list[Stroke]:
        """ページ全体のストロークにリアルな変動を加える。"""
        if not self._config.enabled or not strokes:
            return strokes

        return [self._apply_jitter(stroke) for stroke in strokes]

    def get_line_density_scale(self) -> float:
        """行ごとの文字密度スケールを返す。"""
        if not self._config.enabled:
            return 1.0
        cfg = self._config
        return 1.0 + self._rng.uniform(-cfg.line_density_variation, cfg.line_density_variation)

    def get_char_density_scale(self) -> float:
        """文字ごとの密度スケールを返す。"""
        if not self._config.enabled:
            return 1.0
        cfg = self._config
        return 1.0 + self._rng.uniform(-cfg.char_density_variation, cfg.char_density_variation)

    def get_char_slant(self) -> float:
        """文字ごとの微小傾き角(rad)を返す。手書きの一文字単位の傾き揺らぎ。"""
        if not self._config.enabled:
            return 0.0
        return float(self._rng.normal(0, self._config.slant_variation))

    def augment_char_placement(
        self, x: float, y: float, font_size: float
    ) -> tuple[float, float, float]:
        """個別文字の配置にランダム変動を加える。

        Returns:
            (new_x, new_y, new_font_size)
        """
        if not self._config.enabled:
            return x, y, font_size

        cfg = self._config
        new_x = x + self._rng.normal(0, cfg.spacing_variation)
        new_y = y + self._rng.normal(0, cfg.baseline_drift)
        new_size = font_size * (1 + self._rng.normal(0, cfg.size_variation))
        return new_x, new_y, max(new_size, font_size * 0.8)

    def apply_slant(self, stroke: Stroke, center_x: float, center_y: float) -> Stroke:
        """ストロークにランダムな傾きを加える。"""
        if not self._config.enabled:
            return stroke
        angle = self._rng.normal(0, self._config.slant_variation)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        centered = stroke - np.array([center_x, center_y])
        rotated = centered @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return rotated + np.array([center_x, center_y])

    def random_uniform(self, low: float, high: float) -> float:
        """公開RNGアクセス: [low, high) の一様乱数を返す。"""
        return float(self._rng.uniform(low, high))

    def elastic_distort(self, stroke: Stroke, amplitude: float = 0.002) -> Stroke:
        """ストロークに滑らかな弾性変形を適用。amplitudeはbbox比。"""
        if len(stroke) < 3 or not self._config.enabled:
            return stroke
        bbox_size = max(stroke.max(axis=0) - stroke.min(axis=0))
        if bbox_size < 1e-6:
            return stroke
        disp_amp = amplitude * bbox_size
        n = len(stroke)
        num_ctrl = min(4, n)
        ctrl_dx = self._rng.normal(0, disp_amp, num_ctrl)
        ctrl_dy = self._rng.normal(0, disp_amp, num_ctrl)
        ctrl_t = np.linspace(0, 1, num_ctrl)
        stroke_t = np.linspace(0, 1, n)
        dx = np.interp(stroke_t, ctrl_t, ctrl_dx)
        dy = np.interp(stroke_t, ctrl_t, ctrl_dy)
        return stroke + np.column_stack([dx, dy])

    def apply_tremor(
        self,
        stroke: Stroke,
        freq_range: tuple[float, float] = (3.0, 5.0),
        amplitude: float = 0.01,
    ) -> Stroke:
        """3-5Hz低周波振動を重畳。amplitudeはmm単位。"""
        if len(stroke) < 2 or not self._config.enabled:
            return stroke
        t = np.linspace(0, 1, len(stroke))
        freq = self._rng.uniform(*freq_range)
        phase = self._rng.uniform(0, 2 * np.pi)
        tremor_x = amplitude * np.sin(2 * np.pi * freq * t + phase)
        tremor_y = amplitude * np.sin(2 * np.pi * freq * t + phase + np.pi / 3)
        return stroke + np.column_stack([tremor_x, tremor_y])

    def _apply_jitter(self, stroke: Stroke) -> Stroke:
        """ストロークに微振動を加える。"""
        if len(stroke) < 2:
            return stroke
        noise = self._rng.normal(0, self._config.jitter_amplitude, size=stroke.shape)
        # 自然な見た目のためノイズを平滑化（移動平均）
        kernel_size = min(3, len(stroke))
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            for dim in range(stroke.shape[1]):
                noise[:, dim] = np.convolve(noise[:, dim], kernel, mode="same")
        return stroke + noise
