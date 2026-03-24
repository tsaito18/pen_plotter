from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Stroke = NDArray[np.float64]


@dataclass
class AugmentConfig:
    """リアルさパラメータ。値は標準偏差（mm単位）。"""

    baseline_drift: float = 0.3
    size_variation: float = 0.05
    slant_variation: float = 0.04
    jitter_amplitude: float = 0.08
    spacing_variation: float = 0.2
    line_density_variation: float = 0.1
    enabled: bool = True


class HandwritingAugmenter:
    def __init__(self, config: AugmentConfig | None = None, seed: int | None = None) -> None:
        self._config = config or AugmentConfig()
        self._rng = np.random.default_rng(seed)

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
        return 1.0 + self._rng.uniform(
            -cfg.line_density_variation, cfg.line_density_variation
        )

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
