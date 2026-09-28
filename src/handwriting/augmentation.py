"""手書きらしい揺らぎ（配置の揺らぎ・字形の微小変形）を与える。

配置の揺らぎ（行/文字ベースライン・字間・サイズ・傾き）は 1/f ノイズの系列で
与え、隣接文字・隣接行で揺らぎが相関する自然なうねりを作る。字形の微小変形
（弾性変形・手ブレ）はストローク単位で乱数を引く。全乱数は本クラスが持つ
``rng`` から引くため、``seed`` を与えれば再現可能になる。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.geometry import Stroke
from src.handwriting.pink_noise import PinkNoise1D


@dataclass
class AugmentConfig:
    """揺らぎの強度。長さ系は標準偏差（mm）、傾きは rad。"""

    baseline_drift: float = 0.3
    size_variation: float = 0.05
    slant_variation: float = 0.02
    spacing_variation: float = 0.2
    line_density_variation: float = 0.05
    char_density_variation: float = 0.02
    enabled: bool = True
    # 白色ではなく 1/f(ピンク)ノイズで揺らぎを相関させ、行のうねり・字間のばらつきを
    # 自然な低周波にする。False なら独立な正規乱数。
    use_pink_noise: bool = True
    pink_octaves: int = 16

    def scaled(self, messiness: float) -> AugmentConfig:
        """レイアウト揺らぎ 4 項目（ベースライン・字間・サイズ・傾き）を一括倍率した設定。

        GUI の「汚さ」スライダー用。``messiness=1.0`` で素の値、``0`` で揺らぎなし。
        密度系は字形の質感なので据え置く。
        """
        return replace(
            self,
            baseline_drift=self.baseline_drift * messiness,
            spacing_variation=self.spacing_variation * messiness,
            size_variation=self.size_variation * messiness,
            slant_variation=self.slant_variation * messiness,
        )


class HandwritingAugmenter:
    # 1/f ストリームごとの seed 派生オフセット（同一 seed から互いに独立な系列を得る）
    _PINK_STREAMS = ("line_baseline", "char_baseline", "spacing", "size", "slant")

    def __init__(self, config: AugmentConfig | None = None, seed: int | None = None) -> None:
        self.config = config or AugmentConfig()
        self.rng = np.random.default_rng(seed)
        self._pink: dict[str, PinkNoise1D] = {}
        if self.config.use_pink_noise:
            for offset, name in enumerate(self._PINK_STREAMS):
                stream_seed = None if seed is None else seed + offset
                self._pink[name] = PinkNoise1D(octaves=self.config.pink_octaves, seed=stream_seed)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _draw(self, stream: str, amp: float) -> float:
        if self.config.use_pink_noise:
            return amp * self._pink[stream].sample()
        return float(self.rng.normal(0, amp))

    # --- 配置の揺らぎ（組版が 1 行/1 文字ごとに呼ぶ） ---

    def next_line_baseline(self) -> float:
        """次の行のベースラインオフセット(mm)。"""
        if not self.enabled:
            return 0.0
        return self._draw("line_baseline", self.config.baseline_drift * 0.7)

    def next_char_baseline(self) -> float:
        """行内の次の文字のベースラインオフセット(mm)。"""
        if not self.enabled:
            return 0.0
        return self._draw("char_baseline", self.config.baseline_drift * 0.5)

    def next_char_spacing(self) -> float:
        """次の文字の字間オフセット(mm)。"""
        if not self.enabled:
            return 0.0
        return self._draw("spacing", self.config.spacing_variation)

    def next_char_size_scale(self) -> float:
        """次の文字のサイズ倍率（1.0 中心）。"""
        if not self.enabled:
            return 1.0
        return 1.0 + self._draw("size", self.config.size_variation)

    def next_char_slant(self) -> float:
        """次の文字の傾き角(rad)。"""
        if not self.enabled:
            return 0.0
        return self._draw("slant", self.config.slant_variation)

    def line_density_scale(self) -> float:
        """行ごとの文字密度倍率。"""
        if not self.enabled:
            return 1.0
        v = self.config.line_density_variation
        return 1.0 + self.rng.uniform(-v, v)

    def char_density_scale(self) -> float:
        """文字ごとの密度倍率。"""
        if not self.enabled:
            return 1.0
        v = self.config.char_density_variation
        return 1.0 + self.rng.uniform(-v, v)

    # --- 字形の微小変形 ---

    def elastic_distort(self, stroke: Stroke, amplitude: float = 0.002) -> Stroke:
        """少数の制御点で補間した滑らかな弾性変形。``amplitude`` は bbox 比。"""
        if len(stroke) < 3 or not self.enabled:
            return stroke
        bbox_size = max(stroke.max(axis=0) - stroke.min(axis=0))
        if bbox_size < 1e-6:
            return stroke
        disp_amp = amplitude * bbox_size
        n = len(stroke)
        num_ctrl = min(4, n)
        ctrl_dx = self.rng.normal(0, disp_amp, num_ctrl)
        ctrl_dy = self.rng.normal(0, disp_amp, num_ctrl)
        ctrl_t = np.linspace(0, 1, num_ctrl)
        stroke_t = np.linspace(0, 1, n)
        dx = np.interp(stroke_t, ctrl_t, ctrl_dx)
        dy = np.interp(stroke_t, ctrl_t, ctrl_dy)
        return stroke + np.column_stack([dx, dy])

    def apply_tremor(
        self,
        stroke: Stroke,
        spatial_freq_range: tuple[float, float] = (0.3, 0.5),
        amplitude: float = 0.01,
    ) -> Stroke:
        """手ブレ振動を実弧長(mm)基準で重畳する。

        位相を弧長で進めるため、画の長短によらず波長が一定（既定 2.0-3.3mm）の
        緩いうねりになる（全長正規化だと短い横棒ほど高周波のさざ波になる）。

        Args:
            stroke: 入力ストローク(mm)。
            spatial_freq_range: 空間周波数の範囲(cycles/mm)。
            amplitude: 振幅(mm)。x/y に位相差 π/3 で楕円的に揺らす。
        """
        if len(stroke) < 2 or not self.enabled:
            return stroke
        seg_len = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        if s[-1] < 1e-9:
            return stroke
        spatial_freq = self.rng.uniform(*spatial_freq_range)
        phase = self.rng.uniform(0, 2 * np.pi)
        omega = 2 * np.pi * spatial_freq * s
        tremor_x = amplitude * np.sin(omega + phase)
        tremor_y = amplitude * np.sin(omega + phase + np.pi / 3)
        return stroke + np.column_stack([tremor_x, tremor_y])
