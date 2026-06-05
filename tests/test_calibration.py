"""終端Zリフト キャリブレーション用 G-code 生成のテスト。"""

import numpy as np

from src.gcode.calibration import build_calibration_gcode
from src.gcode.config import PlotterConfig


def _harai_stroke() -> np.ndarray:
    """点数の多い右下がり斜めストローク（払い用）。"""
    xs = np.linspace(0.0, 10.0, 11)
    return np.column_stack([xs, xs])


def _tome_stroke() -> np.ndarray:
    """横画（とめ用）。"""
    xs = np.linspace(0.0, 10.0, 11)
    return np.column_stack([xs, np.full_like(xs, 10.0)])


def _g1_lines(lines):
    return [ln for ln in lines if ln.startswith("G1 ")]


def _x_values(lines):
    xs = []
    for ln in lines:
        for part in ln.split():
            if part.startswith("X"):
                xs.append(float(part[1:]))
    return xs


def _min_z(lines):
    # 描画 G1 行(「G1 X...」)のみ対象。ペン制御「G1G90 Z..」は除外。
    zs = []
    for ln in _g1_lines(lines):
        for part in ln.split():
            if part.startswith("Z"):
                zs.append(float(part[1:]))
    return min(zs) if zs else None


class TestBuildCalibrationGcode:
    def test_has_header_and_footer_once(self):
        strokes = [_harai_stroke()]
        finishes = ["harai"]
        lines = build_calibration_gcode(strokes, finishes, [2.0, 2.5])
        text = "\n".join(lines)
        assert text.count("$H") == 1  # ホーミングは1回
        assert text.count("G92") == 1

    def test_empty_z_values_only_header_footer(self):
        strokes = [_harai_stroke()]
        lines = build_calibration_gcode(strokes, ["harai"], [])
        assert "$H" in "\n".join(lines)
        assert _g1_lines(lines) == []  # 描画なし

    def test_variants_offset_in_x(self):
        """各 Z 版は X 方向にオフセットされ重ならない。"""
        strokes = [_harai_stroke()]
        lines = build_calibration_gcode(strokes, ["harai"], [2.5, 2.0], spacing_mm=50.0)
        xs = _x_values(_g1_lines(lines))
        # 2版なので X の広がりが単一字幅(約10mm)より明らかに大きい
        assert max(xs) - min(xs) > 40.0

    def test_z_differs_per_variant(self):
        """払いの終端Zが版ごとに異なる（finish_lift_z が効く）。"""
        strokes = [_harai_stroke()]
        # 版を1つずつ生成して終端Zを比較
        low = build_calibration_gcode(strokes, ["harai"], [1.8])
        high = build_calibration_gcode(strokes, ["harai"], [2.8])
        assert _min_z(low) < _min_z(high)

    def test_z_within_safe_range(self):
        strokes = [_harai_stroke()]
        cfg = PlotterConfig()
        lines = build_calibration_gcode(strokes, ["harai"], [2.0], base_config=cfg)
        mz = _min_z(lines)
        assert cfg.finish_lift_z - 1.0 <= mz <= cfg.pen_down_z

    def test_tome_emits_no_z(self):
        strokes = [_tome_stroke()]
        lines = build_calibration_gcode(strokes, ["tome"], [2.0, 2.5])
        assert _min_z(lines) is None  # とめはZリフトなし

    def test_finish_count_scales_with_variants(self):
        """版数ぶん同じ字が繰り返される。"""
        strokes = [_harai_stroke()]
        one = _g1_lines(build_calibration_gcode(strokes, ["harai"], [2.0]))
        three = _g1_lines(build_calibration_gcode(strokes, ["harai"], [2.0, 2.2, 2.4]))
        assert len(three) == 3 * len(one)
