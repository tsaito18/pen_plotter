"""手書きの揺らぎ（augmentation / 1/f ノイズ）と筆遣い（finishing）。"""

from __future__ import annotations

import numpy as np
import pytest

from src.handwriting.augmentation import AugmentConfig, HandwritingAugmenter
from src.handwriting.finishing import (
    CONNECT,
    CONNECT_CONTACT,
    HANE,
    HARAI,
    NONE,
    TOME,
    arc_length_from_end,
    classify_finish,
    contact_profile,
    entry_modulation,
    infer_finish_from_stroke,
    insert_connections,
    pressure_modulation,
)
from src.handwriting.pink_noise import PinkNoise1D


def _max_turn(stroke: np.ndarray) -> float:
    """隣り合う区間の向きの変化の最大値(rad)。"""
    d = np.diff(stroke, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi).max())


def _lag1_autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    return float((x[:-1] * x[1:]).mean() / x.var())


# --- 幾何字形の手書き化 ---


def test_hand_drawn_rounds_corners_and_bends_straight_lines():
    z = np.array([[0.1, 0.9], [0.9, 0.9], [0.1, 0.1], [0.9, 0.1]])  # 「Z」の 1 筆
    (out,) = HandwritingAugmenter(seed=0).hand_drawn([z])
    assert _max_turn(out) < 0.6 * _max_turn(z)  # 角が丸くなる
    assert np.allclose(out[[0, -1]], z[[0, -1]], atol=0.06)  # 始点・終点はほぼ同じ
    top = out[out[:, 1] > 0.8]
    assert np.ptp(top[:, 1]) > 0.003  # 横棒が定規の直線ではない
    again = HandwritingAugmenter(seed=0).hand_drawn([z])[0]
    other = HandwritingAugmenter(seed=1).hand_drawn([z])[0]
    assert np.allclose(out, again)
    assert out.shape != other.shape or not np.allclose(out, other)


def test_hand_drawn_keeps_small_dots_small_and_is_off_when_disabled():
    t = np.linspace(0, 2 * np.pi, 12)
    dot = np.column_stack([0.5 + 0.05 * np.cos(t), 0.2 + 0.05 * np.sin(t)])
    (out,) = HandwritingAugmenter(seed=0).hand_drawn([dot])
    assert np.abs(out.mean(axis=0) - dot.mean(axis=0)).max() < 0.02
    off = HandwritingAugmenter(AugmentConfig(enabled=False), seed=0)
    assert off.hand_drawn([dot])[0] is dot


# --- 1/f ノイズ ---


def test_pink_noise_is_correlated_and_unit_scale():
    x = PinkNoise1D(seed=0).samples(4000)
    assert _lag1_autocorr(x) > 0.5
    assert 0.5 < x.std() < 1.5


def test_pink_noise_psd_slope_is_near_minus_one():
    x = PinkNoise1D(octaves=16, seed=1).samples(2**14)
    freqs = np.fft.rfftfreq(len(x))[1:]
    psd = np.abs(np.fft.rfft(x - x.mean()))[1:] ** 2
    band = (freqs > 1e-3) & (freqs < 1e-1)
    slope = np.polyfit(np.log(freqs[band]), np.log(psd[band]), 1)[0]
    assert -1.6 < slope < -0.5


def test_pink_noise_seed_reproducible():
    assert np.array_equal(PinkNoise1D(seed=3).samples(50), PinkNoise1D(seed=3).samples(50))
    assert not np.array_equal(PinkNoise1D(seed=3).samples(50), PinkNoise1D(seed=4).samples(50))


# --- 配置の揺らぎ ---


def test_disabled_augmenter_is_neutral():
    aug = HandwritingAugmenter(AugmentConfig(enabled=False), seed=0)
    assert aug.next_line_baseline() == 0.0
    assert aug.next_char_baseline() == 0.0
    assert aug.next_char_spacing() == 0.0
    assert aug.next_char_size_scale() == 1.0
    assert aug.next_char_slant() == 0.0
    assert aug.line_density_scale() == 1.0
    assert aug.char_density_scale() == 1.0
    stroke = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    assert aug.elastic_distort(stroke) is stroke
    assert aug.apply_tremor(stroke) is stroke


def test_same_seed_gives_same_sequence():
    a, b = HandwritingAugmenter(seed=7), HandwritingAugmenter(seed=7)
    seq_a = [
        (a.next_char_spacing(), a.next_char_slant(), a.char_density_scale()) for _ in range(20)
    ]
    seq_b = [
        (b.next_char_spacing(), b.next_char_slant(), b.char_density_scale()) for _ in range(20)
    ]
    assert seq_a == seq_b


def test_char_baseline_drifts_smoothly_with_pink_noise():
    pink = HandwritingAugmenter(seed=0)
    white = HandwritingAugmenter(AugmentConfig(use_pink_noise=False), seed=0)
    pink_series = np.array([pink.next_char_baseline() for _ in range(3000)])
    white_series = np.array([white.next_char_baseline() for _ in range(3000)])
    assert _lag1_autocorr(pink_series) > 0.5
    assert abs(_lag1_autocorr(white_series)) < 0.1


def test_density_scales_stay_within_configured_range():
    aug = HandwritingAugmenter(seed=0)
    line = [aug.line_density_scale() for _ in range(200)]
    char = [aug.char_density_scale() for _ in range(200)]
    assert all(0.95 <= v <= 1.05 for v in line)
    assert all(0.98 <= v <= 1.02 for v in char)


@pytest.mark.parametrize("messiness", [0.0, 0.5, 2.0])
def test_messiness_scales_layout_jitter_only(messiness: float):
    base = AugmentConfig()
    scaled = base.scaled(messiness)
    assert scaled.baseline_drift == pytest.approx(base.baseline_drift * messiness)
    assert scaled.spacing_variation == pytest.approx(base.spacing_variation * messiness)
    assert scaled.size_variation == pytest.approx(base.size_variation * messiness)
    assert scaled.slant_variation == pytest.approx(base.slant_variation * messiness)
    assert scaled.char_density_variation == base.char_density_variation


# --- 字形の微小変形 ---


def test_elastic_distort_is_small_and_shape_preserving():
    aug = HandwritingAugmenter(seed=0)
    stroke = np.column_stack([np.linspace(0, 10, 30), np.zeros(30)])
    out = aug.elastic_distort(stroke, amplitude=0.01)
    assert out.shape == stroke.shape
    assert 0 < np.abs(out - stroke).max() < 0.5


def test_tremor_wavelength_is_independent_of_stroke_length():
    """位相を実弧長で進めるので、短い画でも高周波のさざ波にならない。"""
    aug = HandwritingAugmenter(seed=0)

    def zero_crossings(length: float) -> float:
        stroke = np.column_stack([np.linspace(0, length, 400), np.zeros(400)])
        dy = aug.apply_tremor(stroke, spatial_freq_range=(0.4, 0.4), amplitude=0.01)[:, 1]
        return np.count_nonzero(np.diff(np.sign(dy))) / length

    assert zero_crossings(10.0) == pytest.approx(zero_crossings(40.0), rel=0.2)


def test_tremor_amplitude_is_bounded():
    aug = HandwritingAugmenter(seed=0)
    stroke = np.column_stack([np.linspace(0, 20, 100), np.zeros(100)])
    assert np.abs(aug.apply_tremor(stroke, amplitude=0.05) - stroke).max() <= 0.05 + 1e-9


# --- 筆法の分類・推定 ---


@pytest.mark.parametrize(
    ("kvg_type", "expected"),
    [
        ("㇒", HARAI),
        ("㇏", HARAI),
        ("㇚", HANE),
        ("㇐", TOME),
        ("㇒/a", HARAI),  # スラッシュ付き variant
        ("㇑a", TOME),  # 接尾辞付き variant
        ("", NONE),
        ("?", NONE),
    ],
)
def test_classify_finish(kvg_type: str, expected: str):
    assert classify_finish(kvg_type) == expected


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([(0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (-0.5, 0.5), (-1, 1)], HANE),  # 縦画→跳ね上げ
        ([(0, 10), (3, 6), (6, 3), (9, 0)], HARAI),  # 斜めに滑らかに流れる
        ([(0, 5), (5, 5), (10, 5)], TOME),  # 横画
        ([(0, 10), (0, 5), (0, 0)], TOME),  # 縦にまっすぐ
    ],
)
def test_infer_finish_from_trajectory(points, expected):
    assert infer_finish_from_stroke(np.array(points, dtype=float)) == expected


# --- 接触率（Z リフト・線幅の単一ソース） ---


def test_arc_length_from_end():
    arc = arc_length_from_end(np.array([[0, 0], [3, 4], [3, 10]], dtype=float))
    assert arc.tolist() == [11.0, 6.0, 0.0]


def test_harai_contact_lifts_only_near_the_end():
    stroke = np.column_stack([np.linspace(0, 20, 201), np.zeros(201)])
    contact = contact_profile(HARAI, arc_length_from_end(stroke), lift_length=1.5)
    assert contact[0] == 1.0 and contact[-1] == 0.0
    assert np.all(np.diff(contact) <= 1e-12)
    assert np.all(contact[stroke[:, 0] < 18.5] == 1.0)


def test_hane_holds_contact_longer_than_harai():
    stroke = np.column_stack([np.linspace(0, 10, 101), np.zeros(101)])
    arc = arc_length_from_end(stroke)
    harai = contact_profile(HARAI, arc, lift_length=1.5)
    hane = contact_profile(HANE, arc, lift_length=1.5)
    assert np.all(hane >= harai - 1e-12)


def test_short_stroke_keeps_a_solid_head():
    """短い画でリフト区間が全長を食い尽くし、全体が薄くならない。"""
    stroke = np.column_stack([np.linspace(0, 1.0, 11), np.zeros(11)])
    contact = contact_profile(HARAI, arc_length_from_end(stroke), lift_length=1.5)
    assert np.all(contact[:5] == 1.0)


@pytest.mark.parametrize("finish", [TOME, NONE])
def test_tome_and_none_keep_full_contact(finish: str):
    arc = arc_length_from_end(np.column_stack([np.arange(5.0), np.zeros(5)]))
    assert np.all(contact_profile(finish, arc, 1.5) == 1.0)


def test_connect_stroke_has_constant_light_contact():
    arc = arc_length_from_end(np.column_stack([np.arange(5.0), np.zeros(5)]))
    assert np.all(contact_profile(CONNECT, arc, 1.5) == CONNECT_CONTACT)


def test_pressure_modulation_darker_on_downstrokes():
    down = np.column_stack([np.zeros(50), np.linspace(10, 0, 50)])
    up = down[::-1]
    assert pressure_modulation(down, 0.5).mean() > pressure_modulation(up, 0.5).mean()
    assert np.all(pressure_modulation(down, 0.0) == 1.0)
    assert np.all((0.5 <= pressure_modulation(down, 0.5)) & (pressure_modulation(down, 0.5) <= 1))


def test_entry_modulation_ramps_up_from_the_start():
    stroke = np.column_stack([np.linspace(0, 5, 51), np.zeros(51)])
    mult = entry_modulation(stroke, entry_length=1.0, strength=0.6)
    assert mult[0] == pytest.approx(0.4)
    assert np.all(mult[stroke[:, 0] >= 1.0] == 1.0)
    assert np.all(entry_modulation(stroke, 1.0, 0.0) == 1.0)


# --- 連綿 ---


def _two_strokes(gap: float) -> list[np.ndarray]:
    return [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0 + gap, 0.0], [2.0 + gap, 0.0]])]


def test_connections_join_close_strokes_only():
    rng = np.random.default_rng(0)
    near, near_f = insert_connections(_two_strokes(0.1), [TOME, TOME], 1.0, 4.5, rng)
    far, _ = insert_connections(_two_strokes(10.0), [TOME, TOME], 1.0, 4.5, rng)
    assert len(near) == 3 and near_f[1] == CONNECT
    assert len(far) == 2


def test_no_connection_after_sweeping_stroke_or_when_disabled():
    rng = np.random.default_rng(0)
    after_harai, _ = insert_connections(_two_strokes(0.01), [HARAI, TOME], 1.0, 4.5, rng)
    disabled, _ = insert_connections(_two_strokes(0.01), [TOME, TOME], 0.0, 4.5, rng)
    assert len(after_harai) == 2 and len(disabled) == 2
