"""kvg:type 分類と終端加工のテスト。"""

import numpy as np

from src.model.stroke_finishing import (
    HANE,
    HARAI,
    NONE,
    TOME,
    FinishingConfig,
    apply_finishing,
    apply_hane,
    apply_harai,
    apply_tome,
    classify_finish,
    classify_finishes,
    contact_profile,
)


class TestClassifyFinish:
    def test_known_types(self):
        assert classify_finish("㇐") == TOME  # H 横画
        assert classify_finish("㇑") == TOME  # S 縦画
        assert classify_finish("㇔") == TOME  # D 点
        assert classify_finish("㇒") == HARAI  # P 左払い
        assert classify_finish("㇏") == HARAI  # N 右払い
        assert classify_finish("㇀") == HARAI  # T 提
        assert classify_finish("㇚") == HANE  # SG 縦はね
        assert classify_finish("㇆") == HANE  # HZG 横折钩
        assert classify_finish("㇄") == TOME  # SW 竖弯（鉤なし折れ）

    def test_variant_slash_stripped(self):
        assert classify_finish("㇒/a") == HARAI
        assert classify_finish("㇏/b") == HARAI

    def test_variant_suffix_char(self):
        # KanjiVG では "㇑a" のように接尾辞文字付きの variant 表記がある
        assert classify_finish("㇑a") == TOME
        assert classify_finish("㇚b") == HANE

    def test_unknown_and_empty_return_none(self):
        assert classify_finish("") == NONE
        assert classify_finish("???") == NONE
        assert classify_finish("xyz") == NONE

    def test_classify_finishes_list(self):
        assert classify_finishes(["㇐", "㇒", ""]) == [TOME, HARAI, NONE]

    def test_classify_finishes_empty(self):
        assert classify_finishes([]) == []


class TestApplyHarai:
    def test_extends_endpoint_along_tangent(self):
        # 右向き直線。払いで終端が接線(右)方向に伸び、点数が増える
        stroke = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        out = apply_harai(stroke, scale=10.0, config=FinishingConfig())
        assert len(out) > len(stroke)
        assert out[-1][0] > stroke[-1][0]
        assert np.isclose(out[-1][1], 0.0, atol=1e-6)

    def test_zero_length_terminal_is_safe(self):
        # 終端が重複点でも接線が定義できず落ちない
        stroke = np.array([[0, 0], [1, 0], [1, 0]], dtype=float)
        out = apply_harai(stroke, scale=10.0, config=FinishingConfig())
        assert out is not None


class TestApplyHane:
    def test_adds_hook_off_tangent(self):
        # 下向き縦画。はねで終端が接線と別方向へ折れる
        stroke = np.array([[0, 3], [0, 2], [0, 1], [0, 0]], dtype=float)
        out = apply_hane(stroke, scale=10.0, config=FinishingConfig())
        assert len(out) > len(stroke)
        seg_before = stroke[-1] - stroke[-2]
        seg_after = out[-1] - out[-2]
        cos = np.dot(seg_before, seg_after) / (
            np.linalg.norm(seg_before) * np.linalg.norm(seg_after)
        )
        assert cos < 0.9  # 進行方向から明確に折れる


class TestApplyTome:
    def test_identity_by_default(self):
        stroke = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        out = apply_tome(stroke, scale=10.0, config=FinishingConfig())
        assert np.array_equal(out, stroke)


class TestApplyFinishing:
    def test_dispatches_by_finish(self):
        strokes = [
            np.array([[0, 0], [1, 0]], dtype=float),
            np.array([[0, 3], [0, 0]], dtype=float),
        ]
        out = apply_finishing(strokes, [HARAI, HANE], scale=10.0)
        assert len(out[0]) > 2  # harai 延長
        assert len(out[1]) > 2  # hane フック

    def test_none_is_identity(self):
        strokes = [np.array([[0, 0], [1, 0]], dtype=float)]
        out = apply_finishing(strokes, [NONE], scale=10.0)
        assert np.array_equal(out[0], strokes[0])

    def test_length_mismatch_safe(self):
        strokes = [
            np.array([[0, 0], [1, 0]], dtype=float),
            np.array([[0, 0], [0, 1]], dtype=float),
        ]
        out = apply_finishing(strokes, [HARAI], scale=10.0)  # finishes 不足
        assert len(out) == 2
        assert np.array_equal(out[1], strokes[1])  # 不足分は none=恒等

    def test_disabled_is_noop(self):
        strokes = [np.array([[0, 0], [1, 0], [2, 0]], dtype=float)]
        cfg = FinishingConfig(enabled=False)
        out = apply_finishing(strokes, [HARAI], scale=10.0, config=cfg)
        assert np.array_equal(out[0], strokes[0])

    def test_short_stroke_guard(self):
        strokes = [np.array([[0, 0]], dtype=float)]  # 1点
        out = apply_finishing(strokes, [HARAI], scale=10.0)
        assert np.array_equal(out[0], strokes[0])  # 落ちない


class TestContactProfile:
    def test_none_and_tome_all_contact(self):
        for finish in (NONE, TOME):
            prof = contact_profile(finish, n_points=10, lift_points=5)
            assert len(prof) == 10
            assert np.allclose(prof, 1.0)

    def test_harai_monotonic_decreasing_tail(self):
        prof = contact_profile(HARAI, n_points=12, lift_points=5, harai_min=0.15)
        assert len(prof) == 12
        # 先頭は完全接触、末尾は最小接触へ
        assert prof[0] == 1.0
        assert np.isclose(prof[-1], 0.15, atol=1e-6)
        # 全体が単調非増加
        assert np.all(np.diff(prof) <= 1e-9)
        # リフト区間より前(7点)は接触1.0のまま
        assert np.allclose(prof[:7], 1.0)

    def test_hane_drops_lower_than_harai(self):
        harai = contact_profile(HARAI, n_points=12, lift_points=5, harai_min=0.15)
        hane = contact_profile(HANE, n_points=12, lift_points=5, hane_min=0.2)
        assert hane[0] == 1.0
        assert np.isclose(hane[-1], 0.2, atol=1e-6)
        assert np.all(np.diff(hane) <= 1e-9)  # 単調非増加
        # はねは二乗カーブで急峻：リフト区間の中間点で払いより接触が小さい
        mid = 9  # 末尾5点(index7..11)の中央
        assert hane[mid] < harai[mid]

    def test_lift_points_exceeds_length_safe(self):
        prof = contact_profile(HARAI, n_points=3, lift_points=10)
        assert len(prof) == 3
        assert prof[0] == 1.0
        assert prof[-1] < 1.0

    def test_short_stroke_guard(self):
        prof = contact_profile(HARAI, n_points=1, lift_points=5)
        assert len(prof) == 1
        assert np.allclose(prof, 1.0)
