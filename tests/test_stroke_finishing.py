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
