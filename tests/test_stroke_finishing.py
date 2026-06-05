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
    arc_length_from_end,
    classify_finish,
    classify_finishes,
    contact_profile,
    infer_finish_from_stroke,
    infer_finishes,
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
    def test_extends_along_existing_hook_direction(self):
        # KanjiVG の鈎は既にフック形状を含む。終端接線(=フック向き)へ延長し、
        # 回転して逆向きに飛ばないこと（二重フックのバグ回帰防止）。
        # 左上へ一貫して跳ねる終端を模した点列。
        stroke = np.array([[3, 0], [2, 1], [1, 2], [0, 3]], dtype=float)
        out = apply_hane(stroke, scale=10.0, config=FinishingConfig())
        assert len(out) > len(stroke)
        seg_before = stroke[-1] - stroke[-2]
        seg_after = out[-1] - out[-2]
        cos = np.dot(seg_before, seg_after) / (
            np.linalg.norm(seg_before) * np.linalg.norm(seg_after)
        )
        assert cos > 0.9  # 既存フック方向をそのまま継続（回転しない）

    def test_zero_length_terminal_is_safe(self):
        stroke = np.array([[0, 0], [1, 0], [1, 0]], dtype=float)
        out = apply_hane(stroke, scale=10.0, config=FinishingConfig())
        assert out is not None


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


class TestArcLengthFromEnd:
    def test_terminal_is_zero(self):
        pts = np.array([[0, 0], [3, 0], [7, 0], [10, 0]], dtype=float)
        arc = arc_length_from_end(pts)
        assert arc[-1] == 0.0
        assert np.isclose(arc[0], 10.0)  # 始点は全長
        assert np.all(np.diff(arc) < 0)  # 終端へ向かって単調減少

    def test_matches_segment_lengths(self):
        pts = np.array([[0, 0], [0, 3], [4, 3]], dtype=float)  # 3 + 4
        arc = arc_length_from_end(pts)
        assert np.allclose(arc, [7.0, 4.0, 0.0])

    def test_single_point_safe(self):
        arc = arc_length_from_end(np.array([[1.0, 1.0]]))
        assert list(arc) == [0.0]


class TestContactProfile:
    """距離(mm)ベース: arc_from_end と lift_length(mm) で contact を決める。"""

    def _arc(self, n, total):
        # 等間隔ストロークの終端からの弧長(mm)
        return np.linspace(total, 0.0, n)

    def test_none_and_tome_all_contact(self):
        arc = self._arc(10, 9.0)
        for finish in (NONE, TOME):
            prof = contact_profile(finish, arc, lift_length=3.0)
            assert len(prof) == 10
            assert np.allclose(prof, 1.0)

    def test_harai_terminal_zero_and_monotonic(self):
        arc = self._arc(20, 19.0)
        prof = contact_profile(HARAI, arc, lift_length=5.0)
        assert len(prof) == 20
        assert np.isclose(prof[-1], 0.0, atol=1e-9)  # 終端は完全リフト
        assert np.all(np.diff(prof) <= 1e-9)  # 単調非増加
        # リフト区間(終端5mm)より手前は接触1.0
        assert np.allclose(prof[arc > 5.0 + 1e-9], 1.0)

    def test_size_independent(self):
        # 全長が lift_length/max_lift_fraction(=5mm) を超える「十分長い画」なら、
        # 点数や全長が違っても終端付近の contact は一致（固定 mm のリフト区間）。
        small = contact_profile(HARAI, self._arc(12, 6.0), lift_length=2.5)
        large = contact_profile(HARAI, self._arc(40, 20.0), lift_length=2.5)
        # 終端から 2.5mm 地点(=リフト境界)で両者 contact≈1.0、終端で 0
        assert np.isclose(small[-1], 0.0, atol=1e-9)
        assert np.isclose(large[-1], 0.0, atol=1e-9)

        # 終端から 1.25mm(リフト中間)地点の contact が一致(線形なら≈0.5)
        def contact_at(prof, arc, d):
            # arc は終端0→始点全長で降順。np.interp 用に昇順化して補間。
            return float(np.interp(d, arc[::-1], prof[::-1]))

        cs = contact_at(small, self._arc(12, 6.0), 1.25)
        cl = contact_at(large, self._arc(40, 20.0), 1.25)
        assert abs(cs - cl) < 1e-6  # サイズ非依存（十分長い画では arc/lift_length のみ）
        assert abs(cs - 0.5) < 1e-6

    def test_hane_steeper_than_harai(self):
        arc = self._arc(20, 19.0)
        harai = contact_profile(HARAI, arc, lift_length=5.0)
        hane = contact_profile(HANE, arc, lift_length=5.0)
        # リフト中間(終端2.5mm付近)ではねの方が接触小(二乗で急)
        i = int(np.argmin(np.abs(arc - 2.5)))
        assert hane[i] < harai[i]

    def test_short_stroke_keeps_solid_head(self):
        # 短い画(全長2mm)に長いlift_lengthを与えても、リフトは全長の
        # max_lift_fraction(=50%)で頭打ち。始点側は完全接触を保ち「全体が
        # 薄くなる」のを防ぐ（はね・払い分類された短画のかすれ対策）。
        arc = self._arc(5, 2.0)
        prof = contact_profile(HARAI, arc, lift_length=10.0)
        assert len(prof) == 5
        assert np.isclose(prof[-1], 0.0, atol=1e-9)  # 終端は完全リフト
        assert np.isclose(prof[0], 1.0, atol=1e-9)  # 始点は完全接触（濃い）
        # 始点側ほぼ半分は接触1.0（全長2mmの内、終端1mmのみリフト）
        assert np.allclose(prof[arc >= 1.0 + 1e-9], 1.0)

    def test_short_stroke_not_faint_over_whole_length(self):
        # 回帰: lift_length より短い画でも、平均接触が高く保たれる
        # （旧実装は始点から薄くなり字が「めっちゃうすく」なっていた）。
        arc = self._arc(10, 1.5)
        prof = contact_profile(HARAI, arc, lift_length=2.5)
        assert prof[0] == 1.0
        assert float(prof.mean()) > 0.6

    def test_short_stroke_guard(self):
        prof = contact_profile(HARAI, np.array([0.0]), lift_length=5.0)
        assert len(prof) == 1
        assert np.allclose(prof, 1.0)


class TestInferFinish:
    """軌跡からの筆法推定（かな用、kvg:type が無い字向け）。Y-UP。"""

    def test_vertical_hook_flicks_up_is_hane(self):
        # 縦に下りてから左上へ跳ね上げる＝はね
        stroke = np.array(
            [[0, 4], [0, 3], [0, 2], [0, 1], [0, 0], [-0.5, 0.5], [-1.0, 1.0]],
            dtype=float,
        )
        assert infer_finish_from_stroke(stroke) == HANE

    def test_diagonal_smooth_sweep_is_harai(self):
        # 右下へ滑らかに流れる長い斜め＝払い
        stroke = np.array([[0, 5], [1, 4], [2, 3], [3, 2], [4, 1], [5, 0]], dtype=float)
        assert infer_finish_from_stroke(stroke) == HARAI

    def test_horizontal_is_tome(self):
        stroke = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        assert infer_finish_from_stroke(stroke) == TOME

    def test_vertical_straight_down_is_tome(self):
        # 跳ねずに真下で止まる縦画＝とめ
        stroke = np.array([[0, 4], [0, 3], [0, 2], [0, 1], [0, 0]], dtype=float)
        assert infer_finish_from_stroke(stroke) == TOME

    def test_short_stroke_is_tome(self):
        assert infer_finish_from_stroke(np.array([[0, 0], [1, 0]], dtype=float)) == TOME

    def test_infer_finishes_list(self):
        strokes = [
            np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float),  # tome
            np.array([[0, 5], [1, 4], [2, 3], [3, 2], [4, 1], [5, 0]], dtype=float),  # harai
        ]
        assert infer_finishes(strokes) == [TOME, HARAI]

    def test_infer_finishes_empty(self):
        assert infer_finishes([]) == []
