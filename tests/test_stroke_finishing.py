"""kvg:type 分類と終端加工のテスト。"""

import numpy as np
import pytest

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
    entry_modulation,
    infer_finish_from_stroke,
    infer_finishes,
    pressure_modulation,
)


class TestEntryModulation:
    """入筆: 始筆の接触をランプさせ、軽く入って濃くなる筆の入りを作る。"""

    def _stroke(self, n=20, total=10.0):
        return np.column_stack([np.linspace(0.0, total, n), np.zeros(n)])

    def test_zero_strength_identity(self):
        m = entry_modulation(self._stroke(), entry_length=1.0, strength=0.0)
        assert np.allclose(m, 1.0)

    def test_start_lighter_then_full(self):
        m = entry_modulation(self._stroke(), entry_length=2.0, strength=0.5)
        assert m[0] == pytest.approx(0.5)  # 始点 = 1 - strength
        assert m[-1] == pytest.approx(1.0)  # entry_length 以降は満額
        assert np.all(np.diff(m) >= -1e-9)  # 単調非減少

    def test_short_stroke_safe(self):
        assert np.allclose(entry_modulation(np.array([[0.0, 0.0]]), 1.0, 0.5), 1.0)


class TestPressureModulation:
    """画内の筆圧（濃淡）変調。決定論的で preview/G-code が一致する。"""

    def _down(self, n=20):
        # 下ろし画（Y-UP で Y 減少）
        return np.column_stack([np.zeros(n), np.linspace(10.0, 0.0, n)])

    def _up(self, n=20):
        return np.column_stack([np.zeros(n), np.linspace(0.0, 10.0, n)])

    def test_zero_amplitude_is_identity(self):
        m = pressure_modulation(self._down(), amplitude=0.0)
        assert np.allclose(m, 1.0)
        assert len(m) == 20

    def test_range_within_bounds(self):
        amp = 0.4
        m = pressure_modulation(self._down(), amplitude=amp)
        assert m.max() <= 1.0 + 1e-9
        assert m.min() >= 1.0 - amp - 1e-9

    def test_deterministic(self):
        s = self._down()
        assert np.allclose(pressure_modulation(s, 0.4), pressure_modulation(s, 0.4))

    def test_downstroke_darker_than_upstroke(self):
        # 下ろしは濃く(接触大)、上げは薄く(接触小)＝人の筆圧
        down = pressure_modulation(self._down(), 0.4).mean()
        up = pressure_modulation(self._up(), 0.4).mean()
        assert down > up

    def test_short_stroke_safe(self):
        assert np.allclose(pressure_modulation(np.array([[0.0, 0.0]]), 0.4), 1.0)


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
        # 曲線(イーズイン)の補間誤差を抑えるため十分密にサンプリングする。
        small_arc, large_arc = self._arc(97, 6.0), self._arc(161, 20.0)
        small = contact_profile(HARAI, small_arc, lift_length=2.5)
        large = contact_profile(HARAI, large_arc, lift_length=2.5)
        # 終端から 2.5mm 地点(=リフト境界)で両者 contact≈1.0、終端で 0
        assert np.isclose(small[-1], 0.0, atol=1e-9)
        assert np.isclose(large[-1], 0.0, atol=1e-9)

        # 終端から 1.25mm(リフト中間)地点の contact が一致
        def contact_at(prof, arc, d):
            # arc は終端0→始点全長で降順。np.interp 用に昇順化して補間。
            return float(np.interp(d, arc[::-1], prof[::-1]))

        cs = contact_at(small, small_arc, 1.25)
        cl = contact_at(large, large_arc, 1.25)
        # サイズ非依存（式は arc/eff_lift のみ）。差は補間サンプル密度の違いのみ。
        assert abs(cs - cl) < 2e-3
        # 払いは二乗イーズイン contact=1-(1-t)^2。t=0.5 → 1-0.25=0.75
        assert abs(cs - 0.75) < 2e-3

    def test_harai_ease_in_quadratic(self):
        # 払いの上がりは線形でなく二乗カーブ（イーズイン）: 終端付近まで接触を
        # 高く保ち、終端直前で曲線的に抜ける。lift=(1-t)^2 で「x^2 みたいに上がる」。
        # 全長(12) > lift_length/max_lift_fraction(=10) なので eff_lift=lift_length=5。
        arc = self._arc(241, 12.0)
        prof = contact_profile(HARAI, arc, lift_length=5.0)

        def contact_at(d):
            return float(np.interp(d, arc[::-1], prof[::-1]))

        # 終端0・境界1、各 t での値が 1-(1-t)^2 に一致
        for d, t in [(0.0, 0.0), (1.25, 0.25), (2.5, 0.5), (3.75, 0.75), (5.0, 1.0)]:
            assert abs(contact_at(d) - (1.0 - (1.0 - t) ** 2)) < 1e-3
        # 線形(=t)より各中間点で接触が高い（=上がりが緩やか・遅い）
        assert contact_at(2.5) > 0.5

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
