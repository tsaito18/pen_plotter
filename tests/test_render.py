"""字形（幾何・KanjiVG・ユーザー筆跡）と文字描画・数式描画・プレビュー。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.gcode.config import PlotterConfig
from src.glyphs.geometric import LATIN_GLYPHS, SYMBOL_GLYPHS, latin_glyph, symbol_glyph
from src.glyphs.sources import KanjiVGStore, UserStrokeDB
from src.handwriting.augmentation import HandwritingAugmenter
from src.handwriting.finishing import HANE, HARAI, NONE, TOME
from src.layout.placement import CharPlacement, MathSpec
from src.render.char_renderer import CharRenderer, _enforce_horizontal_rise, waver_scale
from src.render.math_image import formula_aspect, formula_ink_em, render_latex_to_strokes
from src.render.positioning import X_HEIGHT, position_strokes
from src.render.preview import (
    WIDTH_MIN_RATIO,
    render_page_preview,
    stroke_contact,
    stroke_widths,
)

LS = 8.0


def _at(char: str, fs: float = 5.0, **kw) -> CharPlacement:
    return CharPlacement(char, 10.0, 100.0, fs, **kw)


def _renderer(**kw) -> CharRenderer:
    return CharRenderer(line_spacing=LS, **kw)


def _bbox(strokes: list[np.ndarray]) -> tuple[float, float, float, float]:
    pts = np.concatenate(strokes)
    return (*pts.min(axis=0), *pts.max(axis=0))


# --- 幾何字形 ---


def test_all_symbol_glyphs_are_valid_unit_strokes():
    bad = []
    for char in SYMBOL_GLYPHS:
        strokes = symbol_glyph(char)
        valid = strokes and all(s.ndim == 2 and s.shape[1] == 2 and len(s) >= 2 for s in strokes)
        x0, y0, x1, y1 = _bbox(strokes) if valid else (0, 0, 0, 0)
        # ζ 等は下へ伸びる
        if not valid or not (-0.1 <= x0 and x1 <= 1.1 and -0.25 <= y0 and y1 <= 1.1):
            bad.append(char)
    assert bad == []


def test_latin_glyphs_follow_x_height_cap_and_descender_bands():
    assert set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") <= set(LATIN_GLYPHS)
    top = {c: _bbox(latin_glyph(c))[3] for c in "acemnosuvwxz"}
    assert max(top.values()) - min(top.values()) < 0.05  # 小文字 x-height が揃う
    assert all(_bbox(latin_glyph(c))[1] < 0 for c in "gpqy")  # ディセンダ
    assert _bbox(latin_glyph("H"))[3] == pytest.approx(0.95)


def test_curly_braces_point_outward():
    left, right = symbol_glyph("{"), symbol_glyph("}")
    assert left and right
    (lx0, _, lx1, _), (rx0, _, rx1, _) = _bbox(left), _bbox(right)
    tip = np.concatenate(left)[np.argmin(np.concatenate(left)[:, 0])]
    assert tip[1] == pytest.approx(0.5, abs=0.05)  # { は中央の尖りが左へ出る
    assert np.isclose(lx0 + lx1, 2 - (rx0 + rx1))  # } は { の左右反転


def test_omega_is_one_rounded_stroke_without_a_top_bar():
    (omega,) = symbol_glyph("ω")  # 横棒があると ϖ に見える
    assert omega[0, 1] > 0.5 and omega[-1, 1] > 0.5  # 両端は上
    assert omega[:, 1].min() < 0.2


def test_s_is_not_mirrored():
    s = latin_glyph("S")[0]
    upper = s[s[:, 1] > 0.6]
    assert upper[:, 0].min() < 0.3  # 上半分は左に膨らむ（Ƨ ではない）


# --- 字形ソース ---


def test_kanjivg_store_loads_strokes_with_types(kanjivg_dir: Path):
    store = KanjiVGStore(kanjivg_dir)
    strokes, types = store.load("人")
    assert len(strokes) == 2 and types == ["㇒", "㇏"]
    assert store.load("無") == (None, [])
    assert store.load("/") == store.load("..") == (None, [])  # パスとして不正な字
    assert KanjiVGStore(None).load("人") == (None, [])


def test_user_db_prefers_the_most_careful_sample(user_strokes_root: Path):
    db = UserStrokeDB(user_strokes_root / "taro")
    assert "十" in db and len(db) == 2
    assert len(db.best_sample("十")[0]) == 30  # 点数の多いサンプル
    assert db.best_sample("無") is None


# --- 書き癖 ---


def _angle(stroke: np.ndarray) -> float:
    d = stroke[-1] - stroke[0]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def test_writing_style_tilts_horizontals_and_verticals_separately():
    from src.glyphs.style import WritingStyle

    h = np.array([[0.0, 0.5], [1.0, 0.5]])
    v = np.array([[0.5, 1.0], [0.5, 0.0]])
    out_h, out_v = WritingStyle(horizontal_rise_deg=7.0, vertical_lean_deg=-2.0).apply([h, v])
    assert _angle(out_h) == pytest.approx(7.0)  # 横画は右上がり
    assert _angle(out_v) == pytest.approx(-92.0)  # 縦画は上が右へ傾く
    assert WritingStyle().apply([h])[0] is h  # 恒等


def test_writing_style_is_estimated_from_the_users_samples(tmp_path: Path, kanjivg_dir: Path):
    from src.glyphs.style import WritingStyle, estimate_writing_style
    from tests.conftest import line, write_sample

    rise = np.tan(np.radians(7.0)) * 80  # 筆跡は Y-DOWN なので右上がり = y が減る
    for i in range(3):
        write_sample(tmp_path, "十", [line(10, 50, 90, 50 - rise), line(50, 10, 52, 90)], index=i)
    style = estimate_writing_style(UserStrokeDB(tmp_path), KanjiVGStore(kanjivg_dir), min_pairs=3)
    assert style.horizontal_rise_deg == pytest.approx(7.0, abs=0.5)
    assert style.vertical_lean_deg == pytest.approx(1.4, abs=0.5)  # 下端が右へずれた縦画
    empty = estimate_writing_style(UserStrokeDB(None), KanjiVGStore(kanjivg_dir))
    assert empty == WritingStyle()  # データが足りなければ変えない


def test_reference_glyphs_take_on_the_users_style(kanjivg_dir: Path):
    from src.glyphs.style import WritingStyle

    plain = _renderer(kanjivg_dir=kanjivg_dir, writing_style=WritingStyle()).render(_at("一"))
    styled = _renderer(kanjivg_dir=kanjivg_dir, writing_style=WritingStyle(7.0, 0.0))
    (stroke,) = styled.render(_at("一")).strokes
    assert _angle(stroke) == pytest.approx(7.0, abs=0.5)
    assert _angle(plain.strokes[0]) == pytest.approx(2.0, abs=0.5)  # 下限保証の 2° のみ


# --- 部品合成 ---


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[tuple[float, float]]]:
    """「口」を 3 画で（左縦 → 上と右の折れ → 下）。座標は呼び出し側の向きのまま。"""
    from tests.conftest import line

    return [
        line(x0, y1, x0, y0),
        line(x0, y1, x1, y1) + line(x1, y1, x1, y0)[1:],
        line(x0, y0, x1, y0),
    ]


@pytest.fixture
def parts_world(tmp_path: Path) -> tuple[Path, Path]:
    """KanjiVG（Y-UP）に「吅」「品」と部品表、本人は「吅」だけ書いた（Y-DOWN）。"""
    import json

    from src.glyphs.compose import COMPONENTS_FILE
    from tests.conftest import write_sample

    kvg, user = tmp_path / "kvg", tmp_path / "user"
    write_sample(kvg, "吅", _box(1, 3, 4, 7) + _box(6, 3, 9, 7))
    write_sample(kvg, "品", _box(3, 6, 7, 9) + _box(1, 1, 4, 5) + _box(6, 1, 9, 5))
    table = {
        "吅": {"strokes": 6, "parts": [["口", "left", [0, 1, 2]], ["口", "right", [3, 4, 5]]]},
        "品": {
            "strokes": 9,
            "parts": [
                ["口", "top", [0, 1, 2]],
                ["口", "left", [3, 4, 5]],
                ["口", "right", [6, 7, 8]],
            ],
        },
    }
    (kvg / COMPONENTS_FILE).write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    # 本人の「吅」: 右下がりに傾いた手書き（Y-DOWN）
    flip = [[(x * 10, 100 - y * 10 + x) for x, y in s] for s in _box(1, 3, 4, 7) + _box(6, 3, 9, 7)]
    write_sample(user, "吅", flip)
    return kvg, user


def test_unwritten_char_is_built_from_the_users_own_parts(parts_world):
    from src.glyphs.compose import ComponentComposer, load_components

    kvg, user = parts_world
    store = KanjiVGStore(kvg)
    composer = ComponentComposer(load_components(kvg), UserStrokeDB(user), store)
    built = composer.compose("品")
    assert built is not None and len(built) == 9
    reference, _ = store.load("品")
    # 各部品は KanjiVG の部品の位置・大きさに入り、線は本人のもの（参照そのものではない）
    for got, ref in zip(built, reference, strict=True):
        assert np.allclose(_bbox([got]), _bbox([ref]), atol=0.6)
    assert not np.allclose(built[3], reference[3], atol=0.05)
    assert composer.compose("吅") is None  # 本人が書いた字は自分自身からは組み立てない
    assert composer.compose("無") is None


def test_parts_written_in_a_different_stroke_order_are_not_used(parts_world):
    from src.glyphs.compose import ComponentComposer, load_components
    from tests.conftest import write_sample

    kvg, user = parts_world
    shuffled = _box(1, 3, 4, 7)[::-1] + _box(6, 3, 9, 7)[::-1]  # 書き順が逆
    write_sample(user, "吅", [[(x * 10, 100 - y * 10) for x, y in s] for s in shuffled], index=0)
    composer = ComponentComposer(load_components(kvg), UserStrokeDB(user), KanjiVGStore(kvg))
    assert composer.compose("品") is None  # 形の合わない部品は使わず、従来の経路に任せる


def test_renderer_uses_composed_glyphs_for_unwritten_chars(parts_world):
    kvg, user = parts_world
    r = _renderer(kanjivg_dir=kvg, user_strokes_dir=user)
    r.render(_at("品"))
    r.render(_at("吅"))
    assert r.coverage.composed == ["品"] and r.coverage.user_strokes == ["吅"]
    assert r.ink_width_ratio("品") is not None


# --- 配置 ---


def test_cjk_fits_cell_and_is_vertically_centered():
    tall = [np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1.0]])]
    (placed,) = position_strokes(tall, _at("漢"), LS)
    _x0, y0, _x1, y1 = _bbox([placed])
    assert y1 - y0 == pytest.approx(5.0)
    assert (y0 + y1) / 2 == pytest.approx(100.0 + LS / 2)


def _latin(char: str, strokes: list[np.ndarray]) -> tuple[float, float, float, float]:
    return _bbox(position_strokes(strokes, _at(char), LS))


def test_small_kana_sits_on_the_bottom_line_of_the_other_chars():
    from src.layout.char_metrics import effective_char_scale

    box = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
    kanji_bottom = _bbox(position_strokes(box, _at("漢"), LS))[1]
    small = _bbox(position_strokes(box, _at("っ", fs=5.0 * effective_char_scale("っ")), LS))
    assert small[1] == pytest.approx(kanji_bottom, abs=0.15)  # 下に落ち込まない


def test_latin_letters_sit_in_case_bands_whatever_the_sample_size():
    """本人サンプルは大きさがバラバラ（s が A より大きい等）でも、字種の帯に揃う。"""
    small_e = [np.array([[0.0, 0.0], [0.3, 0.1], [0.0, 0.3]])]
    big_e = [s * 3 for s in small_e]
    assert _latin("e", small_e) == pytest.approx(_latin("e", big_e))
    tall_box = [np.array([[0.0, 0.0], [0.4, 1.0]])]
    e, cap, p = (_latin(c, tall_box) for c in "eAp")
    baseline = cap[1]
    assert e[1] == pytest.approx(baseline)  # 小文字も大文字も同じベースライン
    assert (e[3] - baseline) == pytest.approx(0.55 * (cap[3] - baseline), rel=0.02)  # x-height
    assert p[1] < baseline < p[3] < cap[3]  # ディセンダ
    assert _latin("A", latin_glyph("A"))[1] == pytest.approx(baseline)  # 幾何英字も同じ帯
    # ギリシャ文字も英字と同じ帯（ω を漢字大に引き伸ばすと ∞、π は ∏ に見える）
    omega, phi = _latin("ω", symbol_glyph("ω")), _latin("φ", symbol_glyph("φ"))
    assert omega[1] == pytest.approx(baseline) and omega[3] == pytest.approx(e[3])
    assert phi[1] < baseline and phi[3] > e[3]


def test_wide_latin_samples_are_narrowed_to_their_advance():
    from src.layout.char_metrics import effective_char_scale, halfwidth_advance

    flat_m = [np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0], [3.0, 0.5]])]
    x0, _y0, x1, _y1 = _latin("m", flat_m)
    body = 5.0 / effective_char_scale("m")  # 字種・密度の倍率を外した本文サイズ
    assert x1 - x0 <= halfwidth_advance("m") * body
    assert x0 >= 10.0


def test_operators_are_small_and_sit_on_the_math_axis():
    """= + - 等をセル幅いっぱいに伸ばすと隣の字に触れる（ω=2 が一続きに見える）。"""
    cap = _latin("A", [np.array([[0.0, 0.0], [0.4, 1.0]])])
    baseline, cap_h = cap[1], cap[3] - cap[1]
    for op in "=+-<>×":
        x0, y0, x1, y1 = _bbox(position_strokes(symbol_glyph(op), _at(op), LS))
        assert x1 - x0 < 0.45 * cap_h
        assert baseline < (y0 + y1) / 2 < baseline + X_HEIGHT * cap_h


def test_nearly_equal_sign_is_an_operator_with_two_dots():
    """≒ は = に点を 2 つ（左上・右下）添えた形で、演算子と同じ軸に置く。"""
    strokes = symbol_glyph("≒")
    assert strokes is not None and len(strokes) == 4
    eq = _bbox(position_strokes(symbol_glyph("="), _at("="), LS))
    ne = _bbox(position_strokes(strokes, _at("≒"), LS))
    assert (ne[1] + ne[3]) / 2 == pytest.approx((eq[1] + eq[3]) / 2, abs=0.3)


def test_superscript_digits_are_small_and_raised(kanjivg_dir):
    """m/s² の ² は数字 2 を小さくして右肩に上げる（欠けて空白にならない）。"""
    r = _renderer(kanjivg_dir=kanjivg_dir)
    two = _bbox(r.render(_at("2")).strokes)
    sup = _bbox(r.render(_at("²")).strokes)
    minus = r.render(_at("⁻")).strokes
    assert r.coverage.missing_glyphs == []
    assert minus
    assert sup[3] - sup[1] < 0.7 * (two[3] - two[1])
    assert sup[1] > (two[1] + two[3]) / 2 - 0.1  # 下端が通常の数字の中ほどより上


def test_brackets_hug_the_text_inside_them():
    tall = [np.array([[0.0, 0.0], [-0.2, 0.5], [0.0, 1.0]])]
    opening = _bbox(position_strokes(tall, _at("（"), LS))
    closing = _bbox(position_strokes(tall, _at("）"), LS))
    cell_mid = 10.0 + 5.0 * 0.95 / 2
    assert opening[0] > cell_mid  # 開き括弧はセルの右（中身の側）に寄る
    assert closing[2] < cell_mid
    height = opening[3] - opening[1]
    assert 0.7 * 5.0 < height < 0.95 * 5.0  # 漢字より少し小さい
    assert (opening[1] + opening[3]) / 2 == pytest.approx(100.0 + LS / 2)


def test_corner_brackets_are_upright_and_sit_at_the_top_or_bottom():
    left = position_strokes(symbol_glyph("「"), _at("「"), LS)
    right = position_strokes(symbol_glyph("」"), _at("」"), LS)
    (stroke,) = left
    assert stroke[0, 1] == pytest.approx(stroke[1, 1])  # 横画から書き始め
    assert stroke[-1, 1] < stroke[0, 1]  # 縦画は下へ（┌ の形。└ ではない）
    mid = 100.0 + LS / 2
    assert _bbox(left)[1] > mid - 0.5 and _bbox(right)[3] < mid + 0.5
    assert _bbox(left)[0] > _bbox(right)[0]  # 「は右寄り、」は左寄り


def test_brackets_prefer_the_users_own_sample(tmp_path: Path):
    from tests.conftest import line, write_sample

    write_sample(tmp_path, "（", [line(30, 10, 20, 50) + line(20, 50, 30, 90)[1:]])
    r = _renderer(user_strokes_dir=tmp_path)
    r.render(_at("（"))
    r.render(_at("("))  # 半角も全角の本人サンプルで描く
    r.render(_at("）"))
    assert r.coverage.user_strokes == ["（", "("]
    assert r.coverage.geometric == ["）"]


def test_glyph_is_centered_in_the_slot_the_typesetter_reserved():
    box = [np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])]
    x0, _, x1, _ = _bbox(position_strokes(box, _at("り", advance=8.0), LS))
    assert (x0 + x1) / 2 == pytest.approx(10.0 + 4.0)


def test_slant_rotates_about_glyph_center():
    stroke = [np.array([[0.5, 0.0], [0.5, 1.0]])]
    upright = position_strokes(stroke, _at("丨"), LS)[0]
    tilted = position_strokes(stroke, _at("丨", slant=0.1), LS)[0]
    assert tilted.mean(axis=0) == pytest.approx(upright.mean(axis=0))
    assert tilted[0, 0] != pytest.approx(upright[0, 0])


def test_period_and_comma_are_easy_to_tell_apart():
    """「．」は小さな点、「，」は頭の点から左下へ払う形（どちらも短い斜線だと「、」と紛れる）。"""
    from src.layout.char_metrics import effective_char_scale

    fs_period, fs_comma = (5.0 * effective_char_scale(c) for c in "．，")
    (dot,) = position_strokes(symbol_glyph("．"), _at("．", fs=fs_period), LS)
    (comma,) = position_strokes(symbol_glyph("，"), _at("，", fs=fs_comma), LS)
    kanji_bottom = 100.0 + (LS - 5.0) / 2
    dx0, dy0, dx1, dy1 = _bbox([dot])
    assert max(dx1 - dx0, dy1 - dy0) < 0.5  # ペン幅で塗りつぶされる小さな点
    assert np.linalg.norm(dot[0] - dot[-1]) < 0.1  # 線ではなく閉じた点
    assert abs((dy0 + dy1) / 2 - kanji_bottom) < 0.4  # 字の下端（ベースライン）に置く
    assert (dx0 + dx1) / 2 < 10.0 + 5.0 * 0.95 / 2  # 枠の左寄り
    _, cy0, _, cy1 = _bbox([comma])
    assert cy1 - cy0 > 2.5 * (dy1 - dy0)  # 点より明らかに縦に長い
    head, tail = comma[0], comma[-1]
    assert head[1] > tail[1] and tail[0] < head[0]  # 頭から左下へ払う
    assert cy0 < kanji_bottom < cy1  # 頭はベースライン付近、尾はその下へ


# --- 描画経路 ---


def test_geometric_symbols_and_comma_harai():
    r = _renderer()
    plus = r.render(_at("＋"))  # 全角は半角記号の字形へ
    comma = r.render(_at("，"))
    assert len(plus.strokes) == 2 and plus.finishes == [NONE, NONE]
    assert comma.finishes == [HARAI]
    assert r.coverage.geometric == ["＋", "，"]


def test_route_priority_user_then_ml_then_kanjivg(kanjivg_dir, user_strokes_root, tiny_checkpoint):
    from src.model.inference import StrokeInference

    engine = StrokeInference.from_user_strokes(tiny_checkpoint, user_strokes_root / "taro")
    r = _renderer(
        kanjivg_dir=kanjivg_dir, user_strokes_dir=user_strokes_root / "taro", inference=engine
    )
    for ch in "十人a2λ無":
        r.render(_at(ch))
    cov = r.coverage
    assert cov.user_strokes == ["十", "a"]
    assert cov.ml_inference == ["人"]
    assert cov.kanjivg == ["2"]  # 数字は ML 変形しない
    assert cov.geometric == ["λ"]  # 幾何字形は ML より優先
    assert cov.missing_glyphs == ["無"]


def test_kanjivg_route_applies_brush_finishes(kanjivg_dir):
    rendered = _renderer(kanjivg_dir=kanjivg_dir).render(_at("人"))
    assert rendered.finishes == [HARAI, HARAI]  # 筆法は実機の Z リフトとプレビューの線幅に使う
    raw, _ = KanjiVGStore(kanjivg_dir).load("人")
    placed = position_strokes(raw, _at("人"), LS)
    # 払いを接線方向へ伸ばさない（本人の払いは KanjiVG と同じかやや短い。伸ばすと字が尖る）
    assert np.allclose(rendered.strokes[0][-1], placed[0][-1], atol=1e-6)
    digit = _renderer(kanjivg_dir=kanjivg_dir).render(_at("2"))
    assert set(digit.finishes) == {NONE}


def test_geometric_glyphs_are_drawn_by_hand_not_by_ruler():
    """サンプルの無い英字・記号も、定規の直線ではなく毎回少し違う手書きの線になる。"""
    r = _renderer(augmenter=HandwritingAugmenter(seed=0))
    z1, z2 = (r.render(_at("Z")).strokes[0] for _ in range(2))
    assert z1.shape != z2.shape or not np.allclose(z1, z2)
    ruler = position_strokes(latin_glyph("Z"), _at("Z"), LS)[0]
    d = np.diff(z1, axis=0)
    turns = np.abs((np.diff(np.arctan2(d[:, 1], d[:, 0])) + np.pi) % (2 * np.pi) - np.pi)
    assert len(z1) > len(ruler) and turns.max() < 2.0  # 角が丸い（素の Z は 135° の折れ）
    eq = r.render(_at("=")).strokes
    assert all(np.ptp(s[:, 1]) > 1e-3 for s in eq)  # 等号の横棒も完全な水平線ではない


def test_finishing_and_variation_do_not_change_the_char_size(kanjivg_dir):
    """払いの延長や揺らぎの後も、配置で決めた大きさを保つ（経路で大きさが変わらない）。"""
    raw, _ = KanjiVGStore(kanjivg_dir).load("人")
    expected = _bbox(position_strokes(raw, _at("人"), LS))
    r = _renderer(kanjivg_dir=kanjivg_dir, augmenter=HandwritingAugmenter(seed=0))
    for _ in range(5):
        x0, y0, x1, y1 = _bbox(r.render(_at("人")).strokes)
        assert max(x1 - x0, y1 - y0) == pytest.approx(expected[2] - expected[0], rel=0.02)


def test_line_segment_and_whitespace():
    r = _renderer()
    line = r.render(CharPlacement("", 0, 0, 5, line_segment=(0, 1, 10, 1)))
    assert line.strokes[0].tolist() == [[0, 1], [10, 1]]
    assert r.render(_at(" ")).strokes == [] and r.coverage.skipped == [" "]


def test_japanese_only_mode_skips_latin_symbols_and_math(kanjivg_dir):
    r = _renderer(kanjivg_dir=kanjivg_dir, japanese_only=True)
    math = CharPlacement("", 0, 0, 5, math=MathSpec("x^2", (0, 0, 5, 5)))
    for p in [_at("a"), _at("+"), math]:
        assert r.render(p).strokes == []
    for ch in ["人", "2", "，"]:
        assert r.render(_at(ch)).strokes
    table_line = CharPlacement("", 0, 0, 5, line_segment=(0, 0, 1, 0))
    assert r.render(table_line).strokes


def test_distortion_and_instance_variation_make_repeats_differ(kanjivg_dir):
    clean = _renderer(kanjivg_dir=kanjivg_dir)
    wavy = _renderer(
        kanjivg_dir=kanjivg_dir, augmenter=HandwritingAugmenter(seed=0), instance_variation=1.0
    )
    a1, a2 = clean.render(_at("十")).strokes, clean.render(_at("十")).strokes
    b1, b2 = wavy.render(_at("十")).strokes, wavy.render(_at("十")).strokes
    assert all(np.array_equal(x, y) for x, y in zip(a1, a2, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(b1, b2, strict=True))


def test_waver_scale_decreases_for_dense_characters():
    assert waver_scale(5) == waver_scale(10) == 1.0
    assert 0.5 < waver_scale(15) < 1.0
    assert waver_scale(30) == 0.5


@pytest.mark.parametrize(
    ("stroke", "changed"),
    [
        (np.array([[0.0, 0.0], [5.0, -0.2]]), True),  # 右下がりの横棒は起こす
        (np.array([[0.0, 0.0], [5.0, 1.0]]), False),  # 既に右上がり
        (np.array([[0.0, 0.0], [0.0, 5.0]]), False),  # 縦画
        (np.array([[0.0, 0.0], [0.5, 0.0]]), False),  # 短すぎる
    ],
)
def test_horizontal_strokes_get_minimum_rise(stroke: np.ndarray, changed: bool):
    (out,) = _enforce_horizontal_rise([stroke], font_size=5.0)
    assert (not np.allclose(out, stroke)) == changed
    if changed:
        angle = np.degrees(np.arctan2(out[1, 1] - out[0, 1], out[1, 0] - out[0, 0]))
        assert angle == pytest.approx(2.0)


# --- 数式描画 ---


def test_formula_metrics():
    assert formula_aspect(r"x + y + z") > formula_aspect(r"\frac{a}{b}") > 0
    assert formula_ink_em("u") < formula_ink_em("A") < 1.0  # 小文字は em より小さい
    assert formula_aspect("") == 0.0


def test_inline_math_renders_inside_its_box():
    strokes = render_latex_to_strokes(r"E = mc^2", (10.0, 100.0, 20.0, 5.0), "baseline")
    x0, _y0, x1, _y1 = _bbox(strokes)
    assert len(strokes) >= 5
    assert 10.0 - 0.1 <= x0 and x1 <= 10.0 + 5.0 * formula_aspect("E = mc^2") + 0.1


def _handwritten(src: str, align: str = "center", bar_y: float | None = None, fs: float = 5.0):
    from src.layout.mathtext import (
        MATH_BLOCK_CAP_RATIO,
        MATH_INLINE_CAP_RATIO,
        handwrite_draw_width_mm,
    )

    ratio = MATH_INLINE_CAP_RATIO if align == "baseline" else MATH_BLOCK_CAP_RATIO
    w = handwrite_draw_width_mm(src, fs, ratio)
    spec = MathSpec(src, (10.0, 100.0, w, 10.0), align, handwritten=True, fraction_bar_y=bar_y)
    return CharPlacement("", 10.0, 100.0, fs, math=spec)


def _straight_horizontal(strokes):
    return [s for s in strokes if len(s) == 2 and abs(s[0, 1] - s[1, 1]) < 1e-9]


def test_handwritten_fraction_puts_numerator_over_a_drawn_bar(kanjivg_dir):
    r = _renderer(kanjivg_dir=kanjivg_dir)
    placement = _handwritten(r"\dfrac{a}{\sqrt{b}}", bar_y=105.0)
    strokes = r.render(placement).strokes
    (bar,) = [s for s in _straight_horizontal(strokes) if np.ptp(s[:, 0]) > 1.0]
    assert bar[0, 1] == pytest.approx(105.0)  # 分数線は指定の罫線上
    above = [s for s in strokes if s.min(axis=0)[1] > 105.0]
    x0, _, x1, _ = _bbox(above)
    # 分母の √ の中身を寄せても、分子は分数線の中央のまま
    assert (x0 + x1) / 2 == pytest.approx(bar[:, 0].mean(), abs=0.6)
    assert any(s.min(axis=0)[1] < 105.0 and len(s) == 4 for s in strokes)  # √ の折れ線
    assert r.coverage.geometric == [r"\dfrac{a}{\sqrt{b}}"]


def test_handwritten_root_roof_covers_its_content():
    r = _renderer()
    strokes = r.render(_handwritten(r"\sqrt{x+1}")).strokes
    (root,) = [s for s in strokes if len(s) == 4]
    content = [s for s in strokes if s is not root]
    _x0, _y0, x1, y1 = _bbox(content)
    assert root[2, 0] < _bbox(content)[0] + 0.3  # 中身は屋根の左端から始まる
    assert x1 - 0.2 < root[3, 0] < x1 + 0.8  # 屋根は中身を右端まで覆う（余白は小さく）
    assert root[2, 1] == pytest.approx(root[3, 1]) and root[2, 1] > y1


def test_handwritten_inline_math_sits_on_the_body_baseline():
    r = _renderer()
    strokes = r.render(_handwritten(r"x^{2}", "baseline")).strokes
    body_x = _bbox(r.render(_at("x", fs=5.0 * 0.8)).strokes)
    xs = _bbox(strokes)
    assert xs[1] == pytest.approx(body_x[1], abs=0.25)  # x の下端が本文の x と揃う
    assert xs[3] > body_x[3] + 0.5  # 上付きの 2 は上へ出る
    assert 10.0 - 0.3 <= xs[0] and xs[2] <= 10.0 + placement_width(r"x^{2}") + 0.3


def test_handwritten_accents_and_primes_are_drawn_over_their_letters():
    r = _renderer()
    plain = r.render(_handwritten(r"x", "baseline")).strokes  # ベースラインが動かない配置で比べる
    for src in (r"\bar{x}", r"\hat{x}", r"\vec{x}", r"\dot{x}", r"\tilde{x}"):
        strokes = r.render(_handwritten(src, "baseline")).strokes
        assert len(strokes) > len(plain), src  # アクセントが消えない
        top = max(s[:, 1].max() for s in strokes)
        assert top > _bbox(plain)[3] + 0.2, src  # 字の上に乗る
    prime = r.render(_handwritten(r"f'")).strokes
    f_only = r.render(_handwritten(r"f")).strokes
    assert len(prime) == len(f_only) + 1  # プライムは短い 1 画（0 ではない）


def test_handwritten_decimal_point_is_a_small_dot_like_the_body_period():
    r = _renderer()
    strokes = r.render(_handwritten(r"3.14", "baseline")).strokes
    # 本文の「．」と同じ小さな丸（線にしない）
    dots = [s for s in strokes if np.allclose(s[0], s[-1]) and np.ptp(s, axis=0).max() < 0.5]
    assert len(dots) == 1


def test_handwritten_large_brackets_span_their_content():
    r = _renderer()
    for src in (r"\left[\frac{a}{b}\right]", r"\left\{\frac{a}{b}\right\}"):
        strokes = r.render(_handwritten(src)).strokes
        inner = r.render(_handwritten(r"\frac{a}{b}")).strokes
        bracket_h = max(np.ptp(s[:, 1]) for s in strokes)
        assert bracket_h > 0.9 * (_bbox(inner)[3] - _bbox(inner)[1]), src


def placement_width(src: str) -> float:
    from src.layout.mathtext import MATH_INLINE_CAP_RATIO, handwrite_draw_width_mm

    return handwrite_draw_width_mm(src, 5.0, MATH_INLINE_CAP_RATIO)


# --- プレビュー ---


def test_preview_width_tapers_like_the_z_lift():
    cfg = PlotterConfig(finish_strength=1.0)
    stroke = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
    tome, harai, hane = (stroke_widths(stroke, f, cfg) for f in (TOME, HARAI, HANE))
    assert len(set(tome)) == 1
    assert harai[-1] < harai[0] and hane[-1] < hane[0]
    half = stroke_widths(stroke, HARAI, PlotterConfig(finish_strength=0.5))
    assert harai[-1] < half[-1] < half[0]  # 強さに比例して抜ける
    assert len(set(stroke_widths(stroke, HARAI, PlotterConfig()))) == 1  # 既定は抜かない
    assert stroke_widths(stroke[:1], HARAI, cfg) == []


def test_preview_width_is_the_contact_ratio_scaled():
    cfg = PlotterConfig(finish_strength=1.0)  # 払いを抜く設定（既定は抜かない）
    stroke = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
    contact = stroke_contact(stroke, HARAI, cfg)
    assert contact.shape == (49,) and contact.max() == 1.0 and contact[-1] < 1.0
    pen = cfg.pen_width_mm
    expected = pen * (WIDTH_MIN_RATIO + (1 - WIDTH_MIN_RATIO) * contact)  # 線幅は mm
    assert np.allclose(stroke_widths(stroke, HARAI, cfg), expected)
    # 既定のペン幅は本人の手書きレポートのスキャン実測（同じペンで描く）
    assert PlotterConfig().pen_width_mm == pytest.approx(0.35)
    assert stroke_contact(stroke[:1], HARAI, cfg).shape == (0,)


def test_preview_draws_lines_at_the_real_pen_width(tmp_path: Path):
    """プレビュー画像上の線の太さが、ペン幅(mm)と同じ実寸になる（図の大きさに依らない）。"""
    from PIL import Image

    path = tmp_path / "page.png"
    stroke = np.column_stack([np.linspace(20, 190, 200), np.full(200, 150.0)])
    cfg = PlotterConfig(pen_width_mm=0.6)
    render_page_preview([stroke], [NONE], path, config=cfg)
    img = np.asarray(Image.open(path).convert("L"))
    frame = np.where((img < 60).mean(axis=0) > 0.5)[0]  # 用紙の外枠（縦線）
    px_per_mm = (frame.max() - frame.min()) / 210
    column = img[:, img.shape[1] // 2] < 128
    rows = np.where(column)[0]
    line = rows[(rows > img.shape[0] * 0.3) & (rows < img.shape[0] * 0.7)]
    assert len(line) / px_per_mm == pytest.approx(0.6, abs=0.12)


def test_render_page_preview_writes_png(tmp_path: Path):
    path = tmp_path / "page.png"
    stroke = np.column_stack([np.linspace(10, 50, 20), np.full(20, 100.0)])
    render_page_preview([stroke, stroke[:1]], [HARAI], path, config=PlotterConfig())
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- 字形欠損・置換の不具合（回帰） ---


@pytest.mark.parametrize(("fullwidth", "ascii_char"), [("２", "2"), ("＜", "<"), ("＞", ">")])
def test_fullwidth_chars_use_ascii_glyphs(kanjivg_dir, fullwidth: str, ascii_char: str):
    r = _renderer(kanjivg_dir=kanjivg_dir)
    assert len(r.render(_at(fullwidth)).strokes) == len(r.render(_at(ascii_char)).strokes) > 0
    assert r.coverage.missing_glyphs == []


def test_substituted_chars_keep_their_slant():
    r = _renderer()
    upright = r.render(_at("＝")).strokes
    tilted = r.render(_at("＝", slant=0.2)).strokes
    assert not np.allclose(upright[0], tilted[0])


def test_two_digit_circled_numbers_draw_both_digits(kanjivg_dir):
    r = _renderer(kanjivg_dir=kanjivg_dir)
    one = r.render(_at("①")).strokes
    twelve = r.render(_at("⑫")).strokes
    assert len(one) == 1 + 1  # 円 + 「1」
    assert len(twelve) == 1 + 1 + 2  # 円 + 「1」 + 「2」
    inner = np.concatenate(twelve[1:])
    circle = twelve[0]
    assert circle[:, 0].min() < inner[:, 0].min() and inner[:, 0].max() < circle[:, 0].max()
