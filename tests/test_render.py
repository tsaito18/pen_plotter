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
from src.render.preview import render_page_preview, stroke_widths

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
    assert KanjiVGStore(None).load("人") == (None, [])


def test_user_db_prefers_the_most_careful_sample(user_strokes_root: Path):
    db = UserStrokeDB(user_strokes_root / "taro")
    assert "十" in db and len(db) == 2
    assert len(db.best_sample("十")[0]) == 30  # 点数の多いサンプル
    assert db.best_sample("無") is None


# --- 配置 ---


def test_cjk_fits_cell_and_is_vertically_centered():
    tall = [np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1.0]])]
    (placed,) = position_strokes(tall, _at("漢"), LS)
    _x0, y0, _x1, y1 = _bbox([placed])
    assert y1 - y0 == pytest.approx(5.0)
    assert (y0 + y1) / 2 == pytest.approx(100.0 + LS / 2)


def _latin(char: str, strokes: list[np.ndarray]) -> tuple[float, float, float, float]:
    return _bbox(position_strokes(strokes, _at(char), LS))


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


def test_slant_rotates_about_glyph_center():
    stroke = [np.array([[0.5, 0.0], [0.5, 1.0]])]
    upright = position_strokes(stroke, _at("丨"), LS)[0]
    tilted = position_strokes(stroke, _at("丨", slant=0.1), LS)[0]
    assert tilted.mean(axis=0) == pytest.approx(upright.mean(axis=0))
    assert tilted[0, 0] != pytest.approx(upright[0, 0])


def test_period_is_a_short_dash_not_a_circle():
    (dot,) = position_strokes(symbol_glyph("．"), _at("．"), LS)
    assert len(dot) == 2 and np.linalg.norm(dot[1] - dot[0]) < 1.0


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
    assert rendered.finishes == [HARAI, HARAI]
    raw, _ = KanjiVGStore(kanjivg_dir).load("人")
    assert len(rendered.strokes[0]) > len(raw[0])  # 払いの延長
    digit = _renderer(kanjivg_dir=kanjivg_dir).render(_at("2"))
    assert set(digit.finishes) == {NONE}


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


# --- プレビュー ---


def test_preview_width_tapers_like_the_z_lift():
    cfg = PlotterConfig()
    stroke = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
    tome, harai, hane = (stroke_widths(stroke, f, cfg) for f in (TOME, HARAI, HANE))
    assert len(set(tome)) == 1
    assert harai[-1] < harai[0] and hane[-1] < hane[0]
    assert stroke_widths(stroke[:1], HARAI, cfg) == []


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
