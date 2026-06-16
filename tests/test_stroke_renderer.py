"""StrokeRenderer の単体テスト。"""

import json

import numpy as np
import pytest

from src.layout.typesetter import CharPlacement
from src.ui.stroke_renderer import StrokeRenderer


def _create_user_stroke_json(base_dir, char, strokes, suffix="001", stroke_types=None):
    char_dir = base_dir / char
    char_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "character": char,
        "strokes": [
            [
                {"x": float(p[0]), "y": float(p[1]), "pressure": 1.0, "timestamp": 0.0}
                for p in stroke
            ]
            for stroke in strokes
        ],
        "metadata": {},
    }
    if stroke_types is not None:
        data["stroke_types"] = stroke_types
    (char_dir / f"{char}_{suffix}.json").write_text(json.dumps(data), encoding="utf-8")


class TestStrokeRendererInit:
    def test_creates_without_args(self):
        renderer = StrokeRenderer()
        assert renderer._user_stroke_db == {}
        assert renderer._inference is None

    def test_loads_user_stroke_db(self, tmp_path):
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(user_dir, "あ", [[[0, 0], [100, 100]]])
        renderer = StrokeRenderer(user_strokes_dir=user_dir)
        assert "あ" in renderer._user_stroke_db


class TestStrokeRendererMethods:
    def test_normalize_strokes_to_unit(self):
        strokes = [np.array([[0.0, 0.0], [100.0, 100.0]])]
        result = StrokeRenderer._normalize_strokes_to_unit(strokes)
        all_pts = np.concatenate(result)
        assert all_pts.min() >= -0.01
        assert all_pts.max() <= 1.01

    def test_rect_fallback(self):
        from src.layout.typesetter import CharPlacement

        p = CharPlacement(char="漢", x=10.0, y=20.0, font_size=5.0)
        result = StrokeRenderer._rect_fallback(p)
        assert len(result) == 1
        assert result[0].shape == (5, 2)

    def test_position_strokes_slant_rotates(self):
        from src.layout.typesetter import CharPlacement

        renderer = StrokeRenderer()
        strokes = [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.5], [1.0, 0.5]])]
        straight = renderer._position_strokes(
            strokes, CharPlacement(char="a", x=10.0, y=20.0, font_size=5.0, slant=0.0)
        )
        slanted = renderer._position_strokes(
            strokes, CharPlacement(char="a", x=10.0, y=20.0, font_size=5.0, slant=0.2)
        )
        # 傾きで座標が変わる（文字内の全画が回転）
        assert not np.allclose(np.concatenate(straight), np.concatenate(slanted))
        # 点数・本数は不変
        assert [s.shape for s in straight] == [s.shape for s in slanted]

    def _rendered_height(self, strokes):
        pts = np.concatenate(strokes, axis=0)
        return float(pts[:, 1].max() - pts[:, 1].min())

    def test_ascii_lowercase_shorter_than_uppercase(self):
        """論理フィットで小文字 a (x-height) は大文字 A (cap height) より低く描かれる。"""
        renderer = StrokeRenderer()
        fs = 5.0
        lower = renderer._ascii_letter_strokes("a")
        upper = renderer._ascii_letter_strokes("A")
        lower_pos = renderer._position_strokes(
            lower, CharPlacement(char="a", x=10.0, y=20.0, font_size=fs), logical_ascii=True
        )
        upper_pos = renderer._position_strokes(
            upper, CharPlacement(char="A", x=10.0, y=20.0, font_size=fs), logical_ascii=True
        )

        assert self._rendered_height(lower_pos) < self._rendered_height(upper_pos)

    def test_ascii_uppercase_near_cap_height(self):
        """大文字 A は cap height 基準で論理高が font_size の概ね 0.8〜1.0 倍になる。

        横は半角セル幅で抑えるため厳密 font_size には届かないが、x-height 系の
        小文字より明確に高い。
        """
        renderer = StrokeRenderer()
        fs = 5.0
        upper = renderer._ascii_letter_strokes("A")
        upper_pos = renderer._position_strokes(
            upper, CharPlacement(char="A", x=10.0, y=20.0, font_size=fs), logical_ascii=True
        )

        h = self._rendered_height(upper_pos)
        assert 0.8 * fs <= h <= 1.0 * fs

    def test_ascii_descender_extends_below_baseline(self):
        """ディセンダ字 g は論理フィットでもベースライン下へ伸びる形を保つ。"""
        renderer = StrokeRenderer()
        fs = 5.0
        glyph = renderer._ascii_letter_strokes("g")
        no_desc = renderer._ascii_letter_strokes("o")
        g_pos = renderer._position_strokes(
            glyph, CharPlacement(char="g", x=10.0, y=20.0, font_size=fs), logical_ascii=True
        )
        o_pos = renderer._position_strokes(
            no_desc, CharPlacement(char="o", x=10.0, y=20.0, font_size=fs), logical_ascii=True
        )
        # g のディセンダは o の最下点より下に出る
        assert np.concatenate(g_pos, axis=0)[:, 1].min() < np.concatenate(o_pos, axis=0)[:, 1].min()

    def test_logical_ascii_default_off_preserves_bbox_fit(self):
        """logical_ascii 既定 False は実bboxフィットで、論理フィットと結果が異なる。

        小文字 a は実bboxフィットだとセル内で拡大され、論理フィット(x-height 抑制)
        より高く描かれる。
        """
        renderer = StrokeRenderer()
        fs = 5.0
        glyph = renderer._ascii_letter_strokes("a")
        bbox_pos = renderer._position_strokes(
            glyph, CharPlacement(char="a", x=10.0, y=20.0, font_size=fs)
        )
        logical_pos = renderer._position_strokes(
            glyph, CharPlacement(char="a", x=10.0, y=20.0, font_size=fs), logical_ascii=True
        )
        # 実bboxフィットは小文字 a を論理フィット(x-height)より高く拡大する
        assert self._rendered_height(bbox_pos) > self._rendered_height(logical_pos)

    def test_cjk_unaffected_by_logical_fit(self):
        """CJK 字形は logical_ascii の有無に関わらず実bboxフィットのまま（非回帰）。"""
        renderer = StrokeRenderer()
        fs = 5.0
        # 縦長字形（横律速を避け実bboxフィットで高さ＝font_size になる形）。
        strokes = [np.array([[0.3, 0.0], [0.5, 1.0]]), np.array([[0.3, 0.0], [0.6, 0.5]])]
        default_pos = renderer._position_strokes(
            strokes, CharPlacement(char="漢", x=10.0, y=20.0, font_size=fs)
        )
        flagged_pos = renderer._position_strokes(
            strokes, CharPlacement(char="漢", x=10.0, y=20.0, font_size=fs), logical_ascii=True
        )
        # CJK は logical_ascii を無視し、実bboxフィット（高さ＝font_size）を維持
        assert self._rendered_height(default_pos) == pytest.approx(fs, rel=0.02)
        assert np.allclose(np.concatenate(default_pos), np.concatenate(flagged_pos))

    # 単位系字形の x-height 上端基準。_position_ascii_logical は字形 y をそのまま
    # 論理高に写すため、x-height 小文字の上端が高いと「大文字混じり」に見える。
    _XHEIGHT_TOP = 0.70
    # 描画位置のディセンダ体（g/p/q/y）本体上端は x-height に揃えるが、circle 近似
    # の数値誤差を許容する上限。
    _XHEIGHT_TOP_MAX = 0.72

    @staticmethod
    def _glyph_top(strokes):
        return float(np.concatenate(strokes, axis=0)[:, 1].max())

    @staticmethod
    def _glyph_bottom(strokes):
        return float(np.concatenate(strokes, axis=0)[:, 1].min())

    @pytest.mark.parametrize("char", ["a", "c", "e", "o", "n", "u", "r", "v", "w", "z", "x", "m"])
    def test_xheight_lowercase_top_capped(self, char):
        """x-height 小文字の上端 y が x-height 基準（≈0.72 以下）に収まる。"""
        renderer = StrokeRenderer()
        top = self._glyph_top(renderer._ascii_letter_strokes(char))
        assert top <= self._XHEIGHT_TOP_MAX, f"{char!r} top={top}"

    @pytest.mark.parametrize("char", ["g", "p", "q", "y"])
    def test_descender_body_top_capped(self, char):
        """ディセンダ体（g/p/q/y）の本体上端も x-height 基準に揃う。"""
        renderer = StrokeRenderer()
        top = self._glyph_top(renderer._ascii_letter_strokes(char))
        assert top <= self._XHEIGHT_TOP_MAX, f"{char!r} top={top}"

    @pytest.mark.parametrize("char", ["g", "p", "q", "y", "j"])
    def test_descender_extends_below_baseline_in_unit(self, char):
        """ディセンダ体の下端は単位系でベースライン(y=0)より下に出る。"""
        renderer = StrokeRenderer()
        bottom = self._glyph_bottom(renderer._ascii_letter_strokes(char))
        assert bottom < 0.0, f"{char!r} bottom={bottom}"

    @pytest.mark.parametrize("char", ["b", "d", "h", "k", "l"])
    def test_ascender_lowercase_top_high(self, char):
        """アセンダ小文字（b/d/h/k/l）の上端は cap height 付近（≥0.85）。"""
        renderer = StrokeRenderer()
        top = self._glyph_top(renderer._ascii_letter_strokes(char))
        assert top >= 0.85, f"{char!r} top={top}"

    def test_t_top_in_ascender_band(self):
        """t は伝統的に cap より少し低いアセンダ帯（0.78〜0.95）の上端を持つ。"""
        renderer = StrokeRenderer()
        top = self._glyph_top(renderer._ascii_letter_strokes("t"))
        assert 0.78 <= top <= 0.95, f"t top={top}"

    def test_xheight_lowercase_tops_uniform(self):
        """代表的な x-height 小文字群の上端 y が互いに揃う（最大-最小が小さい）。"""
        renderer = StrokeRenderer()
        chars = ["a", "c", "e", "o", "n", "u"]
        tops = [self._glyph_top(renderer._ascii_letter_strokes(ch)) for ch in chars]
        assert max(tops) - min(tops) < 0.12, dict(zip(chars, tops))

    def test_math_symbol_strokes(self):
        renderer = StrokeRenderer()
        result = renderer._math_symbol_strokes("π")
        assert result is not None
        assert len(result) >= 1

    def test_simple_punct_strokes(self):
        renderer = StrokeRenderer()
        result = renderer._simple_punct_strokes("。")
        assert result is not None

    def test_fullwidth_punct_match_touten_kuten(self):
        """全角「，」「．」は読点「、」句点「。」と同一形で描く（本文正規化先）。"""
        renderer = StrokeRenderer()
        comma = renderer._simple_punct_strokes("，")
        touten = renderer._simple_punct_strokes("、")
        assert comma is not None and touten is not None
        assert np.allclose(comma[0], touten[0])

        period = renderer._simple_punct_strokes("．")
        kuten = renderer._simple_punct_strokes("。")
        assert period is not None and kuten is not None
        assert np.allclose(period[0], kuten[0])

    @pytest.mark.parametrize("char", ["、", "，", ","])
    def test_comma_punct_slants_left_down_and_stays_short(self, char):
        renderer = StrokeRenderer()
        strokes = renderer._simple_punct_strokes(char)
        assert strokes is not None

        raw = strokes[0]
        assert raw.shape == (2, 2)
        assert raw[1, 0] < raw[0, 0]
        assert raw[1, 1] < raw[0, 1]

        placement = CharPlacement(char=char, x=10.0, y=20.0, font_size=6.0)
        positioned = renderer._position_strokes(strokes, placement)[0]
        length = np.linalg.norm(positioned[1] - positioned[0])
        assert 2.5 <= length <= 2.9
        assert length == pytest.approx(2.7, rel=0.06)
        assert 1.55 <= np.ptp(positioned[:, 0]) <= 1.7
        assert 2.1 <= np.ptp(positioned[:, 1]) <= 2.35
        assert positioned[1, 0] < positioned[0, 0]
        assert positioned[1, 1] < positioned[0, 1]

    @pytest.mark.parametrize("char", ["、", "，", ","])
    def test_comma_punct_finish_is_harai(self, char):
        renderer = StrokeRenderer()
        placement = CharPlacement(char=char, x=10.0, y=20.0, font_size=6.0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert len(strokes) == 1
        assert finishes == ["harai"]

    def test_simple_paren_strokes(self):
        from src.layout.typesetter import CharPlacement

        renderer = StrokeRenderer()
        p = CharPlacement(char="(", x=10.0, y=20.0, font_size=5.0)
        result = renderer._simple_paren_strokes("(", p)
        assert result is not None

    def test_paren_open_direction(self):
        # 「(」は中央が左に凸（開口は右向き）、「)」は中央が右に凸（開口は左向き）。
        # 単位セル(0..1)の生座標で膨らみの向きを検証する。
        from src.layout.typesetter import CharPlacement

        renderer = StrokeRenderer()
        p = CharPlacement(char="(", x=0.0, y=0.0, font_size=5.0)

        open_pts = renderer._simple_paren_strokes("（", p)[0]
        close_pts = renderer._simple_paren_strokes("）", p)[0]
        open_mid_x = open_pts[len(open_pts) // 2][0]
        open_end_x = open_pts[0][0]
        close_mid_x = close_pts[len(close_pts) // 2][0]
        close_end_x = close_pts[0][0]
        assert open_mid_x < open_end_x  # 「(」: 中央が端より左
        assert close_mid_x > close_end_x  # 「)」: 中央が端より右

    def test_renders_line_segment(self):
        from src.layout.typesetter import CharPlacement

        renderer = StrokeRenderer()
        placement = CharPlacement(
            char="",
            x=0,
            y=0,
            font_size=8.0,
            page=0,
            line_segment=(10.0, 20.0, 50.0, 20.0),
        )
        strokes = renderer.generate_char_strokes(placement)
        assert len(strokes) == 1
        arr = strokes[0]
        assert arr.shape == (2, 2)
        assert arr[0].tolist() == [10.0, 20.0]
        assert arr[1].tolist() == [50.0, 20.0]

    @pytest.mark.parametrize("char", ["A", "+"])
    def test_skip_non_japanese_skips_non_japanese_chars(self, char):
        renderer = StrokeRenderer(skip_non_japanese=True)
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert strokes == []
        assert finishes == []
        assert char in renderer._last_coverage.skipped

    @pytest.mark.parametrize("char", ["あ", "漢", "カ", "ー", "ヴ"])
    def test_skip_non_japanese_keeps_japanese_chars(self, char):
        renderer = StrokeRenderer(skip_non_japanese=True)
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert strokes == []
        assert finishes == []
        assert char not in renderer._last_coverage.skipped
        assert char in renderer._last_coverage.missing_glyphs

    @pytest.mark.parametrize("char", ["0", "1", "９"])
    def test_skip_non_japanese_keeps_missing_digits_as_blank(self, char):
        renderer = StrokeRenderer(skip_non_japanese=True)
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert strokes == []
        assert finishes == []
        assert char not in renderer._last_coverage.skipped
        assert char in renderer._last_coverage.missing_glyphs

    @pytest.mark.parametrize("char", ["、", "。", "，", "．", ",", "."])
    def test_skip_non_japanese_keeps_punctuation(self, char):
        renderer = StrokeRenderer(skip_non_japanese=True)
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert len(strokes) > 0
        assert len(strokes) == len(finishes)
        assert char not in renderer._last_coverage.skipped

    def test_skip_non_japanese_skips_math_source(self):
        renderer = StrokeRenderer(skip_non_japanese=True)
        placement = CharPlacement(
            char="$",
            x=0.0,
            y=0.0,
            font_size=8.0,
            page=0,
            math_source="E=mc^2",
            math_bbox=(0.0, 0.0, 20.0, 8.0),
        )

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert strokes == []
        assert finishes == []
        assert "$" in renderer._last_coverage.skipped

    def test_skip_non_japanese_keeps_table_line_segments(self):
        renderer = StrokeRenderer(skip_non_japanese=True)
        placement = CharPlacement(
            char="",
            x=0.0,
            y=0.0,
            font_size=8.0,
            page=0,
            line_segment=(10.0, 20.0, 50.0, 20.0),
        )

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert len(strokes) == 1
        assert finishes == ["none"]

    def test_skip_non_japanese_skips_math_line_segments(self):
        renderer = StrokeRenderer(skip_non_japanese=True)
        placement = CharPlacement(
            char="",
            x=0.0,
            y=0.0,
            font_size=8.0,
            page=0,
            role="fraction",
            line_segment=(10.0, 20.0, 50.0, 20.0),
        )

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert strokes == []
        assert finishes == []

    def test_inline_math_baseline_aligns_with_body_line(self):
        """インライン数式(matplotlib画像経路)のベースラインが本文文字の下端に揃う。

        本文文字は placement.y を行ボックス下端とみなし line_spacing 内で縦中央に
        配置する(_position_strokes)。数式画像も同じ下端ラインへ揃えないと、行高と
        数式インク高の差の半分だけ下にずれる(issue #24)。
        """
        from pathlib import Path

        from src.layout.page_layout import PageConfig
        from src.layout.typesetter import Typesetter

        if not Path("data/strokes/国").exists():
            pytest.skip("KanjiVG reference strokes (data/strokes) not available")

        cfg = PageConfig()
        ts = Typesetter(cfg, font_size=4.5)
        placements = ts.typeset(r"国$E=mc^2$")[0]

        renderer = StrokeRenderer(page_config=cfg, kanjivg_dir=Path("data/strokes"))
        kanji_p = next(p for p in placements if p.char == "国")
        math_p = next(p for p in placements if getattr(p, "math_source", None))

        k_strokes, _ = renderer.generate_char_strokes_with_finishes(kanji_p)
        m_strokes, _ = renderer.generate_char_strokes_with_finishes(math_p)
        assert k_strokes and m_strokes

        kanji_bottom = min(float(s[:, 1].min()) for s in k_strokes)
        math_bottom = min(float(s[:, 1].min()) for s in m_strokes)

        # ベースライン揃え: 数式の下端(≈ベースライン)が本文漢字の下端に揃う
        assert math_bottom == pytest.approx(kanji_bottom, abs=0.6)


class TestAsciiMathSymbols:
    """数式で頻出する ASCII 記号は矩形フォールバックではなく幾何で描画する。"""

    @pytest.mark.parametrize(
        "char", ["+", "-", "=", "<", ">", "*", "/", ":", ";", "!", "?", "%", "[", "]", "~"]
    )
    def test_ascii_math_renders_geometric(self, char):
        from src.layout.typesetter import CharPlacement

        renderer = StrokeRenderer()
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)
        strokes = renderer.generate_char_strokes(placement)

        assert len(strokes) > 0
        for s in strokes:
            assert s.dtype == np.float64
            assert s.ndim == 2 and s.shape[1] == 2

        # 矩形フォールバックは「5 点で閉じた単一ストローク」として返る
        is_rect_fallback = len(strokes) == 1 and strokes[0].shape == (5, 2)
        assert not is_rect_fallback, f"'{char}' fell back to rect"

    def test_ascii_math_internal_unit_box(self):
        """_ascii_math_strokes は単位正方形 (0,0)-(1,1) 内に座標を返す。"""
        renderer = StrokeRenderer()
        for char in ["+", "-", "=", "<", ">", "*", "/", ":", ";", "!", "?", "%"]:
            result = renderer._ascii_math_strokes(char)
            assert result is not None, f"'{char}' returned None"
            assert len(result) >= 1
            for s in result:
                assert s.dtype == np.float64
                assert s.shape[1] == 2
                assert s.min() >= -0.05 and s.max() <= 1.05

    def test_ascii_math_returns_none_for_non_math(self):
        renderer = StrokeRenderer()
        # ひらがな・漢字・他の記号は対象外
        assert renderer._ascii_math_strokes("あ") is None
        assert renderer._ascii_math_strokes("漢") is None
        assert renderer._ascii_math_strokes("(") is None
        assert renderer._ascii_math_strokes(",") is None

    def test_plus_has_two_strokes(self):
        """`+` は横棒と縦棒の 2 ストローク。"""
        renderer = StrokeRenderer()
        result = renderer._ascii_math_strokes("+")
        assert result is not None
        assert len(result) == 2

    def test_equals_has_two_horizontal_strokes(self):
        """`=` は上下 2 本の横棒。"""
        renderer = StrokeRenderer()
        result = renderer._ascii_math_strokes("=")
        assert result is not None
        assert len(result) == 2

    def test_percent_has_diagonal_and_two_circles(self):
        """`%` は斜線 1 本＋左上・右下の小円 2 本の計 3 ストローク。"""
        renderer = StrokeRenderer()
        result = renderer._ascii_math_strokes("%")
        assert result is not None
        assert len(result) == 3
        # 斜線は右上がり（始点が左下・終点が右上）
        diag = result[0]
        assert diag[0][0] < diag[-1][0] and diag[0][1] < diag[-1][1]
        # 小円 2 本は閉じている（始点≈終点）
        for circle in result[1:]:
            assert np.allclose(circle[0], circle[-1], atol=1e-6)

    @pytest.mark.parametrize("char", ["①", "②", "④", "⑩", "℃", "°"])
    def test_composite_symbols_not_rect_fallback(self, char):
        """丸数字・℃・° は合成字形で描画され□にならない。"""
        from pathlib import Path

        renderer = StrokeRenderer(kanjivg_dir=Path("data/strokes"))
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)
        strokes = renderer.generate_char_strokes(placement)
        assert len(strokes) >= 1
        assert char not in renderer._last_coverage.rect_fallback

    @pytest.mark.parametrize("char", ["S", "s"])
    def test_s_not_mirrored(self, char):
        """S/s は上が左・下が右に膨らむ正しい向き（左右反転 Ƨ でない）。"""
        renderer = StrokeRenderer()
        stroke = renderer._ascii_letter_strokes(char)[0]
        # 最も左へ張り出す点は上側(yが大)、最も右は下側(yが小)
        left_pt = stroke[np.argmin(stroke[:, 0])]
        right_pt = stroke[np.argmax(stroke[:, 0])]
        assert left_pt[1] > right_pt[1]

    def test_fullwidth_normalized_to_ascii(self):
        """全角プラスも幾何ルートで処理される（_CHAR_SUBSTITUTIONS 経由）。"""
        renderer = StrokeRenderer()
        placement = CharPlacement(char="＋", x=0.0, y=0.0, font_size=8.0, page=0)
        strokes = renderer.generate_char_strokes(placement)
        is_rect_fallback = len(strokes) == 1 and strokes[0].shape == (5, 2)
        assert not is_rect_fallback


class _FailingInference:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or AssertionError("ML inference should not be called")
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise self.exc


class _RecordingInference:
    """deform_scale を記録し、参照ストロークをそのまま返すフェイク。"""

    def __init__(self) -> None:
        self.deform_scales: list[float] = []

    def generate(self, style_sample, *, reference_strokes=None, deform_scale=1.0, **kwargs):
        self.deform_scales.append(deform_scale)
        return [np.asarray(s, dtype=float) for s in (reference_strokes or [])]


class TestWaverScaleByStrokeCount:
    """画数が多い字ほど揺らぎ(distortion/ML offset)を落として固まりを防ぐ。"""

    def test_waver_scale_decreasing_and_clamped(self):
        f = StrokeRenderer._waver_scale
        assert f(1) == pytest.approx(1.0)
        assert f(10) == pytest.approx(1.0)  # 閾値以下は満額
        assert f(30) == pytest.approx(0.5)  # 高画数は下限(経路間の質感を揃え 0.3→0.5)
        # 単調非増加
        vals = [f(n) for n in range(1, 31)]
        assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
        # 中間(15画)は 1.0 と 0.5 の間
        assert 0.5 < f(15) < 1.0

    def test_apply_distortion_zero_scale_is_identity(self):
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter

        r = StrokeRenderer(augmenter=HandwritingAugmenter(AugmentConfig(), seed=0))
        s = [np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]], dtype=float)]
        out = r._apply_distortion(s, waver_scale=0.0)
        assert np.allclose(out[0], s[0])  # 揺らぎ0なら不変

    def test_instance_variation_makes_repeats_differ(self):
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter

        r = StrokeRenderer(
            augmenter=HandwritingAugmenter(AugmentConfig(), seed=0),
            instance_variation=0.6,
        )
        s = [np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]], dtype=float)]
        a = r._apply_instance_variation([s[0].copy()])
        b = r._apply_instance_variation([s[0].copy()])
        assert not np.allclose(a[0], b[0])  # RNG が進み毎回違う

    def test_instance_variation_zero_is_identity(self):
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter

        r = StrokeRenderer(
            augmenter=HandwritingAugmenter(AugmentConfig(), seed=0),
            instance_variation=0.0,
        )
        s = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]], dtype=float)
        out = r._apply_instance_variation([s.copy()])
        assert np.allclose(out[0], s)  # 強度0は恒等

    def test_instance_variation_disabled_aug_is_identity(self):
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter

        r = StrokeRenderer(
            augmenter=HandwritingAugmenter(AugmentConfig(enabled=False), seed=0),
            instance_variation=0.6,
        )
        s = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]], dtype=float)
        out = r._apply_instance_variation([s.copy()])
        assert np.allclose(out[0], s)  # クリーンモード(aug無効)は変動なし

    def test_dense_char_gets_smaller_deform_scale(self):
        from pathlib import Path

        r = StrokeRenderer(kanjivg_dir=Path("data/strokes"))
        rec = _RecordingInference()
        r._inference = rec
        for ch in ["一", "機"]:  # 1画 vs 16画
            r.generate_char_strokes(CharPlacement(char=ch, x=0.0, y=0.0, font_size=4.5, page=0))
        simple, dense = rec.deform_scales
        assert simple == pytest.approx(1.0)
        assert dense < simple  # 多画字はML変形を縮小


class TestGeometricFallbackOrder:
    """句読点・括弧は ML 推論より前に幾何描画する。"""

    @pytest.mark.parametrize("char", ["「", "」", "『", "』", "（", "）", "(", ")", "."])
    def test_brackets_and_punctuation_render_geometric_not_rect(self, char):
        renderer = StrokeRenderer()
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)

        strokes = renderer.generate_char_strokes(placement)

        assert len(strokes) > 0
        is_rect_fallback = len(strokes) == 1 and strokes[0].shape == (5, 2)
        assert not is_rect_fallback, f"'{char}' fell back to rect"
        assert renderer._last_coverage.geometric == [char]
        assert renderer._last_coverage.rect_fallback == []

    @pytest.mark.parametrize("char", ["「", "」", "『", "』", "（", "）", "(", ")"])
    def test_geometric_brackets_do_not_call_ml_inference(self, char):
        renderer = StrokeRenderer()
        fake = _FailingInference()
        renderer._inference = fake
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)

        renderer.generate_char_strokes(placement)

        assert fake.calls == 0
        assert renderer._last_coverage.geometric == [char]

    def test_inference_without_reference_is_blank_and_recorded_as_missing(self):
        renderer = StrokeRenderer()
        fake = _FailingInference(ValueError("missing reference"))
        renderer._inference = fake
        placement = CharPlacement(char="漢", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes = renderer.generate_char_strokes(placement)

        assert fake.calls == 0
        assert strokes == []
        assert renderer._last_coverage.missing_glyphs == ["漢"]
        assert renderer._last_coverage.rect_fallback == []

    def test_lambda_without_reference_uses_geometric_before_ml(self):
        renderer = StrokeRenderer()
        fake = _FailingInference()
        renderer._inference = fake
        placement = CharPlacement(char="λ", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes = renderer.generate_char_strokes(placement)

        assert fake.calls == 0
        assert len(strokes) > 0
        assert renderer._last_coverage.geometric == ["λ"]
        assert renderer._last_coverage.ml_inference == []
        assert renderer._last_coverage.rect_fallback == []

    def test_is_ml_deformable_excludes_digits(self):
        # モデルはCJK(漢字/かな)のみ訓練。数字はML変形で字形が壊れるため除外。
        for d in "0123456789":
            assert StrokeRenderer._is_ml_deformable(d) is False
        assert StrokeRenderer._is_ml_deformable("永") is True
        assert StrokeRenderer._is_ml_deformable("あ") is True

    def test_digits_bypass_ml_use_reference(self):
        from pathlib import Path

        renderer = StrokeRenderer(kanjivg_dir=Path("data/strokes"))
        fake = _FailingInference()
        renderer._inference = fake
        placement = CharPlacement(char="2", x=0.0, y=0.0, font_size=20.0, page=0)

        strokes = renderer.generate_char_strokes(placement)

        assert fake.calls == 0  # 数字はML変形を呼ばない
        assert renderer._last_coverage.ml_inference == []
        assert "2" in renderer._last_coverage.kanjivg
        assert len(strokes) >= 1

    @pytest.mark.parametrize("word", ["cos", "sin", "log", "dx"])
    def test_math_words_render_geometric(self, word):
        renderer = StrokeRenderer()
        placement = CharPlacement(char=word, x=0.0, y=0.0, font_size=8.0, page=0)

        strokes = renderer.generate_char_strokes(placement)

        assert len(strokes) >= len(word)
        assert renderer._last_coverage.geometric == [word]
        assert renderer._last_coverage.rect_fallback == []

    @pytest.mark.parametrize(
        "char",
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    )
    def test_single_ascii_letter_renders_geometric(self, char):
        renderer = StrokeRenderer()
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)

        strokes = renderer.generate_char_strokes(placement)

        assert len(strokes) > 0
        assert renderer._last_coverage.geometric == [char]
        assert renderer._last_coverage.rect_fallback == []


class TestExtendedMathSymbols:
    """LaTeX シンボルマップ追加で頻出する Unicode 記号がストロークになることを確認する。"""

    @pytest.mark.parametrize(
        "char",
        ["ω", "π", "θ", "α", "β", "γ", "λ", "μ", "ε", "σ", "Σ", "Π", "Ω", "×", "·", "→", "∫", "∂"],
    )
    def test_math_symbol_renders(self, char):
        from src.layout.typesetter import CharPlacement

        renderer = StrokeRenderer()
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)
        strokes = renderer.generate_char_strokes(placement)
        assert len(strokes) > 0, f"'{char}' returned no strokes"
        # 矩形フォールバックは「5 点で閉じた単一ストローク」として返る
        is_rect_fallback = len(strokes) == 1 and strokes[0].shape == (5, 2)
        assert not is_rect_fallback, f"'{char}' fell back to rect"

    @pytest.mark.parametrize(
        "char", ["β", "γ", "λ", "μ", "ε", "σ", "Σ", "Π", "Ω", "×", "·", "→", "∫", "∂"]
    )
    def test_extended_unicode_in_unit_box(self, char):
        renderer = StrokeRenderer()
        result = renderer._math_symbol_strokes(char)
        assert result is not None, f"'{char}' returned None"
        for s in result:
            assert s.shape[1] == 2
            assert s.min() >= -0.05 and s.max() <= 1.05


# 払い(㇒)を含む 2 ストローク漢字の KanjiVG fixture（Y-UP, 0..1 正規化済み）。
# 第1画は横画(㇐=とめ), 第2画は左払い(㇒=harai)。各 5 点で len>=2 条件を満たす。
_HARAI_STROKES = [
    [[0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5], [0.9, 0.5]],
    [[0.5, 0.9], [0.45, 0.7], [0.4, 0.5], [0.3, 0.3], [0.15, 0.1]],
]
_HARAI_TYPES = ["㇐", "㇒"]


class TestKanjiVGReferenceFinishing:
    """KanjiVG 参照経路への終端加工配線（とめ・はね・払い）。"""

    def test_load_reference_strokes_returns_strokes_and_types(self, tmp_path):
        kanjivg_dir = tmp_path / "strokes"
        _create_user_stroke_json(kanjivg_dir, "ノ", _HARAI_STROKES, stroke_types=_HARAI_TYPES)
        renderer = StrokeRenderer(kanjivg_dir=kanjivg_dir)

        strokes, types = renderer._load_reference_strokes("ノ")

        assert strokes is not None
        assert len(strokes) == 2
        assert types == _HARAI_TYPES

    def test_load_reference_strokes_missing_char_returns_none(self, tmp_path):
        kanjivg_dir = tmp_path / "strokes"
        kanjivg_dir.mkdir()
        renderer = StrokeRenderer(kanjivg_dir=kanjivg_dir)

        strokes, types = renderer._load_reference_strokes("ノ")

        assert strokes is None
        assert types == []

    def test_load_reference_strokes_sync_filters_short_strokes(self, tmp_path):
        # 1 点しかない第2画は len>=2 で除外され、types も同期して除外される
        kanjivg_dir = tmp_path / "strokes"
        strokes_in = [
            [[0.1, 0.5], [0.9, 0.5]],
            [[0.5, 0.5]],  # 1 点 → 除外
            [[0.5, 0.9], [0.2, 0.1]],
        ]
        types_in = ["㇐", "㇔", "㇒"]
        _create_user_stroke_json(kanjivg_dir, "X", strokes_in, stroke_types=types_in)
        renderer = StrokeRenderer(kanjivg_dir=kanjivg_dir)

        strokes, types = renderer._load_reference_strokes("X")

        assert len(strokes) == 2
        # 除外された index 1 (㇔) を飛ばし、㇐ と ㇒ が残る（順序保持）
        assert types == ["㇐", "㇒"]

    def test_load_kanjivg_json_returns_strokes_and_types(self, tmp_path):
        kanjivg_dir = tmp_path / "strokes"
        _create_user_stroke_json(kanjivg_dir, "ノ", _HARAI_STROKES, stroke_types=_HARAI_TYPES)
        renderer = StrokeRenderer(kanjivg_dir=kanjivg_dir)
        placement = CharPlacement(char="ノ", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, types = renderer._load_kanjivg_json(placement)

        assert strokes is not None
        assert len(strokes) == 2
        assert types == _HARAI_TYPES

    def test_safety_net_finishing_changes_harai_terminal(self, tmp_path):
        # inference=None・kanjivg_dir のみ → safety-net 経路 (_load_kanjivg_json)
        kanjivg_dir = tmp_path / "strokes"
        _create_user_stroke_json(kanjivg_dir, "ノ", _HARAI_STROKES, stroke_types=_HARAI_TYPES)
        placement = CharPlacement(char="ノ", x=10.0, y=20.0, font_size=8.0, page=0)

        renderer_on = StrokeRenderer(kanjivg_dir=kanjivg_dir, enable_finishing=True)
        renderer_off = StrokeRenderer(kanjivg_dir=kanjivg_dir, enable_finishing=False)

        on = renderer_on.generate_char_strokes(placement)
        off = renderer_off.generate_char_strokes(placement)

        assert renderer_on._last_coverage.kanjivg == ["ノ"]
        assert renderer_off._last_coverage.kanjivg == ["ノ"]
        # 払い(㇒)の第2画は終端へ点が延長され座標が変わる
        harai_on = on[1]
        harai_off = off[1]
        assert harai_on.shape[0] > harai_off.shape[0]
        # 加工なし版の終端点と加工あり版の末尾点は別位置
        assert not np.allclose(harai_on[-1], harai_off[-1])

    def test_finishing_disabled_leaves_strokes_identical(self, tmp_path):
        kanjivg_dir = tmp_path / "strokes"
        _create_user_stroke_json(kanjivg_dir, "ノ", _HARAI_STROKES, stroke_types=_HARAI_TYPES)
        placement = CharPlacement(char="ノ", x=10.0, y=20.0, font_size=8.0, page=0)
        renderer_off = StrokeRenderer(kanjivg_dir=kanjivg_dir, enable_finishing=False)

        off = renderer_off.generate_char_strokes(placement)

        # 加工なしならどのストロークも点数が増えていない（元 5 点のまま）
        for s in off:
            assert s.shape[0] == 5

    def test_direct_stroke_path_not_finished(self, tmp_path):
        # user_strokes 登録字は _direct_stroke 経路で加工が掛からない
        user_dir = tmp_path / "user_strokes"
        kanjivg_dir = tmp_path / "strokes"
        _create_user_stroke_json(user_dir, "ノ", _HARAI_STROKES)
        # KanjiVG にも同字を置くが direct が優先されるので使われない
        _create_user_stroke_json(kanjivg_dir, "ノ", _HARAI_STROKES, stroke_types=_HARAI_TYPES)
        placement = CharPlacement(char="ノ", x=10.0, y=20.0, font_size=8.0, page=0)
        renderer = StrokeRenderer(
            user_strokes_dir=user_dir, kanjivg_dir=kanjivg_dir, enable_finishing=True
        )

        strokes = renderer.generate_char_strokes(placement)

        assert renderer._last_coverage.user_strokes == ["ノ"]
        assert renderer._last_coverage.kanjivg == []
        # direct 経路は加工で点が増えていない（元の点数を維持）
        for s in strokes:
            assert s.shape[0] == 5


class TestGenerateCharStrokesWithFinishes:
    """generate_char_strokes_with_finishes の (strokes, finishes) 並走契約。"""

    def test_returns_strokes_and_finishes_len_match_geometric(self):
        renderer = StrokeRenderer()
        placement = CharPlacement(char="+", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert len(strokes) == len(finishes)
        assert finishes == ["none"] * len(strokes)

    def test_geometric_paths_all_none(self):
        renderer = StrokeRenderer()
        # 句読点 / ASCII数式 / 括弧 / 数式記号 / ASCIIレター / 数式ワード
        for char in ("。", "=", "(", "×", "A", "α"):
            placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)
            strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)
            assert len(strokes) == len(finishes), f"{char}: len mismatch"
            assert set(finishes) <= {"none"}, f"{char}: non-none finish leaked"

    def test_math_word_all_none(self):
        renderer = StrokeRenderer()
        placement = CharPlacement(char="cos", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert len(strokes) == len(finishes)
        assert finishes == ["none"] * len(strokes)

    def test_line_segment_returns_none_finish(self):
        renderer = StrokeRenderer()
        placement = CharPlacement(
            char="", x=0.0, y=0.0, font_size=8.0, page=0, line_segment=(0.0, 0.0, 1.0, 1.0)
        )

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert len(strokes) == 1
        assert finishes == ["none"]

    def test_skip_render_returns_empty_pair(self):
        renderer = StrokeRenderer()
        placement = CharPlacement(char=" ", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert strokes == []
        assert finishes == []

    def test_missing_glyph_returns_empty_pair(self):
        # 参照なし・推論なし → 未収録として空白化
        renderer = StrokeRenderer()
        placement = CharPlacement(char="漢", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert strokes == []
        assert finishes == []
        assert renderer._last_coverage.missing_glyphs == ["漢"]

    def test_direct_stroke_all_none(self, tmp_path):
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(user_dir, "ノ", _HARAI_STROKES)
        renderer = StrokeRenderer(user_strokes_dir=user_dir)
        placement = CharPlacement(char="ノ", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert renderer._last_coverage.user_strokes == ["ノ"]
        assert len(strokes) == len(finishes)
        assert finishes == ["none"] * len(strokes)

    def test_kanjivg_safety_net_emits_harai(self, tmp_path):
        kanjivg_dir = tmp_path / "strokes"
        _create_user_stroke_json(kanjivg_dir, "ノ", _HARAI_STROKES, stroke_types=_HARAI_TYPES)
        renderer = StrokeRenderer(kanjivg_dir=kanjivg_dir)
        placement = CharPlacement(char="ノ", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert renderer._last_coverage.kanjivg == ["ノ"]
        assert len(strokes) == len(finishes)
        # 第1画=横画(㇐)→tome, 第2画=左払い(㇒)→harai
        assert finishes == ["tome", "harai"]

    def test_backward_compat_generate_char_strokes_returns_strokes_only(self, tmp_path):
        kanjivg_dir = tmp_path / "strokes"
        _create_user_stroke_json(kanjivg_dir, "ノ", _HARAI_STROKES, stroke_types=_HARAI_TYPES)
        renderer = StrokeRenderer(kanjivg_dir=kanjivg_dir)
        placement = CharPlacement(char="ノ", x=0.0, y=0.0, font_size=8.0, page=0)

        result = renderer.generate_char_strokes(placement)

        # ラッパーは strokes（list[Stroke]）のみ返す（タプルではない）
        assert isinstance(result, list)
        assert not isinstance(result, tuple)
        strokes, _ = renderer.generate_char_strokes_with_finishes(placement)
        assert len(result) == len(strokes)


class TestAsciiLetterHandwriting:
    """本文 ASCII 英字の手書き揺らぎ（きれいすぎ解消）。"""

    def test_ascii_letter_gets_distortion_with_augmenter(self):
        from src.layout.typesetter import CharPlacement
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter

        clean = StrokeRenderer()
        hand = StrokeRenderer(augmenter=HandwritingAugmenter(AugmentConfig(), seed=0))
        cp = CharPlacement(char="x", x=0.0, y=0.0, font_size=10.0)
        s_clean = clean.generate_char_strokes(cp)
        s_hand = hand.generate_char_strokes(cp)
        assert len(s_clean) == len(s_hand)
        # augmenter 有りでは素の幾何字形から変位する（きれいすぎない）
        assert any(not np.allclose(c, h) for c, h in zip(s_clean, s_hand))

    def test_ascii_letter_no_augmenter_is_clean(self):
        from src.layout.typesetter import CharPlacement

        r = StrokeRenderer()  # augmenter なし
        cp = CharPlacement(char="x", x=0.0, y=0.0, font_size=10.0)
        a = r.generate_char_strokes(cp)
        b = r.generate_char_strokes(cp)
        # augmenter 無しなら毎回同一（distortion no-op）
        assert all(np.allclose(x, y) for x, y in zip(a, b))


class TestAsciiLetterPrefersUserStroke:
    """英字はユーザー実筆跡サンプルがあれば幾何フォントより優先（英語くそ対策）。"""

    def test_letter_with_user_sample_uses_direct(self, tmp_path):
        from src.layout.typesetter import CharPlacement

        user_dir = tmp_path / "user_strokes"
        # ユーザーが書いた "A"（特徴的な三角形状）
        _create_user_stroke_json(
            user_dir, "A", [[[0, 100], [50, 0]], [[50, 0], [100, 100]], [[25, 50], [75, 50]]]
        )
        r = StrokeRenderer(user_strokes_dir=user_dir)
        r._last_coverage.geometric.clear()
        r._last_coverage.user_strokes.clear()
        r.generate_char_strokes(CharPlacement(char="A", x=0, y=0, font_size=7.0))
        # 幾何でなく実筆跡経路（user_strokes）に入る
        assert "A" in r._last_coverage.user_strokes
        assert "A" not in r._last_coverage.geometric

    def test_letter_without_sample_falls_back_to_geometric(self, tmp_path):
        from src.layout.typesetter import CharPlacement

        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(user_dir, "A", [[[0, 100], [50, 0]]])
        r = StrokeRenderer(user_strokes_dir=user_dir)
        r._last_coverage.geometric.clear()
        r._last_coverage.user_strokes.clear()
        # "Z" はサンプル無し → 幾何フォントへフォールバック
        r.generate_char_strokes(CharPlacement(char="Z", x=0, y=0, font_size=7.0))
        assert "Z" in r._last_coverage.geometric


def _slope_left_to_right(stroke):
    """ストロークの x 昇順での始点→終点の傾き(rad, 左→右方向)を返す。"""
    s = stroke[np.argsort(stroke[:, 0])]
    dx = s[-1, 0] - s[0, 0]
    dy = s[-1, 1] - s[0, 1]
    return float(np.arctan2(dy, dx))


class TestEnforceHorizontalRise:
    """横棒の右上がり下限保証（日本語手書きの右上がり習性の再現）。"""

    def test_downward_horizontal_is_lifted_to_min_rise(self):
        # 右下がりの水平画（mm, Y-UP=上が大）。矯正で +RISE_MIN 以上に起こす。
        renderer = StrokeRenderer()
        fs = 5.0
        strokes = [np.array([[0.0, 5.0], [5.0, 4.5]], dtype=np.float64)]
        out = renderer._enforce_horizontal_rise(strokes, fs)
        assert _slope_left_to_right(out[0]) >= StrokeRenderer._RISE_MIN_ANGLE - 1e-6

    def test_flat_horizontal_is_lifted_to_min_rise(self):
        renderer = StrokeRenderer()
        fs = 5.0
        strokes = [np.array([[0.0, 5.0], [5.0, 5.0]], dtype=np.float64)]
        out = renderer._enforce_horizontal_rise(strokes, fs)
        assert _slope_left_to_right(out[0]) >= StrokeRenderer._RISE_MIN_ANGLE - 1e-6

    def test_already_steep_rise_is_unchanged(self):
        # 既に十分右上がり(+5°)の横棒は起こし過ぎない＝ほぼ不変。
        renderer = StrokeRenderer()
        fs = 5.0
        ang = np.deg2rad(5.0)
        end_y = 5.0 + 5.0 * np.tan(ang)
        strokes = [np.array([[0.0, 5.0], [5.0, end_y]], dtype=np.float64)]
        before = _slope_left_to_right(strokes[0])
        out = renderer._enforce_horizontal_rise([strokes[0].copy()], fs)
        assert np.allclose(out[0], strokes[0])
        assert _slope_left_to_right(out[0]) == pytest.approx(before)

    def test_vertical_stroke_passes_through(self):
        # 縦画(x_range 小)は横棒判定されず素通り。
        renderer = StrokeRenderer()
        fs = 5.0
        strokes = [np.array([[2.5, 0.0], [2.4, 5.0]], dtype=np.float64)]
        out = renderer._enforce_horizontal_rise([strokes[0].copy()], fs)
        assert np.allclose(out[0], strokes[0])

    def test_diagonal_stroke_passes_through(self):
        # 斜め画(/, y_range 大)は横棒判定されず素通り。
        renderer = StrokeRenderer()
        fs = 5.0
        strokes = [np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)]
        out = renderer._enforce_horizontal_rise([strokes[0].copy()], fs)
        assert np.allclose(out[0], strokes[0])

    def test_short_stroke_passes_through(self):
        # x_range < font_size*0.25 の短い画は素通り。
        renderer = StrokeRenderer()
        fs = 5.0
        strokes = [np.array([[0.0, 5.0], [0.5, 4.6]], dtype=np.float64)]
        out = renderer._enforce_horizontal_rise([strokes[0].copy()], fs)
        assert np.allclose(out[0], strokes[0])

    def test_empty_and_single_point_pass_through(self):
        renderer = StrokeRenderer()
        fs = 5.0
        empty = np.zeros((0, 2), dtype=np.float64)
        single = np.array([[1.0, 1.0]], dtype=np.float64)
        out = renderer._enforce_horizontal_rise([empty, single], fs)
        assert out[0].shape == (0, 2)
        assert np.allclose(out[1], single)

    def test_rotation_preserves_point_count(self):
        renderer = StrokeRenderer()
        fs = 5.0
        strokes = [np.array([[0.0, 5.0], [2.5, 4.9], [5.0, 4.5]], dtype=np.float64)]
        out = renderer._enforce_horizontal_rise(strokes, fs)
        assert out[0].shape == strokes[0].shape


class TestWaverPathUnification:
    """描画経路ごとの揺らぎ強度差を縮め「きれい/汚い混在」を緩和する。"""

    def test_geometric_letter_waver_value(self):
        # 幾何 ASCII 英字経路は中間値 1.5（旧 3.0 から引き下げ）
        assert StrokeRenderer._WAVER_GEOMETRIC == pytest.approx(1.5)

    def test_math_image_waver_value(self):
        # 数式ブロック画像経路は 2.5（旧 6.0 から引き下げ、本文寄りに）
        assert StrokeRenderer._WAVER_MATH_IMAGE == pytest.approx(2.5)

    def test_symbol_waver_value(self):
        # 記号・句読点・括弧など旧 distortion 無し経路へ乗せる微量値 0.4
        assert StrokeRenderer._WAVER_SYMBOL == pytest.approx(0.4)

    def test_waver_floor_raised(self):
        # 画数逓減の下限を 0.3→0.5 へ（多画字と少画字の質感差 最大3倍→2倍）
        assert StrokeRenderer._WAVER_FLOOR == pytest.approx(0.5)

    def test_waver_scale_floor_matches_constant(self):
        f = StrokeRenderer._waver_scale
        assert f(30) == pytest.approx(0.5)  # 高画数は新下限
        assert f(10) == pytest.approx(1.0)  # 閾値以下は満額
        assert 0.5 < f(15) < 1.0  # 中間は下限と満額の間


class TestSymbolMicroWaver:
    """記号・句読点・括弧の幾何経路に微量揺らぎを乗せ定規直線感を消す。"""

    def test_symbol_path_gets_distortion(self):
        from src.layout.typesetter import CharPlacement
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter

        clean = StrokeRenderer()
        hand = StrokeRenderer(augmenter=HandwritingAugmenter(AugmentConfig(), seed=0))
        cp = CharPlacement(char="=", x=0.0, y=0.0, font_size=10.0)
        s_clean = clean.generate_char_strokes(cp)
        s_hand = hand.generate_char_strokes(cp)
        assert len(s_clean) == len(s_hand)
        # augmenter 有りで素の幾何字形から変位する（ツルツルでない）
        assert any(not np.allclose(c, h) for c, h in zip(s_clean, s_hand))

    def test_paren_path_gets_distortion(self):
        from src.layout.typesetter import CharPlacement
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter

        clean = StrokeRenderer()
        hand = StrokeRenderer(augmenter=HandwritingAugmenter(AugmentConfig(), seed=0))
        cp = CharPlacement(char="(", x=0.0, y=0.0, font_size=10.0)
        s_clean = clean.generate_char_strokes(cp)
        s_hand = hand.generate_char_strokes(cp)
        assert any(not np.allclose(c, h) for c, h in zip(s_clean, s_hand))

    def test_symbol_path_no_augmenter_is_clean(self):
        from src.layout.typesetter import CharPlacement

        r = StrokeRenderer()  # augmenter なし → 微量揺らぎも no-op
        cp = CharPlacement(char="=", x=0.0, y=0.0, font_size=10.0)
        a = r.generate_char_strokes(cp)
        b = r.generate_char_strokes(cp)
        assert all(np.allclose(x, y) for x, y in zip(a, b))

    def test_period_dot_not_broken_by_waver(self):
        # 句点の点(極短ストローク)は閾値未満で素のまま＝破綻しない
        from src.layout.typesetter import CharPlacement
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter

        clean = StrokeRenderer()
        hand = StrokeRenderer(augmenter=HandwritingAugmenter(AugmentConfig(), seed=0))
        cp = CharPlacement(char="。", x=0.0, y=0.0, font_size=10.0)
        s_clean = clean.generate_char_strokes(cp)
        s_hand = hand.generate_char_strokes(cp)
        assert len(s_clean) == len(s_hand)
        # 極短の点は揺らぎを当てず素のまま（破綻防止）
        assert all(np.allclose(c, h) for c, h in zip(s_clean, s_hand))

    def test_apply_symbol_distortion_keeps_short_stroke(self):
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter

        r = StrokeRenderer(augmenter=HandwritingAugmenter(AugmentConfig(), seed=0))
        # 長い横棒＋極短の点。点は素のまま、横棒は変位する。
        long_bar = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        dot = np.array([[5.0, 5.0], [5.05, 5.0]], dtype=np.float64)
        out = r._apply_symbol_distortion([long_bar.copy(), dot.copy()])
        assert np.allclose(out[1], dot)  # 極短は破綻させない
        assert not np.allclose(out[0], long_bar)  # 長い画は揺らぐ

    def test_apply_symbol_distortion_no_augmenter_identity(self):
        r = StrokeRenderer()  # augmenter なし
        s = [np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)]
        out = r._apply_symbol_distortion([s[0].copy()])
        assert np.allclose(out[0], s[0])


class TestDirectStrokeSampleFixed:
    """同じ文字には常に同じベースサンプルを使い品質の極端な振れを防ぐ。"""

    def test_same_char_uses_same_base_sample(self, tmp_path):
        user_dir = tmp_path / "user_strokes"
        # 同じ "山" に形の異なる2サンプル（点数も差をつける）
        _create_user_stroke_json(user_dir, "山", [[[0, 0], [0, 100]]], suffix="001")
        _create_user_stroke_json(
            user_dir,
            "山",
            [[[0, 0], [0, 50], [0, 100]], [[50, 0], [50, 100]]],
            suffix="002",
        )
        r = StrokeRenderer(user_strokes_dir=user_dir)
        # _apply_stroke_variation の乱数を揃え、ベース選択が固定かを純粋に検証する
        np.random.seed(0)
        a = r._direct_stroke("山")
        np.random.seed(0)
        b = r._direct_stroke("山")
        assert a is not None and b is not None
        assert len(a) == len(b)
        assert all(np.allclose(x, y) for x, y in zip(a, b))

    def test_best_sample_is_most_points(self, tmp_path):
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(user_dir, "川", [[[0, 0], [0, 100]]], suffix="001")
        # 総点数の多い 002 が「丁寧」として選ばれる
        _create_user_stroke_json(
            user_dir,
            "川",
            [[[0, 0], [0, 50], [0, 100]], [[50, 0], [50, 50], [50, 100]]],
            suffix="002",
        )
        r = StrokeRenderer(user_strokes_dir=user_dir)
        r._direct_stroke("川")
        assert r._direct_choice_cache["川"] == 1

    def test_different_chars_have_separate_cache(self, tmp_path):
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(user_dir, "一", [[[0, 50], [100, 50]]])
        _create_user_stroke_json(user_dir, "二", [[[0, 30], [100, 30]], [[0, 70], [100, 70]]])
        r = StrokeRenderer(user_strokes_dir=user_dir)
        r._direct_stroke("一")
        r._direct_stroke("二")
        assert "一" in r._direct_choice_cache
        assert "二" in r._direct_choice_cache
