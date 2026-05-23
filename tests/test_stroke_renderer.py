"""StrokeRenderer の単体テスト。"""
import json

import numpy as np
import pytest

from src.layout.typesetter import CharPlacement
from src.ui.stroke_renderer import StrokeRenderer


def _create_user_stroke_json(base_dir, char, strokes, suffix="001"):
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

    def test_math_symbol_strokes(self):
        renderer = StrokeRenderer()
        result = renderer._math_symbol_strokes("π")
        assert result is not None
        assert len(result) >= 1

    def test_simple_punct_strokes(self):
        renderer = StrokeRenderer()
        result = renderer._simple_punct_strokes("。")
        assert result is not None

    def test_simple_paren_strokes(self):
        from src.layout.typesetter import CharPlacement

        renderer = StrokeRenderer()
        p = CharPlacement(char="(", x=10.0, y=20.0, font_size=5.0)
        result = renderer._simple_paren_strokes("(", p)
        assert result is not None

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


class TestAsciiMathSymbols:
    """数式で頻出する ASCII 記号は矩形フォールバックではなく幾何で描画する。"""

    @pytest.mark.parametrize(
        "char", ["+", "-", "=", "<", ">", "*", "/", ":", ";", "!", "?"]
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
        for char in ["+", "-", "=", "<", ">", "*", "/", ":", ";", "!", "?"]:
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

    def test_inference_without_reference_is_skipped_and_falls_back_to_rect(self):
        renderer = StrokeRenderer()
        fake = _FailingInference(ValueError("missing reference"))
        renderer._inference = fake
        placement = CharPlacement(char="漢", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes = renderer.generate_char_strokes(placement)

        assert fake.calls == 0
        assert len(strokes) == 1
        assert strokes[0].shape == (5, 2)
        assert renderer._last_coverage.rect_fallback == ["漢"]

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

    @pytest.mark.parametrize("char", ["ω", "π", "θ", "α", "β", "γ", "λ", "μ", "ε", "σ", "Σ", "Π", "Ω", "×", "·", "→", "∫", "∂"])
    def test_math_symbol_renders(self, char):
        from src.layout.typesetter import CharPlacement

        renderer = StrokeRenderer()
        placement = CharPlacement(char=char, x=0.0, y=0.0, font_size=8.0, page=0)
        strokes = renderer.generate_char_strokes(placement)
        assert len(strokes) > 0, f"'{char}' returned no strokes"
        # 矩形フォールバックは「5 点で閉じた単一ストローク」として返る
        is_rect_fallback = len(strokes) == 1 and strokes[0].shape == (5, 2)
        assert not is_rect_fallback, f"'{char}' fell back to rect"

    @pytest.mark.parametrize("char", ["β", "γ", "λ", "μ", "ε", "σ", "Σ", "Π", "Ω", "×", "·", "→", "∫", "∂"])
    def test_extended_unicode_in_unit_box(self, char):
        renderer = StrokeRenderer()
        result = renderer._math_symbol_strokes(char)
        assert result is not None, f"'{char}' returned None"
        for s in result:
            assert s.shape[1] == 2
            assert s.min() >= -0.05 and s.max() <= 1.05
