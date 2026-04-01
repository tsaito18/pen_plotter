"""StrokeRenderer の単体テスト。"""
import json

import numpy as np
import pytest

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
