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
        assert f(30) == pytest.approx(0.3)  # 高画数は下限
        # 単調非増加
        vals = [f(n) for n in range(1, 31)]
        assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
        # 中間(15画)は 1.0 と 0.3 の間
        assert 0.3 < f(15) < 1.0

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
        for char in ("、", "=", "(", "×", "A", "α"):
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

    def test_rect_fallback_returns_none_finish(self):
        # 参照なし・推論なし → 矩形フォールバック
        renderer = StrokeRenderer()
        placement = CharPlacement(char="漢", x=0.0, y=0.0, font_size=8.0, page=0)

        strokes, finishes = renderer.generate_char_strokes_with_finishes(placement)

        assert len(strokes) == len(finishes)
        assert finishes == ["none"] * len(strokes)

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
