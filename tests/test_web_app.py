import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.layout.typesetter import CharPlacement
from src.model.augmentation import HandwritingAugmenter
from src.ui.web_app import PlotterPipeline


def _create_kanjivg_json(base_dir, char, num_strokes=3, num_points=8):
    """テスト用のKanjiVG JSONファイルを作成。

    ユーザーストロークと幾何的に類似した形状を生成し、
    StrokeAlignerのquality_thresholdを通過できるようにする。
    """
    char_dir = base_dir / char
    char_dir.mkdir(parents=True, exist_ok=True)
    templates = [
        [(0, 0), (25, 0), (50, 0), (50, 12), (50, 25), (50, 37), (50, 50), (50, 50)],
        [(60, 0), (60, 7), (60, 14), (60, 21), (60, 28), (60, 35), (60, 42), (60, 50)],
        [(0, 60), (7, 60), (14, 60), (21, 60), (28, 60), (35, 60), (42, 60), (50, 60)],
    ]
    strokes = []
    for s in range(num_strokes):
        template = templates[s % len(templates)]
        points = template[:num_points]
        stroke = [
            {"x": float(p[0]), "y": float(p[1]), "pressure": 1.0, "timestamp": 0.0} for p in points
        ]
        strokes.append(stroke)
    data = {"character": char, "strokes": strokes, "metadata": {"source": "kanjivg"}}
    (char_dir / f"{char}_0.json").write_text(json.dumps(data), encoding="utf-8")


class TestPlotterPipeline:
    @pytest.fixture
    def pipeline(self):
        return PlotterPipeline()

    def test_text_to_placements(self, pipeline):
        placements = pipeline.text_to_placements("あいうえお")
        assert len(placements) > 0
        assert len(placements[0]) == 5  # 5文字

    def test_placements_to_strokes(self, pipeline):
        placements = pipeline.text_to_placements("あい")
        strokes = pipeline.placements_to_strokes(placements[0])
        assert len(strokes) > 0
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2

    def test_strokes_to_gcode(self, pipeline):
        placements = pipeline.text_to_placements("あ")
        strokes = pipeline.placements_to_strokes(placements[0])
        gcode = pipeline.strokes_to_gcode(strokes)
        assert len(gcode) > 0
        text = "\n".join(gcode)
        assert "G90" in text
        assert "M2" in text

    def test_generate_preview(self, pipeline, tmp_path):
        preview_path = tmp_path / "preview.png"
        pipeline.generate_preview("テスト", save_path=preview_path)
        assert preview_path.exists()

    def test_generate_gcode_file(self, pipeline, tmp_path):
        gcode_path = tmp_path / "output.gcode"
        pipeline.generate_gcode_file("テスト文字列", save_path=gcode_path)
        assert gcode_path.exists()
        content = gcode_path.read_text()
        assert "G90" in content

    def test_empty_text(self, pipeline, tmp_path):
        preview_path = tmp_path / "empty.png"
        pipeline.generate_preview("", save_path=preview_path)
        assert preview_path.exists()

    def test_multiline_text(self, pipeline):
        placements = pipeline.text_to_placements("あいう\nかきく")
        assert len(placements) >= 1
        # 改行で行が分かれている
        all_chars = placements[0]
        ys = set(p.y for p in all_chars)
        assert len(ys) >= 2

    def test_pipeline_end_to_end(self, pipeline, tmp_path):
        # テキスト入力からG-codeファイル生成まで一気通貫
        gcode_path = tmp_path / "e2e.gcode"
        preview_path = tmp_path / "e2e.png"
        pipeline.generate_gcode_file("Hello テスト", save_path=gcode_path)
        pipeline.generate_preview("Hello テスト", save_path=preview_path)
        assert gcode_path.exists()
        assert preview_path.exists()


class TestFallbackStrokes:
    """3段階フォールバックのテスト。"""

    def test_default_pipeline_unchanged(self):
        """checkpoint/kanjivg_dir未指定時は従来の矩形フォールバック。"""
        pipeline = PlotterPipeline()
        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=5.0)
        strokes = pipeline.placements_to_strokes([placement])
        assert len(strokes) == 1
        # 矩形は5点（始点に戻る閉じた四角形）
        assert strokes[0].shape == (5, 2)
        assert np.allclose(strokes[0][0], strokes[0][-1])

    def test_pipeline_kanjivg_fallback(self, tmp_path):
        """KanjiVG JSONが存在する文字はKanjiVGストロークを使用。"""
        _create_kanjivg_json(tmp_path, "あ", num_strokes=3, num_points=8)

        pipeline = PlotterPipeline(kanjivg_dir=tmp_path)
        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=5.0)
        strokes = pipeline.placements_to_strokes([placement])

        assert len(strokes) == 3
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            assert s.shape[0] == 8

        all_pts = np.concatenate(strokes, axis=0)
        rendered_w = all_pts[:, 0].max() - all_pts[:, 0].min()
        rendered_h = all_pts[:, 1].max() - all_pts[:, 1].min()
        assert rendered_w <= placement.font_size + 0.01
        assert rendered_h <= placement.font_size + 0.01

    def test_pipeline_kanjivg_missing_char_falls_to_rect(self, tmp_path):
        """KanjiVGにファイルがない文字は矩形フォールバック。"""
        pipeline = PlotterPipeline(kanjivg_dir=tmp_path)
        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=5.0)
        strokes = pipeline.placements_to_strokes([placement])
        assert len(strokes) == 1
        assert strokes[0].shape == (5, 2)

    def test_pipeline_inference_fallback(self, tmp_path):
        """MLモデルが読み込まれている場合はML推論を優先使用。"""
        mock_strokes = [
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
            np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]),
        ]

        pipeline = PlotterPipeline()
        mock_inference = MagicMock()
        mock_inference.generate.return_value = mock_strokes
        pipeline._inference = mock_inference
        pipeline._style_sample = MagicMock()
        pipeline._temperature = 0.8

        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=5.0)
        strokes = pipeline.placements_to_strokes([placement])

        mock_inference.generate.assert_called_once()
        assert len(strokes) == 2
        for s in strokes:
            assert isinstance(s, np.ndarray)

    def test_position_strokes(self):
        """_position_strokesがアスペクト比保持・セル中央配置で正しく動作する。"""
        pipeline = PlotterPipeline()
        normalized = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        ]
        # 漢字(scale=1.0)でテスト
        placement = CharPlacement(char="漢", x=10.0, y=20.0, font_size=6.0)
        result = pipeline._position_strokes(normalized, placement)

        assert len(result) == 1
        all_pts = np.concatenate(result, axis=0)
        rendered_w = all_pts[:, 0].max() - all_pts[:, 0].min()
        rendered_h = all_pts[:, 1].max() - all_pts[:, 1].min()
        assert np.isclose(rendered_w, 6.0, atol=0.01)
        assert np.isclose(rendered_h, 6.0, atol=0.01)
        center_x = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
        center_y = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
        assert np.isclose(center_x, 10.0 + 3.0, atol=0.01)
        line_spacing = pipeline._page_config.line_spacing
        expected_y = 20.0 + (line_spacing - 6.0) / 2 + 3.0
        assert np.isclose(center_y, expected_y, atol=0.01)

    def test_position_strokes_hiragana_scaled(self):
        """平仮名は漢字より小さくスケーリングされる。"""
        pipeline = PlotterPipeline()
        normalized = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        ]
        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=6.0)
        result = pipeline._position_strokes(normalized, placement)

        all_pts = np.concatenate(result, axis=0)
        rendered_w = all_pts[:, 0].max() - all_pts[:, 0].min()
        rendered_h = all_pts[:, 1].max() - all_pts[:, 1].min()
        expected_size = 6.0 * 0.93
        assert rendered_w <= expected_size + 0.1
        assert rendered_h <= expected_size + 0.1

    def test_position_strokes_halfwidth(self):
        """半角文字のアスペクト比が保持され、セル内で中央配置される。"""
        pipeline = PlotterPipeline()
        normalized = [
            np.array([[0.0, 0.0], [0.5, 1.0], [0.5, 0.0]]),
        ]
        placement = CharPlacement(char="C", x=10.0, y=20.0, font_size=6.0)
        result = pipeline._position_strokes(normalized, placement)

        assert len(result) == 1
        all_pts = np.concatenate(result, axis=0)
        rendered_w = all_pts[:, 0].max() - all_pts[:, 0].min()
        rendered_h = all_pts[:, 1].max() - all_pts[:, 1].min()
        assert rendered_h > rendered_w
        assert np.isclose(rendered_h, 6.0, atol=0.01)

    def test_halfwidth_wide_char_constrained(self):
        """幅広の半角文字がセル幅(0.6*fs)を超えないこと。"""
        pipeline = PlotterPipeline()
        # 正方形ストローク（幅=高さ）→ セル幅に制約される
        normalized = [
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        ]
        placement = CharPlacement(char="W", x=10.0, y=20.0, font_size=6.0)
        result = pipeline._position_strokes(normalized, placement)

        all_pts = np.concatenate(result, axis=0)
        rendered_w = all_pts[:, 0].max() - all_pts[:, 0].min()
        cell_width = 6.0 * 0.6
        assert rendered_w <= cell_width + 0.01

    def test_inference_v2_with_reference(self, tmp_path):
        """V2推論時にreference_strokesがKanjiVGから渡される。"""
        _create_kanjivg_json(tmp_path, "あ", num_strokes=3, num_points=8)

        mock_strokes = [
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        ]
        pipeline = PlotterPipeline(kanjivg_dir=tmp_path)
        mock_inference = MagicMock()
        mock_inference.generate.return_value = mock_strokes
        pipeline._inference = mock_inference
        pipeline._style_sample = MagicMock()
        pipeline._temperature = 0.8

        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=5.0)
        result = pipeline.placements_to_strokes([placement])

        assert len(result) > 0
        mock_inference.generate.assert_called_once()
        call_kwargs = mock_inference.generate.call_args
        assert call_kwargs.kwargs.get("reference_strokes") is not None
        ref = call_kwargs.kwargs["reference_strokes"]
        assert isinstance(ref, list)
        assert len(ref) == 3
        for arr in ref:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (8, 2)

    def test_inference_v1_backward_compatible(self):
        """V1推論（KanjiVGなし）ではreference_strokes=Noneが渡される。"""
        mock_strokes = [
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        ]
        pipeline = PlotterPipeline()
        mock_inference = MagicMock()
        mock_inference.generate.return_value = mock_strokes
        pipeline._inference = mock_inference
        pipeline._style_sample = MagicMock()
        pipeline._temperature = 0.8

        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=5.0)
        pipeline.placements_to_strokes([placement])

        call_kwargs = mock_inference.generate.call_args
        assert call_kwargs.kwargs.get("reference_strokes") is None

    def test_load_reference_strokes(self, tmp_path):
        """_load_reference_strokesがKanjiVG JSONからNDArrayリストを返す。"""
        _create_kanjivg_json(tmp_path, "あ", num_strokes=2, num_points=5)

        pipeline = PlotterPipeline(kanjivg_dir=tmp_path)
        ref = pipeline._load_reference_strokes("あ")

        assert ref is not None
        assert len(ref) == 2
        for arr in ref:
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.float64
            assert arr.shape == (5, 2)

    def test_load_reference_strokes_missing_char(self, tmp_path):
        """存在しない文字の参照ストロークはNoneを返す。"""
        pipeline = PlotterPipeline(kanjivg_dir=tmp_path)
        assert pipeline._load_reference_strokes("ん") is None

    def test_load_reference_strokes_no_kanjivg_dir(self):
        """kanjivg_dir未設定時はNoneを返す。"""
        pipeline = PlotterPipeline()
        assert pipeline._load_reference_strokes("あ") is None

    def test_style_sample_from_user_strokes(self, tmp_path):
        """ユーザーストロークが存在する場合、実データからstyle_sampleを生成する。"""
        import torch

        user_dir = tmp_path / "user_strokes"
        char_dir = user_dir / "あ"
        char_dir.mkdir(parents=True)
        stroke_data = {
            "character": "あ",
            "strokes": [
                [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 0.5}],
                [{"x": 3.0, "y": 3.0}, {"x": 4.0, "y": 4.0}],
            ],
            "metadata": {},
        }
        (char_dir / "あ_001.json").write_text(json.dumps(stroke_data), encoding="utf-8")

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)

        assert isinstance(pipeline._style_sample, torch.Tensor)
        assert pipeline._style_sample.dim() == 3  # (1, seq_len, 3)
        assert pipeline._style_sample.shape[0] == 1
        assert pipeline._style_sample.shape[2] == 3
        assert pipeline._style_sample.shape[1] > 0
        assert not torch.all(pipeline._style_sample == 0)

    def test_style_sample_fallback_no_user_strokes(self):
        """ユーザーストロークが存在しない場合、ゼロベクトルにフォールバック。"""
        import torch

        pipeline = PlotterPipeline()

        assert isinstance(pipeline._style_sample, torch.Tensor)
        assert torch.all(pipeline._style_sample == 0)

    def test_style_sample_fallback_empty_dir(self, tmp_path):
        """空のユーザーストロークディレクトリではゼロベクトルにフォールバック。"""
        import torch

        empty_dir = tmp_path / "empty_strokes"
        empty_dir.mkdir()

        pipeline = PlotterPipeline(user_strokes_dir=empty_dir)

        assert isinstance(pipeline._style_sample, torch.Tensor)
        assert torch.all(pipeline._style_sample == 0)

    def test_style_sample_explicit_overrides_user_strokes(self, tmp_path):
        """明示的にstyle_sampleが渡された場合はそれを使う。"""
        import torch

        user_dir = tmp_path / "user_strokes"
        char_dir = user_dir / "い"
        char_dir.mkdir(parents=True)
        stroke_data = {
            "character": "い",
            "strokes": [[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]],
            "metadata": {},
        }
        (char_dir / "い_001.json").write_text(json.dumps(stroke_data), encoding="utf-8")

        explicit = torch.ones(1, 5, 3)
        pipeline = PlotterPipeline(user_strokes_dir=user_dir, style_sample=explicit)

        assert torch.equal(pipeline._style_sample, explicit)

    def test_style_sample_multiple_chars(self, tmp_path):
        """複数文字のユーザーストロークがある場合もロードできる。"""
        import torch

        user_dir = tmp_path / "user_strokes"
        for char in ["あ", "い"]:
            char_dir = user_dir / char
            char_dir.mkdir(parents=True)
            stroke_data = {
                "character": char,
                "strokes": [[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]],
                "metadata": {},
            }
            (char_dir / f"{char}_001.json").write_text(json.dumps(stroke_data), encoding="utf-8")

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)

        assert isinstance(pipeline._style_sample, torch.Tensor)
        assert pipeline._style_sample.dim() == 3
        assert not torch.all(pipeline._style_sample == 0)


class TestInterCharShift:
    """文字間の微小水平シフトのテスト。"""

    def _make_pipeline_with_two_chars(self, tmp_path):
        """2文字分のKanjiVGデータとパイプラインを作成するヘルパー。"""
        _create_kanjivg_json(tmp_path, "あ", num_strokes=2, num_points=5)
        _create_kanjivg_json(tmp_path, "い", num_strokes=2, num_points=5)
        pipeline = PlotterPipeline(kanjivg_dir=tmp_path)
        return pipeline

    def _make_placements(self, chars, font_size=6.0, spacing=6.0):
        """指定文字列のCharPlacementリストを作成。"""
        return [
            CharPlacement(char=c, x=10.0 + i * spacing, y=20.0, font_size=font_size)
            for i, c in enumerate(chars)
        ]

    def test_consecutive_chars_shifted(self, tmp_path):
        """連続文字でストロークが微量シフトされる。"""
        pipeline = self._make_pipeline_with_two_chars(tmp_path)

        single_placement = self._make_placements("い")
        single_strokes = pipeline.placements_to_strokes(single_placement)
        single_xs = np.concatenate(single_strokes, axis=0)[:, 0]

        pair_placements = self._make_placements("あい")
        pair_strokes = pipeline.placements_to_strokes(pair_placements)
        # 2文字目のストロークだけ取得（1文字目は2ストローク）
        second_char_strokes = pair_strokes[2:]
        pair_xs = np.concatenate(second_char_strokes, axis=0)[:, 0]

        # 2文字目の位置が異なるので直接比較ではなく、相対位置で比較
        # single_placement の "い" は x=10.0、pair の "い" は x=16.0
        single_relative = single_xs - 10.0
        pair_relative = pair_xs - 16.0

        # シフトが適用されていれば相対位置が異なる（前の文字方向に寄る）
        shift = single_relative.mean() - pair_relative.mean()
        assert shift != 0.0 or pair_relative.mean() < single_relative.mean()

    def test_shift_amount_within_range(self, tmp_path):
        """シフト量が0〜0.2mmの範囲内。"""
        from src.model.augmentation import AugmentConfig

        _create_kanjivg_json(tmp_path, "あ", num_strokes=2, num_points=5)
        _create_kanjivg_json(tmp_path, "い", num_strokes=2, num_points=5)
        cfg = AugmentConfig(enabled=False)

        for seed in range(20):
            np.random.seed(seed)
            pipeline = PlotterPipeline(kanjivg_dir=tmp_path)
            pipeline._typesetter._augmenter = HandwritingAugmenter(config=cfg)

            baseline_placement = [CharPlacement(char="い", x=16.0, y=20.0, font_size=6.0)]
            baseline_strokes = pipeline.placements_to_strokes(baseline_placement)
            baseline_min_x = np.concatenate(baseline_strokes, axis=0)[:, 0].min()

            pair_placements = self._make_placements("あい")
            pair_strokes = pipeline.placements_to_strokes(pair_placements)
            second_char_strokes = pair_strokes[2:]
            shifted_min_x = np.concatenate(second_char_strokes, axis=0)[:, 0].min()

            shift = baseline_min_x - shifted_min_x
            assert -0.01 <= shift <= 0.2 + 0.01, (
                f"seed={seed}: shift={shift:.4f} is outside [0, 0.2] range"
            )

    def test_shift_varies_randomly(self, tmp_path):
        """シフト量がランダムに変動する。"""
        pipeline = self._make_pipeline_with_two_chars(tmp_path)

        baseline_placement = [CharPlacement(char="い", x=16.0, y=20.0, font_size=6.0)]
        baseline_strokes = pipeline.placements_to_strokes(baseline_placement)
        baseline_min_x = np.concatenate(baseline_strokes, axis=0)[:, 0].min()

        shifts = []
        for seed in range(30):
            np.random.seed(seed)
            pair_placements = self._make_placements("あい")
            pair_strokes = pipeline.placements_to_strokes(pair_placements)
            second_char_strokes = pair_strokes[2:]
            shifted_min_x = np.concatenate(second_char_strokes, axis=0)[:, 0].min()
            shifts.append(baseline_min_x - shifted_min_x)

        unique_shifts = set(round(s, 6) for s in shifts)
        assert len(unique_shifts) > 1, f"All shift amounts are identical: {shifts[:5]}"

    def test_first_char_not_shifted(self, tmp_path):
        """最初の文字はシフトされない（distortion無効で比較）。"""
        from src.model.augmentation import AugmentConfig

        _create_kanjivg_json(tmp_path, "あ", num_strokes=2, num_points=5)
        _create_kanjivg_json(tmp_path, "い", num_strokes=2, num_points=5)
        cfg = AugmentConfig(enabled=False)
        pipeline = PlotterPipeline(kanjivg_dir=tmp_path)
        pipeline._typesetter._augmenter = HandwritingAugmenter(config=cfg)

        single_placement = self._make_placements("あ")
        single_strokes = pipeline.placements_to_strokes(single_placement)
        single_xs = np.concatenate(single_strokes, axis=0)[:, 0]

        pair_placements = self._make_placements("あい")
        pair_strokes = pipeline.placements_to_strokes(pair_placements)
        first_char_strokes = pair_strokes[:2]
        pair_xs = np.concatenate(first_char_strokes, axis=0)[:, 0]

        np.testing.assert_array_almost_equal(single_xs, pair_xs)

    def test_shift_disabled_when_no_augmenter(self):
        """augmenter無しの場合シフトなし。"""
        pipeline1 = PlotterPipeline()
        pipeline2 = PlotterPipeline()

        placements = [
            CharPlacement(char="あ", x=10.0, y=20.0, font_size=6.0),
            CharPlacement(char="い", x=16.0, y=20.0, font_size=6.0),
        ]

        np.random.seed(42)
        strokes1 = pipeline1.placements_to_strokes(placements)
        np.random.seed(42)
        strokes2 = pipeline2.placements_to_strokes(placements)

        for s1, s2 in zip(strokes1, strokes2):
            np.testing.assert_array_equal(s1, s2)


def _create_user_stroke_json(base_dir, char, strokes, suffix="001"):
    """テスト用のユーザーストロークJSONファイルを作成。"""
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


class TestUserStrokeDB:
    """_load_user_stroke_db のテスト。"""

    def test_load_multiple_chars(self, tmp_path):
        """複数文字・複数サンプルが正しくロードされる。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "あ",
            [[[10, 20], [30, 40], [50, 60]], [[70, 80], [90, 100]]],
            suffix="001",
        )
        _create_user_stroke_json(
            user_dir,
            "あ",
            [[[15, 25], [35, 45], [55, 65]]],
            suffix="002",
        )
        _create_user_stroke_json(
            user_dir,
            "い",
            [[[100, 200], [300, 400]]],
            suffix="001",
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        db = pipeline._user_stroke_db

        assert "あ" in db
        assert "い" in db
        assert len(db["あ"]) == 2  # 2サンプル
        assert len(db["い"]) == 1  # 1サンプル

        for samples in db.values():
            for strokes in samples:
                for stroke in strokes:
                    assert isinstance(stroke, np.ndarray)
                    assert stroke.ndim == 2
                    assert stroke.shape[1] == 2

    def test_empty_dir(self, tmp_path):
        """空ディレクトリの場合、空辞書。"""
        user_dir = tmp_path / "empty_strokes"
        user_dir.mkdir()
        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        assert pipeline._user_stroke_db == {}

    def test_none_dir(self):
        """user_strokes_dir=None の場合、空辞書。"""
        pipeline = PlotterPipeline()
        assert pipeline._user_stroke_db == {}

    def test_stroke_coordinates_xy_only(self, tmp_path):
        """ストロークは x, y 座標のみで (N, 2) 配列。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "う",
            [[[10.5, 20.3], [30.1, 40.7], [50.9, 60.2]]],
        )
        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        stroke = pipeline._user_stroke_db["う"][0][0]
        assert stroke.shape == (3, 2)
        np.testing.assert_allclose(stroke[0], [10.5, 20.3])
        np.testing.assert_allclose(stroke[1], [30.1, 40.7])


class TestDirectStrokeUsage:
    """直接ストローク使用のテスト（_generate_char_strokes 冒頭分岐）。"""

    def test_direct_stroke_skips_ml_inference(self, tmp_path):
        """_user_stroke_db に文字がある場合、ML推論が呼ばれない。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "あ",
            [[[0, 0], [100, 0], [100, 100]], [[0, 100], [50, 50]]],
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        mock_inference = MagicMock()
        mock_inference.generate.return_value = [np.array([[0.1, 0.2], [0.3, 0.4]])]
        pipeline._inference = mock_inference
        pipeline._style_sample = MagicMock()

        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=5.0)
        strokes = pipeline._generate_char_strokes(placement)

        mock_inference.generate.assert_not_called()
        assert len(strokes) > 0

    def test_single_sample_used_directly(self, tmp_path):
        """1サンプルの場合、そのストロークがそのまま使われる。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "え",
            [[[0, 0], [100, 0]], [[0, 100], [100, 100]]],
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        placement = CharPlacement(char="え", x=10.0, y=20.0, font_size=5.0)
        strokes = pipeline._generate_char_strokes(placement)

        assert len(strokes) == 2
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2

    def test_strokes_positioned_correctly(self, tmp_path):
        """ストロークが _position_strokes() で正しく配置される。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "漢",
            [[[0, 0], [100, 0], [100, 100], [0, 100]]],
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        placement = CharPlacement(char="漢", x=10.0, y=20.0, font_size=6.0)

        np.random.seed(0)
        strokes = pipeline._generate_char_strokes(placement)
        all_pts = np.concatenate(strokes, axis=0)

        assert all_pts[:, 0].min() >= 9.0
        assert all_pts[:, 0].max() <= 17.0
        assert all_pts[:, 1].min() >= 19.0
        assert all_pts[:, 1].max() <= 29.0

    def test_missing_char_falls_through(self, tmp_path):
        """_user_stroke_db にない文字は従来通りフォールバック。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "あ",
            [[[0, 0], [100, 100]]],
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        placement = CharPlacement(char="か", x=10.0, y=20.0, font_size=5.0)
        strokes = pipeline._generate_char_strokes(placement)

        assert len(strokes) == 1
        assert strokes[0].shape == (5, 2)


class TestStrokeSynthesis:
    """ストローク合成（複数サンプルからのランダム組み合わせ）のテスト。"""

    def test_synthesis_from_multiple_samples(self, tmp_path):
        """複数サンプルがある場合、ストロークが混合される可能性がある。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "お",
            [[[0, 0], [10, 10]], [[20, 20], [30, 30]]],
            suffix="001",
        )
        _create_user_stroke_json(
            user_dir,
            "お",
            [[[100, 100], [110, 110]], [[120, 120], [130, 130]]],
            suffix="002",
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)

        results = []
        for seed in range(50):
            np.random.seed(seed)
            placement = CharPlacement(char="お", x=10.0, y=20.0, font_size=5.0)
            strokes = pipeline._generate_char_strokes(placement)
            assert len(strokes) == 2
            results.append(strokes)

        first_concat = np.concatenate(results[0], axis=0)
        any_different = False
        for r in results[1:]:
            r_concat = np.concatenate(r, axis=0)
            if not np.allclose(first_concat, r_concat, atol=1e-6):
                any_different = True
                break
        assert any_different, "全結果が同一: 合成またはバリエーションが機能していない"

    def test_synthesis_uses_min_stroke_count(self, tmp_path):
        """合成はサンプル間の最小ストローク数を使う。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "か",
            [[[0, 0], [10, 10]], [[20, 20], [30, 30]]],
            suffix="001",
        )
        _create_user_stroke_json(
            user_dir,
            "か",
            [[[0, 0], [10, 10]], [[20, 20], [30, 30]], [[40, 40], [50, 50]]],
            suffix="002",
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        placement = CharPlacement(char="か", x=10.0, y=20.0, font_size=5.0)

        np.random.seed(42)
        strokes = pipeline._generate_char_strokes(placement)
        assert len(strokes) == 2  # min(2, 3) = 2


class TestDirectStrokeGeometricVariation:
    """直接ストロークの幾何バリエーション適用テスト。"""

    def test_variation_produces_different_results(self, tmp_path):
        """同じ文字を複数回生成して結果が異なる。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "き",
            [[[0, 0], [50, 0], [50, 50], [0, 50], [25, 75]]],
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        placement = CharPlacement(char="き", x=10.0, y=20.0, font_size=5.0)

        results = []
        for seed in range(10):
            np.random.seed(seed)
            strokes = pipeline._generate_char_strokes(placement)
            results.append(np.concatenate(strokes, axis=0))

        any_different = False
        for i in range(1, len(results)):
            if not np.allclose(results[0], results[i], atol=1e-6):
                any_different = True
                break
        assert any_different

    def test_variation_preserves_stroke_structure(self, tmp_path):
        """バリエーション後もストローク数・点数は変わらない。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "く",
            [[[0, 0], [50, 25], [100, 0]], [[10, 50], [90, 80]]],
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        placement = CharPlacement(char="く", x=10.0, y=20.0, font_size=5.0)

        for seed in range(5):
            np.random.seed(seed)
            strokes = pipeline._generate_char_strokes(placement)
            assert len(strokes) == 2
            assert strokes[0].shape[0] == 3
            assert strokes[1].shape[0] == 2

    def test_variation_within_reasonable_bounds(self, tmp_path):
        """変形が合理的な範囲内。"""
        user_dir = tmp_path / "user_strokes"
        _create_user_stroke_json(
            user_dir,
            "漢",
            [[[0, 0], [100, 0], [100, 100], [0, 100]]],
        )

        pipeline = PlotterPipeline(user_strokes_dir=user_dir)
        placement = CharPlacement(char="漢", x=10.0, y=20.0, font_size=6.0)

        for seed in range(20):
            np.random.seed(seed)
            strokes = pipeline._generate_char_strokes(placement)
            all_pts = np.concatenate(strokes, axis=0)
            assert all_pts[:, 0].min() >= 10.0 - 3.0
            assert all_pts[:, 0].max() <= 10.0 + 6.0 + 3.0
            assert all_pts[:, 1].min() >= 20.0 - 3.0
            assert all_pts[:, 1].max() <= 20.0 + 8.0 + 3.0


class TestMathSymbolStrokes:
    """_math_symbol_strokes() のテスト。"""

    @pytest.fixture
    def pipeline(self):
        return PlotterPipeline()

    @pytest.mark.parametrize("char", ["ω", "φ", "π", "θ", "α", "Δ"])
    def test_greek_letters_return_strokes(self, pipeline, char):
        """各ギリシャ文字がストロークを返す。"""
        result = pipeline._math_symbol_strokes(char)
        assert result is not None
        assert len(result) >= 1
        for s in result:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            assert s.shape[1] == 2

    @pytest.mark.parametrize("char", ["±", "≈", "∞"])
    def test_math_symbols_return_strokes(self, pipeline, char):
        """各数学記号がストロークを返す。"""
        result = pipeline._math_symbol_strokes(char)
        assert result is not None
        assert len(result) >= 1
        for s in result:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            assert s.shape[1] == 2

    def test_unsupported_char_returns_none(self, pipeline):
        """未対応の文字はNoneを返す。"""
        assert pipeline._math_symbol_strokes("あ") is None
        assert pipeline._math_symbol_strokes("A") is None

    @pytest.mark.parametrize("char", ["ω", "φ", "π", "θ", "α", "Δ", "±", "≈", "∞"])
    def test_strokes_in_unit_range(self, pipeline, char):
        """全ストロークの座標が0-1範囲内。"""
        result = pipeline._math_symbol_strokes(char)
        assert result is not None
        all_pts = np.concatenate(result, axis=0)
        assert all_pts.min() >= -0.05
        assert all_pts.max() <= 1.05

    def test_math_symbol_used_in_fallback_chain(self):
        """_generate_char_strokes() でKanjiVGの前に数式記号が使われる。"""
        pipeline = PlotterPipeline()
        placement = CharPlacement(char="π", x=10.0, y=20.0, font_size=6.0)
        strokes = pipeline._generate_char_strokes(placement)
        assert len(strokes) >= 1
        # 矩形(5点閉)ではない
        assert not (len(strokes) == 1 and strokes[0].shape == (5, 2))


class TestCharPlacementRole:
    """CharPlacement.role フィールドのテスト。"""

    def test_default_role_is_none(self):
        p = CharPlacement(char="x", x=0.0, y=0.0, font_size=6.0)
        assert p.role is None

    def test_role_can_be_set(self):
        p = CharPlacement(char="1", x=0.0, y=0.0, font_size=4.0, role="numerator")
        assert p.role == "numerator"

    def test_place_math_copies_role(self):
        """_place_math() が MathPlacement.role を CharPlacement.role にコピーする。"""
        from src.layout.typesetter import Typesetter
        from src.layout.page_layout import PageConfig

        ts = Typesetter(PageConfig(), font_size=7.0)
        output: list[CharPlacement] = []
        ts._place_math(r"\frac{a}{b}", 0.0, 0.0, 0, output)
        roles = [p.role for p in output]
        assert "numerator" in roles
        assert "denominator" in roles


class TestFractionLine:
    """分数線（水平線）のテスト。"""

    def test_fraction_line_inserted(self):
        """role=numerator の直後に水平線ストロークが挿入される。"""
        pipeline = PlotterPipeline()
        placements = [
            CharPlacement(char="1", x=10.0, y=15.0, font_size=4.0, role="numerator"),
            CharPlacement(char="2", x=10.0, y=25.0, font_size=4.0, role="denominator"),
        ]
        strokes = pipeline.placements_to_strokes(placements)
        # 分数線が追加されているはず
        # "1" の矩形(5点) + 分数線(2点) + "2" の矩形(5点)
        has_horizontal_line = False
        for s in strokes:
            if s.shape[0] == 2:
                dy = abs(s[1, 1] - s[0, 1])
                dx = abs(s[1, 0] - s[0, 0])
                if dx > 0.5 and dy < 0.5:
                    has_horizontal_line = True
                    break
        assert has_horizontal_line, "分数線が見つからない"

    def test_no_fraction_line_without_role(self):
        """role=None の場合は分数線なし。"""
        pipeline = PlotterPipeline()
        placements = [
            CharPlacement(char="1", x=10.0, y=20.0, font_size=6.0),
            CharPlacement(char="2", x=16.0, y=20.0, font_size=6.0),
        ]
        strokes = pipeline.placements_to_strokes(placements)
        # 矩形のみ（各5点）
        for s in strokes:
            assert s.shape[0] != 2 or abs(s[1, 0] - s[0, 0]) < 0.5


class TestMultiPagePreview:
    """マルチページプレビューのテスト。"""

    @pytest.fixture
    def pipeline(self):
        return PlotterPipeline()

    def test_single_page_returns_list_with_original_path(self, pipeline, tmp_path):
        """1ページのテキストは元のパスをそのまま使い、list[Path]で返す。"""
        save_path = tmp_path / "preview.png"
        result = pipeline.generate_preview("テスト", save_path=save_path)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == save_path
        assert save_path.exists()

    def test_multipage_returns_numbered_paths(self, pipeline, tmp_path):
        """複数ページになるテキストはページ番号付きファイルを返す。"""
        # line_spacing=8mm, content_height≈252mm → 約31行/ページ, chars_per_line≈28
        # 32行以上で2ページになる
        long_text = "\n".join(["あ" * 20] * 40)
        save_path = tmp_path / "preview.png"
        result = pipeline.generate_preview(long_text, save_path=save_path)
        assert isinstance(result, list)
        assert len(result) >= 2
        for p in result:
            assert p.exists()
        assert result[0] == tmp_path / "preview_p1.png"
        assert result[1] == tmp_path / "preview_p2.png"

    def test_empty_text_returns_single_page(self, pipeline, tmp_path):
        """空テキストは1ページ（空ページ）を返す。"""
        save_path = tmp_path / "empty.png"
        result = pipeline.generate_preview("", save_path=save_path)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == save_path
        assert save_path.exists()

    def test_existing_single_page_test_compat(self, pipeline, tmp_path):
        """既存テストと同じ使い方でファイルが生成される（後方互換）。"""
        preview_path = tmp_path / "preview.png"
        result = pipeline.generate_preview("テスト", save_path=preview_path)
        assert preview_path.exists()
        assert result[0] == preview_path


class TestGradioGallery:
    """Gradio UIがGalleryコンポーネントを使用するテスト。"""

    @pytest.fixture
    def app(self):
        try:
            import gradio as gr  # noqa: F401
        except ImportError:
            pytest.skip("gradio not installed")
        pipeline = PlotterPipeline()
        return pipeline.create_app()

    def test_create_app_has_gallery(self, app):
        """create_appがgr.Galleryコンポーネントを含む。"""
        import gradio as gr

        gallery_found = False
        for block in app.blocks.values():
            if isinstance(block, gr.Gallery):
                gallery_found = True
                break
        assert gallery_found, "gr.Gallery component not found in app"

    def test_create_app_no_single_image(self, app):
        """gr.Imageがプレビュー用に使われていない。"""
        import gradio as gr

        for block in app.blocks.values():
            if isinstance(block, gr.Image):
                if hasattr(block, "label") and block.label == "Preview":
                    pytest.fail("gr.Image with label 'Preview' should be replaced by gr.Gallery")


class TestPageNumber:
    """ページ番号描画のテスト。"""

    def test_page_number_draws_text(self, tmp_path):
        """page_number指定時にax.textが呼ばれること。"""
        from unittest.mock import patch

        pipeline = PlotterPipeline()
        save_path = tmp_path / "page_num.png"
        strokes = [np.array([[10.0, 10.0], [20.0, 20.0]])]
        ruled_lines = []

        with patch("matplotlib.axes.Axes.text") as mock_text:
            pipeline._preview_with_ruled_lines(strokes, ruled_lines, save_path, page_number=1)
            mock_text.assert_called_once()
            call_args = mock_text.call_args
            assert "P. 1" in str(call_args)

    def test_page_number_none_no_text(self, tmp_path):
        """page_number=Noneではax.textが呼ばれないこと。"""
        from unittest.mock import patch

        pipeline = PlotterPipeline()
        save_path = tmp_path / "no_page_num.png"
        strokes = [np.array([[10.0, 10.0], [20.0, 20.0]])]
        ruled_lines = []

        with patch("matplotlib.axes.Axes.text") as mock_text:
            pipeline._preview_with_ruled_lines(strokes, ruled_lines, save_path, page_number=None)
            mock_text.assert_not_called()

    def test_page_number_position(self, tmp_path):
        """ページ番号がページ下部中央に配置されること。"""
        from unittest.mock import patch

        pipeline = PlotterPipeline()
        save_path = tmp_path / "pos.png"

        with patch("matplotlib.axes.Axes.text") as mock_text:
            pipeline._preview_with_ruled_lines([], [], save_path, page_number=3)
            call_args, call_kwargs = mock_text.call_args
            x, y, text = call_args[0], call_args[1], call_args[2]
            paper_w = pipeline._plotter_config.paper_width
            paper_h = pipeline._plotter_config.paper_height
            margin_bottom = pipeline._page_config.margin_bottom
            assert abs(x - paper_w / 2) < 0.1
            assert abs(y - (paper_h - margin_bottom / 2)) < 0.1
            assert text == "P. 3"

    def test_page_number_format(self, tmp_path):
        """ページ番号が "P. N" 形式であること。"""
        from unittest.mock import patch

        pipeline = PlotterPipeline()
        save_path = tmp_path / "fmt.png"

        with patch("matplotlib.axes.Axes.text") as mock_text:
            pipeline._preview_with_ruled_lines([], [], save_path, page_number=12)
            call_args = mock_text.call_args
            assert call_args[0][2] == "P. 12"


# Stroke synthesis tests removed — synthesis was abandoned due to
# persistent artifacts (double strokes, missing dakuten) caused by
# stroke count/order differences between samples.

    pass
