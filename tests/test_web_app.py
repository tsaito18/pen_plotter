import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.layout.typesetter import CharPlacement
from src.ui.web_app import PlotterPipeline


def _create_kanjivg_json(base_dir, char, num_strokes=3, num_points=8):
    """テスト用のKanjiVG JSONファイルを作成。"""
    char_dir = base_dir / char
    char_dir.mkdir(parents=True, exist_ok=True)
    strokes = []
    for s in range(num_strokes):
        stroke = [
            {"x": float(i + s), "y": float(i * 0.5 + s), "pressure": 1.0, "timestamp": 0.0}
            for i in range(num_points)
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
        expected_size = 6.0 * 0.88
        assert np.isclose(rendered_w, expected_size, atol=0.01)
        assert np.isclose(rendered_h, expected_size, atol=0.01)

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
            (char_dir / f"{char}_001.json").write_text(
                json.dumps(stroke_data), encoding="utf-8"
            )

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
        pipeline = self._make_pipeline_with_two_chars(tmp_path)

        # シフトなし（単独文字）の基準位置を取得
        baseline_placement = [
            CharPlacement(char="い", x=16.0, y=20.0, font_size=6.0)
        ]
        baseline_strokes = pipeline.placements_to_strokes(baseline_placement)
        baseline_min_x = np.concatenate(baseline_strokes, axis=0)[:, 0].min()

        for seed in range(20):
            np.random.seed(seed)
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

        baseline_placement = [
            CharPlacement(char="い", x=16.0, y=20.0, font_size=6.0)
        ]
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
        assert len(unique_shifts) > 1, (
            f"All shift amounts are identical: {shifts[:5]}"
        )

    def test_first_char_not_shifted(self, tmp_path):
        """最初の文字はシフトされない。"""
        pipeline = self._make_pipeline_with_two_chars(tmp_path)

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
