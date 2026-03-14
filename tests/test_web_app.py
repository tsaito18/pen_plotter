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
        half = placement.font_size / 2.0
        assert all_pts[:, 0].min() >= placement.x - 0.01
        assert all_pts[:, 0].max() <= placement.x + placement.font_size + 0.01
        assert all_pts[:, 1].min() >= placement.y - half - 0.01
        assert all_pts[:, 1].max() <= placement.y + half + 0.01

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
        """_position_strokesがスケーリングと平行移動を正しく行う。"""
        pipeline = PlotterPipeline()
        normalized = [
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        ]
        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=6.0)
        result = pipeline._position_strokes(normalized, placement)

        assert len(result) == 1
        # (0,0) → (10.0, 20.0 - 3.0) = (10.0, 17.0)
        assert np.allclose(result[0][0], [10.0, 17.0])
        # (1,1) → (10.0 + 6.0, 17.0 + 6.0) = (16.0, 23.0)
        assert np.allclose(result[0][-1], [16.0, 23.0])

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
