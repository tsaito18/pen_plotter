from unittest.mock import MagicMock

import numpy as np
import pytest

from src.layout.typesetter import CharPlacement
from src.ui.web_app import PlotterPipeline

SAMPLE_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109" viewBox="0 0 109 109">
  <g id="kvg:StrokePaths" style="fill:none;stroke:#000000;stroke-width:3;">
    <g id="kvg:03042" kvg:element="あ">
      <path id="kvg:03042-s1" d="M 30,20 C 30,50 30,80 30,90"/>
      <path id="kvg:03042-s2" d="M 20,50 L 80,50"/>
      <path id="kvg:03042-s3" d="M 50,30 C 60,60 70,80 80,90"/>
    </g>
  </g>
</svg>"""


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
        """KanjiVGファイルが存在する文字はKanjiVGストロークを使用。"""
        # 「あ」= U+3042 → 03042.svg
        svg_path = tmp_path / "03042.svg"
        svg_path.write_text(SAMPLE_SVG)

        pipeline = PlotterPipeline(kanjivg_dir=tmp_path)
        placement = CharPlacement(char="あ", x=10.0, y=20.0, font_size=5.0)
        strokes = pipeline.placements_to_strokes([placement])

        assert len(strokes) == 3  # SVGに3ストロークある
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            # 矩形ではない（5点の閉じた四角形ではない）
            assert s.shape[0] != 5 or not np.allclose(s[0], s[-1])

        # 配置位置の範囲内にある
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
