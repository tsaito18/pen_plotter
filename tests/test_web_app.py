from pathlib import Path
import numpy as np
import pytest
from src.ui.web_app import PlotterPipeline


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
