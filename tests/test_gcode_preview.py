import matplotlib

matplotlib.use("Agg")

from pathlib import Path

from src.gcode.generator import GCodeGenerator, Stroke
from src.gcode.preview import preview_gcode, preview_strokes


class TestPreviewStrokes:
    def test_saves_image_file(self, tmp_path: Path, square_stroke: Stroke):
        save_path = tmp_path / "preview_strokes.png"
        preview_strokes([square_stroke], save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPreviewGcode:
    def test_saves_image_file(self, tmp_path: Path, sample_gcode: list[str]):
        save_path = tmp_path / "preview_gcode.png"
        preview_gcode(sample_gcode, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
