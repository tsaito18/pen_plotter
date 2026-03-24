from pathlib import Path

from src.gcode.generator import GCodeGenerator, Stroke


class TestGCodeGeneratorSquare:
    """四角形ストロークからのG-code生成テスト"""

    def test_square_contains_required_commands(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        lines = gcode_generator.generate([square_stroke])
        text = "\n".join(lines)
        for cmd in ["G90", "G21", "M3", "M5", "G0", "G1", "M2"]:
            assert cmd in text, f"{cmd} が G-code に含まれていない"

    def test_square_has_move_to_start(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        lines = gcode_generator.generate([square_stroke])
        g0_lines = [l for l in lines if l.startswith("G0 X")]
        assert len(g0_lines) >= 1

    def test_square_has_draw_lines(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        lines = gcode_generator.generate([square_stroke])
        g1_lines = [l for l in lines if l.startswith("G1")]
        assert len(g1_lines) == 4


class TestGCodeGeneratorEmpty:
    """空ストロークリストのテスト"""

    def test_empty_strokes_has_header_and_footer(self, gcode_generator: GCodeGenerator):
        lines = gcode_generator.generate([])
        text = "\n".join(lines)
        assert "G90" in text
        assert "G21" in text
        assert "M2" in text

    def test_empty_strokes_returns_list(self, gcode_generator: GCodeGenerator):
        lines = gcode_generator.generate([])
        assert isinstance(lines, list)
        assert len(lines) > 0


class TestGCodeGeneratorSave:
    """ファイル保存テスト"""

    def test_save_creates_file(
        self, tmp_path: Path, gcode_generator: GCodeGenerator, line_stroke: Stroke
    ):
        lines = gcode_generator.generate([line_stroke])
        filepath = tmp_path / "output.gcode"
        gcode_generator.save(lines, filepath)
        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "G90" in content

    def test_save_creates_subdirectory(
        self, tmp_path: Path, gcode_generator: GCodeGenerator
    ):
        lines = gcode_generator.generate([])
        filepath = tmp_path / "sub" / "dir" / "output.gcode"
        gcode_generator.save(lines, filepath)
        assert filepath.exists()


class TestGCodeGeneratorVarySpeed:
    """フィードレート変調テスト"""

    def test_vary_speed_default_on(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        """vary_speed=True（デフォルト）でフィードレートが変調される"""
        lines = gcode_generator.generate([square_stroke], vary_speed=True)
        g1_lines = [l for l in lines if l.startswith("G1")]
        feed_rates = []
        for l in g1_lines:
            for part in l.split():
                if part.startswith("F"):
                    feed_rates.append(float(part[1:]))
        assert len(set(feed_rates)) > 1, "フィードレートが全て同じ"

    def test_vary_speed_off_uniform(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        """vary_speed=Falseで従来通り均一フィードレート"""
        lines = gcode_generator.generate([square_stroke], vary_speed=False)
        g1_lines = [l for l in lines if l.startswith("G1")]
        feed_rates = set()
        for l in g1_lines:
            for part in l.split():
                if part.startswith("F"):
                    feed_rates.add(float(part[1:]))
        assert len(feed_rates) == 1, "vary_speed=Falseなのにフィードレートが変わっている"

    def test_vary_speed_start_slower(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        """始点付近のフィードレートが draw_speed より遅い"""
        lines = gcode_generator.generate([square_stroke], vary_speed=True)
        g1_lines = [l for l in lines if l.startswith("G1")]
        first_feed = None
        for part in g1_lines[0].split():
            if part.startswith("F"):
                first_feed = float(part[1:])
        assert first_feed is not None
        assert first_feed < gcode_generator.config.draw_speed

    def test_vary_speed_end_faster(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        """終点付近のフィードレートが draw_speed より速い"""
        lines = gcode_generator.generate([square_stroke], vary_speed=True)
        g1_lines = [l for l in lines if l.startswith("G1")]
        last_feed = None
        for part in g1_lines[-1].split():
            if part.startswith("F"):
                last_feed = float(part[1:])
        assert last_feed is not None
        assert last_feed > gcode_generator.config.draw_speed

    def test_vary_speed_two_point_stroke(
        self, gcode_generator: GCodeGenerator, line_stroke: Stroke
    ):
        """2点ストローク（1セグメント）でもクラッシュしない"""
        lines = gcode_generator.generate([line_stroke], vary_speed=True)
        g1_lines = [l for l in lines if l.startswith("G1")]
        assert len(g1_lines) == 1
