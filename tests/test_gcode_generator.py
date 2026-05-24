from pathlib import Path

from src.gcode.generator import GCodeGenerator, Stroke


class TestGCodeGeneratorSquare:
    """四角形ストロークからのG-code生成テスト"""

    def test_square_contains_required_commands(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        lines = gcode_generator.generate([square_stroke])
        text = "\n".join(lines)
        for cmd in ["$H", "G92", "G90", "G0", "G1"]:
            assert cmd in text, f"{cmd} が G-code に含まれていない"

    def test_square_contains_pen_commands(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        lines = gcode_generator.generate([square_stroke])
        text = "\n".join(lines)
        assert gcode_generator.config.pen_down_command in text
        assert gcode_generator.config.pen_up_command in text

    def test_square_has_move_to_start(self, gcode_generator: GCodeGenerator, square_stroke: Stroke):
        lines = gcode_generator.generate([square_stroke])
        g0_lines = [line for line in lines if line.startswith("G0 X")]
        assert len(g0_lines) >= 1

    def test_square_has_draw_lines(self, gcode_generator: GCodeGenerator, square_stroke: Stroke):
        lines = gcode_generator.generate([square_stroke])
        g1_lines = [line for line in lines if line.startswith("G1 ")]
        assert len(g1_lines) == 4

    def test_pen_down_count_matches_stroke_count(
        self,
        gcode_generator: GCodeGenerator,
        line_stroke: Stroke,
        square_stroke: Stroke,
    ):
        """ストローク数だけ pen_down_command が現れる（各ストローク開始時に1回）"""
        lines = gcode_generator.generate([line_stroke, square_stroke])
        text = "\n".join(lines)
        pen_down = gcode_generator.config.pen_down_command
        assert text.count(pen_down) == 2

    def test_pen_down_after_g0_before_g1(
        self,
        gcode_generator: GCodeGenerator,
        line_stroke: Stroke,
        square_stroke: Stroke,
    ):
        """各 pen_down_command 行は直前が G0 X 移動、直後が G1 描画であること"""
        lines = gcode_generator.generate([line_stroke, square_stroke])
        pen_down = gcode_generator.config.pen_down_command
        pen_down_indices = [i for i, line in enumerate(lines) if line == pen_down]
        assert len(pen_down_indices) == 2
        for idx in pen_down_indices:
            assert idx > 0, "pen_down が先頭行にある"
            assert idx < len(lines) - 1, "pen_down が末尾行にある"
            prev_line = lines[idx - 1]
            next_line = lines[idx + 1]
            assert prev_line.startswith("G0 X"), (
                f"pen_down の直前が G0 X ではない: {prev_line!r}"
            )
            assert next_line.startswith("G1"), (
                f"pen_down の直後が G1 ではない: {next_line!r}"
            )

    def test_pen_up_in_final_section(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        """退避前にペンを上げるため、最終3行以内に pen_up_command が現れる"""
        lines = gcode_generator.generate([square_stroke])
        pen_up = gcode_generator.config.pen_up_command
        tail = lines[-3:]
        assert pen_up in tail, f"末尾3行に pen_up_command がない: {tail}"

    def test_pen_up_before_g0_travel(
        self,
        gcode_generator: GCodeGenerator,
        line_stroke: Stroke,
        square_stroke: Stroke,
    ):
        """全 G0 トラベル移動の直前が pen_up_command であること（ホーミング直後の初回 G0 を除く）"""
        lines = gcode_generator.generate([line_stroke, square_stroke])
        pen_up = gcode_generator.config.pen_up_command
        g0_indices = [i for i, line in enumerate(lines) if line.startswith("G0 X")]
        # ホーミング直後の最初の G0 はペンが上の状態なので除外可、2 個目以降の G0 を検査
        assert len(g0_indices) >= 2, "G0 トラベルが2回以上必要"
        for idx in g0_indices[1:]:
            assert idx > 0
            prev_line = lines[idx - 1]
            assert prev_line == pen_up, (
                f"G0 トラベル直前が pen_up_command ではない: {prev_line!r}"
            )


class TestGCodeGeneratorEmpty:
    """空ストロークリストのテスト"""

    def test_empty_strokes_has_header_and_footer(self, gcode_generator: GCodeGenerator):
        lines = gcode_generator.generate([])
        text = "\n".join(lines)
        assert "$H" in text
        assert "G92" in text
        assert "G90" in text

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

    def test_save_creates_subdirectory(self, tmp_path: Path, gcode_generator: GCodeGenerator):
        lines = gcode_generator.generate([])
        filepath = tmp_path / "sub" / "dir" / "output.gcode"
        gcode_generator.save(lines, filepath)
        assert filepath.exists()


class TestGCodeGeneratorVarySpeed:
    """フィードレート変調テスト"""

    def test_vary_speed_default_on(self, gcode_generator: GCodeGenerator, square_stroke: Stroke):
        """vary_speed=True（デフォルト）でフィードレートが変調される"""
        lines = gcode_generator.generate([square_stroke], vary_speed=True)
        g1_lines = [line for line in lines if line.startswith("G1 ")]
        feed_rates = []
        for line in g1_lines:
            for part in line.split():
                if part.startswith("F"):
                    feed_rates.append(float(part[1:]))
        assert len(set(feed_rates)) > 1, "フィードレートが全て同じ"

    def test_vary_speed_off_uniform(self, gcode_generator: GCodeGenerator, square_stroke: Stroke):
        """vary_speed=Falseで従来通り均一フィードレート"""
        lines = gcode_generator.generate([square_stroke], vary_speed=False)
        g1_lines = [line for line in lines if line.startswith("G1 ")]
        feed_rates = set()
        for line in g1_lines:
            for part in line.split():
                if part.startswith("F"):
                    feed_rates.add(float(part[1:]))
        assert len(feed_rates) == 1, "vary_speed=Falseなのにフィードレートが変わっている"

    def test_vary_speed_start_slower(self, gcode_generator: GCodeGenerator, square_stroke: Stroke):
        """始点付近のフィードレートが draw_speed より遅い"""
        lines = gcode_generator.generate([square_stroke], vary_speed=True)
        g1_lines = [line for line in lines if line.startswith("G1 ")]
        first_feed = None
        for part in g1_lines[0].split():
            if part.startswith("F"):
                first_feed = float(part[1:])
        assert first_feed is not None
        assert first_feed < gcode_generator.config.draw_speed

    def test_vary_speed_end_slower(self, gcode_generator: GCodeGenerator, square_stroke: Stroke):
        """終点付近のフィードレートが draw_speed より遅い（S字カーブで減速）"""
        lines = gcode_generator.generate([square_stroke], vary_speed=True)
        g1_lines = [line for line in lines if line.startswith("G1 ")]
        last_feed = None
        for part in g1_lines[-1].split():
            if part.startswith("F"):
                last_feed = float(part[1:])
        assert last_feed is not None
        assert last_feed < gcode_generator.config.draw_speed

    def test_vary_speed_middle_fastest(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        """中盤のフィードレートが始点・終点より速い（ベル型S字カーブ）"""
        lines = gcode_generator.generate([square_stroke], vary_speed=True)
        g1_lines = [line for line in lines if line.startswith("G1 ")]
        feed_rates = []
        for line in g1_lines:
            for part in line.split():
                if part.startswith("F"):
                    feed_rates.append(float(part[1:]))
        assert len(feed_rates) >= 3
        mid = len(feed_rates) // 2
        assert feed_rates[mid] > feed_rates[0]
        assert feed_rates[mid] > feed_rates[-1]

    def test_vary_speed_feed_rate_is_integer(
        self, gcode_generator: GCodeGenerator, square_stroke: Stroke
    ):
        """F値が整数であること（小数点を含まない）"""
        lines = gcode_generator.generate([square_stroke], vary_speed=True)
        g1_lines = [line for line in lines if line.startswith("G1 ")]
        for line in g1_lines:
            for part in line.split():
                if part.startswith("F"):
                    f_value = part[1:]
                    assert "." not in f_value, f"F値に小数点が含まれている: {part}"

    def test_vary_speed_two_point_stroke(
        self, gcode_generator: GCodeGenerator, line_stroke: Stroke
    ):
        """2点ストローク（1セグメント）でもクラッシュしない"""
        lines = gcode_generator.generate([line_stroke], vary_speed=True)
        g1_lines = [line for line in lines if line.startswith("G1 ")]
        assert len(g1_lines) == 1
