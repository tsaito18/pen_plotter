from dataclasses import replace
from pathlib import Path

import numpy as np

from src.gcode.config import PlotterConfig
from src.gcode.generator import GCodeGenerator, Stroke


def _finish_only_generator() -> GCodeGenerator:
    """終端加工を単独検証するための、画内筆圧変調・入筆を 0 にした generator。"""
    return GCodeGenerator(replace(PlotterConfig(), pressure_variation=0.0, entry_taper=0.0))


def _g1_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("G1 ")]


def _z_values(g1_lines: list[str]) -> list[float]:
    zs = []
    for line in g1_lines:
        for part in line.split():
            if part.startswith("Z"):
                zs.append(float(part[1:]))
    return zs


def _long_stroke() -> Stroke:
    """終端リフトが見えるだけの点数を持つ右向き直線(11点)。"""
    xs = np.linspace(0.0, 50.0, 11)
    return np.column_stack([xs, np.zeros_like(xs)])


class TestConnectFinish:
    """連綿(つなぎ画 finish='connect')はペンを上げず継続して描く。"""

    def _pen_up_count(self, lines: list[str], cfg) -> int:
        return sum(1 for line in lines if line == cfg.pen_up_command)

    def test_connect_skips_pen_up_for_connection_and_next(self):
        g = _finish_only_generator()
        s0 = np.array([[0.0, 0.0], [5.0, 0.0]])
        conn = np.array([[5.0, 0.0], [6.0, 1.0]])  # つなぎ画
        s1 = np.array([[6.0, 1.0], [10.0, 1.0]])
        connected = g.generate([s0, conn, s1], finishes=["tome", "connect", "tome"])
        plain = g.generate([s0, conn, s1], finishes=["tome", "tome", "tome"])
        # 連綿では つなぎ画＋直後の画 がペンを上げず継続 → pen_up が2回少ない
        assert self._pen_up_count(connected, g.config) == self._pen_up_count(plain, g.config) - 2

    def test_connect_emits_constant_faint_z(self):
        g = _finish_only_generator()
        s0 = np.array([[0.0, 0.0], [5.0, 0.0]])
        conn = np.array([[5.0, 0.0], [6.0, 0.0]])
        s1 = np.array([[6.0, 0.0], [10.0, 0.0]])
        lines = g.generate([s0, conn, s1], finishes=["tome", "connect", "tome"])
        # つなぎ画は接触 < 1 ＝ Z 付き（薄い）。Z は finish_lift_z〜pen_down_z 内
        zs = _z_values(_g1_lines(lines))
        assert len(zs) > 0
        for z in zs:
            assert g.config.finish_lift_z - 1e-6 <= z <= g.config.pen_down_z + 1e-6


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
            assert prev_line.startswith("G0 X"), f"pen_down の直前が G0 X ではない: {prev_line!r}"
            assert next_line.startswith("G1"), f"pen_down の直後が G1 ではない: {next_line!r}"

    def test_pen_up_in_final_section(self, gcode_generator: GCodeGenerator, square_stroke: Stroke):
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
            assert prev_line == pen_up, f"G0 トラベル直前が pen_up_command ではない: {prev_line!r}"


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


class TestGCodeGeneratorFinishLift:
    """終端Zリフト（払い・はねの接触圧抜き）テスト"""

    def test_finishes_none_matches_legacy(self, gcode_generator: GCodeGenerator):
        """finishes 無し/None/["none"] は従来出力と完全一致（後方互換）"""
        stroke = _long_stroke()
        legacy = gcode_generator.generate([stroke])
        assert gcode_generator.generate([stroke], finishes=None) == legacy
        assert gcode_generator.generate([stroke], finishes=["none"]) == legacy

    def test_tome_matches_legacy(self):
        """とめは終端加工なし（筆圧変調を切れば Z 付き G1 を出さず従来と一致）"""
        gen = _finish_only_generator()
        stroke = _long_stroke()
        legacy = gen.generate([stroke])
        tome = gen.generate([stroke], finishes=["tome"])
        assert tome == legacy
        assert _z_values(_g1_lines(tome)) == []

    def test_harai_emits_z_in_terminal_g1(self, gcode_generator: GCodeGenerator):
        """払いは終端の G1 に Z を含む"""
        stroke = _long_stroke()
        lines = gcode_generator.generate([stroke], finishes=["harai"])
        zs = _z_values(_g1_lines(lines))
        assert len(zs) > 0

    def test_harai_z_within_safe_range(self, gcode_generator: GCodeGenerator):
        """Z は finish_lift_z 〜 pen_down_z の範囲内"""
        cfg = gcode_generator.config
        stroke = _long_stroke()
        lines = gcode_generator.generate([stroke], finishes=["harai"])
        zs = _z_values(_g1_lines(lines))
        for z in zs:
            assert cfg.finish_lift_z - 1e-6 <= z <= cfg.pen_down_z + 1e-6

    def test_harai_z_monotonic_toward_lift(self):
        """終端へ向かって Z が単調に持ち上がる（筆圧変調を切った単独検証）"""
        gen = _finish_only_generator()
        cfg = gen.config
        stroke = _long_stroke()
        lines = gen.generate([stroke], finishes=["harai"])
        zs = _z_values(_g1_lines(lines))
        assert all(b <= a + 1e-9 for a, b in zip(zs, zs[1:]))  # 単調非増加
        assert zs[-1] < cfg.pen_down_z  # 接触高さから持ち上がっている
        # 先端でも接触を残す設計なので少なくとも半分は持ち上がる
        assert zs[-1] <= cfg.pen_down_z - 0.5 * (cfg.pen_down_z - cfg.finish_lift_z)

    def test_hane_emits_z(self, gcode_generator: GCodeGenerator):
        """はねも終端 G1 に Z を含む"""
        stroke = _long_stroke()
        lines = gcode_generator.generate([stroke], finishes=["hane"])
        assert len(_z_values(_g1_lines(lines))) > 0

    def test_finishes_shorter_than_strokes_safe(self, gcode_generator: GCodeGenerator):
        """finishes が strokes より短い場合、不足分は none 扱いで落ちない"""
        stroke = _long_stroke()
        lines = gcode_generator.generate([stroke, stroke], finishes=["harai"])
        assert isinstance(lines, list)

    def test_harai_terminal_feed_slower(self, gcode_generator: GCodeGenerator):
        """払いの終端区間は速度倍率で減速する"""
        cfg = gcode_generator.config
        stroke = _long_stroke()
        lines = gcode_generator.generate([stroke], finishes=["harai"], vary_speed=False)
        g1 = _g1_lines(lines)
        last_feed = float(g1[-1].split("F")[1])
        assert last_feed < cfg.draw_speed


class TestSimplifyStroke:
    """RDP ストローク簡略化（実機スタッター抑制）。"""

    def test_collinear_points_collapse_to_endpoints(self):
        from src.gcode.generator import simplify_stroke

        line = np.stack([np.linspace(0, 3.5, 32), np.zeros(32)], axis=1)
        out = simplify_stroke(line, tolerance_mm=0.05)
        assert len(out) == 2
        assert np.allclose(out[0], line[0])
        assert np.allclose(out[-1], line[-1])

    def test_curve_preserved_within_tolerance(self):
        from src.gcode.generator import simplify_stroke

        t = np.linspace(0, 1, 32)
        curve = np.stack([3.5 * t, 0.5 * np.sin(np.pi * t)], axis=1)
        out = simplify_stroke(curve, tolerance_mm=0.05)
        # 32点 → 大幅削減しつつ端点保持
        assert 3 <= len(out) < 32
        assert np.allclose(out[0], curve[0]) and np.allclose(out[-1], curve[-1])

    def test_zero_tolerance_is_noop(self):
        from src.gcode.generator import simplify_stroke

        t = np.linspace(0, 1, 10)
        s = np.stack([t, t**2], axis=1)
        out = simplify_stroke(s, tolerance_mm=0.0)
        assert len(out) == len(s)

    def test_generator_reduces_g1_segments(self):
        # 直線的なストロークは G1 が大幅に減る（スタッター抑制）
        line = np.stack([np.linspace(2, 6, 32), np.full(32, 100.0)], axis=1)
        many = GCodeGenerator(PlotterConfig(simplify_tolerance_mm=0.0))
        few = GCodeGenerator(PlotterConfig(simplify_tolerance_mm=0.05))
        n_many = sum(1 for s in many._stroke_to_gcode(line) if s.startswith("G1 X"))
        n_few = sum(1 for s in few._stroke_to_gcode(line) if s.startswith("G1 X"))
        assert n_few < n_many
