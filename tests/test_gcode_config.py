from src.gcode.config import PlotterConfig


class TestPlotterConfigDefaults:
    def test_work_area(self, plotter_config: PlotterConfig):
        assert plotter_config.work_area_width == 210.0
        assert plotter_config.work_area_height == 297.0

    def test_paper_settings(self, plotter_config: PlotterConfig):
        assert plotter_config.paper_origin_x == 0.0
        assert plotter_config.paper_origin_y == 0.0
        assert plotter_config.paper_width == 210.0
        assert plotter_config.paper_height == 297.0

    def test_speed_settings(self, plotter_config: PlotterConfig):
        assert plotter_config.travel_speed == 5000.0
        assert plotter_config.draw_speed == 1000.0

    def test_pen_commands(self, plotter_config: PlotterConfig):
        assert plotter_config.pen_down_command == "G1G90 Z3.5 F5000"
        assert plotter_config.pen_up_command == "G1G90 Z0.5 F5000"
        assert plotter_config.pen_delay == 0.0

    def test_decimal_places(self, plotter_config: PlotterConfig):
        assert plotter_config.decimal_places == 2


class TestPenDelayGcode:
    def test_default_delay(self, plotter_config: PlotterConfig):
        result = plotter_config.pen_delay_gcode()
        assert result == ""

    def test_custom_delay(self):
        cfg = PlotterConfig(pen_delay=0.5)
        result = cfg.pen_delay_gcode()
        assert result == "G4 P500"

    def test_zero_delay(self):
        cfg = PlotterConfig(pen_delay=0.0)
        result = cfg.pen_delay_gcode()
        assert result == ""

    def test_format_starts_with_g4(self):
        result = PlotterConfig(pen_delay=0.15).pen_delay_gcode()
        assert result.startswith("G4 P")
