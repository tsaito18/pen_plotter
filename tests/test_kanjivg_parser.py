import numpy as np
import pytest
from src.collector.kanjivg_parser import KanjiVGParser, parse_svg_path


# KanjiVG形式の最小限サンプルSVG
SAMPLE_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109" viewBox="0 0 109 109">
  <g id="kvg:StrokePaths_04e00" style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">
    <g id="kvg:04e00" kvg:element="丠">
      <g id="kvg:04e00-g1">
        <path id="kvg:04e00-s1" d="M 30,20 C 30,50 30,80 30,90"/>
        <path id="kvg:04e00-s2" d="M 70,20 L 70,90"/>
      </g>
    </g>
  </g>
</svg>"""


class TestParseSvgPath:
    def test_move_and_line(self):
        points = parse_svg_path("M 10,20 L 30,40")
        assert len(points) >= 2
        assert np.allclose(points[0], [10.0, 20.0])
        assert np.allclose(points[-1], [30.0, 40.0])

    def test_move_and_curve(self):
        points = parse_svg_path("M 10,20 C 15,25 25,35 30,40")
        assert len(points) >= 2
        assert np.allclose(points[0], [10.0, 20.0])
        assert np.allclose(points[-1], [30.0, 40.0])

    def test_multiple_segments(self):
        points = parse_svg_path("M 0,0 L 10,10 L 20,0")
        assert len(points) >= 3

    def test_empty_path(self):
        points = parse_svg_path("")
        assert len(points) == 0

    def test_parse_smooth_bezier_absolute(self):
        points = parse_svg_path("M 10,20 S 30,40 50,60")
        assert len(points) >= 2
        assert np.allclose(points[0], [10.0, 20.0])
        assert np.allclose(points[-1], [50.0, 60.0])

    def test_parse_smooth_bezier_relative(self):
        points = parse_svg_path("M 10,20 s 20,20 40,40")
        assert len(points) >= 2
        assert np.allclose(points[0], [10.0, 20.0])
        assert np.allclose(points[-1], [50.0, 60.0])

    def test_parse_smooth_bezier_after_cubic(self):
        """C followed by s: reflected control point should differ from current."""
        points = parse_svg_path("M 0,0 C 10,0 20,10 20,20 s 10,10 20,0")
        assert len(points) >= 10
        assert np.allclose(points[-1], [40.0, 20.0])
        xs = points[:, 0]
        assert xs.max() - xs.min() > 30

    def test_parse_gaku_stroke_6(self):
        d = "M37.25,46.5c1,0.25,3.75,0.25,5.5,-0.25s18.25,-4,20,-4s2.75,0.75,1,2.25S54.5,53.5,53,54.75"
        points = parse_svg_path(d)
        assert len(points) > 20
        assert points[-1][0] == pytest.approx(53.0, abs=0.1)
        assert points[-1][1] == pytest.approx(54.75, abs=0.1)
        assert points[:, 0].max() - points[:, 0].min() > 10


class TestKanjiVGParser:
    def test_parse_svg_string(self):
        parser = KanjiVGParser()
        strokes = parser.parse_svg(SAMPLE_SVG)
        assert len(strokes) == 2  # 2つのpath要素

    def test_stroke_is_numpy_array(self):
        parser = KanjiVGParser()
        strokes = parser.parse_svg(SAMPLE_SVG)
        for stroke in strokes:
            assert isinstance(stroke, np.ndarray)
            assert stroke.ndim == 2
            assert stroke.shape[1] == 2

    def test_parse_svg_file(self, tmp_path):
        svg_file = tmp_path / "test.svg"
        svg_file.write_text(SAMPLE_SVG, encoding="utf-8")
        parser = KanjiVGParser()
        strokes = parser.parse_file(svg_file)
        assert len(strokes) == 2

    def test_normalize_to_unit(self):
        parser = KanjiVGParser()
        strokes = parser.parse_svg(SAMPLE_SVG)
        normalized = parser.normalize(strokes, target_size=10.0)
        for stroke in normalized:
            assert stroke[:, 0].max() <= 10.0
            assert stroke[:, 1].max() <= 10.0
            assert stroke[:, 0].min() >= 0.0
            assert stroke[:, 1].min() >= 0.0
