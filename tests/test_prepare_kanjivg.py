from __future__ import annotations

import json
from pathlib import Path

from src.collector.data_format import StrokePoint, StrokeSample


MINIMAL_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109" viewBox="0 0 109 109">
  <g id="kvg:StrokePaths_04e00" style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">
    <g id="kvg:04e00" kvg:element="丠">
      <g id="kvg:04e00-g1">
        <path id="kvg:04e00-s1" d="M 30,20 L 30,90"/>
        <path id="kvg:04e00-s2" d="M 70,20 L 70,90"/>
      </g>
    </g>
  </g>
</svg>"""


class TestHexCodepointToChar:
    def test_basic_cjk(self):
        from scripts.prepare_kanjivg import hex_filename_to_char

        assert hex_filename_to_char("0904e") == "過"

    def test_hiragana(self):
        from scripts.prepare_kanjivg import hex_filename_to_char

        assert hex_filename_to_char("03042") == "あ"

    def test_uppercase_hex(self):
        from scripts.prepare_kanjivg import hex_filename_to_char

        assert hex_filename_to_char("0904E") == "過"

    def test_short_hex(self):
        from scripts.prepare_kanjivg import hex_filename_to_char

        assert hex_filename_to_char("41") == "A"


class TestConvertSingleSvg:
    def test_converts_svg_to_stroke_sample(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_single_svg

        svg_file = tmp_path / "0904e.svg"
        svg_file.write_text(MINIMAL_SVG, encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = convert_single_svg(svg_file, output_dir, target_size=10.0, num_points=8)

        assert sample is not None
        assert sample.character == "過"
        assert len(sample.strokes) == 2

    def test_strokes_contain_stroke_points(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_single_svg

        svg_file = tmp_path / "0904e.svg"
        svg_file.write_text(MINIMAL_SVG, encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = convert_single_svg(svg_file, output_dir, target_size=10.0, num_points=8)

        for stroke in sample.strokes:
            for pt in stroke:
                assert isinstance(pt, StrokePoint)
                assert pt.pressure == 1.0
                assert pt.timestamp == 0.0

    def test_resampling_applied(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_single_svg

        svg_file = tmp_path / "0904e.svg"
        svg_file.write_text(MINIMAL_SVG, encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = convert_single_svg(svg_file, output_dir, target_size=10.0, num_points=16)

        for stroke in sample.strokes:
            assert len(stroke) == 16

    def test_normalization_applied(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_single_svg

        svg_file = tmp_path / "0904e.svg"
        svg_file.write_text(MINIMAL_SVG, encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = convert_single_svg(svg_file, output_dir, target_size=10.0, num_points=8)

        for stroke in sample.strokes:
            xs = [pt.x for pt in stroke]
            ys = [pt.y for pt in stroke]
            assert max(xs) <= 10.0 + 0.01
            assert max(ys) <= 10.0 + 0.01
            assert min(xs) >= -0.01
            assert min(ys) >= -0.01

    def test_saves_json_file(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_single_svg

        svg_file = tmp_path / "0904e.svg"
        svg_file.write_text(MINIMAL_SVG, encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        convert_single_svg(svg_file, output_dir, target_size=10.0, num_points=8)

        char_dir = output_dir / "過"
        assert char_dir.exists()
        json_files = list(char_dir.glob("過_*.json"))
        assert len(json_files) == 1

        loaded = StrokeSample.load(json_files[0])
        assert loaded.character == "過"

    def test_skip_non_hex_filename(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_single_svg

        svg_file = tmp_path / "not_hex.svg"
        svg_file.write_text(MINIMAL_SVG, encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = convert_single_svg(svg_file, output_dir, target_size=10.0, num_points=8)
        assert result is None


class TestBatchConversion:
    def _create_svg(self, directory: Path, hex_name: str) -> Path:
        svg_file = directory / f"{hex_name}.svg"
        svg_file.write_text(MINIMAL_SVG, encoding="utf-8")
        return svg_file

    def test_converts_multiple_files(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_kanjivg_to_samples

        svg_dir = tmp_path / "svgs"
        svg_dir.mkdir()
        self._create_svg(svg_dir, "03042")  # あ
        self._create_svg(svg_dir, "03044")  # い
        self._create_svg(svg_dir, "03046")  # う

        output_dir = tmp_path / "output"
        count = convert_kanjivg_to_samples(svg_dir, output_dir, target_size=10.0, num_points=8)

        assert count == 3

    def test_output_directory_structure(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_kanjivg_to_samples

        svg_dir = tmp_path / "svgs"
        svg_dir.mkdir()
        self._create_svg(svg_dir, "03042")  # あ
        self._create_svg(svg_dir, "03044")  # い

        output_dir = tmp_path / "output"
        convert_kanjivg_to_samples(svg_dir, output_dir, target_size=10.0, num_points=8)

        assert (output_dir / "あ").is_dir()
        assert (output_dir / "い").is_dir()
        assert len(list((output_dir / "あ").glob("あ_*.json"))) == 1
        assert len(list((output_dir / "い").glob("い_*.json"))) == 1

    def test_dataset_compatible_format(self, tmp_path: Path):
        """StrokeDatasetで読み込めるJSON形式であることを確認。"""
        from scripts.prepare_kanjivg import convert_kanjivg_to_samples

        svg_dir = tmp_path / "svgs"
        svg_dir.mkdir()
        self._create_svg(svg_dir, "03042")  # あ

        output_dir = tmp_path / "output"
        convert_kanjivg_to_samples(svg_dir, output_dir, target_size=10.0, num_points=8)

        json_file = list((output_dir / "あ").glob("*.json"))[0]
        data = json.loads(json_file.read_text(encoding="utf-8"))

        assert "character" in data
        assert "strokes" in data
        assert isinstance(data["strokes"], list)
        for stroke in data["strokes"]:
            assert isinstance(stroke, list)
            for pt in stroke:
                assert "x" in pt
                assert "y" in pt
                assert "pressure" in pt

    def test_skips_non_svg_files(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_kanjivg_to_samples

        svg_dir = tmp_path / "svgs"
        svg_dir.mkdir()
        self._create_svg(svg_dir, "03042")
        (svg_dir / "readme.txt").write_text("ignore me")

        output_dir = tmp_path / "output"
        count = convert_kanjivg_to_samples(svg_dir, output_dir, target_size=10.0, num_points=8)

        assert count == 1

    def test_empty_directory(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_kanjivg_to_samples

        svg_dir = tmp_path / "svgs"
        svg_dir.mkdir()
        output_dir = tmp_path / "output"

        count = convert_kanjivg_to_samples(svg_dir, output_dir, target_size=10.0, num_points=8)
        assert count == 0


class TestConvertXml:
    """KanjiVG統合XMLからの変換テスト。"""

    MINIMAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<kanjivg xmlns:kvg='http://kanjivg.tagaini.net'>
<kanji id="kvg:kanji_04e00">
<g id="kvg:04e00" kvg:element="一">
  <path id="kvg:04e00-s1" d="M 10,50 L 90,50"/>
</g>
</kanji>
<kanji id="kvg:kanji_04e8c">
<g id="kvg:04e8c" kvg:element="二">
  <path id="kvg:04e8c-s1" d="M 20,30 L 80,30"/>
  <path id="kvg:04e8c-s2" d="M 10,70 L 90,70"/>
</g>
</kanji>
</kanjivg>"""

    def test_converts_xml_to_samples(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_xml_to_samples

        xml_file = tmp_path / "kanjivg.xml"
        xml_file.write_text(self.MINIMAL_XML, encoding="utf-8")
        output_dir = tmp_path / "output"

        count = convert_xml_to_samples(xml_file, output_dir, target_size=10.0, num_points=8)

        assert count == 2

    def test_xml_creates_correct_dirs(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_xml_to_samples

        xml_file = tmp_path / "kanjivg.xml"
        xml_file.write_text(self.MINIMAL_XML, encoding="utf-8")
        output_dir = tmp_path / "output"

        convert_xml_to_samples(xml_file, output_dir, target_size=10.0, num_points=8)

        assert (output_dir / "一").is_dir()
        assert (output_dir / "二").is_dir()
        assert len(list((output_dir / "一").glob("*.json"))) == 1
        assert len(list((output_dir / "二").glob("*.json"))) == 1

    def test_xml_stroke_count(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_xml_to_samples

        xml_file = tmp_path / "kanjivg.xml"
        xml_file.write_text(self.MINIMAL_XML, encoding="utf-8")
        output_dir = tmp_path / "output"

        convert_xml_to_samples(xml_file, output_dir, target_size=10.0, num_points=8)

        sample = StrokeSample.load(list((output_dir / "二").glob("*.json"))[0])
        assert sample.character == "二"
        assert len(sample.strokes) == 2

    def test_xml_dataset_compatible(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_xml_to_samples

        xml_file = tmp_path / "kanjivg.xml"
        xml_file.write_text(self.MINIMAL_XML, encoding="utf-8")
        output_dir = tmp_path / "output"

        convert_xml_to_samples(xml_file, output_dir, target_size=10.0, num_points=8)

        json_file = list((output_dir / "一").glob("*.json"))[0]
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert "character" in data
        assert "strokes" in data
        assert isinstance(data["strokes"][0], list)
        assert "x" in data["strokes"][0][0]


class TestMetadata:
    def test_metadata_contains_source(self, tmp_path: Path):
        from scripts.prepare_kanjivg import convert_single_svg

        svg_file = tmp_path / "0904e.svg"
        svg_file.write_text(MINIMAL_SVG, encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample = convert_single_svg(svg_file, output_dir, target_size=10.0, num_points=8)

        assert sample.metadata["source"] == "kanjivg"
