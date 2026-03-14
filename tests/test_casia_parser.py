from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.collector.casia_parser import CASIAParser, CASIASample


def _make_pot_sample(char: str, strokes: list[list[tuple[int, int]]]) -> bytes:
    """Create a binary .pot sample for testing."""
    tag = char.encode("gbk")
    stroke_data = b""
    for stroke in strokes:
        for x, y in stroke:
            stroke_data += struct.pack("<hh", x, y)
        stroke_data += struct.pack("<hh", -1, 0)
    stroke_data += struct.pack("<hh", -1, -1)
    total = 4 + len(tag) + 2 + len(stroke_data)
    return struct.pack("<I", total) + tag + struct.pack("<H", len(strokes)) + stroke_data


def _make_pot_file(samples: list[bytes]) -> bytes:
    return b"".join(samples)


class TestParsePotFile:
    def test_parse_single_sample(self, tmp_path: Path) -> None:
        strokes = [[(100, 200), (150, 250)], [(300, 400), (350, 450)]]
        pot_data = _make_pot_file([_make_pot_sample("\u4e00", strokes)])
        pot_file = tmp_path / "test.pot"
        pot_file.write_bytes(pot_data)

        parser = CASIAParser()
        samples = parser.parse_pot_file(pot_file)

        assert len(samples) == 1
        assert samples[0].character == "\u4e00"
        assert len(samples[0].strokes) == 2

    def test_parse_multiple_samples(self, tmp_path: Path) -> None:
        chars = ["\u4e00", "\u4e8c", "\u4e09"]
        pot_samples = []
        for ch in chars:
            pot_samples.append(_make_pot_sample(ch, [[(10, 20), (30, 40)]]))
        pot_data = _make_pot_file(pot_samples)
        pot_file = tmp_path / "test.pot"
        pot_file.write_bytes(pot_data)

        parser = CASIAParser()
        samples = parser.parse_pot_file(pot_file)

        assert len(samples) == 3
        assert [s.character for s in samples] == chars

    def test_stroke_coordinates(self, tmp_path: Path) -> None:
        strokes = [[(100, 200), (150, 250), (200, 300)]]
        pot_data = _make_pot_file([_make_pot_sample("\u4e00", strokes)])
        pot_file = tmp_path / "test.pot"
        pot_file.write_bytes(pot_data)

        parser = CASIAParser()
        samples = parser.parse_pot_file(pot_file)

        stroke = samples[0].strokes[0]
        assert stroke.shape == (3, 2)
        np.testing.assert_array_equal(stroke[0], [100, 200])
        np.testing.assert_array_equal(stroke[1], [150, 250])
        np.testing.assert_array_equal(stroke[2], [200, 300])


class TestNormalize:
    def test_normalize_flips_y(self) -> None:
        parser = CASIAParser()
        strokes = [np.array([[0, 0], [100, 100]], dtype=np.float64)]
        normalized = parser.normalize(strokes, target_size=1.0)

        assert normalized[0][0, 1] == pytest.approx(1.0)
        assert normalized[0][1, 1] == pytest.approx(0.0)

    def test_normalize_preserves_aspect(self) -> None:
        parser = CASIAParser()
        strokes = [np.array([[0, 0], [200, 100]], dtype=np.float64)]
        normalized = parser.normalize(strokes, target_size=1.0)

        assert normalized[0][:, 0].max() == pytest.approx(1.0)
        y_range = normalized[0][:, 1].max() - normalized[0][:, 1].min()
        assert y_range == pytest.approx(0.5)


class TestConvertToStrokeSamples:
    def test_convert_to_stroke_samples(self, tmp_path: Path) -> None:
        samples = [
            CASIASample(
                character="\u4e00",
                strokes=[
                    np.array([[0, 0], [50, 50], [100, 100]], dtype=np.float64),
                ],
            ),
            CASIASample(
                character="\u4e00",
                strokes=[
                    np.array([[10, 10], [60, 60], [110, 110]], dtype=np.float64),
                ],
            ),
        ]

        count = CASIAParser.convert_to_stroke_samples(
            samples, output_dir=tmp_path, target_size=10.0, num_points=8
        )

        assert count == 2
        char_dir = tmp_path / "\u4e00"
        assert char_dir.is_dir()
        json_files = sorted(char_dir.glob("*.json"))
        assert len(json_files) == 2

        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert data["character"] == "\u4e00"
        assert len(data["strokes"]) == 1
        assert len(data["strokes"][0]) == 8


class TestEdgeCases:
    def test_gbk_decode_error_skipped(self, tmp_path: Path) -> None:
        good_sample = _make_pot_sample("\u4e00", [[(10, 20), (30, 40)]])
        bad_tag = b"\xff\xfe"
        bad_stroke_data = struct.pack("<hh", 10, 20) + struct.pack("<hh", -1, 0)
        bad_stroke_data += struct.pack("<hh", -1, -1)
        bad_total = 4 + 2 + 2 + len(bad_stroke_data)
        bad_sample = struct.pack("<I", bad_total) + bad_tag + struct.pack("<H", 1) + bad_stroke_data

        pot_data = bad_sample + good_sample
        pot_file = tmp_path / "test.pot"
        pot_file.write_bytes(pot_data)

        parser = CASIAParser()
        samples = parser.parse_pot_file(pot_file)

        assert len(samples) == 1
        assert samples[0].character == "\u4e00"

    def test_empty_pot_file(self, tmp_path: Path) -> None:
        pot_file = tmp_path / "empty.pot"
        pot_file.write_bytes(b"")

        parser = CASIAParser()
        samples = parser.parse_pot_file(pot_file)

        assert samples == []
