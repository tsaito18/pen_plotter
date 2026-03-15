"""CASIA変換スクリプトのテスト。"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from scripts.prepare_casia import convert_pot_directory


def _make_pot_sample(char: str, strokes: list[list[tuple[int, int]]]) -> bytes:
    """テスト用 .pot バイナリサンプルを作成。"""
    tag = char.encode("gbk")
    stroke_data = b""
    for stroke in strokes:
        for x, y in stroke:
            stroke_data += struct.pack("<hh", x, y)
        stroke_data += struct.pack("<hh", -1, 0)
    stroke_data += struct.pack("<hh", -1, -1)
    total = 4 + len(tag) + 2 + len(stroke_data)
    return struct.pack("<I", total) + tag + struct.pack("<H", len(strokes)) + stroke_data


class TestConvertPotDirectory:
    def test_converts_pot_files(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        pot_data = _make_pot_sample("\u4e00", [[(100, 200), (300, 400)]])
        pot_data += _make_pot_sample("\u4e8c", [[(10, 20), (30, 40)], [(50, 60), (70, 80)]])
        (input_dir / "test.pot").write_bytes(pot_data)

        count = convert_pot_directory(input_dir, output_dir, target_size=10.0, num_points=8)

        assert count == 2
        assert (output_dir / "\u4e00").is_dir()
        assert (output_dir / "\u4e8c").is_dir()

    def test_output_json_format(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        pot_data = _make_pot_sample("\u4e00", [[(100, 200), (300, 400)]])
        (input_dir / "test.pot").write_bytes(pot_data)

        convert_pot_directory(input_dir, output_dir, target_size=10.0, num_points=8)

        json_files = list((output_dir / "\u4e00").glob("*.json"))
        assert len(json_files) == 1

        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert data["character"] == "\u4e00"
        assert "strokes" in data
        assert len(data["strokes"][0]) == 8
        assert data["metadata"]["source"] == "casia"

    def test_multiple_pot_files(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        pot1 = _make_pot_sample("\u4e00", [[(100, 200), (300, 400)]])
        pot2 = _make_pot_sample("\u4e09", [[(10, 20), (30, 40)]])
        (input_dir / "a.pot").write_bytes(pot1)
        (input_dir / "b.pot").write_bytes(pot2)

        count = convert_pot_directory(input_dir, output_dir, target_size=10.0, num_points=8)

        assert count == 2

    def test_multiple_samples_same_char(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        pot_data = _make_pot_sample("\u4e00", [[(100, 200), (300, 400)]])
        pot_data += _make_pot_sample("\u4e00", [[(150, 250), (350, 450)]])
        (input_dir / "test.pot").write_bytes(pot_data)

        count = convert_pot_directory(input_dir, output_dir, target_size=10.0, num_points=8)

        assert count == 2
        json_files = list((output_dir / "\u4e00").glob("*.json"))
        assert len(json_files) == 2

    def test_empty_directory(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        count = convert_pot_directory(input_dir, output_dir)

        assert count == 0

    def test_resampling_applied(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        pot_data = _make_pot_sample(
            "\u4e00", [[(0, 0), (25, 25), (50, 50), (75, 75), (100, 100)]]
        )
        (input_dir / "test.pot").write_bytes(pot_data)

        convert_pot_directory(input_dir, output_dir, target_size=10.0, num_points=16)

        json_files = list((output_dir / "\u4e00").glob("*.json"))
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert len(data["strokes"][0]) == 16
