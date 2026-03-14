"""CASIA-OLHWDB .pot (isolated character) ファイルのパーサー。"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.collector.data_format import StrokePoint, StrokeSample
from src.collector.stroke_recorder import StrokeRecorder


@dataclass
class CASIASample:
    character: str
    strokes: list[NDArray[np.float64]]


class CASIAParser:
    """CASIA-OLHWDB .pot 形式のバイナリファイルを解析する。"""

    def parse_pot_file(self, path: Path) -> list[CASIASample]:
        """Parse a single .pot file, returning all character samples."""
        data = path.read_bytes()
        samples: list[CASIASample] = []
        offset = 0

        while offset < len(data):
            if offset + 4 > len(data):
                break

            sample_size = struct.unpack_from("<I", data, offset)[0]
            if sample_size < 8 or offset + sample_size > len(data):
                break

            sample_data = data[offset : offset + sample_size]
            offset += sample_size

            sample = self._parse_sample(sample_data)
            if sample is not None:
                samples.append(sample)

        return samples

    def _parse_sample(self, data: bytes) -> CASIASample | None:
        pos = 4
        tag_bytes = data[pos : pos + 2]
        pos += 2

        try:
            character = tag_bytes.decode("gbk")
        except (UnicodeDecodeError, ValueError):
            return None

        stroke_number = struct.unpack_from("<H", data, pos)[0]
        pos += 2

        strokes: list[NDArray[np.float64]] = []
        for _ in range(stroke_number):
            points: list[tuple[int, int]] = []
            while pos + 4 <= len(data):
                x, y = struct.unpack_from("<hh", data, pos)
                pos += 4
                if x == -1 and y == 0:
                    break
                if x == -1 and y == -1:
                    break
                points.append((x, y))
            if points:
                strokes.append(np.array(points, dtype=np.float64))

        return CASIASample(character=character, strokes=strokes)

    def normalize(self, strokes: list[NDArray], target_size: float = 1.0) -> list[NDArray]:
        """Normalize all strokes to [0, target_size] range, preserving aspect ratio.

        Y axis is flipped (CASIA uses top-left origin, plotter uses bottom-left).
        """
        if not strokes:
            return []

        all_points = np.concatenate(strokes, axis=0)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        ranges = maxs - mins
        scale = target_size / ranges.max() if ranges.max() > 0 else 1.0

        result = []
        for stroke in strokes:
            normalized = (stroke - mins) * scale
            normalized[:, 1] = target_size - normalized[:, 1]
            result.append(normalized)
        return result

    @staticmethod
    def convert_to_stroke_samples(
        samples: list[CASIASample],
        output_dir: Path,
        target_size: float = 10.0,
        num_points: int = 32,
    ) -> int:
        """Convert CASIA samples to StrokeSample JSON files compatible with StrokeDataset.

        Returns number of samples converted.
        """
        parser = CASIAParser()
        recorder = StrokeRecorder(target_size=target_size, output_dir=output_dir)

        char_counters: dict[str, int] = {}
        count = 0

        for sample in samples:
            char = sample.character
            normalized = parser.normalize(sample.strokes, target_size=target_size)

            stroke_point_lists: list[list[StrokePoint]] = []
            for stroke_arr in normalized:
                points = [StrokePoint(x=float(pt[0]), y=float(pt[1])) for pt in stroke_arr]
                resampled = recorder.resample_points(points, num_points=num_points)
                stroke_point_lists.append(resampled)

            stroke_sample = StrokeSample(
                character=char,
                strokes=stroke_point_lists,
                metadata={"source": "casia"},
            )

            idx = char_counters.get(char, 0)
            char_counters[char] = idx + 1

            char_dir = output_dir / char
            char_dir.mkdir(parents=True, exist_ok=True)
            filepath = char_dir / f"{char}_{idx}.json"
            stroke_sample.save(filepath)
            count += 1

        return count
