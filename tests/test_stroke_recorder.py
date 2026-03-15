import json
from pathlib import Path
import numpy as np
import pytest
from src.collector.stroke_recorder import StrokeRecorder
from src.collector.data_format import StrokePoint, StrokeSample


class TestNormalize:
    def test_size_normalization(self):
        recorder = StrokeRecorder(target_size=10.0)
        points = [
            StrokePoint(x=0, y=0),
            StrokePoint(x=100, y=0),
            StrokePoint(x=100, y=100),
            StrokePoint(x=0, y=100),
        ]
        normalized = recorder.normalize_points(points)
        xs = [p.x for p in normalized]
        ys = [p.y for p in normalized]
        assert max(xs) - min(xs) <= 10.0
        assert max(ys) - min(ys) <= 10.0

    def test_centering(self):
        recorder = StrokeRecorder(target_size=10.0)
        points = [
            StrokePoint(x=50, y=50),
            StrokePoint(x=150, y=150),
        ]
        normalized = recorder.normalize_points(points)
        xs = [p.x for p in normalized]
        ys = [p.y for p in normalized]
        center_x = (max(xs) + min(xs)) / 2
        center_y = (max(ys) + min(ys)) / 2
        assert center_x == pytest.approx(5.0, abs=0.1)
        assert center_y == pytest.approx(5.0, abs=0.1)

    def test_preserves_pressure(self):
        recorder = StrokeRecorder(target_size=10.0)
        points = [
            StrokePoint(x=0, y=0, pressure=0.3),
            StrokePoint(x=100, y=100, pressure=0.9),
        ]
        normalized = recorder.normalize_points(points)
        assert normalized[0].pressure == 0.3
        assert normalized[1].pressure == 0.9

    def test_single_point(self):
        recorder = StrokeRecorder(target_size=10.0)
        points = [StrokePoint(x=50, y=50)]
        normalized = recorder.normalize_points(points)
        assert len(normalized) == 1


class TestResample:
    def test_resample_increases_points(self):
        recorder = StrokeRecorder()
        points = [
            StrokePoint(x=0, y=0),
            StrokePoint(x=100, y=100),
        ]
        resampled = recorder.resample_points(points, num_points=10)
        assert len(resampled) == 10

    def test_resample_preserves_endpoints(self):
        recorder = StrokeRecorder()
        points = [
            StrokePoint(x=0, y=0),
            StrokePoint(x=10, y=10),
        ]
        resampled = recorder.resample_points(points, num_points=5)
        assert resampled[0].x == pytest.approx(0.0)
        assert resampled[0].y == pytest.approx(0.0)
        assert resampled[-1].x == pytest.approx(10.0)
        assert resampled[-1].y == pytest.approx(10.0)


class TestSaveLoad:
    def test_save_sample(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = StrokeSample(
            character="あ",
            strokes=[[StrokePoint(x=0, y=0), StrokePoint(x=5, y=5)]],
        )
        filepath = recorder.save_sample(sample)
        assert filepath.exists()
        assert filepath.suffix == ".json"

    def test_load_samples_for_char(self, tmp_path: Path):
        out_dir = tmp_path / "test_strokes"
        recorder = StrokeRecorder(output_dir=out_dir)
        for i in range(3):
            sample = StrokeSample(
                character="い",
                strokes=[[StrokePoint(x=i, y=i), StrokePoint(x=i+5, y=i+5)]],
                metadata={"variant": i},
            )
            recorder.save_sample(sample)
        loaded = recorder.load_samples("い")
        assert len(loaded) == 3

    def test_list_characters(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)
        for ch in ["あ", "い", "う"]:
            sample = StrokeSample(
                character=ch,
                strokes=[[StrokePoint(x=0, y=0), StrokePoint(x=1, y=1)]],
            )
            recorder.save_sample(sample)
        chars = recorder.list_characters()
        assert set(chars) == {"あ", "い", "う"}
