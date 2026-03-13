import json
from pathlib import Path
import pytest
from src.collector.data_format import StrokePoint, StrokeSample


class TestStrokePoint:
    def test_creation(self):
        pt = StrokePoint(x=10.0, y=20.0, pressure=0.5, timestamp=100.0)
        assert pt.x == 10.0
        assert pt.y == 20.0
        assert pt.pressure == 0.5
        assert pt.timestamp == 100.0

    def test_to_dict(self):
        pt = StrokePoint(x=1.0, y=2.0, pressure=0.8, timestamp=0.0)
        d = pt.to_dict()
        assert d == {"x": 1.0, "y": 2.0, "pressure": 0.8, "timestamp": 0.0}

    def test_from_dict(self):
        d = {"x": 5.0, "y": 6.0, "pressure": 1.0, "timestamp": 50.0}
        pt = StrokePoint.from_dict(d)
        assert pt.x == 5.0
        assert pt.pressure == 1.0

    def test_default_pressure_and_timestamp(self):
        pt = StrokePoint(x=0.0, y=0.0)
        assert pt.pressure == 1.0
        assert pt.timestamp == 0.0


class TestStrokeSample:
    def test_creation(self):
        points = [StrokePoint(x=0, y=0), StrokePoint(x=1, y=1)]
        sample = StrokeSample(
            character="あ",
            strokes=[points],
        )
        assert sample.character == "あ"
        assert len(sample.strokes) == 1
        assert len(sample.strokes[0]) == 2

    def test_to_json_and_back(self):
        points1 = [StrokePoint(x=0, y=0, pressure=0.5, timestamp=0.0),
                    StrokePoint(x=10, y=10, pressure=1.0, timestamp=100.0)]
        points2 = [StrokePoint(x=5, y=5), StrokePoint(x=15, y=15)]
        sample = StrokeSample(
            character="漢",
            strokes=[points1, points2],
            metadata={"source": "ipad", "variant": 1},
        )
        json_str = sample.to_json()
        restored = StrokeSample.from_json(json_str)
        assert restored.character == "漢"
        assert len(restored.strokes) == 2
        assert restored.strokes[0][0].pressure == 0.5
        assert restored.metadata["source"] == "ipad"

    def test_save_and_load(self, tmp_path: Path):
        sample = StrokeSample(
            character="テ",
            strokes=[[StrokePoint(x=0, y=0), StrokePoint(x=5, y=5)]],
        )
        filepath = tmp_path / "test_sample.json"
        sample.save(filepath)
        assert filepath.exists()
        loaded = StrokeSample.load(filepath)
        assert loaded.character == "テ"
        assert len(loaded.strokes) == 1

    def test_multiple_strokes(self):
        strokes = [
            [StrokePoint(x=i, y=i) for i in range(5)],
            [StrokePoint(x=i*2, y=i*2) for i in range(3)],
            [StrokePoint(x=i*3, y=i*3) for i in range(4)],
        ]
        sample = StrokeSample(character="書", strokes=strokes)
        json_str = sample.to_json()
        restored = StrokeSample.from_json(json_str)
        assert len(restored.strokes) == 3
        assert len(restored.strokes[0]) == 5
        assert len(restored.strokes[1]) == 3
