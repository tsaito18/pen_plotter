from pathlib import Path

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
                strokes=[[StrokePoint(x=i, y=i), StrokePoint(x=i + 5, y=i + 5)]],
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


def _make_sample(
    character: str = "あ", num_strokes: int = 2, points_per_stroke: int = 3
) -> StrokeSample:
    """テスト用のサンプルデータを生成する。"""
    strokes = [
        [StrokePoint(x=float(s + i), y=float(s + i)) for i in range(points_per_stroke)]
        for s in range(num_strokes)
    ]
    return StrokeSample(character=character, strokes=strokes)


def _make_sample_with_ts(
    character: str = "あ",
    num_strokes: int = 2,
    points_per_stroke: int = 5,
    bbox_size: float = 200.0,
    total_time_ms: float = 2000.0,
) -> StrokeSample:
    """タイムスタンプ・座標を制御可能なテスト用サンプル。"""
    strokes = []
    time_per_point = total_time_ms / max(num_strokes * points_per_stroke - 1, 1)
    t = 0.0
    for s in range(num_strokes):
        stroke = []
        for i in range(points_per_stroke):
            frac = (s * points_per_stroke + i) / max(num_strokes * points_per_stroke - 1, 1)
            stroke.append(
                StrokePoint(
                    x=100.0 + frac * bbox_size,
                    y=100.0 + frac * bbox_size,
                    pressure=1.0,
                    timestamp=t,
                )
            )
            t += time_per_point
        strokes.append(stroke)
    return StrokeSample(character=character, strokes=strokes)


class TestDeleteSample:
    def test_delete_existing_sample(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = _make_sample("あ")
        filepath = recorder.save_sample(sample)
        filename = filepath.name

        result = recorder.delete_sample("あ", filename)

        assert result is True
        assert not filepath.exists()

    def test_delete_nonexistent_sample(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)

        result = recorder.delete_sample("あ", "nonexistent_12345.json")

        assert result is False

    def test_delete_validates_filename(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)

        with pytest.raises(ValueError):
            recorder.delete_sample("あ", "../evil.json")

        with pytest.raises(ValueError):
            recorder.delete_sample("あ", "../../etc/passwd")

    def test_delete_all_samples(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)
        for _ in range(3):
            recorder.save_sample(_make_sample("い"))

        deleted_count = recorder.delete_all_samples("い")

        assert deleted_count == 3
        char_dir = tmp_path / "い"
        assert not char_dir.exists()

    def test_delete_all_empty_character(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)

        deleted_count = recorder.delete_all_samples("ん")

        assert deleted_count == 0


class TestGetSampleInfo:
    def test_get_sample_info_returns_metadata(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = _make_sample("あ", num_strokes=2, points_per_stroke=4)
        recorder.save_sample(sample)

        infos = recorder.get_sample_info("あ")

        assert len(infos) == 1
        info = infos[0]
        assert "filename" in info
        assert info["stroke_count"] == 2
        assert info["point_count"] == 8
        assert "timestamp" in info

    def test_get_sample_info_includes_strokes(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = _make_sample("あ", num_strokes=3, points_per_stroke=5)
        recorder.save_sample(sample)

        infos = recorder.get_sample_info("あ")

        assert "strokes" in infos[0]
        assert len(infos[0]["strokes"]) == 3
        assert len(infos[0]["strokes"][0]) == 5

    def test_get_sample_info_empty_character(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)

        infos = recorder.get_sample_info("ん")

        assert infos == []

    def test_get_sample_info_sorted_by_time(self, tmp_path: Path):
        recorder = StrokeRecorder(output_dir=tmp_path)
        for _ in range(3):
            recorder.save_sample(_make_sample("あ"))

        infos = recorder.get_sample_info("あ")

        assert len(infos) == 3
        timestamps = [info["timestamp"] for info in infos]
        assert timestamps == sorted(timestamps)


class TestFindAnomalies:
    def test_normal_sample_no_anomalies(self, tmp_path: Path):
        """正常なサンプル → 空リスト"""
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = _make_sample_with_ts(
            "あ", num_strokes=2, points_per_stroke=5, bbox_size=200.0, total_time_ms=2000.0
        )
        recorder.save_sample(sample)

        anomalies = recorder.find_anomalies()

        assert anomalies == []

    def test_too_few_points(self, tmp_path: Path):
        """合計ポイント数 < 5 → "点数少" を含む"""
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = _make_sample_with_ts(
            "あ", num_strokes=1, points_per_stroke=3, bbox_size=200.0, total_time_ms=2000.0
        )
        recorder.save_sample(sample)

        anomalies = recorder.find_anomalies()

        assert len(anomalies) == 1
        assert "点数少" in anomalies[0]["reasons"]

    def test_too_many_strokes(self, tmp_path: Path):
        """ストローク数 > 10 → "画数多" を含む"""
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = _make_sample_with_ts(
            "あ", num_strokes=11, points_per_stroke=5, bbox_size=200.0, total_time_ms=5000.0
        )
        recorder.save_sample(sample)

        anomalies = recorder.find_anomalies()

        assert len(anomalies) == 1
        assert "画数多" in anomalies[0]["reasons"]

    def test_too_small_bbox(self, tmp_path: Path):
        """バウンディングボックス < 51.2px → "描画領域小" を含む"""
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = _make_sample_with_ts(
            "あ", num_strokes=2, points_per_stroke=5, bbox_size=30.0, total_time_ms=2000.0
        )
        recorder.save_sample(sample)

        anomalies = recorder.find_anomalies()

        assert len(anomalies) == 1
        assert "描画領域小" in anomalies[0]["reasons"]

    def test_too_fast(self, tmp_path: Path):
        """描画時間 < 500ms → "描画時間短" を含む"""
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = _make_sample_with_ts(
            "あ", num_strokes=2, points_per_stroke=5, bbox_size=200.0, total_time_ms=300.0
        )
        recorder.save_sample(sample)

        anomalies = recorder.find_anomalies()

        assert len(anomalies) == 1
        assert "描画時間短" in anomalies[0]["reasons"]

    def test_single_point_stroke(self, tmp_path: Path):
        """1点のみのストロークがある → "単点ストローク" を含む"""
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = StrokeSample(
            character="あ",
            strokes=[
                [
                    StrokePoint(x=100.0, y=100.0, pressure=1.0, timestamp=0.0),
                    StrokePoint(x=200.0, y=200.0, pressure=1.0, timestamp=500.0),
                    StrokePoint(x=300.0, y=300.0, pressure=1.0, timestamp=1000.0),
                    StrokePoint(x=350.0, y=350.0, pressure=1.0, timestamp=1500.0),
                ],
                [StrokePoint(x=150.0, y=150.0, pressure=1.0, timestamp=2000.0)],
            ],
        )
        recorder.save_sample(sample)

        anomalies = recorder.find_anomalies()

        assert len(anomalies) == 1
        assert "単点ストローク" in anomalies[0]["reasons"]

    def test_multiple_anomalies(self, tmp_path: Path):
        """複数の異常条件に該当 → 全ての理由が含まれる"""
        recorder = StrokeRecorder(output_dir=tmp_path)
        sample = StrokeSample(
            character="あ",
            strokes=[
                [StrokePoint(x=100.0, y=100.0, pressure=1.0, timestamp=0.0)],
                [StrokePoint(x=105.0, y=105.0, pressure=1.0, timestamp=100.0)],
            ],
        )
        recorder.save_sample(sample)

        anomalies = recorder.find_anomalies()

        assert len(anomalies) == 1
        reasons = anomalies[0]["reasons"]
        assert "点数少" in reasons
        assert "描画領域小" in reasons
        assert "描画時間短" in reasons
        assert "単点ストローク" in reasons
