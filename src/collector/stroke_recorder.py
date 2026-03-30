from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.collector.data_format import StrokePoint, StrokeSample


class StrokeRecorder:
    def __init__(
        self,
        target_size: float = 10.0,
        output_dir: Path | None = None,
    ) -> None:
        self.target_size = target_size
        self.output_dir = output_dir or Path("data/strokes")

    def normalize_points(self, points: list[StrokePoint]) -> list[StrokePoint]:
        if len(points) <= 1:
            return (
                [
                    StrokePoint(
                        x=self.target_size / 2,
                        y=self.target_size / 2,
                        pressure=points[0].pressure,
                        timestamp=points[0].timestamp,
                    )
                ]
                if points
                else []
            )

        xs = np.array([p.x for p in points])
        ys = np.array([p.y for p in points])

        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        scale_denom = max(x_range, y_range)

        if scale_denom == 0:
            return [
                StrokePoint(
                    x=self.target_size / 2,
                    y=self.target_size / 2,
                    pressure=p.pressure,
                    timestamp=p.timestamp,
                )
                for p in points
            ]

        scale = self.target_size / scale_denom
        xs_scaled = (xs - xs.min()) * scale
        ys_scaled = (ys - ys.min()) * scale

        # target_size x target_size 領域の中心に配置
        offset_x = (self.target_size - (xs_scaled.max() - xs_scaled.min())) / 2
        offset_y = (self.target_size - (ys_scaled.max() - ys_scaled.min())) / 2

        return [
            StrokePoint(
                x=float(xs_scaled[i] + offset_x),
                y=float(ys_scaled[i] + offset_y),
                pressure=points[i].pressure,
                timestamp=points[i].timestamp,
            )
            for i in range(len(points))
        ]

    def resample_points(self, points: list[StrokePoint], num_points: int = 32) -> list[StrokePoint]:
        if len(points) < 2:
            return points

        xs = np.array([p.x for p in points])
        ys = np.array([p.y for p in points])
        pressures = np.array([p.pressure for p in points])
        timestamps = np.array([p.timestamp for p in points])

        diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        cumulative = np.concatenate(([0.0], np.cumsum(diffs)))
        total_length = cumulative[-1]

        if total_length == 0:
            return points

        target_dists = np.linspace(0, total_length, num_points)

        new_xs = np.interp(target_dists, cumulative, xs)
        new_ys = np.interp(target_dists, cumulative, ys)
        new_pressures = np.interp(target_dists, cumulative, pressures)
        new_timestamps = np.interp(target_dists, cumulative, timestamps)

        return [
            StrokePoint(
                x=float(new_xs[i]),
                y=float(new_ys[i]),
                pressure=float(new_pressures[i]),
                timestamp=float(new_timestamps[i]),
            )
            for i in range(num_points)
        ]

    def save_sample(self, sample: StrokeSample) -> Path:
        char_dir = self.output_dir / sample.character
        char_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1_000_000)
        filename = f"{sample.character}_{timestamp}.json"
        filepath = char_dir / filename

        sample.save(filepath)
        return filepath

    def load_samples(self, character: str) -> list[StrokeSample]:
        char_dir = self.output_dir / character
        if not char_dir.exists():
            return []
        return [StrokeSample.load(p) for p in sorted(char_dir.glob("*.json"))]

    def delete_sample(self, character: str, filename: str) -> bool:
        if "/" in filename or ".." in filename or not re.match(r"^.+_\d+\.json$", filename):
            raise ValueError(f"Invalid filename: {filename}")
        filepath = self.output_dir / character / filename
        if not filepath.exists():
            return False
        filepath.unlink()
        return True

    def set_metadata(self, character: str, filename: str, key: str, value: object) -> bool:
        if "/" in filename or ".." in filename or not re.match(r"^.+_\d+\.json$", filename):
            raise ValueError(f"Invalid filename: {filename}")
        filepath = self.output_dir / character / filename
        if not filepath.exists():
            return False
        sample = StrokeSample.load(filepath)
        sample.metadata[key] = value
        sample.save(filepath)
        return True

    def delete_all_samples(self, character: str) -> int:
        char_dir = self.output_dir / character
        if not char_dir.exists():
            return 0
        files = list(char_dir.glob("*.json"))
        for f in files:
            f.unlink()
        if not any(char_dir.iterdir()):
            char_dir.rmdir()
        return len(files)

    def get_sample_info(self, character: str) -> list[dict]:
        char_dir = self.output_dir / character
        if not char_dir.exists():
            return []
        infos = []
        for path in char_dir.glob("*.json"):
            sample = StrokeSample.load(path)
            match = re.search(r"_(\d+)\.json$", path.name)
            timestamp = int(match.group(1)) if match else 0
            point_count = sum(len(stroke) for stroke in sample.strokes)
            infos.append(
                {
                    "filename": path.name,
                    "stroke_count": len(sample.strokes),
                    "point_count": point_count,
                    "timestamp": timestamp,
                    "strokes": [[p.to_dict() for p in stroke] for stroke in sample.strokes],
                }
            )
        infos.sort(key=lambda x: x["timestamp"])
        return infos

    def find_stroke_mismatches(self) -> list[dict]:
        """同一文字内で画数が不一致のサンプルを検出する。"""
        results: list[dict] = []
        for character in self.list_characters():
            char_dir = self.output_dir / character
            if not char_dir.exists():
                continue

            sample_infos: list[dict] = []
            for path in sorted(char_dir.glob("*.json")):
                sample = StrokeSample.load(path)
                if sample.metadata.get("ignore_stroke_mismatch"):
                    continue
                stroke_count = len(sample.strokes)
                point_count = sum(len(stroke) for stroke in sample.strokes)
                sample_infos.append(
                    {
                        "filename": path.name,
                        "stroke_count": stroke_count,
                        "point_count": point_count,
                        "strokes": [
                            [p.to_dict() for p in stroke] for stroke in sample.strokes
                        ],
                    }
                )

            if len(sample_infos) < 2:
                continue

            counts = Counter(s["stroke_count"] for s in sample_infos)
            mode_count = counts.most_common(1)[0][0]

            has_outlier = any(s["stroke_count"] != mode_count for s in sample_infos)
            if not has_outlier:
                continue

            for s in sample_infos:
                s["is_outlier"] = s["stroke_count"] != mode_count

            results.append(
                {
                    "character": character,
                    "mode_count": mode_count,
                    "samples": sample_infos,
                }
            )
        return results

    def find_anomalies(self, canvas_size: float = 512.0) -> list[dict]:
        """全文字をスキャンし異常サンプルを検出する。"""
        min_bbox = canvas_size * 0.1
        anomalies: list[dict] = []
        for character in self.list_characters():
            char_dir = self.output_dir / character
            if not char_dir.exists():
                continue
            for path in sorted(char_dir.glob("*.json")):
                sample = StrokeSample.load(path)
                if sample.metadata.get("ignore_anomaly"):
                    continue

                stroke_count = len(sample.strokes)
                point_count = sum(len(stroke) for stroke in sample.strokes)
                strokes_dicts = [
                    [p.to_dict() for p in stroke] for stroke in sample.strokes
                ]

                reasons: list[str] = []
                if point_count < 5:
                    reasons.append("点数少")
                if stroke_count > 10:
                    reasons.append("画数多")

                all_x: list[float] = []
                all_y: list[float] = []
                for stroke in strokes_dicts:
                    for pt in stroke:
                        all_x.append(pt["x"])
                        all_y.append(pt["y"])
                if all_x:
                    x_range = max(all_x) - min(all_x)
                    y_range = max(all_y) - min(all_y)
                    if max(x_range, y_range) < min_bbox:
                        reasons.append("描画領域小")

                if strokes_dicts:
                    first_ts = strokes_dicts[0][0]["timestamp"]
                    last_ts = strokes_dicts[-1][-1]["timestamp"]
                    if last_ts - first_ts < 500:
                        reasons.append("描画時間短")

                for stroke in strokes_dicts:
                    if len(stroke) == 1:
                        reasons.append("単点ストローク")
                        break

                if reasons:
                    anomalies.append(
                        {
                            "character": character,
                            "filename": path.name,
                            "stroke_count": stroke_count,
                            "point_count": point_count,
                            "reasons": reasons,
                            "strokes": strokes_dicts,
                        }
                    )
        return anomalies

    def list_characters(self) -> list[str]:
        if not self.output_dir.exists():
            return []
        return [
            d.name
            for d in sorted(self.output_dir.iterdir())
            if d.is_dir() and any(d.glob("*.json"))
        ]
