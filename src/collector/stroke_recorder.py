from __future__ import annotations

import time
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
            return [
                StrokePoint(
                    x=self.target_size / 2,
                    y=self.target_size / 2,
                    pressure=points[0].pressure,
                    timestamp=points[0].timestamp,
                )
            ] if points else []

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

    def resample_points(
        self, points: list[StrokePoint], num_points: int = 32
    ) -> list[StrokePoint]:
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
        return [
            StrokeSample.load(p)
            for p in sorted(char_dir.glob("*.json"))
        ]

    def list_characters(self) -> list[str]:
        if not self.output_dir.exists():
            return []
        return [
            d.name
            for d in sorted(self.output_dir.iterdir())
            if d.is_dir() and any(d.glob("*.json"))
        ]
