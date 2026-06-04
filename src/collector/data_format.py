from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StrokePoint:
    x: float
    y: float
    pressure: float = 1.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "pressure": self.pressure,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StrokePoint:
        return cls(
            x=d["x"],
            y=d["y"],
            pressure=d["pressure"],
            timestamp=d["timestamp"],
        )


@dataclass
class StrokeSample:
    character: str
    strokes: list[list[StrokePoint]]
    metadata: dict = field(default_factory=dict)
    stroke_types: list[str] = field(default_factory=list)
    """ストローク単位の筆画タイプ（KanjiVG の raw kvg:type 文字列）。

    ``strokes`` と並走し、index が対応する。空リストの場合は筆画タイプ不明
    （旧フォーマット・ユーザー収集データ等）として扱う。
    """

    def to_json(self) -> str:
        return json.dumps(
            {
                "character": self.character,
                "strokes": [[pt.to_dict() for pt in stroke] for stroke in self.strokes],
                "metadata": self.metadata,
                "stroke_types": self.stroke_types,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, json_str: str) -> StrokeSample:
        data = json.loads(json_str)
        return cls(
            character=data["character"],
            strokes=[[StrokePoint.from_dict(pt) for pt in stroke] for stroke in data["strokes"]],
            metadata=data.get("metadata", {}),
            stroke_types=data.get("stroke_types", []),
        )

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> StrokeSample:
        return cls.from_json(path.read_text(encoding="utf-8"))
