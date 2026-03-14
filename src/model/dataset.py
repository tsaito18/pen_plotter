"""KanjiVG + ユーザーサンプルのペアリングデータセット。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class StrokeDataset(Dataset):
    """文字ごとのサブディレクトリからストロークJSONを読み込むデータセット。

    ディレクトリ構成:
        data_dir/
            あ/
                あ_0.json
                あ_1.json
            い/
                い_0.json
    """

    def __init__(self, data_dir: Path | list[Path]) -> None:
        self.samples: list[tuple[str, Path]] = []
        dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        for d in dirs:
            d = Path(d)
            if not d.is_dir():
                continue
            for char_dir in sorted(d.iterdir()):
                if char_dir.is_dir():
                    for f in sorted(char_dir.glob("*.json")):
                        self.samples.append((char_dir.name, f))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        character, filepath = self.samples[idx]
        data = json.loads(filepath.read_text(encoding="utf-8"))
        points = []
        for stroke in data["strokes"]:
            for pt in stroke:
                points.append([pt["x"], pt["y"], pt.get("pressure", 1.0)])
        strokes_tensor = torch.tensor(points, dtype=torch.float32)
        return {"strokes": strokes_tensor, "character": character}


def collate_strokes(batch: list[dict]) -> dict:
    """可変長ストロークシーケンスをパディングしてバッチ化する。"""
    strokes = [item["strokes"] for item in batch]
    lengths = torch.tensor([s.shape[0] for s in strokes])
    padded = pad_sequence(strokes, batch_first=True, padding_value=0.0)
    characters = [item["character"] for item in batch]
    return {"strokes": padded, "lengths": lengths, "characters": characters}
