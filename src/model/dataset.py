"""KanjiVG + ユーザーサンプルのペアリングデータセット。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.model.data_utils import strokes_to_deltas


class StrokeDataset(Dataset):
    """文字ごとのサブディレクトリからストロークJSONを読み込むデータセット。

    各ストロークが個別のサンプルとなる（ストローク単位生成）。

    ディレクトリ構成:
        data_dir/
            あ/
                あ_0.json
                あ_1.json
            い/
                い_0.json
    """

    def __init__(self, data_dir: Path | list[Path]) -> None:
        self.char_samples: list[tuple[str, Path]] = []
        dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        for d in dirs:
            d = Path(d)
            if not d.is_dir():
                continue
            for char_dir in sorted(d.iterdir()):
                if char_dir.is_dir():
                    for f in sorted(char_dir.glob("*.json")):
                        self.char_samples.append((char_dir.name, f))

        self.samples: list[tuple[int, int, int]] = []
        for char_idx, (ch, filepath) in enumerate(self.char_samples):
            data = json.loads(filepath.read_text(encoding="utf-8"))
            n_strokes = len(data["strokes"])
            for stroke_idx in range(n_strokes):
                stroke = data["strokes"][stroke_idx]
                if len(stroke) >= 2:
                    self.samples.append((char_idx, stroke_idx, n_strokes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        char_idx, stroke_idx, num_strokes = self.samples[idx]
        character, filepath = self.char_samples[char_idx]
        data = json.loads(filepath.read_text(encoding="utf-8"))

        stroke_points = data["strokes"][stroke_idx]
        pts = [(pt["x"], pt["y"]) for pt in stroke_points]
        deltas = torch.zeros(len(pts), 2, dtype=torch.float32)
        for i in range(1, len(pts)):
            deltas[i, 0] = pts[i][0] - pts[i - 1][0]
            deltas[i, 1] = pts[i][1] - pts[i - 1][1]

        eos = torch.zeros(len(pts), 1, dtype=torch.float32)
        eos[-1, 0] = 1.0

        style_tensor = strokes_to_deltas(data["strokes"])

        return {
            "stroke_deltas": deltas,
            "eos": eos,
            "stroke_index": stroke_idx,
            "num_strokes": num_strokes,
            "character": character,
            "style_strokes": style_tensor,
        }


def collate_strokes(batch: list[dict]) -> dict:
    """可変長ストロークシーケンスをパディングしてバッチ化する。"""
    stroke_deltas = [item["stroke_deltas"] for item in batch]
    eos_list = [item["eos"] for item in batch]
    style_strokes = [item["style_strokes"] for item in batch]

    stroke_lengths = torch.tensor([s.shape[0] for s in stroke_deltas])
    style_lengths = torch.tensor([s.shape[0] for s in style_strokes])
    stroke_indices = torch.tensor([item["stroke_index"] for item in batch])

    padded_deltas = pad_sequence(stroke_deltas, batch_first=True, padding_value=0.0)
    padded_eos = pad_sequence(eos_list, batch_first=True, padding_value=0.0)
    padded_style = pad_sequence(style_strokes, batch_first=True, padding_value=0.0)

    characters = [item["character"] for item in batch]
    return {
        "stroke_deltas": padded_deltas,
        "eos": padded_eos,
        "stroke_indices": stroke_indices,
        "style_strokes": padded_style,
        "stroke_lengths": stroke_lengths,
        "style_lengths": style_lengths,
        "characters": characters,
    }
