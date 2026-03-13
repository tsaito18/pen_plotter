import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.dataset import StrokeDataset, collate_strokes


def _make_sample_file(path: Path, character: str, strokes_data: list) -> Path:
    """テスト用のサンプルJSONファイルを作成"""
    data = {
        "character": character,
        "strokes": strokes_data,
        "metadata": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestStrokeDataset:
    @pytest.fixture
    def sample_dir(self, tmp_path):
        """テスト用サンプルディレクトリ"""
        for ch in ["あ", "い"]:
            for i in range(3):
                stroke = [{"x": float(j), "y": float(j), "pressure": 1.0, "timestamp": float(j * 10)}
                          for j in range(10)]
                _make_sample_file(
                    tmp_path / ch / f"{ch}_{i}.json",
                    ch,
                    [[{"x": s["x"], "y": s["y"], "pressure": s["pressure"], "timestamp": s["timestamp"]} for s in stroke]],
                )
        return tmp_path

    def test_dataset_length(self, sample_dir):
        ds = StrokeDataset(sample_dir)
        assert len(ds) == 6  # 2文字 x 3サンプル

    def test_getitem_returns_tensors(self, sample_dir):
        ds = StrokeDataset(sample_dir)
        item = ds[0]
        assert "strokes" in item
        assert "character" in item
        assert isinstance(item["strokes"], torch.Tensor)
        assert item["strokes"].ndim == 2  # (seq_len, features)

    def test_stroke_features(self, sample_dir):
        ds = StrokeDataset(sample_dir)
        item = ds[0]
        # features: x, y, pressure (最低3つ)
        assert item["strokes"].shape[1] >= 3

    def test_character_label(self, sample_dir):
        ds = StrokeDataset(sample_dir)
        item = ds[0]
        assert isinstance(item["character"], str)


class TestCollateStrokes:
    def test_collate_pads_sequences(self):
        batch = [
            {"strokes": torch.randn(5, 3), "character": "あ"},
            {"strokes": torch.randn(10, 3), "character": "い"},
        ]
        collated = collate_strokes(batch)
        assert collated["strokes"].shape == (2, 10, 3)  # バッチ, 最大長, 特徴

    def test_collate_returns_lengths(self):
        batch = [
            {"strokes": torch.randn(5, 3), "character": "あ"},
            {"strokes": torch.randn(8, 3), "character": "い"},
        ]
        collated = collate_strokes(batch)
        assert "lengths" in collated
        assert collated["lengths"].tolist() == [5, 8]

    def test_collate_characters(self):
        batch = [
            {"strokes": torch.randn(5, 3), "character": "あ"},
            {"strokes": torch.randn(5, 3), "character": "い"},
        ]
        collated = collate_strokes(batch)
        assert collated["characters"] == ["あ", "い"]
