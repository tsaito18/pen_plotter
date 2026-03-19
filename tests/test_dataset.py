import json
from pathlib import Path
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
        # 2 chars * 3 samples * 1 stroke each = 6 stroke-level samples
        assert len(ds) == 6

    def test_getitem_returns_tensors(self, sample_dir):
        ds = StrokeDataset(sample_dir)
        item = ds[0]
        assert "stroke_deltas" in item
        assert "eos" in item
        assert "stroke_index" in item
        assert "character" in item
        assert isinstance(item["stroke_deltas"], torch.Tensor)
        assert item["stroke_deltas"].ndim == 2

    def test_stroke_features(self, sample_dir):
        ds = StrokeDataset(sample_dir)
        item = ds[0]
        assert item["stroke_deltas"].shape[1] == 2  # dx, dy only
        assert item["eos"].shape[1] == 1

    def test_delta_encoding(self, sample_dir):
        """Strokes use delta coordinates with EOS."""
        ds = StrokeDataset(sample_dir)
        item = ds[0]
        assert item["stroke_deltas"][0, 0] == 0.0
        assert item["stroke_deltas"][0, 1] == 0.0
        assert item["eos"][-1, 0] == 1.0
        if item["eos"].shape[0] > 1:
            assert item["eos"][0, 0] == 0.0

    def test_character_label(self, sample_dir):
        ds = StrokeDataset(sample_dir)
        item = ds[0]
        assert isinstance(item["character"], str)

    def test_style_strokes_full_character(self, sample_dir):
        """style_strokes should be the full character with pen_state (3D)."""
        ds = StrokeDataset(sample_dir)
        item = ds[0]
        assert "style_strokes" in item
        assert item["style_strokes"].shape[1] == 3

    def test_multi_stroke_creates_multiple_samples(self, tmp_path):
        """Multi-stroke character should create multiple samples."""
        ch = "あ"
        s1 = [{"x": float(j), "y": float(j)} for j in range(5)]
        s2 = [{"x": float(j + 10), "y": float(j + 10)} for j in range(4)]
        data = {"character": ch, "strokes": [s1, s2], "metadata": {}}
        d = tmp_path / ch
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{ch}_0.json").write_text(json.dumps(data), encoding="utf-8")

        ds = StrokeDataset(tmp_path)
        assert len(ds) == 2
        assert ds[0]["stroke_index"] == 0
        assert ds[1]["stroke_index"] == 1


class TestCollateStrokes:
    def test_collate_pads_sequences(self):
        batch = [
            {
                "stroke_deltas": torch.randn(5, 2),
                "eos": torch.zeros(5, 1),
                "stroke_index": 0,
                "num_strokes": 1,
                "character": "あ",
                "style_strokes": torch.randn(5, 3),
            },
            {
                "stroke_deltas": torch.randn(10, 2),
                "eos": torch.zeros(10, 1),
                "stroke_index": 0,
                "num_strokes": 1,
                "character": "い",
                "style_strokes": torch.randn(8, 3),
            },
        ]
        collated = collate_strokes(batch)
        assert collated["stroke_deltas"].shape == (2, 10, 2)
        assert collated["eos"].shape == (2, 10, 1)

    def test_collate_returns_lengths(self):
        batch = [
            {
                "stroke_deltas": torch.randn(5, 2),
                "eos": torch.zeros(5, 1),
                "stroke_index": 0,
                "num_strokes": 1,
                "character": "あ",
                "style_strokes": torch.randn(5, 3),
            },
            {
                "stroke_deltas": torch.randn(8, 2),
                "eos": torch.zeros(8, 1),
                "stroke_index": 0,
                "num_strokes": 1,
                "character": "い",
                "style_strokes": torch.randn(8, 3),
            },
        ]
        collated = collate_strokes(batch)
        assert "stroke_lengths" in collated
        assert collated["stroke_lengths"].tolist() == [5, 8]
        assert "stroke_indices" in collated
        assert "style_lengths" in collated

    def test_collate_characters(self):
        batch = [
            {
                "stroke_deltas": torch.randn(5, 2),
                "eos": torch.zeros(5, 1),
                "stroke_index": 0,
                "num_strokes": 1,
                "character": "あ",
                "style_strokes": torch.randn(5, 3),
            },
            {
                "stroke_deltas": torch.randn(5, 2),
                "eos": torch.zeros(5, 1),
                "stroke_index": 0,
                "num_strokes": 1,
                "character": "い",
                "style_strokes": torch.randn(5, 3),
            },
        ]
        collated = collate_strokes(batch)
        assert collated["characters"] == ["あ", "い"]
