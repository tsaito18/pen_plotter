"""Pre-training パイプラインのテスト。"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.model.pretrain import (
    CASIAPairedDataset,
    PairedStrokeDataset,
    PretrainConfig,
    Pretrainer,
    collate_paired,
)


def _make_paired_data(
    base_dir: Path,
    chars: tuple[str, ...] = ("あ", "い"),
    n_samples: int = 2,
    n_points: int = 15,
) -> tuple[Path, Path]:
    """手書きデータと参照データの両ディレクトリを作成する。"""
    hand_dir = base_dir / "hand"
    ref_dir = base_dir / "ref"
    for ch in chars:
        for i in range(n_samples):
            stroke = [
                {"x": float(j), "y": float(j) * 0.5, "pressure": 1.0, "timestamp": float(j * 10)}
                for j in range(n_points)
            ]
            data = {"character": ch, "strokes": [stroke], "metadata": {}}
            d = hand_dir / ch
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{ch}_{i}.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )

        ref_stroke = [
            {"x": float(j) * 0.8, "y": float(j) * 0.3, "pressure": 1.0, "timestamp": 0.0}
            for j in range(n_points)
        ]
        ref_data = {"character": ch, "strokes": [ref_stroke], "metadata": {"source": "kanjivg"}}
        rd = ref_dir / ch
        rd.mkdir(parents=True, exist_ok=True)
        (rd / f"{ch}_0.json").write_text(json.dumps(ref_data, ensure_ascii=False), encoding="utf-8")

    return hand_dir, ref_dir


class TestPretrainConfig:
    def test_defaults(self):
        cfg = PretrainConfig()
        assert cfg.epochs == 50
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 1e-3
        assert cfg.grad_clip_norm == 5.0
        assert cfg.style_dim == 128
        assert cfg.char_dim == 128
        assert cfg.hidden_dim == 128
        assert cfg.num_mixtures == 5


class TestPairedStrokeDataset:
    def test_creation(self, tmp_path):
        hand_dir, ref_dir = _make_paired_data(tmp_path, chars=("あ", "い"), n_samples=3)
        ds = PairedStrokeDataset(hand_dir, ref_dir)
        # Each char has 1 stroke * 3 samples = 3 stroke-level samples per char = 6 total
        assert len(ds) == 6

    def test_only_common_chars(self, tmp_path):
        """hand にあるが ref にない文字は除外される。"""
        hand_dir, ref_dir = _make_paired_data(tmp_path, chars=("あ", "い"), n_samples=2)
        extra_dir = hand_dir / "う"
        extra_dir.mkdir(parents=True, exist_ok=True)
        stroke = [{"x": 0.0, "y": 0.0, "pressure": 1.0, "timestamp": 0.0},
                  {"x": 1.0, "y": 1.0, "pressure": 1.0, "timestamp": 10.0}]
        data = {"character": "う", "strokes": [stroke], "metadata": {}}
        (extra_dir / "う_0.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        ds = PairedStrokeDataset(hand_dir, ref_dir)
        # 2 chars * 2 samples * 1 stroke each = 4
        assert len(ds) == 4
        chars_in_dataset = {ds[i]["character"] for i in range(len(ds))}
        assert "う" not in chars_in_dataset

    def test_getitem(self, tmp_path):
        hand_dir, ref_dir = _make_paired_data(tmp_path, chars=("あ",), n_samples=2, n_points=10)
        ds = PairedStrokeDataset(hand_dir, ref_dir)
        item = ds[0]
        assert "stroke_deltas" in item
        assert "eos" in item
        assert "stroke_index" in item
        assert "num_strokes" in item
        assert "style_strokes" in item
        assert "reference" in item
        assert "character" in item
        assert item["stroke_deltas"].shape == (10, 2)
        assert item["eos"].shape == (10, 1)
        assert item["eos"][-1, 0] == 1.0
        assert item["stroke_index"] == 0
        assert item["style_strokes"].shape[1] == 3
        assert item["reference"].shape[1] == 2
        assert item["character"] == "あ"

    def test_style_strokes_differ_from_strokes(self, tmp_path):
        """style_strokes should come from a different sample than strokes."""
        hand_dir, ref_dir = _make_paired_data(tmp_path, chars=("あ",), n_samples=3, n_points=10)
        ds = PairedStrokeDataset(hand_dir, ref_dir)
        assert len(ds.char_to_indices["あ"]) == 3

    def test_reference_has_separators(self, tmp_path):
        """reference_to_sequence inserts (-1,-1) separators between strokes."""
        hand_dir = tmp_path / "hand"
        ref_dir = tmp_path / "ref"
        ch = "あ"
        stroke = [{"x": float(j), "y": float(j) * 0.5, "pressure": 1.0, "timestamp": 0.0}
                  for j in range(5)]
        data = {"character": ch, "strokes": [stroke], "metadata": {}}
        d = hand_dir / ch
        d.mkdir(parents=True, exist_ok=True)
        import json as _json
        (d / f"{ch}_0.json").write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
        s1 = [{"x": float(j), "y": float(j)} for j in range(3)]
        s2 = [{"x": float(j + 5), "y": float(j + 5)} for j in range(4)]
        ref_data = {"character": ch, "strokes": [s1, s2], "metadata": {}}
        rd = ref_dir / ch
        rd.mkdir(parents=True, exist_ok=True)
        (rd / f"{ch}_0.json").write_text(_json.dumps(ref_data, ensure_ascii=False), encoding="utf-8")
        ds = PairedStrokeDataset(hand_dir, ref_dir)
        item = ds[0]
        ref = item["reference"]
        # 3 points + separator + 4 points = 8
        assert ref.shape == (8, 2)
        assert ref[3, 0].item() == -1.0
        assert ref[3, 1].item() == -1.0

    def test_multi_hand_dirs(self, tmp_path):
        """hand_dir がリストの場合も動作する。"""
        hand1 = tmp_path / "h1"
        hand2 = tmp_path / "h2"
        ref_dir = tmp_path / "ref"

        for ch in ("あ",):
            for hdir in (hand1, hand2):
                d = hdir / ch
                d.mkdir(parents=True, exist_ok=True)
                stroke = [{"x": 0.0, "y": 0.0, "pressure": 1.0, "timestamp": 0.0},
                          {"x": 1.0, "y": 1.0, "pressure": 1.0, "timestamp": 10.0}]
                data = {"character": ch, "strokes": [stroke], "metadata": {}}
                (d / f"{ch}_0.json").write_text(
                    json.dumps(data, ensure_ascii=False), encoding="utf-8"
                )

            rd = ref_dir / ch
            rd.mkdir(parents=True, exist_ok=True)
            ref_data = {
                "character": ch,
                "strokes": [[{"x": 0.0, "y": 0.0, "pressure": 1.0, "timestamp": 0.0},
                              {"x": 1.0, "y": 1.0, "pressure": 1.0, "timestamp": 10.0}]],
                "metadata": {},
            }
            (rd / f"{ch}_0.json").write_text(
                json.dumps(ref_data, ensure_ascii=False), encoding="utf-8"
            )

        ds = PairedStrokeDataset([hand1, hand2], ref_dir)
        assert len(ds) == 2

    def test_multi_stroke_character(self, tmp_path):
        """Multi-stroke character creates multiple samples."""
        hand_dir = tmp_path / "hand"
        ref_dir = tmp_path / "ref"
        ch = "あ"
        s1 = [{"x": float(j), "y": float(j)} for j in range(5)]
        s2 = [{"x": float(j + 10), "y": float(j + 10)} for j in range(4)]
        data = {"character": ch, "strokes": [s1, s2], "metadata": {}}
        d = hand_dir / ch
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{ch}_0.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        ref_data = {"character": ch, "strokes": [s1, s2], "metadata": {}}
        rd = ref_dir / ch
        rd.mkdir(parents=True, exist_ok=True)
        (rd / f"{ch}_0.json").write_text(json.dumps(ref_data, ensure_ascii=False), encoding="utf-8")

        ds = PairedStrokeDataset(hand_dir, ref_dir)
        assert len(ds) == 2  # 2 strokes = 2 samples
        assert ds[0]["stroke_index"] == 0
        assert ds[1]["stroke_index"] == 1
        assert ds[0]["num_strokes"] == 2
        assert ds[1]["num_strokes"] == 2


class TestCollatePaired:
    def test_collate(self, tmp_path):
        hand_dir, ref_dir = _make_paired_data(
            tmp_path, chars=("あ", "い"), n_samples=2, n_points=10
        )
        ds = PairedStrokeDataset(hand_dir, ref_dir)
        batch = [ds[i] for i in range(len(ds))]
        collated = collate_paired(batch)

        assert collated["stroke_deltas"].shape[0] == 4
        assert collated["stroke_deltas"].shape[2] == 2
        assert collated["eos"].shape[0] == 4
        assert collated["eos"].shape[2] == 1
        assert collated["style_strokes"].shape[0] == 4
        assert collated["style_strokes"].shape[2] == 3
        assert collated["reference"].shape[0] == 4
        assert collated["reference"].shape[2] == 2
        assert len(collated["stroke_lengths"]) == 4
        assert len(collated["style_lengths"]) == 4
        assert len(collated["ref_lengths"]) == 4
        assert "stroke_indices" in collated
        assert len(collated["characters"]) == 4


class TestPretrainer:
    @pytest.fixture
    def paired_dirs(self, tmp_path):
        return _make_paired_data(tmp_path, chars=("あ", "い", "う"), n_samples=3, n_points=15)

    def test_creation(self, paired_dirs, tmp_path):
        hand_dir, ref_dir = paired_dirs
        cfg = PretrainConfig(epochs=1, batch_size=2)
        pt = Pretrainer(cfg, hand_dir=hand_dir, ref_dir=ref_dir, output_dir=tmp_path / "out")
        assert pt is not None
        assert pt.char_encoder is not None
        assert pt.style_encoder is not None
        assert pt.generator is not None

    def test_norm_stats_computed(self, paired_dirs, tmp_path):
        hand_dir, ref_dir = paired_dirs
        cfg = PretrainConfig(epochs=1, batch_size=2)
        pt = Pretrainer(cfg, hand_dir=hand_dir, ref_dir=ref_dir, output_dir=tmp_path / "out")
        assert pt.norm_stats is not None
        for key in ("mean_x", "mean_y", "std_x", "std_y"):
            assert key in pt.norm_stats
        assert pt.norm_stats["std_x"] > 0
        assert pt.norm_stats["std_y"] > 0

    def test_ref_norm_stats_computed(self, paired_dirs, tmp_path):
        hand_dir, ref_dir = paired_dirs
        cfg = PretrainConfig(epochs=1, batch_size=2)
        pt = Pretrainer(cfg, hand_dir=hand_dir, ref_dir=ref_dir, output_dir=tmp_path / "out")
        assert pt.ref_norm_stats is not None
        for key in ("mean_x", "mean_y", "std_x", "std_y"):
            assert key in pt.ref_norm_stats
        assert pt.ref_norm_stats["std_x"] > 0
        assert pt.ref_norm_stats["std_y"] > 0

    @pytest.mark.slow
    def test_train_one_epoch(self, paired_dirs, tmp_path):
        hand_dir, ref_dir = paired_dirs
        output_dir = tmp_path / "out"
        cfg = PretrainConfig(epochs=1, batch_size=4)
        pt = Pretrainer(cfg, hand_dir=hand_dir, ref_dir=ref_dir, output_dir=output_dir)
        history = pt.train()
        assert "losses" in history
        assert len(history["losses"]) == 1
        assert history["losses"][0] > 0

    @pytest.mark.slow
    def test_checkpoint_format(self, paired_dirs, tmp_path):
        hand_dir, ref_dir = paired_dirs
        output_dir = tmp_path / "out"
        cfg = PretrainConfig(epochs=1, batch_size=4)
        pt = Pretrainer(cfg, hand_dir=hand_dir, ref_dir=ref_dir, output_dir=output_dir)
        pt.train()

        ckpt_path = output_dir / "pretrain_checkpoint.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "generator_state_dict" in ckpt
        assert "style_encoder_state_dict" in ckpt
        assert "char_encoder_state_dict" in ckpt
        assert "config" in ckpt
        assert "norm_stats" in ckpt
        for key in ("mean_x", "mean_y", "std_x", "std_y"):
            assert key in ckpt["norm_stats"]
        assert "ref_norm_stats" in ckpt
        for key in ("mean_x", "mean_y", "std_x", "std_y"):
            assert key in ckpt["ref_norm_stats"]


def _make_pot_sample_v1(char: str, strokes: list[list[tuple[int, int]]]) -> bytes:
    """V1形式の.potサンプルバイナリを生成するヘルパー。"""
    import struct

    tag = char.encode("gbk")
    stroke_data = b""
    for stroke in strokes:
        for x, y in stroke:
            stroke_data += struct.pack("<hh", x, y)
        stroke_data += struct.pack("<hh", -1, 0)
    stroke_data += struct.pack("<hh", -1, -1)
    total = 4 + len(tag) + 2 + len(stroke_data)
    return struct.pack("<I", total) + tag + struct.pack("<H", len(strokes)) + stroke_data


def _make_ref_json(char: str) -> str:
    """KanjiVG参照用JSONを生成するヘルパー。"""
    stroke = [{"x": float(j) * 0.3, "y": float(j) * 0.7} for j in range(10)]
    return json.dumps({"character": char, "strokes": [stroke], "metadata": {}})


class TestCASIAPairedDataset:
    def _create_pot_file(self, pot_dir: Path, chars_strokes: dict) -> Path:
        """テスト用.potファイルを作成する。"""
        pot_dir.mkdir(parents=True, exist_ok=True)
        pot_path = pot_dir / "test.pot"
        data = b""
        for char, strokes in chars_strokes.items():
            data += _make_pot_sample_v1(char, strokes)
        pot_path.write_bytes(data)
        return pot_path

    def _create_ref_dir(self, ref_dir: Path, chars: list[str]) -> None:
        """テスト用参照ディレクトリを作成する。"""
        for ch in chars:
            d = ref_dir / ch
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{ch}_0.json").write_text(_make_ref_json(ch), encoding="utf-8")

    def test_creation(self, tmp_path):
        pot_dir = tmp_path / "pot"
        ref_dir = tmp_path / "ref"

        sample_strokes = {
            "\u5927": [[(100, 200), (150, 250), (200, 300)]],
            "\u5c0f": [[(50, 50), (100, 100)]],
        }
        self._create_pot_file(pot_dir, sample_strokes)
        self._create_ref_dir(ref_dir, ["\u5927", "\u5c0f"])

        ds = CASIAPairedDataset(pot_dir, ref_dir)
        assert len(ds) > 0

    def test_getitem_format(self, tmp_path):
        pot_dir = tmp_path / "pot"
        ref_dir = tmp_path / "ref"

        sample_strokes = {
            "\u5927": [[(100, 200), (150, 250), (200, 300)]],
        }
        self._create_pot_file(pot_dir, sample_strokes)
        self._create_ref_dir(ref_dir, ["\u5927"])

        ds = CASIAPairedDataset(pot_dir, ref_dir)
        item = ds[0]

        assert "stroke_deltas" in item
        assert "eos" in item
        assert "stroke_index" in item
        assert "num_strokes" in item
        assert "style_strokes" in item
        assert "reference" in item
        assert "character" in item
        assert item["stroke_deltas"].ndim == 2
        assert item["stroke_deltas"].shape[1] == 2
        assert item["eos"].ndim == 2
        assert item["eos"].shape[1] == 1
        assert item["eos"][-1, 0] == 1.0
        assert item["style_strokes"].ndim == 2
        assert item["style_strokes"].shape[1] == 3
        assert item["reference"].ndim == 2
        assert item["reference"].shape[1] == 2
        assert item["character"] == "\u5927"

    def test_only_matched_chars(self, tmp_path):
        """refに存在しない文字はデータセットから除外される。"""
        pot_dir = tmp_path / "pot"
        ref_dir = tmp_path / "ref"

        sample_strokes = {
            "\u5927": [[(100, 200), (150, 250)]],
            "\u5c0f": [[(50, 50), (100, 100)]],
            "\u4e2d": [[(10, 20), (30, 40)]],
        }
        self._create_pot_file(pot_dir, sample_strokes)
        self._create_ref_dir(ref_dir, ["\u5927", "\u5c0f"])

        ds = CASIAPairedDataset(pot_dir, ref_dir)
        chars_in_dataset = {ds[i]["character"] for i in range(len(ds))}
        assert "\u5927" in chars_in_dataset
        assert "\u5c0f" in chars_in_dataset
        assert "\u4e2d" not in chars_in_dataset
        assert len(ds) == 2
