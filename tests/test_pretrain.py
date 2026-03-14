"""Pre-training パイプラインのテスト。"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.model.pretrain import PairedStrokeDataset, PretrainConfig, Pretrainer, collate_paired


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
            # 手書きデータ (x, y, pressure)
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

        # 参照データ (x, y のみ — KanjiVG 形式)
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
        # 2 chars * 3 samples each = 6
        assert len(ds) == 6

    def test_only_common_chars(self, tmp_path):
        """hand にあるが ref にない文字は除外される。"""
        hand_dir, ref_dir = _make_paired_data(tmp_path, chars=("あ", "い"), n_samples=2)
        # hand に「う」を追加するが ref には追加しない
        extra_dir = hand_dir / "う"
        extra_dir.mkdir(parents=True, exist_ok=True)
        stroke = [{"x": 0.0, "y": 0.0, "pressure": 1.0, "timestamp": 0.0}]
        data = {"character": "う", "strokes": [stroke], "metadata": {}}
        (extra_dir / "う_0.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        ds = PairedStrokeDataset(hand_dir, ref_dir)
        # 「う」は ref に存在しないので除外 → 2 chars * 2 samples = 4
        assert len(ds) == 4
        chars_in_dataset = {ds[i]["character"] for i in range(len(ds))}
        assert "う" not in chars_in_dataset

    def test_getitem(self, tmp_path):
        hand_dir, ref_dir = _make_paired_data(tmp_path, chars=("あ",), n_samples=1, n_points=10)
        ds = PairedStrokeDataset(hand_dir, ref_dir)
        item = ds[0]
        assert "strokes" in item
        assert "reference" in item
        assert "character" in item
        assert item["strokes"].shape == (10, 3)
        assert item["reference"].shape[1] == 2
        assert item["character"] == "あ"

    def test_multi_hand_dirs(self, tmp_path):
        """hand_dir がリストの場合も動作する。"""
        hand1 = tmp_path / "h1"
        hand2 = tmp_path / "h2"
        ref_dir = tmp_path / "ref"

        for ch in ("あ",):
            for hdir in (hand1, hand2):
                d = hdir / ch
                d.mkdir(parents=True, exist_ok=True)
                stroke = [{"x": 0.0, "y": 0.0, "pressure": 1.0, "timestamp": 0.0}]
                data = {"character": ch, "strokes": [stroke], "metadata": {}}
                (d / f"{ch}_0.json").write_text(
                    json.dumps(data, ensure_ascii=False), encoding="utf-8"
                )

            rd = ref_dir / ch
            rd.mkdir(parents=True, exist_ok=True)
            ref_data = {
                "character": ch,
                "strokes": [[{"x": 0.0, "y": 0.0, "pressure": 1.0, "timestamp": 0.0}]],
                "metadata": {},
            }
            (rd / f"{ch}_0.json").write_text(
                json.dumps(ref_data, ensure_ascii=False), encoding="utf-8"
            )

        ds = PairedStrokeDataset([hand1, hand2], ref_dir)
        assert len(ds) == 2


class TestCollatePaired:
    def test_collate(self, tmp_path):
        hand_dir, ref_dir = _make_paired_data(
            tmp_path, chars=("あ", "い"), n_samples=1, n_points=10
        )
        ds = PairedStrokeDataset(hand_dir, ref_dir)
        batch = [ds[i] for i in range(len(ds))]
        collated = collate_paired(batch)

        assert collated["strokes"].shape[0] == 2
        assert collated["strokes"].shape[2] == 3
        assert collated["reference"].shape[0] == 2
        assert collated["reference"].shape[2] == 2
        assert len(collated["lengths"]) == 2
        assert len(collated["ref_lengths"]) == 2
        assert len(collated["characters"]) == 2


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
