"""scripts/train_model.py の CLI テスト。"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.train_model import main, parse_args  # noqa: E402


def _make_samples(base_dir: Path, n_chars: int = 2, n_samples: int = 3) -> Path:
    """テスト用のストロークデータを生成。"""
    chars = ["あ", "い", "う"][:n_chars]
    for ch in chars:
        d = base_dir / ch
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_samples):
            stroke = [
                {"x": float(j), "y": float(j) * 0.5, "pressure": 1.0, "timestamp": float(j * 10)}
                for j in range(15)
            ]
            data = {"character": ch, "strokes": [stroke], "metadata": {}}
            (d / f"{ch}_{i}.json").write_text(json.dumps(data), encoding="utf-8")
    return base_dir


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.data_dir == Path("data/strokes")
        assert args.output_dir == Path("data/models")
        assert args.epochs == 50
        assert args.batch_size == 32
        assert args.learning_rate == 1e-3
        assert args.grad_clip_norm == 5.0
        assert args.style_dim == 128
        assert args.hidden_dim == 128
        assert args.num_mixtures == 5

    def test_custom(self):
        args = parse_args(
            [
                "--data-dir",
                "/tmp/data",
                "--output-dir",
                "/tmp/out",
                "--epochs",
                "10",
                "--batch-size",
                "8",
                "--learning-rate",
                "0.01",
                "--grad-clip-norm",
                "3.0",
                "--style-dim",
                "64",
                "--hidden-dim",
                "64",
                "--num-mixtures",
                "3",
            ]
        )
        assert args.data_dir == Path("/tmp/data")
        assert args.output_dir == Path("/tmp/out")
        assert args.epochs == 10
        assert args.batch_size == 8
        assert args.learning_rate == 0.01
        assert args.grad_clip_norm == 3.0
        assert args.style_dim == 64
        assert args.hidden_dim == 64
        assert args.num_mixtures == 3


class TestMain:
    @pytest.mark.slow
    def test_main_creates_checkpoint(self, tmp_path):
        data_dir = _make_samples(tmp_path / "data")
        output_dir = tmp_path / "output"
        result = main(
            [
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
            ]
        )
        assert (output_dir / "checkpoint.pt").exists()
        assert "losses" in result
        assert len(result["losses"]) == 1

    @pytest.mark.slow
    def test_main_prints_summary(self, tmp_path, capsys):
        data_dir = _make_samples(tmp_path / "data")
        output_dir = tmp_path / "output"
        main(
            [
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
            ]
        )
        captured = capsys.readouterr()
        assert "Training complete" in captured.out
        assert "Checkpoint saved" in captured.out

    def test_main_missing_data_dir(self, tmp_path):
        with pytest.raises(SystemExit):
            main(
                [
                    "--data-dir",
                    str(tmp_path / "nonexistent"),
                    "--user-dir",
                    str(tmp_path / "also_nonexistent"),
                    "--output-dir",
                    str(tmp_path / "output"),
                    "--epochs",
                    "1",
                ]
            )
