"""手書きスタイルモデルの訓練 CLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.train import Trainer, TrainConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train handwriting style model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/strokes"),
        help="Directory containing stroke samples (KanjiVG)",
    )
    parser.add_argument(
        "--user-dir",
        type=Path,
        default=Path("data/user_strokes"),
        help="Directory containing user stroke samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory to save checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-mixtures", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    data_dirs = []
    if args.data_dir.exists():
        data_dirs.append(args.data_dir)
    if args.user_dir.exists():
        data_dirs.append(args.user_dir)
    if not data_dirs:
        print(f"Error: no data directories found ({args.data_dir}, {args.user_dir})")
        print("Run 'python3 scripts/prepare_kanjivg.py --download' first to get training data.")
        sys.exit(1)
    total_json = sum(len(list(d.rglob("*.json"))) for d in data_dirs)
    if total_json == 0:
        print("Error: no stroke data found in any data directory")
        sys.exit(1)
    print(f"Data sources: {', '.join(str(d) for d in data_dirs)} ({total_json} files)")
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        style_dim=args.style_dim,
        hidden_dim=args.hidden_dim,
        num_mixtures=args.num_mixtures,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(config=config, data_dir=data_dirs, output_dir=args.output_dir)
    result = trainer.train()
    print(f"Training complete. Final loss: {result['losses'][-1]:.4f}")
    print(f"Checkpoint saved to: {args.output_dir / 'checkpoint.pt'}")
    return result


if __name__ == "__main__":
    main()
