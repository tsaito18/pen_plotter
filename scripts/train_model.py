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
        help="Directory containing stroke samples",
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
    if not args.data_dir.exists():
        print(f"Error: data directory not found: {args.data_dir}")
        print("Run 'python3 scripts/prepare_kanjivg.py --download' first to get training data.")
        sys.exit(1)
    json_files = list(args.data_dir.rglob("*.json"))
    if not json_files:
        print(f"Error: no stroke data found in {args.data_dir}")
        print("Run 'python3 scripts/prepare_kanjivg.py --download' first to get training data.")
        sys.exit(1)
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
    trainer = Trainer(config=config, data_dir=args.data_dir, output_dir=args.output_dir)
    result = trainer.train()
    print(f"Training complete. Final loss: {result['losses'][-1]:.4f}")
    print(f"Checkpoint saved to: {args.output_dir / 'checkpoint.pt'}")
    return result


if __name__ == "__main__":
    main()
