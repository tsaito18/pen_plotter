"""事前学習 CLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.pretrain import PretrainConfig, Pretrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train handwriting model")
    parser.add_argument(
        "--hand-dir",
        type=Path,
        default=Path("data/strokes"),
        help="Directory containing handwritten stroke samples",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=Path("data/strokes"),
        help="Directory containing reference strokes (KanjiVG)",
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
    parser.add_argument("--char-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-mixtures", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    hand_dir = args.hand_dir
    ref_dir = args.ref_dir

    if not hand_dir.exists():
        print(f"Error: hand directory not found: {hand_dir}")
        sys.exit(1)
    if not ref_dir.exists():
        print(f"Error: reference directory not found: {ref_dir}")
        sys.exit(1)

    hand_json = len(list(hand_dir.rglob("*.json")))
    ref_json = len(list(ref_dir.rglob("*.json")))
    if hand_json == 0:
        print(f"Error: no stroke data found in {hand_dir}")
        sys.exit(1)
    if ref_json == 0:
        print(f"Error: no reference data found in {ref_dir}")
        sys.exit(1)

    print(f"Hand data: {hand_dir} ({hand_json} files)")
    print(f"Reference data: {ref_dir} ({ref_json} files)")

    config = PretrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        style_dim=args.style_dim,
        char_dim=args.char_dim,
        hidden_dim=args.hidden_dim,
        num_mixtures=args.num_mixtures,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pretrainer = Pretrainer(
        config=config,
        hand_dir=hand_dir,
        ref_dir=ref_dir,
        output_dir=args.output_dir,
    )
    result = pretrainer.train()
    print(f"Pre-training complete. Final loss: {result['losses'][-1]:.4f}")
    print(f"Checkpoint saved to: {args.output_dir / 'pretrain_checkpoint.pt'}")
    return result


if __name__ == "__main__":
    main()
