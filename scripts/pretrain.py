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
    parser.add_argument("--num-mixtures", type=int, default=20)
    parser.add_argument(
        "--pot-dir",
        type=Path,
        default=None,
        help="Directory containing CASIA .pot files (alternative to --hand-dir)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max CASIA samples to load (0=all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda/xpu). Auto-detect if not specified.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0=main process)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v3",
        choices=["v2", "v3"],
        help="Model architecture version (v2=LSTM+MDN, v3=deformation)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    hand_dir = args.hand_dir
    ref_dir = args.ref_dir
    pot_dir = args.pot_dir

    if pot_dir is not None:
        if not pot_dir.exists():
            print(f"Error: pot directory not found: {pot_dir}")
            sys.exit(1)
        pot_files = list(pot_dir.glob("*.pot"))
        if len(pot_files) == 0:
            print(f"Error: no .pot files found in {pot_dir}")
            sys.exit(1)
        print(f"CASIA data: {pot_dir} ({len(pot_files)} .pot files)")
    else:
        if not hand_dir.exists():
            print(f"Error: hand directory not found: {hand_dir}")
            sys.exit(1)
        hand_json = len(list(hand_dir.rglob("*.json")))
        if hand_json == 0:
            print(f"Error: no stroke data found in {hand_dir}")
            sys.exit(1)
        print(f"Hand data: {hand_dir} ({hand_json} files)")

    if not ref_dir.exists():
        print(f"Error: reference directory not found: {ref_dir}")
        sys.exit(1)
    ref_json = len(list(ref_dir.rglob("*.json")))
    if ref_json == 0:
        print(f"Error: no reference data found in {ref_dir}")
        sys.exit(1)
    print(f"Reference data: {ref_dir} ({ref_json} files)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_version == "v3":
        from src.model.pretrain import DeformationConfig, DeformationPretrainer

        config_v3 = DeformationConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            grad_clip_norm=args.grad_clip_norm,
            style_dim=args.style_dim,
            hidden_dim=args.hidden_dim,
        )
        pretrainer = DeformationPretrainer(
            config=config_v3,
            ref_dir=ref_dir,
            output_dir=args.output_dir,
            device=args.device,
            pot_dir=pot_dir,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            amp=args.amp,
        )
    else:
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
        pretrainer = Pretrainer(
            config=config,
            hand_dir=hand_dir,
            ref_dir=ref_dir,
            output_dir=args.output_dir,
            device=args.device,
            pot_dir=pot_dir,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            amp=args.amp,
        )

    result = pretrainer.train()
    print(f"Pre-training complete. Final loss: {result['losses'][-1]:.4f}")
    print(f"Checkpoint saved to: {args.output_dir / 'pretrain_checkpoint.pt'}")
    return result


if __name__ == "__main__":
    main()
