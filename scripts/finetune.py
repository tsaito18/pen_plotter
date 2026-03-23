"""ユーザー筆跡へのFine-tuning CLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.finetune import FinetuneConfig, Finetuner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune style on user handwriting")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Pre-trained checkpoint path",
    )
    parser.add_argument(
        "--user-dir",
        type=Path,
        default=Path("data/user_strokes"),
        help="Directory containing user stroke samples",
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
        help="Directory to save fine-tuned checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not args.user_dir.exists():
        print(f"Error: user data directory not found: {args.user_dir}")
        sys.exit(1)

    if not args.ref_dir.exists():
        print(f"Error: reference directory not found: {args.ref_dir}")
        sys.exit(1)

    user_json = len(list(args.user_dir.rglob("*.json")))
    if user_json == 0:
        print("Error: no user stroke data found")
        sys.exit(1)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"User data: {args.user_dir} ({user_json} files)")
    print(f"Reference: {args.ref_dir}")

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    is_v3 = "deformer_state_dict" in checkpoint
    del checkpoint

    config = FinetuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
    )

    if is_v3:
        from src.model.finetune import DeformationFinetuner

        print("Detected V3 (deformation) checkpoint")
        finetuner = DeformationFinetuner(
            config=config,
            pretrain_checkpoint=args.checkpoint,
            user_data_dir=args.user_dir,
            ref_dir=args.ref_dir,
            output_dir=args.output_dir,
            device=args.device,
            num_workers=args.num_workers,
        )
    else:
        finetuner = Finetuner(
            config=config,
            pretrain_checkpoint=args.checkpoint,
            user_data_dir=args.user_dir,
            ref_dir=args.ref_dir,
            output_dir=args.output_dir,
            device=args.device,
            num_workers=args.num_workers,
        )

    result = finetuner.train()
    print(f"Fine-tuning complete. Final loss: {result['losses'][-1]:.4f}")
    print(f"Checkpoint saved to: {args.output_dir / 'finetuned.pt'}")
    return result


if __name__ == "__main__":
    main()
