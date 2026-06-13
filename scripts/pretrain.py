"""ユーザー手書きデータによる訓練 CLI（v3-user / UserDeformationTrainer）。

ユーザーの手書きサンプルのみで StrokeDeformer + StyleEncoder をスクラッチ訓練する（CASIA不使用）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector.profiles import list_profiles


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deformer on user handwriting (v3-user)")
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
    parser.add_argument("--hidden-dim", type=int, default=128)
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
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for StrokeDeformer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--use-aligner",
        action="store_true",
        help="Enable stroke alignment (Hungarian + MHD) for stroke order/count mismatch",
    )
    parser.add_argument(
        "--deformer-type",
        type=str,
        default="twostage",
        choices=["offset", "transformer", "twostage"],
        help="Deformer type: offset=MLP, transformer=Transformer, twostage=Affine+Transformer",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    hand_dir = args.hand_dir
    if not hand_dir.exists():
        print(f"Error: hand directory not found: {hand_dir}")
        sys.exit(1)
    profiles = list_profiles(hand_dir)
    hand_data_arg: Path | list[Path] = [p.path for p in profiles] if profiles else hand_dir
    hand_dirs = hand_data_arg if isinstance(hand_data_arg, list) else [hand_data_arg]
    hand_json = sum(len(list(Path(d).rglob("*.json"))) for d in hand_dirs)
    if hand_json == 0:
        print(f"Error: no stroke data found in {hand_dir}")
        sys.exit(1)
    print(f"Hand data: {', '.join(str(d) for d in hand_dirs)} ({hand_json} files)")

    ref_dir = args.ref_dir
    if not ref_dir.exists():
        print(f"Error: reference directory not found: {ref_dir}")
        sys.exit(1)
    ref_json = len(list(ref_dir.rglob("*.json")))
    if ref_json == 0:
        print(f"Error: no reference data found in {ref_dir}")
        sys.exit(1)
    print(f"Reference data: {ref_dir} ({ref_json} files)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from src.model.finetune import UserDeformationTrainer, UserTrainConfig

    config = UserTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        style_dim=args.style_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        deformer_type=args.deformer_type,
    )
    trainer = UserDeformationTrainer(
        config=config,
        user_data_dir=hand_data_arg,
        ref_dir=ref_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers,
        use_aligner=args.use_aligner,
    )

    result = trainer.train()
    print(f"Training complete. Final loss: {result['losses'][-1]:.4f}")
    print(f"Checkpoint saved to: {args.output_dir / 'pretrain_checkpoint.pt'}")
    return result


if __name__ == "__main__":
    main()
