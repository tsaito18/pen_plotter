"""V3 変形モデルの訓練 CLI。

サブコマンド:
    pretrain  ユーザー筆跡だけで変形器＋StyleEncoder を訓練する（pretrain_checkpoint.pt）
    finetune  変形器を凍結し StyleEncoder だけを微調整する（finetuned.pt）

例:
    uv run python scripts/train.py pretrain --epochs 80 --use-aligner
    uv run python scripts/train.py finetune --checkpoint data/models/pretrain_checkpoint.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector.profiles import list_profiles
from src.model.deformers import DEFORMER_TYPES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--user-dir",
        type=Path,
        default=Path("data/user_strokes"),
        help="ユーザー筆跡（プロファイルのルートなら全プロファイルを使う）",
    )
    common.add_argument("--ref-dir", type=Path, default=Path("data/strokes"), help="KanjiVG")
    common.add_argument("--output-dir", type=Path, default=Path("data/models"))
    common.add_argument("--epochs", type=int)
    common.add_argument("--batch-size", type=int)
    common.add_argument("--learning-rate", type=float)
    common.add_argument("--grad-clip-norm", type=float, default=5.0)
    common.add_argument("--device", default=None, help="cpu/cuda/xpu（省略時は自動）")
    common.add_argument("--num-workers", type=int, default=0)
    common.add_argument(
        "--use-aligner", action="store_true", help="画の順序・数の不一致を自動対応付けする"
    )

    pre = sub.add_parser("pretrain", parents=[common], help="変形器＋StyleEncoder を訓練")
    pre.add_argument("--style-dim", type=int, default=128)
    pre.add_argument("--hidden-dim", type=int, default=128)
    pre.add_argument("--dropout", type=float, default=0.2)
    pre.add_argument("--weight-decay", type=float, default=0.01)
    pre.add_argument(
        "--deformer-type",
        default="twostage",
        choices=[t for t in DEFORMER_TYPES if t != "affine"],
    )

    fine = sub.add_parser("finetune", parents=[common], help="StyleEncoder だけを微調整")
    fine.add_argument("--checkpoint", type=Path, required=True, help="pretrain のチェックポイント")
    return parser.parse_args(argv)


def _user_dirs(user_dir: Path) -> list[Path]:
    profiles = list_profiles(user_dir)
    return [p.path for p in profiles] if profiles else [user_dir]


def main(argv: list[str] | None = None) -> list[float]:
    args = parse_args(argv)
    if not args.ref_dir.is_dir() or not any(args.ref_dir.rglob("*.json")):
        sys.exit(f"Error: KanjiVG データが見つかりません: {args.ref_dir}")
    if not args.user_dir.is_dir():
        sys.exit(f"Error: ユーザー筆跡が見つかりません: {args.user_dir}")
    user_dirs = _user_dirs(args.user_dir)
    n_files = sum(len(list(d.rglob("*.json"))) for d in user_dirs)
    if n_files == 0:
        sys.exit(f"Error: ユーザー筆跡がありません: {args.user_dir}")
    print(f"User data: {', '.join(map(str, user_dirs))} ({n_files} files)")

    from src.model.training import (
        DeformationFinetuner,
        FinetuneConfig,
        UserDeformationTrainer,
        UserTrainConfig,
    )

    trainer: UserDeformationTrainer | DeformationFinetuner
    if args.command == "pretrain":
        config = UserTrainConfig(
            epochs=args.epochs or 50,
            batch_size=args.batch_size or 32,
            learning_rate=args.learning_rate or 1e-3,
            grad_clip_norm=args.grad_clip_norm,
            style_dim=args.style_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            deformer_type=args.deformer_type,
        )
        trainer = UserDeformationTrainer(
            config,
            user_dirs=user_dirs,
            ref_dir=args.ref_dir,
            output_dir=args.output_dir,
            device=args.device,
            num_workers=args.num_workers,
            use_aligner=args.use_aligner,
        )
    else:
        if not args.checkpoint.exists():
            sys.exit(f"Error: チェックポイントが見つかりません: {args.checkpoint}")
        config = FinetuneConfig(
            epochs=args.epochs or 20,
            batch_size=args.batch_size or 8,
            learning_rate=args.learning_rate or 5e-4,
            grad_clip_norm=args.grad_clip_norm,
        )
        trainer = DeformationFinetuner(
            config,
            pretrain_checkpoint=args.checkpoint,
            user_dirs=user_dirs,
            ref_dir=args.ref_dir,
            output_dir=args.output_dir,
            device=args.device,
            num_workers=args.num_workers,
            use_aligner=args.use_aligner,
        )

    losses = trainer.train()
    print(f"Training complete. Final loss: {losses[-1]:.4f}" if losses else "No epochs run.")
    print(f"Checkpoint saved to: {trainer.checkpoint_path}")
    return losses


if __name__ == "__main__":
    main()
