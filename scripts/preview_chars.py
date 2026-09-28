"""ML 変形の品質確認: 文字ごとに参照字形と変形サンプルをグリッド画像に並べる。

例:
    uv run python scripts/preview_chars.py --chars あいう学文字 \
        --checkpoint data/models/pretrain_checkpoint.pt --user-strokes-dir data/user_strokes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.collector.profiles import resolve_character_root
from src.glyphs.sources import KanjiVGStore
from src.model.inference import StrokeInference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("data/models/pretrain_checkpoint.pt")
    )
    parser.add_argument("--ref-dir", type=Path, default=Path("data/strokes"))
    parser.add_argument("--user-strokes-dir", type=Path, default=Path("data/user_strokes"))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--chars", default="あいうえお学文字手書")
    parser.add_argument("--samples", type=int, default=3, help="文字あたりのサンプル数")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/preview_chars.png"))
    return parser.parse_args(argv)


def _plot(ax: plt.Axes, strokes: list[np.ndarray], title: str) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(strokes), 1)))
    for i, s in enumerate(strokes):
        ax.plot(s[:, 0], s[:, 1], "-", color=colors[i % len(colors)], linewidth=1.5)
        ax.plot(s[0, 0], s[0, 1], "o", color=colors[i % len(colors)], markersize=3)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.checkpoint.exists():
        sys.exit(f"Error: チェックポイントが見つかりません: {args.checkpoint}")
    store = KanjiVGStore(args.ref_dir)
    refs = {ch: store.load(ch)[0] for ch in args.chars}
    chars = [ch for ch, ref in refs.items() if ref]
    if missing := [ch for ch, ref in refs.items() if not ref]:
        print(f"参照字形が無いため除外: {''.join(missing)}")
    if not chars:
        sys.exit("Error: 描ける文字がありません")

    np.random.seed(args.seed)
    user_dir = resolve_character_root(args.user_strokes_dir, args.profile)
    engine = StrokeInference.from_user_strokes(args.checkpoint, user_dir)

    fig, axes = plt.subplots(
        len(chars),
        1 + args.samples,
        figsize=(3 * (1 + args.samples), 3 * len(chars)),
        squeeze=False,
    )
    for row, ch in enumerate(chars):
        ref = refs[ch]
        _plot(axes[row][0], ref, "reference")
        for col in range(args.samples):
            generated = engine.generate(ref, temperature=args.temperature)
            _plot(axes[row][1 + col], generated, f"#{col + 1}")
    fig.suptitle(f"{args.checkpoint.name}  temperature={args.temperature}")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=120)
    plt.close(fig)
    print(f"保存: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
