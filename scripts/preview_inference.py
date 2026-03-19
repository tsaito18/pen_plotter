"""事前学習済みチェックポイントから手書き文字を生成・プレビューする CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model.inference import StrokeInference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and preview handwritten characters from a pretrained checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/pretrain_checkpoint.pt"),
        help="Path to pretrained checkpoint file",
    )
    parser.add_argument(
        "--char",
        type=str,
        default="あ",
        help="Character to generate",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=Path("data/strokes"),
        help="Directory containing reference strokes",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (lower=more conservative, higher=more varied)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="Number of generation steps",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save plot to file instead of displaying",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate (columns in the plot)",
    )
    return parser.parse_args(argv)


def load_reference_strokes(ref_dir: Path, char: str) -> list[np.ndarray]:
    """参照ストロークをJSONファイルから読み込む。"""
    char_dir = ref_dir / char
    if not char_dir.exists():
        print(f"Error: reference directory not found: {char_dir}")
        print(
            f"Available characters: {sorted(d.name for d in ref_dir.iterdir() if d.is_dir())[:20]}..."
        )
        sys.exit(1)

    ref_files = sorted(char_dir.glob("*.json"))
    if not ref_files:
        print(f"Error: no stroke files found in {char_dir}")
        sys.exit(1)

    ref_data = json.loads(ref_files[0].read_text(encoding="utf-8"))
    strokes = []
    for stroke in ref_data["strokes"]:
        points = np.array([[pt["x"], pt["y"]] for pt in stroke])
        if len(points) >= 2:
            strokes.append(points)
    return strokes


def plot_strokes(ax: plt.Axes, strokes: list[np.ndarray], title: str) -> None:
    """ストロークを軸に描画する。"""
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(strokes), 1)))
    for i, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        ax.plot(stroke[:, 0], stroke[:, 1], "-", color=colors[i % len(colors)], linewidth=1.5)
        ax.plot(stroke[0, 0], stroke[0, 1], "o", color=colors[i % len(colors)], markersize=3)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def _configure_cjk_font() -> None:
    """CJK対応フォントが利用可能であればmatplotlibに設定する。"""
    import matplotlib.font_manager as fm

    cjk_candidates = ["Noto Sans CJK JP", "Noto Serif CJK JP", "IPAexGothic", "TakaoPGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in cjk_candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            return
    import warnings

    warnings.warn("No CJK font found — Japanese characters may not render correctly", stacklevel=2)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_cjk_font()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Character: {args.char}")
    print(f"Temperature: {args.temperature}")
    print(f"Steps: {args.num_steps}")

    ref_strokes = load_reference_strokes(args.ref_dir, args.char)
    print(f"Reference strokes: {len(ref_strokes)} strokes loaded")

    inference = StrokeInference(checkpoint_path=args.checkpoint)
    print("Model loaded successfully")

    from src.model.data_utils import strokes_to_deltas_from_arrays

    style_tensor = strokes_to_deltas_from_arrays(ref_strokes)  # (N, 3) with pen_state for StyleEncoder
    style_sample = style_tensor.unsqueeze(0)  # (1, N, 3)

    n_cols = 1 + args.num_samples
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    plot_strokes(axes[0], ref_strokes, f"Reference: {args.char}")

    for i in range(args.num_samples):
        print(f"Generating sample {i + 1}/{args.num_samples}...")
        generated = inference.generate(
            style_sample=style_sample,
            num_steps=args.num_steps,
            temperature=args.temperature,
            reference_strokes=ref_strokes,
        )
        label = f"Generated: {args.char}" if args.num_samples == 1 else f"Generated #{i + 1}"
        plot_strokes(axes[1 + i], generated, label)
        print(f"  → {len(generated)} strokes generated")

    plt.suptitle(
        f"Character: {args.char}  |  Temperature: {args.temperature}  |  Steps: {args.num_steps}",
        fontsize=12,
    )
    plt.tight_layout()

    if args.output:
        fig.savefig(str(args.output), dpi=150)
        print(f"Saved to {args.output}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
