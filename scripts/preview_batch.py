"""複数文字を一括生成し、品質比較用のグリッド画像を出力する CLI。"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np

from src.model.data_utils import strokes_to_deltas_from_arrays
from src.model.inference import StrokeInference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a grid of handwritten characters for quality comparison"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/pretrain_checkpoint.pt"),
        help="Path to pretrained checkpoint file",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=Path("data/strokes"),
        help="Directory containing reference strokes",
    )
    parser.add_argument(
        "--chars",
        type=str,
        default="あいうえお学文字手書",
        help="Characters to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="Number of generation steps per character",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG file path (default: auto-numbered in /tmp/preview_NNN.png)",
    )
    parser.add_argument(
        "--samples-per-char",
        type=int,
        default=2,
        help="Number of generated samples per character",
    )
    return parser.parse_args(argv)


def _configure_cjk_font() -> None:
    """CJK対応フォントが利用可能であればmatplotlibに設定する。"""
    import matplotlib.font_manager as fm

    cjk_candidates = ["Noto Sans CJK JP", "Noto Serif CJK JP", "IPAexGothic", "TakaoPGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in cjk_candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            return
    warnings.warn("No CJK font found — Japanese characters may not render correctly", stacklevel=2)


def load_reference_strokes(ref_dir: Path, char: str) -> list[np.ndarray] | None:
    """参照ストロークを読み込む。見つからない場合は None を返す。"""
    char_dir = ref_dir / char
    if not char_dir.exists():
        return None

    ref_files = sorted(char_dir.glob("*.json"))
    if not ref_files:
        return None

    ref_data = json.loads(ref_files[0].read_text(encoding="utf-8"))
    strokes = []
    for stroke in ref_data["strokes"]:
        points = np.array([[pt["x"], pt["y"]] for pt in stroke])
        if len(points) >= 2:
            strokes.append(points)
    return strokes if strokes else None


def plot_strokes(ax: plt.Axes, strokes: list[np.ndarray], title: str) -> None:
    """ストロークを軸に描画する。"""
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(strokes), 1)))
    for i, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        ax.plot(stroke[:, 0], stroke[:, 1], "-", color=colors[i % len(colors)], linewidth=1.5)
        ax.plot(stroke[0, 0], stroke[0, 1], "o", color=colors[i % len(colors)], markersize=3)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)


def _auto_output_path() -> Path:
    """Generate auto-numbered output path: /tmp/preview_001.png, 002, ..."""
    for i in range(1, 1000):
        p = Path(f"/tmp/preview_{i:03d}.png")
        if not p.exists():
            return p
    return Path("/tmp/preview_latest.png")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output is None:
        args.output = _auto_output_path()
    _configure_cjk_font()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    chars = list(args.chars)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Characters: {''.join(chars)} ({len(chars)} chars)")
    print(f"Temperature: {args.temperature}, Steps: {args.num_steps}")
    print(f"Samples per char: {args.samples_per_char}")

    valid_chars: list[str] = []
    ref_strokes_map: dict[str, list[np.ndarray]] = {}

    for ch in chars:
        ref = load_reference_strokes(args.ref_dir, ch)
        if ref is None:
            warnings.warn(f"Skipping '{ch}': no reference strokes found in {args.ref_dir / ch}")
            continue
        valid_chars.append(ch)
        ref_strokes_map[ch] = ref

    if not valid_chars:
        print("Error: no valid characters found. Check --ref-dir and --chars.")
        sys.exit(1)

    print(f"Valid characters: {''.join(valid_chars)} ({len(valid_chars)}/{len(chars)})")
    print(f"Loading model...")

    inference = StrokeInference(checkpoint_path=args.checkpoint)
    print("Model loaded successfully")

    n_rows = len(valid_chars)
    n_cols = 1 + args.samples_per_char
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row, ch in enumerate(valid_chars):
        ref = ref_strokes_map[ch]
        plot_strokes(axes[row][0], ref, f"Ref: {ch}")

        style_tensor = strokes_to_deltas_from_arrays(ref)
        style_sample = style_tensor.unsqueeze(0)

        for col in range(args.samples_per_char):
            print(f"  Generating '{ch}' sample {col + 1}/{args.samples_per_char}...")
            generated = inference.generate(
                style_sample=style_sample,
                num_steps=args.num_steps,
                temperature=args.temperature,
                reference_strokes=ref,
            )
            plot_strokes(axes[row][1 + col], generated, f"Gen #{col + 1}: {ch}")

    fig.suptitle(
        f"Batch Preview  |  temp={args.temperature}  steps={args.num_steps}",
        fontsize=14,
    )
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=150)
    print(f"Saved grid image to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
