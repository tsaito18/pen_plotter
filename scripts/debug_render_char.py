"""Dump rendering pipeline stages for selected characters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.collector.data_format import StrokeSample
from src.layout.typesetter import CharPlacement
from src.model.augmentation import HandwritingAugmenter
from src.ui.stroke_renderer import StrokeRenderer

Stroke = np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump StrokeRenderer pipeline stages as PNGs for one or more characters."
    )
    parser.add_argument("chars", help="Characters to inspect, e.g. 均満")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp"),
        help="Directory for debug PNGs",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional model checkpoint. Inference stages are skipped when absent.",
    )
    parser.add_argument(
        "--kanjivg-dir",
        type=Path,
        default=Path("data/strokes"),
        help="Directory containing KanjiVG StrokeSample JSON files",
    )
    parser.add_argument(
        "--user-strokes-dir",
        type=Path,
        default=_default_user_strokes_dir(),
        help="Directory containing user stroke samples for style loading",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    return parser.parse_args(argv)


def _default_user_strokes_dir() -> Path:
    taiga_dir = Path("data/user_strokes/taiga")
    if taiga_dir.exists():
        return taiga_dir
    return Path("data/user_strokes")


def _load_reference_strokes(kanjivg_dir: Path, char: str) -> list[Stroke] | None:
    char_dir = kanjivg_dir / char
    if not char_dir.is_dir():
        return None
    json_files = sorted(char_dir.glob(f"{char}_*.json"))
    if not json_files:
        json_files = sorted(char_dir.glob("*.json"))
    if not json_files:
        return None

    sample = StrokeSample.load(json_files[0])
    strokes: list[Stroke] = []
    for stroke_points in sample.strokes:
        stroke = np.array([[p.x, p.y] for p in stroke_points], dtype=np.float64)
        if len(stroke) >= 2:
            strokes.append(stroke)
    return strokes or None


def _bbox(strokes: list[Stroke]) -> tuple[float, float, float, float] | None:
    if not strokes:
        return None
    points = np.concatenate(strokes, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1]))


def _format_bbox(bounds: tuple[float, float, float, float] | None) -> str:
    if bounds is None:
        return "none"
    x0, y0, x1, y1 = bounds
    return f"({x0:.3f}, {y0:.3f})-({x1:.3f}, {y1:.3f})"


def _print_stage(char: str, stage: str, strokes: list[Stroke]) -> None:
    print(
        f"char={char} stage={stage} strokes={len(strokes)} bbox={_format_bbox(_bbox(strokes))}"
    )


def _next_output_path(output_dir: Path, char: str, stage: str) -> Path:
    index = 1
    while True:
        path = output_dir / f"debug_render_{char}_{stage}_{index}.png"
        if not path.exists():
            return path
        index += 1


def _plot_strokes(strokes: list[Stroke], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(strokes), 1)))

    for index, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        color = colors[index % len(colors)]
        ax.plot(stroke[:, 0], stroke[:, 1], "-", color=color, linewidth=1.4)
        ax.plot(stroke[0, 0], stroke[0, 1], "o", color=color, markersize=3)

    bounds = _bbox(strokes)
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        pad = max(x1 - x0, y1 - y0, 1.0) * 0.08
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"saved={output_path}")


def _emit_stage(output_dir: Path, char: str, stage: str, strokes: list[Stroke]) -> Path:
    _print_stage(char, stage, strokes)
    output_path = _next_output_path(output_dir, char, stage)
    _plot_strokes(strokes, f"{char}: {stage}", output_path)
    return output_path


def _checkpoint_to_use(checkpoint: Path | None) -> Path | None:
    if checkpoint is None:
        return None
    if checkpoint.exists():
        return checkpoint
    print(f"skip inference: checkpoint not found: {checkpoint}")
    return None


def _build_renderer(args: argparse.Namespace) -> StrokeRenderer:
    checkpoint = _checkpoint_to_use(args.checkpoint)
    return StrokeRenderer(
        checkpoint_path=checkpoint,
        kanjivg_dir=args.kanjivg_dir,
        user_strokes_dir=args.user_strokes_dir,
        augmenter=HandwritingAugmenter(seed=args.seed),
    )


def dump_char(args: argparse.Namespace, renderer: StrokeRenderer, char: str) -> None:
    reference = _load_reference_strokes(args.kanjivg_dir, char)
    if reference is None:
        print(f"char={char} skip: reference not found in {args.kanjivg_dir}")
        return

    placement = CharPlacement(char=char, x=20.0, y=20.0, font_size=7.0)

    _emit_stage(args.output_dir, char, "reference", reference)

    inference_strokes: list[Stroke] | None = None
    if renderer._inference is None:
        print("skip inference: no usable checkpoint loaded")
    else:
        inference_strokes = renderer._inference.generate(
            renderer._style_sample,
            num_steps=50,
            temperature=renderer._temperature,
            reference_strokes=reference,
        )
        _emit_stage(args.output_dir, char, "inference_raw", inference_strokes)

    source_for_position = inference_strokes if inference_strokes is not None else reference
    positioned_stage = "positioned" if inference_strokes is not None else "positioned_reference"
    positioned = renderer._position_strokes(source_for_position, placement)
    _emit_stage(args.output_dir, char, positioned_stage, positioned)

    distorted = renderer._apply_distortion(positioned)
    _emit_stage(args.output_dir, char, "distorted", distorted)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    renderer = _build_renderer(args)
    for char in args.chars:
        dump_char(args, renderer, char)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
