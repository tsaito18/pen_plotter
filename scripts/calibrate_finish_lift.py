"""終端Zリフトのキャリブレーション用 G-code を生成する CLI。

同じ字を複数の ``finish_lift_z`` で横並び配置した1本の ``.gcode`` を出力する。
実機（Windows 側 ``run_plotter_gui.py``）で1回送信すれば、払い・はねが「抜け
始める高さ」を1枚の紙で目視比較できる。良かった Z をコードのデフォルトに焼く。

例:
    uv run python scripts/calibrate_finish_lift.py 永 \
        --z-values 2.8 2.5 2.3 2.0 1.8 \
        --kanjivg-dir data/strokes \
        --checkpoint data_examples/models/finetuned.pt \
        --user-strokes-dir data_examples/user_strokes \
        --output /tmp/calib_ei.gcode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gcode.calibration import build_calibration_gcode
from src.gcode.generator import GCodeGenerator
from src.handwriting.finishing import HANE, HARAI
from src.pipeline import PlotterPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("char", help="キャリブに使う1文字（払い・はねを含む字が良い。例: 永）")
    p.add_argument(
        "--z-values",
        type=float,
        nargs="+",
        default=[2.8, 2.5, 2.3, 2.0, 1.8],
        help="試す finish_lift_z のリスト（左から順に並ぶ）",
    )
    p.add_argument("--output", type=Path, default=Path("/tmp/calib_finish_lift.gcode"))
    p.add_argument("--kanjivg-dir", type=Path, default=Path("data/strokes"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--user-strokes-dir", type=Path, default=None)
    p.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="バリアント間の X 間隔(mm)。省略時は字幅の1.6倍",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    pipeline = PlotterPipeline(
        checkpoint_path=args.checkpoint,
        kanjivg_dir=args.kanjivg_dir,
        user_strokes_dir=args.user_strokes_dir,
    )
    pages = pipeline.typeset(args.char)
    if not pages[0]:
        print(f"文字を配置できなかった: {args.char!r}")
        return 1

    page = pipeline.render_page(pages[0])
    strokes, finishes = page.strokes, page.finishes
    if not strokes:
        print(f"ストロークが生成されなかった: {args.char!r}")
        return 1

    n_lift = sum(1 for f in finishes if f in (HARAI, HANE))
    if n_lift == 0:
        print(
            f"警告: {args.char!r} に払い・はねが無く Z リフトが出ない。"
            "永・木・道 など払いを含む字を使うこと。"
        )

    gcode = build_calibration_gcode(strokes, finishes, args.z_values, spacing_mm=args.spacing)
    GCodeGenerator().save(gcode, args.output)

    print(f"保存: {args.output}")
    print(f"文字: {args.char}  ストローク数: {len(strokes)}  払い/はね: {n_lift}")
    print("左から順の finish_lift_z:")
    for i, z in enumerate(args.z_values):
        print(f"  [{i + 1}] z={z}")
    print("実機で送信し、払いが一番きれいに抜ける版の z をデフォルトに採用する。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
