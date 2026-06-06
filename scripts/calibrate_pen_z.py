"""ペンの接触特性（芯が浮き始める Z）を測る水平線 G-code を生成する CLI。

単純な水平線を固定 Z で1本ずつ、上から下へ Z を変えて並べる。実機で描けば
「線が薄くなる/消える行」から芯が浮き始める Z が分かる。これが分かって初めて
払いの ``finish_lift_z`` を実測レンジに合わせられる（永など複雑な字でいきなり
詰めるより先にやるべき土台校正）。

例:
    uv run python scripts/calibrate_pen_z.py \
        --z-start 3.5 --z-stop 0.5 --z-step 0.2 \
        --output /mnt/nas/pen_plotter_calib/calib_pen_z.gcode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gcode.calibration import build_pen_z_calibration
from src.gcode.config import PlotterConfig
from src.gcode.generator import GCodeGenerator


def _frange(start: float, stop: float, step: float) -> list[float]:
    """start から stop まで step 刻み（降順可）。端点 stop を含める。"""
    n = int(round(abs(start - stop) / abs(step))) + 1
    sign = -1.0 if stop < start else 1.0
    return [round(start + sign * abs(step) * i, 3) for i in range(n)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--z-start", type=float, default=3.5, help="最上行のZ（接触側）")
    p.add_argument("--z-stop", type=float, default=0.5, help="最下行のZ（浮き側）")
    p.add_argument("--z-step", type=float, default=0.2, help="Z刻み")
    p.add_argument("--line-length", type=float, default=100.0, help="水平線の長さ(mm)")
    p.add_argument("--x-origin", type=float, default=40.0, help="線の始点X(mm)")
    p.add_argument("--y-top", type=float, default=280.0, help="最上行のY(mm)")
    p.add_argument("--row-spacing", type=float, default=12.0, help="行間隔(mm)")
    p.add_argument("--output", type=Path, default=Path("/tmp/calib_pen_z.gcode"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    z_values = _frange(args.z_start, args.z_stop, args.z_step)

    cfg = PlotterConfig()
    if not (0.0 <= args.y_top - (len(z_values) - 1) * args.row_spacing):
        print("警告: 行数×行間隔が紙の高さを超える可能性。--row-spacing を詰めること。")

    gcode = build_pen_z_calibration(
        z_values,
        base_config=cfg,
        line_length=args.line_length,
        x_origin=args.x_origin,
        y_top=args.y_top,
        row_spacing=args.row_spacing,
    )
    GCodeGenerator().save(gcode, args.output)

    print(f"保存: {args.output}")
    print(f"行数: {len(z_values)}  Z: {z_values[0]} → {z_values[-1]} (step {args.z_step})")
    print("上から下へ Z が下がる。線が消え始める行の Z が『芯が浮き始める高さ』。")
    print("各行の Z（上→下）:")
    print("  " + " ".join(str(z) for z in z_values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
