"""終端Zリフトのキャリブレーション用 G-code 生成。

同じ字を複数の ``finish_lift_z`` で横並びに配置した1本の G-code を生成する。
実機で1回送信すれば、払い・はねの「抜け始める高さ」を1枚の紙で目視比較できる。
確定した値を :class:`PlotterConfig` のデフォルトに焼き戻す運用を想定。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt

from src.gcode.config import PlotterConfig
from src.gcode.generator import GCodeGenerator

Stroke = npt.NDArray[np.float64]


def _strokes_width(strokes: list[Stroke]) -> float:
    """全ストロークを内包する X 方向の幅（mm）。"""
    points = np.concatenate(strokes, axis=0)
    return float(points[:, 0].max() - points[:, 0].min())


def build_pen_z_calibration(
    z_values: list[float],
    *,
    base_config: PlotterConfig | None = None,
    line_length: float = 100.0,
    x_origin: float = 40.0,
    y_top: float = 280.0,
    row_spacing: float = 12.0,
) -> list[str]:
    """ペンの接触特性（芯が浮き始める Z）を測る水平線の G-code を生成する。

    1 本ずつ固定 Z で水平線を引き、上から下へ Z を変えて並べる。実機で描けば
    「線が薄くなる/消える行」から芯が浮き始める Z が読み取れる。払いの
    ``finish_lift_z`` をその実測レンジに合わせる土台にする。

    各行は ``G0`` で始点上空へ移動 → ``G1 Z{z}`` で目標 Z へ下ろす →
    ``G1 X.. Y.. Z{z}`` で水平移動、の順。``z`` が高いほど濃く、低いほど浮いて
    消える。

    Args:
        z_values: 各行の Z 値（上から下の順）。空なら描画なし。
        base_config: プロッタ設定（速度・フィードの基準）。
        line_length: 各水平線の長さ(mm)。
        x_origin: 線の始点 X(mm)。
        y_top: 最上行の Y(mm)。行ごとに ``row_spacing`` ずつ下げる。
        row_spacing: 行間隔(mm)。

    Returns:
        G-code 行のリスト。
    """
    cfg = base_config or PlotterConfig()
    gen = GCodeGenerator(cfg)
    lines: list[str] = gen._header()

    for i, z in enumerate(z_values):
        y = y_top - i * row_spacing
        x1 = x_origin + line_length
        lines.append(cfg.pen_up_command)
        lines.append(
            f"G0 X{gen._format_coord(x_origin)} Y{gen._format_coord(y)} F{cfg.travel_speed:.0f}"
        )
        # 目標 Z まで下ろしてから水平に引く（線全体を固定 Z で描く）。
        lines.append(f"G1 Z{gen._format_coord(z)} F{cfg.pen_z_feed:.0f}")
        lines.append(
            f"G1 X{gen._format_coord(x1)} Y{gen._format_coord(y)} "
            f"Z{gen._format_coord(z)} F{cfg.draw_speed:.0f}"
        )

    lines.extend(gen._footer())
    return lines


def build_calibration_gcode(
    strokes: list[Stroke],
    finishes: list[str],
    z_values: list[float],
    *,
    base_config: PlotterConfig | None = None,
    spacing_mm: float | None = None,
) -> list[str]:
    """同じ字を複数の ``finish_lift_z`` で横並び配置した1本の G-code を返す。

    各バリアントは X 方向に ``spacing_mm`` ずつオフセットして並べる。``finishes``
    で払い・はねに指定された終端だけ Z リフト量が ``z_values[i]`` に応じて変わる
    （とめ/none は不変）。ヘッダ（ホーミング）・フッタは全体で1回ずつ。

    Args:
        strokes: 1文字ぶんの配置後ストローク列（``(N, 2)`` mm 座標）。
        finishes: ``strokes`` と並走する筆法ラベル。
        z_values: 試す ``finish_lift_z`` のリスト。左から順に並ぶ。空なら描画なし。
        base_config: ベースのプロッタ設定。``finish_lift_z`` 以外は共通で使う。
        spacing_mm: バリアント間の X 間隔。省略時は字幅の 1.6 倍。

    Returns:
        G-code 行のリスト。
    """
    base_config = base_config or PlotterConfig()
    header_gen = GCodeGenerator(base_config)
    lines: list[str] = header_gen._header()

    if strokes and z_values:
        gap = spacing_mm if spacing_mm is not None else _strokes_width(strokes) * 1.6
        for i, z in enumerate(z_values):
            gen = GCodeGenerator(replace(base_config, finish_lift_z=z))
            offset = np.array([i * gap, 0.0], dtype=np.float64)
            for stroke, finish in zip(strokes, finishes):
                lines.extend(gen._stroke_to_gcode(stroke + offset, finish=finish))

    lines.extend(header_gen._footer())
    return lines
