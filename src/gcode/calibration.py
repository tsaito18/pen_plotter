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
