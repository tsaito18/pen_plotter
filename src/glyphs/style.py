"""本人の書き癖（横画の右上がり・縦画の傾き）を推定し、参照字形に掛ける。

本人サンプルの無い字は KanjiVG の参照字形（ML 変形を含む）で描くため、教科書体の
まっすぐな字になる。本人の字と KanjiVG を画ごとに比べると、横画は一貫して右上がり、
縦画は一貫して傾いている（例: 本人データ 755 サンプルで横画 +6.9°、縦画 -2.3°）。
この 2 つの角度差を 1 つの 2x2 変換にして参照字形へ掛けると、字形を崩さずに
本人らしい傾きになる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry import Stroke
from src.glyphs.sources import KanjiVGStore, UserStrokeDB

# 参照の画が「横画/縦画」とみなす向きの許容(度)、比べる画の最小長（字の長辺比）
_DIRECTION_TOLERANCE_DEG = 20.0
_MIN_STROKE_RATIO = 0.25
# 書き順・画の対応が崩れた組（角度差がこれを超える）は外れ値として捨てる
_MAX_DIFF_DEG = 30.0


@dataclass(frozen=True)
class WritingStyle:
    """書き癖の角度差（度、反時計回りが正。Y-UP）。

    Attributes:
        horizontal_rise_deg: 横画の向きの差。正なら参照より右上がり。
        vertical_lean_deg: 縦画（下向き）の向きの差。負なら下端が左＝上が右へ傾く。
    """

    horizontal_rise_deg: float = 0.0
    vertical_lean_deg: float = 0.0

    def matrix(self) -> np.ndarray:
        """横向き (1,0) を横画の角度だけ、上向き (0,1) を縦画の角度だけ回す 2x2 行列。"""
        h = np.radians(self.horizontal_rise_deg)
        v = np.radians(self.vertical_lean_deg)
        return np.array([[np.cos(h), -np.sin(v)], [np.sin(h), np.cos(v)]])

    def apply(self, strokes: list[Stroke]) -> list[Stroke]:
        """字形 bbox の中心を固定して変換する（恒等なら入力をそのまま返す）。"""
        if not strokes or (self.horizontal_rise_deg == 0 and self.vertical_lean_deg == 0):
            return strokes
        pts = np.concatenate(strokes, axis=0)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2
        m = self.matrix()
        return [(s - center) @ m.T + center for s in strokes]


def _direction_deg(stroke: Stroke) -> float:
    d = stroke[-1] - stroke[0]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def estimate_writing_style(
    user_db: UserStrokeDB, kanjivg: KanjiVGStore, min_pairs: int = 30
) -> WritingStyle:
    """本人サンプルと KanjiVG の同じ字を、画の順番どおりに対応させて角度差の中央値を取る。

    画数が一致するサンプルだけを使う（収集 UI はお手本の書き順で書かせているため、
    画数が同じなら i 画目どうしが対応する）。対応する画が ``min_pairs`` 未満の向きは 0。
    """
    horizontal: list[float] = []
    vertical: list[float] = []
    for char, samples in user_db.items():
        reference, _ = kanjivg.load(char)
        if reference is None or not _is_cjk(char):
            continue
        ref_span = float(np.ptp(np.concatenate(reference), axis=0).max())
        for sample in samples:
            if len(sample) != len(reference):
                continue
            for user_stroke, ref_stroke in zip(sample, reference, strict=True):
                if len(user_stroke) < 2 or len(ref_stroke) < 2:
                    continue
                if np.linalg.norm(ref_stroke[-1] - ref_stroke[0]) < _MIN_STROKE_RATIO * ref_span:
                    continue
                flipped = user_stroke * np.array([1.0, -1.0])  # 筆跡は Y-DOWN
                ref_angle = _direction_deg(ref_stroke)
                diff = (_direction_deg(flipped) - ref_angle + 180.0) % 360.0 - 180.0
                if abs(diff) > _MAX_DIFF_DEG:
                    continue
                if abs(ref_angle) < _DIRECTION_TOLERANCE_DEG:
                    horizontal.append(diff)
                elif abs(ref_angle + 90.0) < _DIRECTION_TOLERANCE_DEG:
                    vertical.append(diff)
    return WritingStyle(
        horizontal_rise_deg=float(np.median(horizontal)) if len(horizontal) >= min_pairs else 0.0,
        vertical_lean_deg=float(np.median(vertical)) if len(vertical) >= min_pairs else 0.0,
    )


def _is_cjk(char: str) -> bool:
    """かな・漢字（書き癖の推定に使う字）か。"""
    return len(char) == 1 and (0x3040 <= ord(char) <= 0x30FF or 0x4E00 <= ord(char) <= 0x9FFF)
