"""KanjiVG SVGファイルからストローク座標列を抽出するパーサー。"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _parse_coordinate_pair(text: str) -> tuple[float, float]:
    parts = text.split(",")
    return float(parts[0]), float(parts[1])


def _sample_cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    num_points: int = 10,
) -> list[tuple[float, float]]:
    """3次ベジェ曲線をnum_points個の点にサンプリング。"""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def _tokenize_svg_path(d: str) -> list[str]:
    """SVG pathのd属性をコマンドと数値トークンに分割。"""
    return re.findall(r"[MmLlCcSsZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)


def parse_svg_path(d: str) -> NDArray[np.float64]:
    """SVG pathのd属性文字列をパースして(N, 2)座標配列に変換。

    M(moveto), L(lineto), C(cubic bezier), S(smooth cubic bezier)をサポート。
    """
    tokens = _tokenize_svg_path(d)
    if not tokens:
        return np.empty((0, 2), dtype=np.float64)

    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    last_control: tuple[float, float] | None = None
    i = 0

    while i < len(tokens):
        cmd = tokens[i]
        i += 1

        if cmd in ("M", "m"):
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if cmd == "m":
                x += current[0]
                y += current[1]
            current = (x, y)
            points.append(current)
            last_control = None

        elif cmd in ("L", "l"):
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if cmd == "l":
                x += current[0]
                y += current[1]
            current = (x, y)
            points.append(current)
            last_control = None

        elif cmd in ("C", "c"):
            x1, y1 = float(tokens[i]), float(tokens[i + 1])
            x2, y2 = float(tokens[i + 2]), float(tokens[i + 3])
            x3, y3 = float(tokens[i + 4]), float(tokens[i + 5])
            i += 6
            if cmd == "c":
                x1 += current[0]
                y1 += current[1]
                x2 += current[0]
                y2 += current[1]
                x3 += current[0]
                y3 += current[1]
            sampled = _sample_cubic_bezier(current, (x1, y1), (x2, y2), (x3, y3))
            points.extend(sampled[1:])
            last_control = (x2, y2)
            current = (x3, y3)

        elif cmd in ("S", "s"):
            x2, y2 = float(tokens[i]), float(tokens[i + 1])
            x3, y3 = float(tokens[i + 2]), float(tokens[i + 3])
            i += 4
            if cmd == "s":
                x2 += current[0]
                y2 += current[1]
                x3 += current[0]
                y3 += current[1]
            if last_control is not None:
                x1 = 2 * current[0] - last_control[0]
                y1 = 2 * current[1] - last_control[1]
            else:
                x1, y1 = current
            sampled = _sample_cubic_bezier(current, (x1, y1), (x2, y2), (x3, y3))
            points.extend(sampled[1:])
            last_control = (x2, y2)
            current = (x3, y3)

        elif cmd in ("Z", "z"):
            last_control = None

    return np.array(points, dtype=np.float64) if points else np.empty((0, 2), dtype=np.float64)


class KanjiVGParser:
    """KanjiVG形式のSVGからストローク座標列を抽出する。"""

    SVG_NS = "http://www.w3.org/2000/svg"

    @staticmethod
    def _strip_kvg_namespace(svg_string: str) -> str:
        """KanjiVG固有の未宣言kvg:プレフィックスを除去してパース可能にする。"""
        return re.sub(r'\bkvg:\w+="[^"]*"', "", svg_string)

    def parse_svg(self, svg_string: str) -> list[NDArray[np.float64]]:
        """SVG文字列からストロークリストを抽出。"""
        cleaned = self._strip_kvg_namespace(svg_string)
        root = ET.fromstring(cleaned)
        paths = root.findall(f".//{{{self.SVG_NS}}}path")
        strokes = []
        for path in paths:
            d = path.get("d", "")
            points = parse_svg_path(d)
            if len(points) > 0:
                strokes.append(points)
        return strokes

    def parse_file(self, path: Path | str) -> list[NDArray[np.float64]]:
        """SVGファイルからストロークリストを抽出。"""
        svg_text = Path(path).read_text(encoding="utf-8")
        return self.parse_svg(svg_text)

    def normalize(
        self, strokes: list[NDArray[np.float64]], target_size: float = 1.0
    ) -> list[NDArray[np.float64]]:
        """全ストロークを0〜target_sizeの範囲に正規化。

        アスペクト比を保持し、長辺をtarget_sizeに合わせる。
        """
        if not strokes:
            return []

        all_points = np.concatenate(strokes, axis=0)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        ranges = maxs - mins
        scale = target_size / ranges.max() if ranges.max() > 0 else 1.0

        return [(stroke - mins) * scale for stroke in strokes]
