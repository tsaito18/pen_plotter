"""KanjiVG SVGファイルからストローク座標列を抽出するパーサー。"""

from __future__ import annotations

import re
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


def _is_path_command(token: str) -> bool:
    return bool(re.fullmatch(r"[MmLlCcSsZz]", token))


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
    cmd: str | None = None

    while i < len(tokens):
        if _is_path_command(tokens[i]):
            cmd = tokens[i]
            i += 1
        elif cmd is None:
            break

        if cmd in ("M", "m"):
            is_relative = cmd == "m"
            is_first_pair = True
            while i + 1 < len(tokens) and not _is_path_command(tokens[i]):
                x, y = float(tokens[i]), float(tokens[i + 1])
                i += 2
                if is_relative:
                    x += current[0]
                    y += current[1]
                current = (x, y)
                points.append(current)
                if is_first_pair:
                    is_first_pair = False
                    last_control = None
                    cmd = "l" if is_relative else "L"
                    continue
                last_control = None
            last_control = None

        elif cmd in ("L", "l"):
            while i + 1 < len(tokens) and not _is_path_command(tokens[i]):
                x, y = float(tokens[i]), float(tokens[i + 1])
                i += 2
                if cmd == "l":
                    x += current[0]
                    y += current[1]
                current = (x, y)
                points.append(current)
            last_control = None

        elif cmd in ("C", "c"):
            while i + 5 < len(tokens) and not _is_path_command(tokens[i]):
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
            while i + 3 < len(tokens) and not _is_path_command(tokens[i]):
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


# <path> タグから d / kvg:type 属性を抽出する正規表現。
# KanjiVG SVG は xmlns:kvg を宣言しない場合があるため、ET ではなく正規表現で
# d と kvg:type を同時抽出し、ストローク列と筆画タイプ列の順序・フィルタを
# 完全に一致させる。
_PATH_TAG_RE = re.compile(r"<path\b[^>]*>", re.DOTALL)
_D_ATTR_RE = re.compile(r'\bd="([^"]*)"')
_KVG_TYPE_RE = re.compile(r'\bkvg:type="([^"]*)"')


class KanjiVGParser:
    """KanjiVG形式のSVGからストローク座標列を抽出する。"""

    def parse_svg_with_types(self, svg_string: str) -> tuple[list[NDArray[np.float64]], list[str]]:
        """SVG文字列からストロークと筆画タイプ(kvg:type)を抽出。

        Args:
            svg_string: KanjiVG形式のSVG文字列。

        Returns:
            (strokes, stroke_types) のタプル。両リストは index が対応し長さが
            等しい。stroke_types は各ストロークの raw kvg:type 文字列で、属性が
            無い path は空文字列。座標が空の path はどちらからも除外される。
        """
        strokes: list[NDArray[np.float64]] = []
        stroke_types: list[str] = []
        for tag in _PATH_TAG_RE.findall(svg_string):
            d_match = _D_ATTR_RE.search(tag)
            if d_match is None:
                continue
            points = parse_svg_path(d_match.group(1))
            if len(points) == 0:
                continue
            type_match = _KVG_TYPE_RE.search(tag)
            strokes.append(points)
            stroke_types.append(type_match.group(1) if type_match else "")
        return strokes, stroke_types

    def parse_file_with_types(
        self, path: Path | str
    ) -> tuple[list[NDArray[np.float64]], list[str]]:
        """SVGファイルからストロークと筆画タイプを抽出。"""
        svg_text = Path(path).read_text(encoding="utf-8")
        return self.parse_svg_with_types(svg_text)

    def parse_svg(self, svg_string: str) -> list[NDArray[np.float64]]:
        """SVG文字列からストロークリストを抽出。"""
        strokes, _ = self.parse_svg_with_types(svg_string)
        return strokes

    def parse_file(self, path: Path | str) -> list[NDArray[np.float64]]:
        """SVGファイルからストロークリストを抽出。"""
        strokes, _ = self.parse_file_with_types(path)
        return strokes

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
