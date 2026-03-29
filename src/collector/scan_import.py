"""Scan image importer: extract characters from scanned handwriting images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from skimage.morphology import skeletonize


class ScanImporter:
    """スキャン画像から文字を抽出し、ストローク軌跡を復元する。"""

    def __init__(self, line_spacing_mm: float = 8.0) -> None:
        self.line_spacing_mm = line_spacing_mm

    def load_image(self, path: Path) -> np.ndarray:
        """画像をグレースケールで読み込む。"""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img

    def deskew(self, gray: np.ndarray) -> np.ndarray:
        """Hough変換で傾きを検出し、補正した画像を返す。"""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=200,
            minLineLength=gray.shape[1] // 4, maxLineGap=20,
        )
        if lines is None:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 10:
                angles.append(angle)

        if not angles or abs(np.median(angles)) < 0.1:
            return gray

        skew_angle = float(np.median(angles))
        h, w = gray.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        return cv2.warpAffine(gray, matrix, (w, h), borderValue=255)

    def detect_lines(self, gray: np.ndarray) -> list[int]:
        """罫線のY座標リストを返す。

        2段階アプローチ:
        1. 合成画像向け: adaptive threshold + 射影プロファイル（罫線が明確な場合）
        2. スキャン画像向け: 行平均輝度のピーク検出（罫線が薄い場合）
        """
        height, width = gray.shape

        # --- Strategy 1: 明確な罫線（合成画像・濃い罫線）---
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )
        h_profile = np.sum(binary, axis=1).astype(float)

        threshold = width * 0.3 * 255
        candidates = np.where(h_profile > threshold)[0]

        if len(candidates) > 0:
            lines = self._group_candidates(candidates)
            if len(lines) >= 3:
                filtered = self._filter_by_spacing(lines)
                if len(filtered) >= 3:
                    return filtered

        # --- Strategy 2: 薄い罫線（実スキャン画像）---
        margin = min(width // 10, 100)
        center = gray[:, margin : width - margin]
        row_mean = np.mean(center.astype(float), axis=1)
        smoothed = uniform_filter1d(row_mean, size=5)

        # 暗い方向のピーク = 罫線候補
        peaks, _ = find_peaks(-smoothed, distance=30, prominence=0.3)
        if len(peaks) < 3:
            return []

        # 最頻間隔を求めて等間隔の罫線をフィルタ
        spacings = np.diff(peaks)
        median_spacing = np.median(spacings)

        filtered = [int(peaks[0])]
        for i in range(1, len(peaks)):
            gap = peaks[i] - filtered[-1]
            if abs(gap - median_spacing) < median_spacing * 0.4:
                filtered.append(int(peaks[i]))
            elif gap > median_spacing * 1.3:
                filtered.append(int(peaks[i]))

        return filtered if len(filtered) >= 2 else []

    @staticmethod
    def _group_candidates(candidates: np.ndarray) -> list[int]:
        """近接するY座標をグループ化して中心を返す。"""
        lines: list[int] = []
        group_start = candidates[0]
        prev = candidates[0]
        for y in candidates[1:]:
            if y - prev > 5:
                lines.append(int((group_start + prev) // 2))
                group_start = y
            prev = y
        lines.append(int((group_start + prev) // 2))
        return lines

    @staticmethod
    def _filter_by_spacing(lines: list[int]) -> list[int]:
        """等間隔性でフィルタ。"""
        spacings = np.diff(lines)
        median_spacing = np.median(spacings)
        filtered = [lines[0]]
        for i in range(1, len(lines)):
            spacing = lines[i] - filtered[-1]
            if abs(spacing - median_spacing) < median_spacing * 0.5:
                filtered.append(lines[i])
            elif spacing > median_spacing * 1.3:
                filtered.append(lines[i])
        return filtered

    def extract_chars_from_line(
        self, gray: np.ndarray, y_top: int, y_bottom: int
    ) -> list[np.ndarray]:
        """1行分の画像から文字セルを切り出す。"""
        line_img = gray[y_top:y_bottom, :]
        if line_img.size == 0:
            return []

        binary = cv2.adaptiveThreshold(
            line_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

        # 垂直方向の射影プロファイル
        v_profile = np.sum(binary, axis=0).astype(float)
        height = line_img.shape[0]
        threshold = height * 0.05 * 255

        # インク領域（射影 > threshold）のランを検出
        is_ink = v_profile > threshold
        chars: list[np.ndarray] = []

        in_char = False
        x_start = 0
        for x in range(len(is_ink)):
            if is_ink[x] and not in_char:
                x_start = x
                in_char = True
            elif not is_ink[x] and in_char:
                in_char = False
                char_width = x - x_start
                if char_width > 3:
                    cell = gray[y_top:y_bottom, x_start:x]
                    chars.append(cell)
        if in_char:
            char_width = len(is_ink) - x_start
            if char_width > 3:
                cell = gray[y_top:y_bottom, x_start : len(is_ink)]
                chars.append(cell)

        return chars

    def extract_all_chars(self, image_path: Path) -> list[list[np.ndarray]]:
        """画像から全文字を行ごとに抽出。list[行][文字] = numpy画像。"""
        gray = self.load_image(image_path)
        gray = self.deskew(gray)
        lines = self.detect_lines(gray)

        if len(lines) < 2:
            return []

        result: list[list[np.ndarray]] = []
        for i in range(len(lines) - 1):
            y_top = lines[i]
            y_bottom = lines[i + 1]
            chars = self.extract_chars_from_line(gray, y_top, y_bottom)
            if chars:
                result.append(chars)

        return result

    def image_to_strokes(self, char_img: np.ndarray) -> list[np.ndarray]:
        """文字画像から骨格化→ストローク点列を復元する。"""
        if char_img.ndim == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_bool = binary > 0

        if not np.any(binary_bool):
            return []

        skeleton = skeletonize(binary_bool)

        labeled, n_components = ndimage.label(skeleton)

        strokes: list[np.ndarray] = []
        for comp_id in range(1, n_components + 1):
            component_mask = labeled == comp_id
            points = self._trace_skeleton_component(component_mask)
            if len(points) >= 3:
                strokes.append(points)

        return strokes

    def _trace_skeleton_component(self, skeleton_region: np.ndarray) -> np.ndarray:
        """骨格の1連結成分をトレースして順序付き点列にする。"""
        ys, xs = np.where(skeleton_region)
        if len(ys) == 0:
            return np.empty((0, 2), dtype=float)

        if len(ys) == 1:
            return np.array([[xs[0], ys[0]]], dtype=float)

        # 隣接関係を構築
        pixel_set = set(zip(ys.tolist(), xs.tolist()))
        neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for y, x in pixel_set:
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nb = (y + dy, x + dx)
                    if nb in pixel_set:
                        nbrs.append(nb)
            neighbors[(y, x)] = nbrs

        # 端点（隣接数=1）を開始点に
        endpoints = [p for p, nbrs in neighbors.items() if len(nbrs) == 1]
        start = endpoints[0] if endpoints else next(iter(pixel_set))

        # DFSでトレース（分岐点で1枝を最後まで辿ってから次へ）
        visited: set[tuple[int, int]] = set()
        order: list[tuple[int, int]] = []
        stack = [start]
        visited.add(start)

        while stack:
            current = stack.pop()
            order.append(current)
            unvisited = [n for n in neighbors.get(current, []) if n not in visited]
            for nb in unvisited:
                visited.add(nb)
            stack.extend(unvisited)

        # (y, x) → (x, y) に変換
        points = np.array([[x, y] for y, x in order], dtype=float)
        return points
