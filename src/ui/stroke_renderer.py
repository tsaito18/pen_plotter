from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.collector.data_format import StrokeSample
from src.layout.line_breaking import is_halfwidth
from src.layout.page_layout import PageConfig
from src.layout.typesetter import CharPlacement
from src.model.augmentation import HandwritingAugmenter
from dataclasses import dataclass, field

Stroke = npt.NDArray[np.float64]


@dataclass
class CharCoverageReport:
    user_strokes: list[str] = field(default_factory=list)
    ml_inference: list[str] = field(default_factory=list)
    kanjivg: list[str] = field(default_factory=list)
    geometric: list[str] = field(default_factory=list)
    rect_fallback: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

logger = logging.getLogger(__name__)


class StrokeRenderer:
    _SKIP_RENDER = set(" \t\u3000")

    _CHAR_SUBSTITUTIONS: dict[str, str] = {
        "\uff5b": "{",
        "\uff5d": "}",
        "\uff3b": "[",
        "\uff3d": "]",
        "\uff01": "!",
        "\uff1f": "?",
        "\uff1a": ":",
        "\uff1b": ";",
        "\uff1d": "=",
        "\uff0b": "+",
        "\uff0d": "-",
        "\uff0f": "/",
    }

    _SMOOTH_CHARS = set(
        "\u3001\u3002\uff0c\uff0e\u30fb\u30fc\uff5e\u2014\u2015()\uff08\uff09\u300c\u300d\u300e\u300f\u3010\u3011\u3008\u3009\u300a\u300b\u3014\u3015"
    )

    _NOISE_SCALE = 0.3

    _SMALL_KANA = set(
        "\u3041\u3043\u3045\u3047\u3049\u3063\u3083\u3085\u3087\u308e\u30a1\u30a3\u30a5\u30a7\u30a9\u30c3\u30e3\u30e5\u30e7\u30ee"
    )
    _SMALL_PUNCT = set("\u3002\u3001.,")

    def __init__(
        self,
        *,
        checkpoint_path: Path | str | None = None,
        kanjivg_dir: Path | str | None = None,
        style_sample: object | None = None,
        temperature: float = 1.0,
        user_strokes_dir: Path | str | None = None,
        augmenter: HandwritingAugmenter | None = None,
        page_config: PageConfig | None = None,
    ) -> None:
        self._page_config = page_config or PageConfig()
        self._temperature = temperature
        self._augmenter = augmenter

        self._inference = None
        if checkpoint_path is not None:
            cp = Path(checkpoint_path)
            if cp.exists():
                try:
                    from src.model.inference import StrokeInference

                    self._inference = StrokeInference(cp)
                    logger.info("ML inference engine loaded from %s", cp)
                except Exception:
                    logger.warning("Failed to load ML checkpoint: %s", cp, exc_info=True)

        if style_sample is not None:
            self._style_sample = style_sample
        else:
            self._style_sample = self._load_style_from_user_strokes(user_strokes_dir)

        self._user_stroke_db = self._load_user_stroke_db(user_strokes_dir)
        self._last_coverage = CharCoverageReport()

        self._kanjivg_dir: Path | None = None
        if kanjivg_dir is not None:
            d = Path(kanjivg_dir)
            if d.is_dir():
                self._kanjivg_dir = d

    @staticmethod
    def _load_user_stroke_db(
        user_strokes_dir: Path | str | None,
    ) -> dict[str, list[list[Stroke]]]:
        db: dict[str, list[list[Stroke]]] = {}
        if user_strokes_dir is None:
            return db
        user_dir = Path(user_strokes_dir)
        if not user_dir.is_dir():
            return db
        for char_dir in sorted(user_dir.iterdir()):
            if not char_dir.is_dir():
                continue
            char = char_dir.name
            for json_file in sorted(char_dir.glob("*.json")):
                try:
                    sample = StrokeSample.load(json_file)
                    strokes = [
                        np.array([[p.x, p.y] for p in stroke], dtype=np.float64)
                        for stroke in sample.strokes
                    ]
                    db.setdefault(char, []).append(strokes)
                except Exception:
                    logger.warning("Failed to load user stroke: %s", json_file)
        logger.info("Loaded user stroke DB: %d chars", len(db))
        return db

    @staticmethod
    def _load_style_from_user_strokes(
        user_strokes_dir: Path | str | None,
    ) -> object:
        try:
            import torch
        except ImportError:
            return None

        if user_strokes_dir is not None:
            user_dir = Path(user_strokes_dir)
            if user_dir.is_dir():
                import json

                from src.model.data_utils import strokes_to_deltas

                json_files: list[Path] = []
                for char_dir in sorted(user_dir.iterdir()):
                    if char_dir.is_dir():
                        json_files.extend(sorted(char_dir.glob("*.json")))

                if json_files:
                    all_strokes: list[list[dict[str, float]]] = []
                    for jf in json_files:
                        try:
                            data = json.loads(jf.read_text(encoding="utf-8"))
                            all_strokes.extend(data["strokes"])
                        except (json.JSONDecodeError, KeyError):
                            continue

                    if all_strokes:
                        deltas = strokes_to_deltas(all_strokes)
                        style_sample = deltas.unsqueeze(0)
                        logger.info(
                            "Loaded style sample from %d files (%d points)",
                            len(json_files),
                            deltas.shape[0],
                        )
                        return style_sample

        return torch.zeros(1, 10, 3)

    def generate_char_strokes(self, placement: CharPlacement) -> list[Stroke]:
        """Tier 0-4 フォールバックでストローク生成。"""
        cov = self._last_coverage

        if placement.char in self._SKIP_RENDER:
            cov.skipped.append(placement.char)
            return []

        original_char = placement.char
        lookup_char = self._CHAR_SUBSTITUTIONS.get(original_char, original_char)
        if lookup_char != original_char:
            placement = CharPlacement(
                char=lookup_char,
                x=placement.x,
                y=placement.y,
                font_size=placement.font_size,
                page=placement.page,
            )

        is_smooth = original_char in self._SMOOTH_CHARS or lookup_char in self._SMOOTH_CHARS

        punct_strokes = self._simple_punct_strokes(lookup_char)
        if punct_strokes is not None:
            cov.geometric.append(original_char)
            return self._position_strokes(punct_strokes, placement)

        direct = self._direct_stroke(placement.char)
        if direct is not None:
            cov.user_strokes.append(original_char)
            positioned = self._position_strokes(direct, placement)
            return positioned if is_smooth else self._apply_distortion(positioned)

        reference = self._load_reference_strokes(placement.char)

        if self._inference is not None:
            try:
                raw = self._inference.generate(
                    self._style_sample,
                    num_steps=50,
                    temperature=self._temperature,
                    reference_strokes=reference,
                )
                cov.ml_inference.append(original_char)
                positioned = self._position_strokes(raw, placement)
                return positioned if is_smooth else self._apply_distortion(positioned)
            except Exception:
                logger.warning("ML inference failed for '%s'", placement.char, exc_info=True)

        paren_strokes = self._simple_paren_strokes(original_char, placement)
        if paren_strokes is not None:
            cov.geometric.append(original_char)
            return self._position_strokes(paren_strokes, placement)

        math_strokes = self._math_symbol_strokes(lookup_char)
        if math_strokes is not None:
            cov.geometric.append(original_char)
            return self._position_strokes(math_strokes, placement)

        if self._kanjivg_dir is not None:
            char_strokes = self._load_kanjivg_json(placement)
            if char_strokes is not None:
                cov.kanjivg.append(original_char)
                positioned = self._position_strokes(char_strokes, placement)
                return positioned if is_smooth else self._apply_distortion(positioned)

        cov.rect_fallback.append(original_char)
        return self._rect_fallback(placement)

    def _apply_distortion(self, strokes: list[Stroke]) -> list[Stroke]:
        aug = self._augmenter
        if aug is None:
            return strokes
        strokes = [aug.elastic_distort(s) for s in strokes]
        strokes = [aug.apply_tremor(s) for s in strokes]
        return strokes

    def _direct_stroke(self, char: str) -> list[Stroke] | None:
        samples = self._user_stroke_db.get(char)
        if not samples:
            return None
        chosen = samples[np.random.randint(len(samples))]
        normalized = self._normalize_strokes_to_unit(chosen)
        return self._apply_stroke_variation(normalized)

    @staticmethod
    def _normalize_strokes_to_unit(strokes: list[Stroke]) -> list[Stroke]:
        all_pts = np.concatenate(strokes, axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        span = (maxs - mins).max()
        if span < 1e-6:
            return strokes
        center = (mins + maxs) / 2
        result = []
        for s in strokes:
            normalized = (s - center) / span + 0.5
            normalized[:, 1] = 1.0 - normalized[:, 1]
            result.append(normalized)
        return result

    def _apply_stroke_variation(self, strokes: list[Stroke]) -> list[Stroke]:
        ns = self._NOISE_SCALE
        result = []
        for stroke in strokes:
            center = stroke.mean(axis=0)
            centered = stroke - center
            angle = np.random.normal(0, ns * 0.05)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotated = centered @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            sx = 1.0 + np.random.normal(0, ns * 0.03)
            sy = 1.0 + np.random.normal(0, ns * 0.03)
            scaled = rotated * np.array([sx, sy])
            dx = np.random.normal(0, ns * 0.1)
            dy = np.random.normal(0, ns * 0.1)
            result.append(scaled + center + np.array([dx, dy]))
        return result

    def _math_symbol_strokes(self, char: str) -> list[Stroke] | None:
        if char == "\u03c9":
            t = np.linspace(0, 1, 30)
            x = t
            y = 0.3 + 0.3 * np.abs(np.sin(2 * np.pi * t))
            return [np.stack([x, y], axis=1)]
        elif char == "\u03c6":
            angles = np.linspace(0, 2 * np.pi, 24)
            r = 0.3
            circle = np.stack([0.5 + r * np.cos(angles), 0.55 + r * np.sin(angles)], axis=1)
            stem = np.array([[0.5, 0.1], [0.5, 0.9]])
            return [circle, stem]
        elif char == "\u03c0":
            top = np.array([[0.15, 0.25], [0.85, 0.25]])
            left_leg = np.array([[0.35, 0.25], [0.30, 0.85]])
            right_leg = np.array([[0.65, 0.25], [0.70, 0.85]])
            return [top, left_leg, right_leg]
        elif char == "\u03b8":
            angles = np.linspace(0, 2 * np.pi, 24)
            rx, ry = 0.3, 0.4
            ellipse = np.stack([0.5 + rx * np.cos(angles), 0.5 + ry * np.sin(angles)], axis=1)
            bar = np.array([[0.2, 0.5], [0.8, 0.5]])
            return [ellipse, bar]
        elif char == "\u03b1":
            t = np.linspace(0, 2 * np.pi, 30)
            x = 0.5 + 0.3 * np.cos(t) - 0.1 * np.sin(2 * t)
            y = 0.5 + 0.35 * np.sin(t)
            return [np.stack([x, y], axis=1)]
        elif char == "\u0394":
            triangle = np.array([[0.5, 0.1], [0.1, 0.9], [0.9, 0.9], [0.5, 0.1]])
            return [triangle]
        elif char == "\u00b1":
            h_top = np.array([[0.15, 0.2], [0.85, 0.2]])
            h_mid = np.array([[0.15, 0.5], [0.85, 0.5]])
            v_mid = np.array([[0.5, 0.2], [0.5, 0.8]])
            return [h_top, h_mid, v_mid]
        elif char == "\u2248":
            t = np.linspace(0, 2 * np.pi, 20)
            x = np.linspace(0.1, 0.9, 20)
            wave1 = np.stack([x, 0.35 + 0.08 * np.sin(t)], axis=1)
            wave2 = np.stack([x, 0.65 + 0.08 * np.sin(t)], axis=1)
            return [wave1, wave2]
        elif char == "\u221e":
            t = np.linspace(0, 2 * np.pi, 40)
            x = 0.5 + 0.35 * np.cos(t) / (1 + np.sin(t) ** 2)
            y = 0.5 + 0.25 * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
            return [np.stack([x, y], axis=1)]
        return None

    def _simple_punct_strokes(self, char: str) -> list[Stroke] | None:
        if char in ("\u3001", ","):
            return [np.array([[0.6, 0.2], [0.3, 0.8]])]
        elif char in ("\u3002", "."):
            angles = np.linspace(0, 2 * np.pi, 16)
            r = 0.3
            return [np.stack([0.5 + r * np.cos(angles), 0.5 + r * np.sin(angles)], axis=1)]
        elif char == "\u30fb":
            angles = np.linspace(0, 2 * np.pi, 12)
            r = 0.15
            return [np.stack([0.5 + r * np.cos(angles), 0.5 + r * np.sin(angles)], axis=1)]
        return None

    def _simple_paren_strokes(self, char: str, placement: CharPlacement) -> list[Stroke] | None:
        if char in ("(", "\uff08"):
            points = []
            for i in range(20):
                t = i / 19
                x = 0.15 + 0.25 * np.cos(np.pi * (t - 0.5))
                y = 0.1 + 0.8 * t
                points.append([x, y])
            return [np.array(points)]
        elif char in (")", "\uff09"):
            points = []
            for i in range(20):
                t = i / 19
                x = 0.85 - 0.25 * np.cos(np.pi * (t - 0.5))
                y = 0.1 + 0.8 * t
                points.append([x, y])
            return [np.array(points)]
        return None

    def _load_reference_strokes(self, char: str) -> list[Stroke] | None:
        if self._kanjivg_dir is None:
            return None
        char_dir = self._kanjivg_dir / char
        if not char_dir.is_dir():
            return None
        json_files = sorted(char_dir.glob(f"{char}_*.json"))
        if not json_files:
            return None
        try:
            sample = StrokeSample.load(json_files[0])
            return [
                np.array([[p.x, p.y] for p in stroke], dtype=np.float64)
                for stroke in sample.strokes
                if len(stroke) >= 2
            ]
        except Exception:
            return None

    def _load_kanjivg_json(self, placement: CharPlacement) -> list[Stroke] | None:
        char_dir = self._kanjivg_dir / placement.char
        if not char_dir.is_dir():
            return None
        json_files = sorted(char_dir.glob(f"{placement.char}_*.json"))
        if not json_files:
            return None
        try:
            sample = StrokeSample.load(json_files[0])
            strokes = []
            for stroke_points in sample.strokes:
                arr = np.array([[p.x, p.y] for p in stroke_points], dtype=np.float64)
                if len(arr) >= 2:
                    strokes.append(arr)
            return strokes if strokes else None
        except Exception:
            logger.warning("KanjiVG JSON load failed for '%s'", placement.char, exc_info=True)
            return None

    def _char_scale_factor(self, char: str) -> float:
        cp = ord(char)
        if char in self._SMALL_KANA:
            return 0.55
        elif char in self._SMALL_PUNCT:
            return 0.35
        elif 0x3040 <= cp <= 0x309F:
            return 0.88
        elif 0x30A0 <= cp <= 0x30FF:
            return 0.85
        else:
            return 1.0

    def _position_strokes(self, strokes: list[Stroke], placement: CharPlacement) -> list[Stroke]:
        if not strokes:
            return []

        all_pts = np.concatenate(strokes, axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        ranges = maxs - mins

        fs = placement.font_size
        char_scale = self._char_scale_factor(placement.char)
        target_h = fs * char_scale

        if is_halfwidth(placement.char):
            cell_width = fs * 0.6
        else:
            cell_width = fs * char_scale

        scale_w = cell_width / ranges[0] if ranges[0] > 1e-6 else float("inf")
        scale_h = target_h / ranges[1] if ranges[1] > 1e-6 else float("inf")
        scale = min(scale_w, scale_h)

        scaled = [(stroke - mins) * scale for stroke in strokes]
        rendered_w = ranges[0] * scale
        rendered_h = ranges[1] * scale

        if is_halfwidth(placement.char):
            x_offset = placement.x + (fs * 0.6 - rendered_w) / 2
        else:
            x_offset = placement.x + (cell_width - rendered_w) / 2

        line_spacing = self._page_config.line_spacing
        if char_scale < 0.5:
            y_offset = placement.y + 0.1 * line_spacing
        else:
            y_offset = placement.y + (line_spacing - rendered_h) / 2

        offset = np.array([x_offset, y_offset])
        return [stroke + offset for stroke in scaled]

    @staticmethod
    def _rect_fallback(p: CharPlacement) -> list[Stroke]:
        half = p.font_size / 2.0
        x0, y0 = p.x, p.y - half
        x1, y1 = p.x + p.font_size, p.y + half
        rect = np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [x0, y0],
            ]
        )
        return [rect]
