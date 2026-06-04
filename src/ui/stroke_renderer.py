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
from src.model.stroke_finishing import (
    FinishingConfig,
    apply_finishing,
    classify_finishes,
)
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

    _NOISE_SCALE = 0.15

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
        finishing_config: FinishingConfig | None = None,
        enable_finishing: bool = True,
    ) -> None:
        self._page_config = page_config or PageConfig()
        self._temperature = temperature
        self._augmenter = augmenter

        # KanjiVG 参照経路（ML 推論 / safety-net）の終端加工（とめ・はね・払い）。
        # _direct_stroke（人の実筆跡）・幾何描画経路には適用しない。
        if finishing_config is not None:
            self._finishing_config = finishing_config
        elif enable_finishing:
            self._finishing_config = FinishingConfig()
        else:
            self._finishing_config = FinishingConfig(enabled=False)

        # 既存呼び出し元（pipeline.create_app() 等）が UI 構築時に環境引数を
        # 復元できるよう、入力時のパスを保持しておく
        self._checkpoint_path = checkpoint_path
        self._user_strokes_dir = user_strokes_dir

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

        # 分数線・根号の屋根線などの補助線分は文字ではなく単一ストロークとして描画する
        if placement.line_segment is not None:
            x1, y1, x2, y2 = placement.line_segment
            return [np.array([[x1, y1], [x2, y2]], dtype=np.float64)]

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

        ascii_math = self._ascii_math_strokes(lookup_char)
        if ascii_math is not None:
            cov.geometric.append(original_char)
            return self._position_strokes(ascii_math, placement)

        paren_strokes = self._simple_paren_strokes(original_char, placement)
        if paren_strokes is not None:
            cov.geometric.append(original_char)
            return self._position_strokes(paren_strokes, placement)

        math_strokes = self._math_symbol_strokes(lookup_char)
        if math_strokes is not None:
            cov.geometric.append(original_char)
            return self._position_strokes(math_strokes, placement)

        word_strokes = self._math_word_strokes(lookup_char, placement)
        if word_strokes is not None:
            cov.geometric.append(original_char)
            return word_strokes

        letter_strokes = self._ascii_letter_strokes(lookup_char)
        if letter_strokes is not None:
            cov.geometric.append(original_char)
            return self._position_strokes(letter_strokes, placement)

        direct = self._direct_stroke(placement.char)
        if direct is not None:
            cov.user_strokes.append(original_char)
            positioned = self._position_strokes(direct, placement)
            return positioned if is_smooth else self._apply_distortion(positioned)

        reference, ref_types = self._load_reference_strokes(placement.char)

        if self._inference is not None and reference is not None:
            try:
                raw = self._inference.generate(
                    self._style_sample,
                    num_steps=50,
                    temperature=self._temperature,
                    reference_strokes=reference,
                )
                cov.ml_inference.append(original_char)
                positioned = self._position_strokes(raw, placement)
                positioned = apply_finishing(
                    positioned,
                    classify_finishes(ref_types),
                    scale=placement.font_size,
                    config=self._finishing_config,
                )
                return positioned if is_smooth else self._apply_distortion(positioned)
            except Exception:
                logger.warning("ML inference failed for '%s'", placement.char, exc_info=True)

        if self._kanjivg_dir is not None:
            char_strokes, char_types = self._load_kanjivg_json(placement)
            if char_strokes is not None:
                cov.kanjivg.append(original_char)
                positioned = self._position_strokes(char_strokes, placement)
                positioned = apply_finishing(
                    positioned,
                    classify_finishes(char_types),
                    scale=placement.font_size,
                    config=self._finishing_config,
                )
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
        elif char == "\u03b2":  # \u03b2: \u7e26\u68d2 + \u4e0a\u4e0b\u306e\u30eb\u30fc\u30d7
            stem = np.array([[0.25, 0.05], [0.25, 0.95]], dtype=np.float64)
            t = np.linspace(0, 1, 30)
            upper = np.stack(
                [0.25 + 0.45 * np.sin(np.pi * t), 0.5 + 0.4 * (1 - t)], axis=1
            )
            lower = np.stack(
                [0.25 + 0.5 * np.sin(np.pi * t), 0.5 - 0.45 * t], axis=1
            )
            return [stem, upper, lower]
        elif char == "\u03b3":  # \u03b3: \u4e0a\u958b\u304d\u306e y \u5b57
            left = np.array([[0.15, 0.85], [0.5, 0.4]], dtype=np.float64)
            right = np.array([[0.85, 0.85], [0.4, 0.05]], dtype=np.float64)
            return [left, right]
        elif char == "\u03b4":  # \u03b4: \u4e0a\u958b\u304d\u306e\u5186 + \u4e0a\u306b\u3057\u3063\u307d
            t = np.linspace(0.2 * np.pi, 1.8 * np.pi, 30)
            body = np.stack(
                [0.5 + 0.3 * np.cos(t), 0.4 + 0.3 * np.sin(t)], axis=1
            )
            tail = np.array([[0.7, 0.85], [0.55, 0.95]], dtype=np.float64)
            return [body, tail]
        elif char == "\u03b5":  # \u03b5: \u6a2a\u5411\u304d\u306e 3 \u5b57
            t = np.linspace(0.5 * np.pi, 1.5 * np.pi, 16)
            top = np.stack(
                [0.55 - 0.3 * np.sin(t), 0.7 + 0.18 * np.cos(t)], axis=1
            )
            bot = np.stack(
                [0.55 - 0.3 * np.sin(t), 0.3 + 0.18 * np.cos(t)], axis=1
            )
            mid = np.array([[0.3, 0.5], [0.55, 0.5]], dtype=np.float64)
            return [top, mid, bot]
        elif char == "\u03b6":  # \u03b6: \u3072\u3063\u304b\u304d\u306e z + \u4e0b\u306b\u30eb\u30fc\u30d7
            top = np.array(
                [[0.2, 0.9], [0.8, 0.9], [0.3, 0.4]], dtype=np.float64
            )
            t = np.linspace(0, np.pi, 16)
            tail = np.stack(
                [0.5 + 0.25 * np.sin(t), 0.2 - 0.18 * (1 - np.cos(t))], axis=1
            )
            return [top, tail]
        elif char == "\u03b7":  # \u03b7: n \u5b57 + \u53f3\u811a\u3092\u4e0b\u306b\u4f38\u3070\u3059
            stem_left = np.array([[0.2, 0.7], [0.2, 0.05]], dtype=np.float64)
            t = np.linspace(np.pi, 0, 16)
            arch = np.stack(
                [0.5 + 0.3 * np.cos(t), 0.55 + 0.15 * np.sin(t)], axis=1
            )
            stem_right = np.array([[0.8, 0.7], [0.8, 0.2]], dtype=np.float64)
            return [stem_left, arch, stem_right]
        elif char == "\u03bb":  # \u03bb: \u659c\u3081\u306e\u00d7 \uff08/ + \u77ed\u3044 \\uff09
            main = np.array([[0.15, 0.05], [0.85, 0.95]], dtype=np.float64)
            cross = np.array([[0.3, 0.55], [0.1, 0.95]], dtype=np.float64)
            return [main, cross]
        elif char == "\u03bc":  # \u03bc: \u4e0b\u306b\u9577\u3044\u68d2 + n \u5b57
            left_stem = np.array([[0.2, 0.7], [0.2, 0.0]], dtype=np.float64)
            t = np.linspace(np.pi, 0, 16)
            arch = np.stack(
                [0.5 + 0.3 * np.cos(t), 0.4 + 0.3 * np.sin(t)], axis=1
            )
            right_stem = np.array([[0.8, 0.7], [0.8, 0.15]], dtype=np.float64)
            return [left_stem, arch, right_stem]
        elif char == "\u03bd":  # \u03bd: V \u5b57
            return [np.array([[0.15, 0.85], [0.5, 0.1], [0.85, 0.85]], dtype=np.float64)]
        elif char == "\u03c1":  # \u03c1: \u7e26\u68d2 + \u5186
            stem = np.array([[0.3, 0.5], [0.3, 0.0]], dtype=np.float64)
            t = np.linspace(0, 2 * np.pi, 24)
            circle = np.stack(
                [0.5 + 0.25 * np.cos(t), 0.55 + 0.25 * np.sin(t)], axis=1
            )
            return [stem, circle]
        elif char == "\u03c3":  # \u03c3: \u5186 + \u4e0a\u306e\u6a2a\u68d2
            t = np.linspace(0, 2 * np.pi, 24)
            circle = np.stack(
                [0.4 + 0.25 * np.cos(t), 0.4 + 0.25 * np.sin(t)], axis=1
            )
            top = np.array([[0.4, 0.7], [0.85, 0.7]], dtype=np.float64)
            return [circle, top]
        elif char == "\u03c4":  # \u03c4: \u4e0a\u6a2a\u68d2 + \u4e0b\u306b\u66f2\u304c\u308b\u811a
            top = np.array([[0.1, 0.75], [0.9, 0.75]], dtype=np.float64)
            t = np.linspace(0, np.pi / 2, 16)
            stem = np.stack(
                [0.5 + 0.2 * np.sin(t), 0.75 - 0.7 * t / (np.pi / 2)], axis=1
            )
            return [top, stem]
        elif char == "\u03c7":  # \u03c7: \u5927\u304d\u306a \u00d7
            d1 = np.array([[0.15, 0.1], [0.85, 0.85]], dtype=np.float64)
            d2 = np.array([[0.85, 0.1], [0.15, 0.85]], dtype=np.float64)
            return [d1, d2]
        elif char == "\u03c8":  # \u03c8: V + \u7e26\u68d2
            v = np.array(
                [[0.15, 0.75], [0.5, 0.35], [0.85, 0.75]], dtype=np.float64
            )
            stem = np.array([[0.5, 0.95], [0.5, 0.05]], dtype=np.float64)
            return [v, stem]
        elif char == "\u0393":  # \u0393: \u4e0a\u6a2a\u68d2 + \u5de6\u7e26\u68d2
            top = np.array([[0.15, 0.9], [0.85, 0.9]], dtype=np.float64)
            left = np.array([[0.15, 0.9], [0.15, 0.1]], dtype=np.float64)
            return [top, left]
        elif char == "\u039b":  # \u039b: \u4e09\u89d2\u306e\u4e0a (\u0394 \u304b\u3089\u5e95\u8fba\u306a\u3057)
            return [np.array([[0.1, 0.1], [0.5, 0.9], [0.9, 0.1]], dtype=np.float64)]
        elif char == "\u0398":  # \u0398: \u6955\u5186 + \u4e2d\u592e\u6a2a\u68d2
            t = np.linspace(0, 2 * np.pi, 30)
            ellipse = np.stack(
                [0.5 + 0.35 * np.cos(t), 0.5 + 0.4 * np.sin(t)], axis=1
            )
            bar = np.array([[0.3, 0.5], [0.7, 0.5]], dtype=np.float64)
            return [ellipse, bar]
        elif char == "\u03a0":  # \u03a0: \u4e0a\u6a2a\u68d2 + \u5de6\u53f3\u7e26\u68d2
            top = np.array([[0.1, 0.9], [0.9, 0.9]], dtype=np.float64)
            left = np.array([[0.2, 0.9], [0.2, 0.1]], dtype=np.float64)
            right = np.array([[0.8, 0.9], [0.8, 0.1]], dtype=np.float64)
            return [top, left, right]
        elif char == "\u03a3" or char == "\u2211":  # \u03a3 / \u2211: \u6a2a3 + \u6298\u8fd4\u3057
            top = np.array([[0.1, 0.9], [0.9, 0.9]], dtype=np.float64)
            diag1 = np.array([[0.1, 0.9], [0.5, 0.5]], dtype=np.float64)
            diag2 = np.array([[0.5, 0.5], [0.1, 0.1]], dtype=np.float64)
            bot = np.array([[0.1, 0.1], [0.9, 0.1]], dtype=np.float64)
            return [top, diag1, diag2, bot]
        elif char == "\u03a6":  # \u03a6: \u5186 + \u7e26\u68d2
            t = np.linspace(0, 2 * np.pi, 24)
            circle = np.stack(
                [0.5 + 0.3 * np.cos(t), 0.5 + 0.3 * np.sin(t)], axis=1
            )
            stem = np.array([[0.5, 0.95], [0.5, 0.05]], dtype=np.float64)
            return [circle, stem]
        elif char == "\u03a8":  # \u03a8: U + \u7e26\u68d2
            t = np.linspace(np.pi, 2 * np.pi, 16)
            cup = np.stack(
                [0.5 + 0.35 * np.cos(t), 0.55 + 0.25 * np.sin(t)], axis=1
            )
            stem = np.array([[0.5, 0.95], [0.5, 0.05]], dtype=np.float64)
            base = np.array([[0.25, 0.05], [0.75, 0.05]], dtype=np.float64)
            return [cup, stem, base]
        elif char == "\u03a9":  # \u03a9: U\u5b57 + \u4e0b\u306b\u811a
            t = np.linspace(np.pi, 2 * np.pi, 24)
            arch = np.stack(
                [0.5 + 0.35 * np.cos(t), 0.4 + 0.45 * np.sin(t)], axis=1
            )
            left_foot = np.array([[0.15, 0.4], [0.05, 0.1]], dtype=np.float64)
            right_foot = np.array([[0.85, 0.4], [0.95, 0.1]], dtype=np.float64)
            base_l = np.array([[0.05, 0.1], [0.25, 0.1]], dtype=np.float64)
            base_r = np.array([[0.75, 0.1], [0.95, 0.1]], dtype=np.float64)
            return [arch, left_foot, right_foot, base_l, base_r]
        elif char == "\u00d7":  # \u00d7: \u30af\u30ed\u30b9
            d1 = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float64)
            d2 = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float64)
            return [d1, d2]
        elif char == "\u00f7":  # \u00f7: \u6a2a\u68d2 + \u4e0a\u4e0b\u70b9
            bar = np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float64)
            return [bar, self._small_dot(0.5, 0.75), self._small_dot(0.5, 0.25)]
        elif char == "\u2260":  # \u2260: = \u306b\u659c\u3081\u7dda
            top = np.array([[0.2, 0.4], [0.8, 0.4]], dtype=np.float64)
            bot = np.array([[0.2, 0.6], [0.8, 0.6]], dtype=np.float64)
            slash = np.array([[0.7, 0.2], [0.3, 0.8]], dtype=np.float64)
            return [top, bot, slash]
        elif char == "\u2264":  # \u2264: < + \u4e0b\u6a2a\u68d2
            v = np.array([[0.8, 0.25], [0.2, 0.55], [0.8, 0.85]], dtype=np.float64)
            bar = np.array([[0.2, 0.15], [0.8, 0.15]], dtype=np.float64)
            return [v, bar]
        elif char == "\u2265":  # \u2265: > + \u4e0b\u6a2a\u68d2
            v = np.array([[0.2, 0.25], [0.8, 0.55], [0.2, 0.85]], dtype=np.float64)
            bar = np.array([[0.2, 0.15], [0.8, 0.15]], dtype=np.float64)
            return [v, bar]
        elif char == "\u00b7":  # \u00b7: \u4e2d\u592e\u70b9
            return [self._small_dot(0.5, 0.5)]
        elif char == "\u2026":  # \u2026: \u6a2a\u4e26\u3073\u306e 3 \u70b9
            return [self._small_dot(0.2, 0.2), self._small_dot(0.5, 0.2), self._small_dot(0.8, 0.2)]
        elif char == "\u2192":  # \u2192: \u6a2a\u7dda + \u77e2\u3058\u308a
            shaft = np.array([[0.1, 0.5], [0.85, 0.5]], dtype=np.float64)
            head_top = np.array([[0.85, 0.5], [0.65, 0.65]], dtype=np.float64)
            head_bot = np.array([[0.85, 0.5], [0.65, 0.35]], dtype=np.float64)
            return [shaft, head_top, head_bot]
        elif char == "\u2190":  # \u2190: \u6a2a\u7dda + \u5de6\u306e\u77e2\u3058\u308a
            shaft = np.array([[0.15, 0.5], [0.9, 0.5]], dtype=np.float64)
            head_top = np.array([[0.15, 0.5], [0.35, 0.65]], dtype=np.float64)
            head_bot = np.array([[0.15, 0.5], [0.35, 0.35]], dtype=np.float64)
            return [shaft, head_top, head_bot]
        elif char == "\u21d2":  # \u21d2: \u4e8c\u91cd\u7dda + \u77e2\u3058\u308a
            top = np.array([[0.1, 0.55], [0.8, 0.55]], dtype=np.float64)
            bot = np.array([[0.1, 0.45], [0.8, 0.45]], dtype=np.float64)
            head_top = np.array([[0.85, 0.5], [0.65, 0.7]], dtype=np.float64)
            head_bot = np.array([[0.85, 0.5], [0.65, 0.3]], dtype=np.float64)
            return [top, bot, head_top, head_bot]
        elif char == "\u2202":  # \u2202: 6 \u3092\u53cd\u8ee2\u3057\u305f\u5f62\uff08curly d\uff09
            t = np.linspace(0, 2 * np.pi, 30)
            body = np.stack(
                [0.5 + 0.3 * np.cos(t), 0.4 + 0.35 * np.sin(t)], axis=1
            )
            tail = np.array([[0.55, 0.75], [0.85, 0.95]], dtype=np.float64)
            return [body, tail]
        elif char == "\u2207":  # \u2207: \u4e0b\u5411\u304d\u4e09\u89d2
            return [np.array([[0.1, 0.9], [0.9, 0.9], [0.5, 0.1], [0.1, 0.9]], dtype=np.float64)]
        elif char == "\u222b":  # \u222b: \u7e26\u9577\u306e S \u5b57
            t = np.linspace(0, 2 * np.pi, 40)
            x = 0.5 + 0.18 * np.sin(t * 0.5 + np.pi)
            y = np.linspace(0.05, 0.95, 40)
            stroke = np.stack([x, y], axis=1)
            top_hook = np.array([[stroke[-1, 0], stroke[-1, 1]], [0.7, 0.95]], dtype=np.float64)
            bot_hook = np.array([[0.3, 0.05], [stroke[0, 0], stroke[0, 1]]], dtype=np.float64)
            return [bot_hook, stroke, top_hook]
        elif char == "\u220f":  # \u220f: \u03a0 \u3068\u540c\u3058\u5f62
            top = np.array([[0.1, 0.9], [0.9, 0.9]], dtype=np.float64)
            left = np.array([[0.2, 0.9], [0.2, 0.1]], dtype=np.float64)
            right = np.array([[0.8, 0.9], [0.8, 0.1]], dtype=np.float64)
            return [top, left, right]
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

    @staticmethod
    def _small_dot(cx: float, cy: float, r: float = 0.06) -> Stroke:
        """単位正方形内の小円（句点・コロン用）。"""
        angles = np.linspace(0, 2 * np.pi, 12)
        return np.stack(
            [cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1
        ).astype(np.float64)

    def _ascii_math_strokes(self, char: str) -> list[Stroke] | None:
        """ASCII の数式記号・句読点を単位正方形 (0,0)-(1,1) で幾何描画する。

        矩形フォールバックを避けるため _simple_punct_strokes より後・ML 推論より
        前で呼ぶ。座標は np.float64 で統一（後段の _position_strokes 互換）。
        """
        if char == "+":
            h = np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float64)
            v = np.array([[0.5, 0.2], [0.5, 0.8]], dtype=np.float64)
            return [h, v]
        elif char == "-":
            return [np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float64)]
        elif char == "=":
            top = np.array([[0.2, 0.4], [0.8, 0.4]], dtype=np.float64)
            bot = np.array([[0.2, 0.6], [0.8, 0.6]], dtype=np.float64)
            return [top, bot]
        elif char == "<":
            return [np.array([[0.8, 0.2], [0.2, 0.5], [0.8, 0.8]], dtype=np.float64)]
        elif char == ">":
            return [np.array([[0.2, 0.2], [0.8, 0.5], [0.2, 0.8]], dtype=np.float64)]
        elif char == "*":
            cx, cy, r = 0.5, 0.5, 0.3
            arms: list[Stroke] = []
            for k in range(3):
                ang = np.pi * k / 3.0
                dx, dy = r * np.cos(ang), r * np.sin(ang)
                arms.append(
                    np.array([[cx - dx, cy - dy], [cx + dx, cy + dy]], dtype=np.float64)
                )
            return arms
        elif char == "/":
            return [np.array([[0.2, 0.2], [0.8, 0.8]], dtype=np.float64)]
        elif char == ":":
            return [self._small_dot(0.5, 0.7), self._small_dot(0.5, 0.3)]
        elif char == ";":
            tail = np.array([[0.55, 0.3], [0.4, 0.0]], dtype=np.float64)
            return [self._small_dot(0.5, 0.7), tail]
        elif char == "!":
            stem = np.array([[0.5, 0.85], [0.5, 0.25]], dtype=np.float64)
            return [stem, self._small_dot(0.5, 0.05)]
        elif char == "?":
            t = np.linspace(np.pi, 0.0, 16)
            arc = np.stack([0.5 + 0.2 * np.cos(t), 0.725 + 0.125 * np.sin(t)], axis=1).astype(
                np.float64
            )
            stem = np.array([[0.7, 0.6], [0.55, 0.35]], dtype=np.float64)
            return [arc, stem, self._small_dot(0.55, 0.1)]
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
        elif char == "\u300c":
            return [
                np.array([[0.8, 0.15], [0.25, 0.15]], dtype=np.float64),
                np.array([[0.25, 0.15], [0.25, 0.45]], dtype=np.float64),
            ]
        elif char == "\u300d":
            return [
                np.array([[0.75, 0.55], [0.75, 0.85]], dtype=np.float64),
                np.array([[0.75, 0.85], [0.2, 0.85]], dtype=np.float64),
            ]
        elif char == "\u300e":
            return [
                np.array([[0.8, 0.15], [0.25, 0.15], [0.25, 0.45]], dtype=np.float64),
                np.array([[0.65, 0.25], [0.35, 0.25], [0.35, 0.45]], dtype=np.float64),
            ]
        elif char == "\u300f":
            return [
                np.array([[0.75, 0.55], [0.75, 0.85], [0.2, 0.85]], dtype=np.float64),
                np.array([[0.65, 0.55], [0.65, 0.75], [0.35, 0.75]], dtype=np.float64),
            ]
        return None

    def _math_word_strokes(self, word: str, placement: CharPlacement) -> list[Stroke] | None:
        supported = {"cos", "sin", "tan", "log", "ln", "exp", "lim", "dx", "dy", "dt"}
        if word not in supported:
            return None

        strokes: list[Stroke] = []
        advance = placement.font_size * 0.55
        for i, ch in enumerate(word):
            glyph = self._ascii_letter_strokes(ch)
            if glyph is None:
                return None
            char_placement = CharPlacement(
                char=ch,
                x=placement.x + i * advance,
                y=placement.y,
                font_size=placement.font_size,
                page=placement.page,
            )
            strokes.extend(self._position_strokes(glyph, char_placement))
        return strokes

    def _ascii_letter_strokes(self, char: str) -> list[Stroke] | None:
        if char == "c":
            t = np.linspace(0.25 * np.pi, 1.75 * np.pi, 24)
            return [
                np.stack(
                    [0.55 + 0.32 * np.cos(t), 0.5 + 0.32 * np.sin(t)],
                    axis=1,
                ).astype(np.float64)
            ]
        if char == "o":
            t = np.linspace(0, 2 * np.pi, 28)
            return [
                np.stack(
                    [0.5 + 0.3 * np.cos(t), 0.5 + 0.3 * np.sin(t)],
                    axis=1,
                ).astype(np.float64)
            ]
        if char == "s":
            t = np.linspace(0, 1, 30)
            x = 0.5 + 0.28 * np.sin(2 * np.pi * t)
            y = 0.85 - 0.7 * t
            return [np.stack([x, y], axis=1).astype(np.float64)]
        if char == "i":
            stem = np.array([[0.5, 0.25], [0.5, 0.7]], dtype=np.float64)
            return [stem, self._small_dot(0.5, 0.88)]
        if char == "n":
            left = np.array([[0.2, 0.2], [0.2, 0.75]], dtype=np.float64)
            arch = np.array([[0.2, 0.75], [0.5, 0.2], [0.8, 0.75]], dtype=np.float64)
            return [left, arch]
        if char == "t":
            stem = np.array([[0.5, 0.15], [0.5, 0.85]], dtype=np.float64)
            cross = np.array([[0.25, 0.35], [0.75, 0.35]], dtype=np.float64)
            return [stem, cross]
        if char == "a":
            t = np.linspace(0, 2 * np.pi, 24)
            body = np.stack(
                [0.45 + 0.25 * np.cos(t), 0.45 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            tail = np.array([[0.7, 0.2], [0.7, 0.7]], dtype=np.float64)
            return [body, tail]
        if char == "l":
            return [np.array([[0.45, 0.15], [0.45, 0.85]], dtype=np.float64)]
        if char == "g":
            t = np.linspace(0, 2 * np.pi, 24)
            body = np.stack(
                [0.48 + 0.25 * np.cos(t), 0.45 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            tail = np.array([[0.65, 0.45], [0.6, 0.0], [0.35, 0.05]], dtype=np.float64)
            return [body, tail]
        if char == "e":
            t = np.linspace(0.2 * np.pi, 1.8 * np.pi, 24)
            body = np.stack(
                [0.52 + 0.28 * np.cos(t), 0.5 + 0.28 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            bar = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
            return [body, bar]
        if char == "x":
            a = np.array([[0.25, 0.2], [0.75, 0.8]], dtype=np.float64)
            b = np.array([[0.75, 0.2], [0.25, 0.8]], dtype=np.float64)
            return [a, b]
        if char == "p":
            stem = np.array([[0.25, 0.0], [0.25, 0.85]], dtype=np.float64)
            t = np.linspace(-np.pi / 2, np.pi / 2, 18)
            loop = np.stack(
                [0.25 + 0.45 * np.cos(t), 0.6 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            return [stem, loop]
        if char == "m":
            return [
                np.array(
                    [[0.15, 0.2], [0.15, 0.75], [0.38, 0.35], [0.6, 0.75], [0.85, 0.2]],
                    dtype=np.float64,
                )
            ]
        if char == "d":
            stem = np.array([[0.75, 0.1], [0.75, 0.9]], dtype=np.float64)
            t = np.linspace(0, 2 * np.pi, 24)
            body = np.stack(
                [0.48 + 0.25 * np.cos(t), 0.45 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            return [stem, body]
        if char == "y":
            return [
                np.array([[0.2, 0.75], [0.48, 0.4], [0.75, 0.75]], dtype=np.float64),
                np.array([[0.48, 0.4], [0.35, 0.0]], dtype=np.float64),
            ]
        if char == "b":
            stem = np.array([[0.25, 0.0], [0.25, 0.95]], dtype=np.float64)
            t = np.linspace(0, 2 * np.pi, 24)
            body = np.stack(
                [0.48 + 0.25 * np.cos(t), 0.25 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            return [stem, body]
        if char == "f":
            t = np.linspace(0, np.pi / 2, 12)
            hook = np.stack(
                [0.45 + 0.25 * np.sin(t), 0.7 + 0.2 * (1 - np.cos(t))],
                axis=1,
            ).astype(np.float64)
            stem = np.array([[0.45, 0.7], [0.45, 0.0]], dtype=np.float64)
            cross = np.array([[0.25, 0.5], [0.6, 0.5]], dtype=np.float64)
            return [hook, stem, cross]
        if char == "h":
            stem = np.array([[0.25, 0.0], [0.25, 0.95]], dtype=np.float64)
            arch = np.array([[0.25, 0.5], [0.5, 0.7], [0.75, 0.5], [0.75, 0.0]], dtype=np.float64)
            return [stem, arch]
        if char == "j":
            stem = np.array([[0.55, 0.7], [0.55, -0.05], [0.4, -0.15]], dtype=np.float64)
            return [stem, self._small_dot(0.55, 0.88)]
        if char == "k":
            stem = np.array([[0.25, 0.0], [0.25, 0.95]], dtype=np.float64)
            upper = np.array([[0.25, 0.35], [0.7, 0.7]], dtype=np.float64)
            lower = np.array([[0.4, 0.45], [0.75, 0.0]], dtype=np.float64)
            return [stem, upper, lower]
        if char == "q":
            t = np.linspace(0, 2 * np.pi, 24)
            body = np.stack(
                [0.45 + 0.25 * np.cos(t), 0.45 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            tail = np.array([[0.7, 0.45], [0.7, -0.05]], dtype=np.float64)
            return [body, tail]
        if char == "r":
            stem = np.array([[0.3, 0.0], [0.3, 0.7]], dtype=np.float64)
            hook = np.array([[0.3, 0.55], [0.5, 0.7], [0.7, 0.55]], dtype=np.float64)
            return [stem, hook]
        if char == "u":
            t = np.linspace(np.pi, 2 * np.pi, 16)
            cup = np.stack(
                [0.5 + 0.3 * np.cos(t), 0.3 + 0.3 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            right = np.array([[0.8, 0.3], [0.8, 0.0]], dtype=np.float64)
            top_left = np.array([[0.2, 0.7], [0.2, 0.3]], dtype=np.float64)
            return [top_left, cup, right]
        if char == "v":
            return [np.array([[0.2, 0.7], [0.5, 0.0], [0.8, 0.7]], dtype=np.float64)]
        if char == "w":
            return [
                np.array(
                    [[0.1, 0.7], [0.3, 0.0], [0.5, 0.45], [0.7, 0.0], [0.9, 0.7]],
                    dtype=np.float64,
                )
            ]
        if char == "z":
            return [
                np.array([[0.2, 0.7], [0.8, 0.7], [0.2, 0.05], [0.8, 0.05]], dtype=np.float64)
            ]
        if char == "A":
            left = np.array([[0.2, 0.0], [0.5, 0.95]], dtype=np.float64)
            right = np.array([[0.5, 0.95], [0.8, 0.0]], dtype=np.float64)
            cross = np.array([[0.3, 0.35], [0.7, 0.35]], dtype=np.float64)
            return [left, right, cross]
        if char == "B":
            stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
            t = np.linspace(-np.pi / 2, np.pi / 2, 14)
            upper = np.stack(
                [0.2 + 0.4 * np.cos(t), 0.7 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            lower = np.stack(
                [0.2 + 0.45 * np.cos(t), 0.225 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            return [stem, upper, lower]
        if char == "C":
            t = np.linspace(0.25 * np.pi, 1.75 * np.pi, 28)
            return [
                np.stack(
                    [0.55 + 0.38 * np.cos(t), 0.5 + 0.45 * np.sin(t)],
                    axis=1,
                ).astype(np.float64)
            ]
        if char == "D":
            stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
            t = np.linspace(-np.pi / 2, np.pi / 2, 20)
            arc = np.stack(
                [0.2 + 0.55 * np.cos(t), 0.475 + 0.475 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            return [stem, arc]
        if char == "E":
            stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
            top = np.array([[0.2, 0.95], [0.8, 0.95]], dtype=np.float64)
            mid = np.array([[0.2, 0.5], [0.7, 0.5]], dtype=np.float64)
            bot = np.array([[0.2, 0.0], [0.8, 0.0]], dtype=np.float64)
            return [stem, top, mid, bot]
        if char == "F":
            stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
            top = np.array([[0.2, 0.95], [0.8, 0.95]], dtype=np.float64)
            mid = np.array([[0.2, 0.5], [0.7, 0.5]], dtype=np.float64)
            return [stem, top, mid]
        if char == "G":
            t = np.linspace(0.25 * np.pi, 1.75 * np.pi, 28)
            arc = np.stack(
                [0.55 + 0.38 * np.cos(t), 0.5 + 0.45 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            inner = np.array([[0.55, 0.5], [0.85, 0.5], [0.85, 0.1]], dtype=np.float64)
            return [arc, inner]
        if char == "H":
            left = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
            right = np.array([[0.8, 0.0], [0.8, 0.95]], dtype=np.float64)
            cross = np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float64)
            return [left, right, cross]
        if char == "I":
            stem = np.array([[0.5, 0.0], [0.5, 0.95]], dtype=np.float64)
            top = np.array([[0.3, 0.95], [0.7, 0.95]], dtype=np.float64)
            bot = np.array([[0.3, 0.0], [0.7, 0.0]], dtype=np.float64)
            return [stem, top, bot]
        if char == "J":
            stem = np.array([[0.65, 0.95], [0.65, 0.2]], dtype=np.float64)
            t = np.linspace(0, np.pi, 14)
            curl = np.stack(
                [0.45 + 0.2 * np.cos(t), 0.2 - 0.15 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            return [stem, curl]
        if char == "K":
            stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
            upper = np.array([[0.2, 0.5], [0.8, 0.95]], dtype=np.float64)
            lower = np.array([[0.4, 0.6], [0.85, 0.0]], dtype=np.float64)
            return [stem, upper, lower]
        if char == "L":
            stem = np.array([[0.2, 0.95], [0.2, 0.0]], dtype=np.float64)
            bot = np.array([[0.2, 0.0], [0.8, 0.0]], dtype=np.float64)
            return [stem, bot]
        if char == "M":
            return [
                np.array(
                    [[0.15, 0.0], [0.15, 0.95], [0.5, 0.3], [0.85, 0.95], [0.85, 0.0]],
                    dtype=np.float64,
                )
            ]
        if char == "N":
            left = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
            diag = np.array([[0.2, 0.95], [0.8, 0.0]], dtype=np.float64)
            right = np.array([[0.8, 0.0], [0.8, 0.95]], dtype=np.float64)
            return [left, diag, right]
        if char == "O":
            t = np.linspace(0, 2 * np.pi, 32)
            return [
                np.stack(
                    [0.5 + 0.35 * np.cos(t), 0.5 + 0.45 * np.sin(t)],
                    axis=1,
                ).astype(np.float64)
            ]
        if char == "P":
            stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
            t = np.linspace(-np.pi / 2, np.pi / 2, 14)
            loop = np.stack(
                [0.2 + 0.45 * np.cos(t), 0.7 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            return [stem, loop]
        if char == "Q":
            t = np.linspace(0, 2 * np.pi, 32)
            body = np.stack(
                [0.5 + 0.35 * np.cos(t), 0.5 + 0.45 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            tail = np.array([[0.55, 0.2], [0.85, -0.05]], dtype=np.float64)
            return [body, tail]
        if char == "R":
            stem = np.array([[0.2, 0.0], [0.2, 0.95]], dtype=np.float64)
            t = np.linspace(-np.pi / 2, np.pi / 2, 14)
            loop = np.stack(
                [0.2 + 0.45 * np.cos(t), 0.7 + 0.25 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            leg = np.array([[0.4, 0.45], [0.85, 0.0]], dtype=np.float64)
            return [stem, loop, leg]
        if char == "S":
            t = np.linspace(0, 1, 32)
            x = 0.5 + 0.32 * np.sin(2 * np.pi * t)
            y = 0.95 - 0.9 * t
            return [np.stack([x, y], axis=1).astype(np.float64)]
        if char == "T":
            top = np.array([[0.15, 0.95], [0.85, 0.95]], dtype=np.float64)
            stem = np.array([[0.5, 0.95], [0.5, 0.0]], dtype=np.float64)
            return [top, stem]
        if char == "U":
            left = np.array([[0.2, 0.95], [0.2, 0.3]], dtype=np.float64)
            t = np.linspace(np.pi, 2 * np.pi, 18)
            cup = np.stack(
                [0.5 + 0.3 * np.cos(t), 0.3 + 0.3 * np.sin(t)],
                axis=1,
            ).astype(np.float64)
            right = np.array([[0.8, 0.3], [0.8, 0.95]], dtype=np.float64)
            return [left, cup, right]
        if char == "V":
            return [np.array([[0.15, 0.95], [0.5, 0.0], [0.85, 0.95]], dtype=np.float64)]
        if char == "W":
            return [
                np.array(
                    [[0.1, 0.95], [0.3, 0.0], [0.5, 0.6], [0.7, 0.0], [0.9, 0.95]],
                    dtype=np.float64,
                )
            ]
        if char == "X":
            d1 = np.array([[0.2, 0.0], [0.8, 0.95]], dtype=np.float64)
            d2 = np.array([[0.8, 0.0], [0.2, 0.95]], dtype=np.float64)
            return [d1, d2]
        if char == "Y":
            v = np.array([[0.2, 0.95], [0.5, 0.5], [0.8, 0.95]], dtype=np.float64)
            stem = np.array([[0.5, 0.5], [0.5, 0.0]], dtype=np.float64)
            return [v, stem]
        if char == "Z":
            return [
                np.array([[0.2, 0.95], [0.8, 0.95], [0.2, 0.0], [0.8, 0.0]], dtype=np.float64)
            ]
        return None

    @staticmethod
    def _strokes_and_types_from_sample(
        sample: StrokeSample,
    ) -> tuple[list[Stroke], list[str]]:
        """``StrokeSample`` を ``len>=2`` 条件で strokes/types を同期フィルタする。

        推論バッチ化（``inference.py``）は ``len>=2`` のストロークだけを同順で
        返すため、筆画タイプ（``stroke_types``）も同じ条件・同じ順序で間引き、
        ``positioned[j]`` と ``types[j]`` の 1 対 1 対応を保つ。

        Args:
            sample: 読み込んだ ``StrokeSample``。

        Returns:
            ``(strokes, types)``。``types`` は対応する ``stroke_types``（無い・
            不足分は ``""`` で補完し、strokes と同数になる）。
        """
        raw_types = sample.stroke_types
        strokes: list[Stroke] = []
        types: list[str] = []
        for i, stroke_points in enumerate(sample.strokes):
            arr = np.array([[p.x, p.y] for p in stroke_points], dtype=np.float64)
            if len(arr) >= 2:
                strokes.append(arr)
                types.append(raw_types[i] if i < len(raw_types) else "")
        return strokes, types

    def _load_reference_strokes(
        self, char: str
    ) -> tuple[list[Stroke] | None, list[str]]:
        if self._kanjivg_dir is None:
            return None, []
        char_dir = self._kanjivg_dir / char
        if not char_dir.is_dir():
            return None, []
        json_files = sorted(char_dir.glob(f"{char}_*.json"))
        if not json_files:
            return None, []
        try:
            sample = StrokeSample.load(json_files[0])
            strokes, types = self._strokes_and_types_from_sample(sample)
            return (strokes if strokes else None), types
        except Exception:
            return None, []

    def _load_kanjivg_json(
        self, placement: CharPlacement
    ) -> tuple[list[Stroke] | None, list[str]]:
        char_dir = self._kanjivg_dir / placement.char
        if not char_dir.is_dir():
            return None, []
        json_files = sorted(char_dir.glob(f"{placement.char}_*.json"))
        if not json_files:
            return None, []
        try:
            sample = StrokeSample.load(json_files[0])
            strokes, types = self._strokes_and_types_from_sample(sample)
            return (strokes if strokes else None), types
        except Exception:
            logger.warning("KanjiVG JSON load failed for '%s'", placement.char, exc_info=True)
            return None, []

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
            cell_width = fs * 0.55
        else:
            cell_width = fs * min(char_scale, 0.95)

        scale_w = cell_width / ranges[0] if ranges[0] > 1e-6 else float("inf")
        scale_h = target_h / ranges[1] if ranges[1] > 1e-6 else float("inf")
        scale = min(scale_w, scale_h)

        scaled = [(stroke - mins) * scale for stroke in strokes]
        rendered_w = ranges[0] * scale
        rendered_h = ranges[1] * scale

        if is_halfwidth(placement.char):
            x_offset = placement.x + (fs * 0.55 - rendered_w) / 2
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
