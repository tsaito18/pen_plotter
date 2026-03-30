from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.collector.data_format import StrokeSample
from src.collector.kanjivg_parser import KanjiVGParser
from src.gcode.config import PlotterConfig
from src.gcode.generator import GCodeGenerator
from src.gcode.optimizer import optimize_stroke_order
from src.gcode.preview import _draw_stroke_with_width, preview_strokes
from src.layout.line_breaking import is_halfwidth
from src.layout.page_layout import PageConfig
from src.layout.typesetter import CharPlacement, Typesetter
from src.model.augmentation import HandwritingAugmenter

Stroke = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class PlotterPipeline:
    def __init__(
        self,
        page_config: PageConfig | None = None,
        plotter_config: PlotterConfig | None = None,
        checkpoint_path: Path | str | None = None,
        kanjivg_dir: Path | str | None = None,
        style_sample: object | None = None,
        temperature: float = 1.0,
        user_strokes_dir: Path | str | None = None,
    ) -> None:
        self._page_config = page_config or PageConfig(
            paper_size=(210.0, 297.0),
            margin_top=30.0,
            margin_bottom=15.0,
            margin_left=25.0,
            margin_right=15.0,
            line_spacing=8.0,
        )
        self._plotter_config = plotter_config or PlotterConfig(
            work_area_width=220.0,
            work_area_height=310.0,
            paper_origin_x=0.0,
            paper_origin_y=0.0,
            paper_width=self._page_config.paper_size[0],
            paper_height=self._page_config.paper_size[1],
        )
        self._typesetter = Typesetter(
            self._page_config, font_size=6.0, augmenter=HandwritingAugmenter()
        )
        self._generator = GCodeGenerator(self._plotter_config)
        self._temperature = temperature

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

        self._kanjivg_parser: KanjiVGParser | None = None
        self._kanjivg_dir: Path | None = None
        if kanjivg_dir is not None:
            d = Path(kanjivg_dir)
            if d.is_dir():
                self._kanjivg_parser = KanjiVGParser()
                self._kanjivg_dir = d

    @staticmethod
    def _load_user_stroke_db(
        user_strokes_dir: Path | str | None,
    ) -> dict[str, list[list[Stroke]]]:
        """ユーザーストロークJSONを全てロードしてDBを構築。"""
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
        """ユーザーストロークからstyle_sampleテンソルを生成。データなしはゼロベクトル。"""
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
                        style_sample = deltas.unsqueeze(0)  # (1, N, 3)
                        logger.info(
                            "Loaded style sample from %d files (%d points)",
                            len(json_files),
                            deltas.shape[0],
                        )
                        return style_sample

        return torch.zeros(1, 10, 3)

    def text_to_placements(self, text: str) -> list[list[CharPlacement]]:
        return self._typesetter.typeset(text)

    def placements_to_strokes(self, placements: list[CharPlacement]) -> list[Stroke]:
        """各文字のストロークを3段階フォールバックで生成。"""
        strokes: list[Stroke] = []
        prev_end_x: float | None = None
        augmenter = self._typesetter.augmenter
        has_real_renderer = self._inference is not None or self._kanjivg_dir is not None

        for i, p in enumerate(placements):
            char_strokes = self._generate_char_strokes(p)

            if prev_end_x is not None and augmenter is not None and has_real_renderer and char_strokes:
                shift = augmenter.random_uniform(0, 0.1)
                char_strokes = [s - np.array([shift, 0.0]) for s in char_strokes]

            if char_strokes:
                last_stroke = char_strokes[-1]
                prev_end_x = last_stroke[-1, 0]

            strokes.extend(char_strokes)

            if getattr(p, 'role', None) == "numerator":
                next_denom = None
                for j in range(i + 1, len(placements)):
                    if getattr(placements[j], 'role', None) == "denominator":
                        next_denom = placements[j]
                        break
                if next_denom is not None:
                    line_x0 = min(p.x, next_denom.x) - p.font_size * 0.1
                    line_x1 = max(
                        p.x + p.font_size, next_denom.x + next_denom.font_size
                    ) + p.font_size * 0.1
                    line_y = (p.y + next_denom.y) / 2
                    strokes.append(np.array([[line_x0, line_y], [line_x1, line_y]]))

        return strokes

    _SKIP_RENDER = set(" \t　")

    _CHAR_SUBSTITUTIONS: dict[str, str] = {
        '｛': '{',
        '｝': '}',
        '［': '[',
        '］': ']',
        '！': '!',
        '？': '?',
        '：': ':',
        '；': ';',
        '＝': '=',
        '＋': '+',
        '－': '-',
        '／': '/',
    }

    # 一筆系の文字: 弾性変形・tremorを適用しない（滑らかさを保つ）
    _SMOOTH_CHARS = set("、。，．・ー～—―()（）「」『』【】〈〉《》〔〕")

    def _generate_char_strokes(self, placement: CharPlacement) -> list[Stroke]:
        """Tier 0: 句読点/括弧 → Tier 1: 直接ストローク → Tier 2: ML推論 → Tier 3: KanjiVG → Tier 4: 矩形。"""
        if placement.char in self._SKIP_RENDER:
            return []

        original_char = placement.char
        lookup_char = self._CHAR_SUBSTITUTIONS.get(original_char, original_char)
        if lookup_char != original_char:
            placement = CharPlacement(
                char=lookup_char, x=placement.x, y=placement.y,
                font_size=placement.font_size, page=placement.page,
            )

        is_smooth = original_char in self._SMOOTH_CHARS or lookup_char in self._SMOOTH_CHARS

        # Tier 0: 句読点の幾何生成（KanjiVGデータの不具合を回避）
        punct_strokes = self._simple_punct_strokes(lookup_char)
        if punct_strokes is not None:
            return self._position_strokes(punct_strokes, placement)

        # Tier 1: ユーザー直接ストローク
        direct = self._direct_stroke(placement.char)
        if direct is not None:
            positioned = self._position_strokes(direct, placement)
            return positioned if is_smooth else self._apply_distortion(positioned)

        reference = self._load_reference_strokes(placement.char)

        # Tier 2: ML推論
        if self._inference is not None:
            try:
                raw = self._inference.generate(
                    self._style_sample,
                    num_steps=50,
                    temperature=self._temperature,
                    reference_strokes=reference,
                )
                positioned = self._position_strokes(raw, placement)
                return positioned if is_smooth else self._apply_distortion(positioned)
            except Exception:
                logger.warning("ML inference failed for '%s'", placement.char, exc_info=True)

        # Tier 3: 括弧・数式記号の幾何生成
        paren_strokes = self._simple_paren_strokes(original_char, placement)
        if paren_strokes is not None:
            return self._position_strokes(paren_strokes, placement)

        math_strokes = self._math_symbol_strokes(lookup_char)
        if math_strokes is not None:
            return self._position_strokes(math_strokes, placement)

        # Tier 4: KanjiVG
        if self._kanjivg_dir is not None:
            char_strokes = self._load_kanjivg_json(placement)
            if char_strokes is not None:
                positioned = self._position_strokes(char_strokes, placement)
                return positioned if is_smooth else self._apply_distortion(positioned)

        return self._rect_fallback(placement)

    def _apply_distortion(self, strokes: list[Stroke]) -> list[Stroke]:
        """弾性変形+手の震えを全ストロークに適用。"""
        aug = self._typesetter.augmenter
        if aug is None:
            return strokes
        strokes = [aug.elastic_distort(s) for s in strokes]
        strokes = [aug.apply_tremor(s) for s in strokes]
        return strokes

    _NOISE_SCALE = 0.3

    def _direct_stroke(self, char: str) -> list[Stroke] | None:
        """ユーザー直接ストロークを合成して正規化(0-1)で返す。"""
        samples = self._user_stroke_db.get(char)
        if not samples:
            return None

        if len(samples) == 1:
            normalized = self._normalize_strokes_to_unit(samples[0])
        else:
            # 各サンプルを先に正規化してから合成（座標系を統一）
            normalized_samples = [self._normalize_strokes_to_unit(s) for s in samples]
            min_stroke_count = min(len(s) for s in normalized_samples)
            normalized = []
            for i in range(min_stroke_count):
                candidates = [s[i] for s in normalized_samples if i < len(s)]
                normalized.append(candidates[np.random.randint(len(candidates))])

        varied = self._apply_stroke_variation(normalized)
        return varied

    @staticmethod
    def _normalize_strokes_to_unit(strokes: list[Stroke]) -> list[Stroke]:
        """iPad座標のストロークを 0-1 範囲に正規化（Y軸反転: Canvas→組版座標）。"""
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
            normalized[:, 1] = 1.0 - normalized[:, 1]  # Y反転
            result.append(normalized)
        return result

    def _apply_stroke_variation(self, strokes: list[Stroke]) -> list[Stroke]:
        """ストロークごとの幾何バリエーション（inference.py _generate_v3 と同等）。"""
        ns = self._NOISE_SCALE
        result = []
        for stroke in strokes:
            center = stroke.mean(axis=0)
            centered = stroke - center
            angle = np.random.normal(0, ns * 0.15)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotated = centered @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            sx = 1.0 + np.random.normal(0, ns * 0.08)
            sy = 1.0 + np.random.normal(0, ns * 0.08)
            scaled = rotated * np.array([sx, sy])
            dx = np.random.normal(0, ns * 0.3)
            dy = np.random.normal(0, ns * 0.3)
            result.append(scaled + center + np.array([dx, dy]))
        return result

    def _math_symbol_strokes(self, char: str) -> list[Stroke] | None:
        """Generate normalized strokes for math symbols and Greek letters (0-1 range)."""
        if char == "ω":
            t = np.linspace(0, 1, 30)
            x = t
            y = 0.3 + 0.3 * np.abs(np.sin(2 * np.pi * t))
            return [np.stack([x, y], axis=1)]
        elif char == "φ":
            angles = np.linspace(0, 2 * np.pi, 24)
            r = 0.3
            circle = np.stack([0.5 + r * np.cos(angles), 0.55 + r * np.sin(angles)], axis=1)
            stem = np.array([[0.5, 0.1], [0.5, 0.9]])
            return [circle, stem]
        elif char == "π":
            top = np.array([[0.15, 0.25], [0.85, 0.25]])
            left_leg = np.array([[0.35, 0.25], [0.30, 0.85]])
            right_leg = np.array([[0.65, 0.25], [0.70, 0.85]])
            return [top, left_leg, right_leg]
        elif char == "θ":
            angles = np.linspace(0, 2 * np.pi, 24)
            rx, ry = 0.3, 0.4
            ellipse = np.stack([0.5 + rx * np.cos(angles), 0.5 + ry * np.sin(angles)], axis=1)
            bar = np.array([[0.2, 0.5], [0.8, 0.5]])
            return [ellipse, bar]
        elif char == "α":
            t = np.linspace(0, 2 * np.pi, 30)
            x = 0.5 + 0.3 * np.cos(t) - 0.1 * np.sin(2 * t)
            y = 0.5 + 0.35 * np.sin(t)
            return [np.stack([x, y], axis=1)]
        elif char == "Δ":
            triangle = np.array([[0.5, 0.1], [0.1, 0.9], [0.9, 0.9], [0.5, 0.1]])
            return [triangle]
        elif char == "±":
            h_top = np.array([[0.15, 0.2], [0.85, 0.2]])
            h_mid = np.array([[0.15, 0.5], [0.85, 0.5]])
            v_mid = np.array([[0.5, 0.2], [0.5, 0.8]])
            return [h_top, h_mid, v_mid]
        elif char == "≈":
            t = np.linspace(0, 2 * np.pi, 20)
            x = np.linspace(0.1, 0.9, 20)
            wave1 = np.stack([x, 0.35 + 0.08 * np.sin(t)], axis=1)
            wave2 = np.stack([x, 0.65 + 0.08 * np.sin(t)], axis=1)
            return [wave1, wave2]
        elif char == "∞":
            t = np.linspace(0, 2 * np.pi, 40)
            x = 0.5 + 0.35 * np.cos(t) / (1 + np.sin(t) ** 2)
            y = 0.5 + 0.25 * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
            return [np.stack([x, y], axis=1)]
        return None

    def _simple_punct_strokes(self, char: str) -> list[Stroke] | None:
        """Generate normalized strokes for common punctuation (0-1 range)."""
        if char in ('、', ','):
            return [np.array([[0.6, 0.2], [0.3, 0.8]])]
        elif char in ('。', '.'):
            angles = np.linspace(0, 2 * np.pi, 16)
            r = 0.3
            return [np.stack([0.5 + r * np.cos(angles), 0.5 + r * np.sin(angles)], axis=1)]
        elif char == '・':
            angles = np.linspace(0, 2 * np.pi, 12)
            r = 0.15
            return [np.stack([0.5 + r * np.cos(angles), 0.5 + r * np.sin(angles)], axis=1)]
        return None

    def _simple_paren_strokes(self, char: str, placement: CharPlacement) -> list[Stroke] | None:
        """Generate normalized arc strokes for parentheses (0-1 range)."""
        if char in ('(', '（'):
            points = []
            for i in range(20):
                t = i / 19
                x = 0.15 + 0.25 * np.cos(np.pi * (t - 0.5))
                y = 0.1 + 0.8 * t
                points.append([x, y])
            return [np.array(points)]
        elif char in (')', '）'):
            points = []
            for i in range(20):
                t = i / 19
                x = 0.85 - 0.25 * np.cos(np.pi * (t - 0.5))
                y = 0.1 + 0.8 * t
                points.append([x, y])
            return [np.array(points)]
        return None

    def _load_reference_strokes(self, char: str) -> list[Stroke] | None:
        """KanjiVG参照ストロークをNDArrayリストとして読み込む（CharEncoder入力用）。"""
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
        """data/strokes/{char}/{char}_*.json からストロークを読み込む。"""
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

    _SMALL_KANA = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ")
    _SMALL_PUNCT = set("。、.,")

    def _char_scale_factor(self, char: str) -> float:
        """文字種別に応じたスケール係数。

        日本語組版では平仮名・片仮名は漢字よりやや小さく組む。
        """
        cp = ord(char)
        if char in self._SMALL_KANA:
            return 0.55
        elif char in self._SMALL_PUNCT:
            return 0.35
        elif 0x3040 <= cp <= 0x309F:  # Hiragana
            return 0.88
        elif 0x30A0 <= cp <= 0x30FF:  # Katakana
            return 0.85
        else:
            return 1.0

    def _position_strokes(self, strokes: list[Stroke], placement: CharPlacement) -> list[Stroke]:
        """正規化ストロークをCharPlacementの位置・サイズに合わせる。

        アスペクト比を保持し、セル内で中央配置する。
        半角文字はセル幅に収まるよう制約する。
        """
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

        scale_w = cell_width / ranges[0] if ranges[0] > 1e-6 else float('inf')
        scale_h = target_h / ranges[1] if ranges[1] > 1e-6 else float('inf')
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
        """矩形フォールバック（従来実装）。"""
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

    def strokes_to_gcode(self, strokes: list[Stroke]) -> list[str]:
        optimized = optimize_stroke_order(strokes)
        return self._generator.generate(optimized)

    def _preview_with_ruled_lines(
        self,
        strokes: list[Stroke],
        ruled_lines: list[Stroke],
        save_path: str | Path,
        page_number: int | None = None,
    ) -> None:
        """罫線（薄グレー）+ 文字ストローク（青）を高解像度で描画。"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        cfg = self._plotter_config
        fig, ax = plt.subplots(1, 1, figsize=(10, 14))

        paper_rect = patches.Rectangle(
            (cfg.paper_origin_x, cfg.paper_origin_y),
            cfg.paper_width, cfg.paper_height,
            linewidth=1, edgecolor="gray", facecolor="lightyellow", linestyle="--",
        )
        ax.add_patch(paper_rect)

        for line in ruled_lines:
            if len(line) >= 2:
                ax.plot(line[:, 0], line[:, 1], color="#CCCCCC", linewidth=0.3)

        for stroke in strokes:
            if len(stroke) >= 2:
                _draw_stroke_with_width(ax, stroke)

        if page_number is not None:
            page_x = cfg.paper_width / 2
            page_y = cfg.paper_height - self._page_config.margin_bottom / 2
            ax.text(
                page_x, page_y, f"P. {page_number}",
                ha="center", va="center", fontsize=8, color="#666666",
            )

        ax.set_xlim(-2, cfg.paper_width + 2)
        ax.set_ylim(-2, cfg.paper_height + 2)
        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=300)
        plt.close(fig)

    def generate_preview(self, text: str, save_path: str | Path) -> list[Path]:
        save_path = Path(save_path)
        pages = self.text_to_placements(text)
        ruled_lines = self._typesetter._layout.ruled_line_strokes()

        if not pages or not pages[0]:
            self._preview_with_ruled_lines([], ruled_lines, save_path)
            return [save_path]

        if len(pages) == 1:
            strokes = self.placements_to_strokes(pages[0])
            optimized = optimize_stroke_order(strokes)
            self._preview_with_ruled_lines(optimized, ruled_lines, save_path, page_number=1)
            return [save_path]

        stem = save_path.stem
        suffix = save_path.suffix
        parent = save_path.parent
        result: list[Path] = []
        for i, page_placements in enumerate(pages, start=1):
            page_path = parent / f"{stem}_p{i}{suffix}"
            strokes = self.placements_to_strokes(page_placements)
            optimized = optimize_stroke_order(strokes)
            self._preview_with_ruled_lines(optimized, ruled_lines, page_path, page_number=i)
            result.append(page_path)
        return result

    def generate_gcode_file(self, text: str, save_path: str | Path) -> None:
        save_path = Path(save_path)
        pages = self.text_to_placements(text)
        if not pages or not pages[0]:
            gcode = self._generator.generate([])
            self._generator.save(gcode, save_path)
            return
        strokes = self.placements_to_strokes(pages[0])
        gcode = self.strokes_to_gcode(strokes)
        self._generator.save(gcode, save_path)

    def create_app(self):
        try:
            import gradio as gr
        except ImportError:
            return None

        def _on_preview(text: str):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = Path(f.name)
            paths = self.generate_preview(text, save_path=path)
            return [str(p) for p in paths]

        def _on_generate(text: str):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
                path = Path(f.name)
            self.generate_gcode_file(text, save_path=path)
            return str(path)

        with gr.Blocks(title="Pen Plotter") as app:
            gr.Markdown("# Pen Plotter Preview")
            text_input = gr.Textbox(label="Text", lines=5)
            with gr.Row():
                preview_btn = gr.Button("Preview")
                gcode_btn = gr.Button("Generate G-code")
            preview_img = gr.Gallery(label="Preview")
            gcode_file = gr.File(label="G-code")

            preview_btn.click(_on_preview, inputs=text_input, outputs=preview_img)
            gcode_btn.click(_on_generate, inputs=text_input, outputs=gcode_file)

        return app
