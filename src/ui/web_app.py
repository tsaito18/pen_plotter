from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.gcode.config import PlotterConfig
from src.gcode.generator import GCodeGenerator
from src.gcode.optimizer import optimize_stroke_order
from src.layout.page_layout import PageConfig
from src.layout.typesetter import CharPlacement, Typesetter
from src.model.augmentation import HandwritingAugmenter

Stroke = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


from src.ui.stroke_renderer import CharCoverageReport  # noqa: F401 (re-export)


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
            margin_top=48.0,
            margin_bottom=34.0,
            margin_left=5.0,
            margin_right=5.0,
            line_spacing=7.16,
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
            self._page_config, font_size=4.5, augmenter=HandwritingAugmenter()
        )
        self._generator = GCodeGenerator(self._plotter_config)

        from src.ui.preview_renderer import PreviewRenderer
        from src.ui.stroke_renderer import StrokeRenderer

        self._stroke_renderer = StrokeRenderer(
            checkpoint_path=checkpoint_path,
            kanjivg_dir=kanjivg_dir,
            style_sample=style_sample,
            temperature=temperature,
            user_strokes_dir=user_strokes_dir,
            augmenter=self._typesetter.augmenter,
            page_config=self._page_config,
        )

        default_bg = Path("data/report_paper.jpg")
        self._preview_renderer = PreviewRenderer(
            plotter_config=self._plotter_config,
            page_config=self._page_config,
            report_bg_path=default_bg if default_bg.exists() else None,
        )

    # --- StrokeRenderer への委譲プロパティ ---

    @property
    def _inference(self):
        return self._stroke_renderer._inference

    @_inference.setter
    def _inference(self, value):
        self._stroke_renderer._inference = value

    @property
    def _style_sample(self):
        return self._stroke_renderer._style_sample

    @_style_sample.setter
    def _style_sample(self, value):
        self._stroke_renderer._style_sample = value

    @property
    def _temperature(self):
        return self._stroke_renderer._temperature

    @_temperature.setter
    def _temperature(self, value):
        self._stroke_renderer._temperature = value

    @property
    def _user_stroke_db(self):
        return self._stroke_renderer._user_stroke_db

    @property
    def _last_coverage(self):
        return self._stroke_renderer._last_coverage

    @_last_coverage.setter
    def _last_coverage(self, value):
        self._stroke_renderer._last_coverage = value

    @property
    def _kanjivg_dir(self):
        return self._stroke_renderer._kanjivg_dir

    # --- StrokeRenderer へのメソッド委譲 ---

    def _generate_char_strokes(self, placement: CharPlacement) -> list[Stroke]:
        return self._stroke_renderer.generate_char_strokes(placement)

    def _apply_distortion(self, strokes: list[Stroke]) -> list[Stroke]:
        return self._stroke_renderer._apply_distortion(strokes)

    def _direct_stroke(self, char: str) -> list[Stroke] | None:
        return self._stroke_renderer._direct_stroke(char)

    @staticmethod
    def _normalize_strokes_to_unit(strokes: list[Stroke]) -> list[Stroke]:
        from src.ui.stroke_renderer import StrokeRenderer

        return StrokeRenderer._normalize_strokes_to_unit(strokes)

    def _apply_stroke_variation(self, strokes: list[Stroke]) -> list[Stroke]:
        return self._stroke_renderer._apply_stroke_variation(strokes)

    def _math_symbol_strokes(self, char: str) -> list[Stroke] | None:
        return self._stroke_renderer._math_symbol_strokes(char)

    def _simple_punct_strokes(self, char: str) -> list[Stroke] | None:
        return self._stroke_renderer._simple_punct_strokes(char)

    def _simple_paren_strokes(self, char: str, placement: CharPlacement) -> list[Stroke] | None:
        return self._stroke_renderer._simple_paren_strokes(char, placement)

    def _load_reference_strokes(self, char: str) -> list[Stroke] | None:
        return self._stroke_renderer._load_reference_strokes(char)

    def _load_kanjivg_json(self, placement: CharPlacement) -> list[Stroke] | None:
        return self._stroke_renderer._load_kanjivg_json(placement)

    def _position_strokes(self, strokes: list[Stroke], placement: CharPlacement) -> list[Stroke]:
        return self._stroke_renderer._position_strokes(strokes, placement)

    @staticmethod
    def _rect_fallback(p: CharPlacement) -> list[Stroke]:
        from src.ui.stroke_renderer import StrokeRenderer

        return StrokeRenderer._rect_fallback(p)

    @staticmethod
    def _load_user_stroke_db(
        user_strokes_dir: Path | str | None,
    ) -> dict[str, list[list[Stroke]]]:
        from src.ui.stroke_renderer import StrokeRenderer

        return StrokeRenderer._load_user_stroke_db(user_strokes_dir)

    @staticmethod
    def _load_style_from_user_strokes(
        user_strokes_dir: Path | str | None,
    ) -> object:
        from src.ui.stroke_renderer import StrokeRenderer

        return StrokeRenderer._load_style_from_user_strokes(user_strokes_dir)

    # --- PreviewRenderer への委譲 ---

    _REPORT_PAPER_BG: Path | None = None

    @classmethod
    def set_report_paper_bg(cls, path: Path | str | None) -> None:
        cls._REPORT_PAPER_BG = Path(path) if path else None

    def _preview_with_ruled_lines(
        self,
        strokes: list[Stroke],
        ruled_lines: list[Stroke],
        save_path: str | Path,
        page_number: int | None = None,
        page_number_strokes: list[Stroke] | None = None,
    ) -> None:
        if self._REPORT_PAPER_BG:
            self._preview_renderer._report_bg_path = self._REPORT_PAPER_BG
        self._preview_renderer.preview_with_ruled_lines(
            strokes, ruled_lines, save_path,
            page_number=page_number,
            page_number_strokes=page_number_strokes,
        )

    # --- パイプライン本体 ---

    def text_to_placements(self, text: str) -> list[list[CharPlacement]]:
        return self._typesetter.typeset(text)

    def placements_to_strokes(
        self,
        placements: list[CharPlacement],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[Stroke]:
        self._stroke_renderer._augmenter = self._typesetter.augmenter
        self._stroke_renderer._last_coverage = CharCoverageReport()
        strokes: list[Stroke] = []
        prev_end_x: float | None = None
        augmenter = self._typesetter.augmenter
        has_real_renderer = self._inference is not None or self._kanjivg_dir is not None
        total = len(placements)

        for i, p in enumerate(placements):
            if progress_callback:
                progress_callback(
                    i / max(total, 1),
                    f"\u30b9\u30c8\u30ed\u30fc\u30af\u751f\u6210\u4e2d ({i + 1}/{total})",
                )
            char_strokes = self._stroke_renderer.generate_char_strokes(p)

            if (
                prev_end_x is not None
                and augmenter is not None
                and has_real_renderer
                and char_strokes
            ):
                shift = augmenter.random_uniform(0, 0.1)
                char_strokes = [s - np.array([shift, 0.0]) for s in char_strokes]

            if char_strokes:
                last_stroke = char_strokes[-1]
                prev_end_x = last_stroke[-1, 0]

            strokes.extend(char_strokes)

            if getattr(p, "role", None) == "numerator":
                next_denom = None
                for j in range(i + 1, len(placements)):
                    if getattr(placements[j], "role", None) == "denominator":
                        next_denom = placements[j]
                        break
                if next_denom is not None:
                    line_x0 = min(p.x, next_denom.x) - p.font_size * 0.1
                    line_x1 = (
                        max(p.x + p.font_size, next_denom.x + next_denom.font_size)
                        + p.font_size * 0.1
                    )
                    line_y = (p.y + next_denom.y) / 2
                    strokes.append(np.array([[line_x0, line_y], [line_x1, line_y]]))

        return strokes

    def _generate_page_number_strokes(self, page_number: int) -> list[Stroke]:
        """ページ番号を手書きストロークで生成。用紙の「P.」印字の右横に配置。"""
        text = str(page_number)
        x = 30.0   # 「P.」印字の右、下線の上
        y = 7.0    # 下端からの距離（下線の上に載るように）
        font_size = 3.5
        all_strokes: list[Stroke] = []
        for ch in text:
            placement = CharPlacement(char=ch, x=x, y=y, font_size=font_size)
            char_strokes = self._stroke_renderer.generate_char_strokes(placement)
            all_strokes.extend(char_strokes)
            x += font_size * 0.5
        return all_strokes

    def strokes_to_gcode(self, strokes: list[Stroke]) -> list[str]:
        optimized = optimize_stroke_order(strokes)
        return self._generator.generate(optimized)

    def generate_gcode(
        self,
        text: str,
        save_path: str | Path,
    ) -> Path:
        """テキストからG-codeを生成して保存。ストロークは書き順（文字順）を保持。"""
        save_path = Path(save_path)
        pages = self.text_to_placements(text)
        if not pages or not pages[0]:
            self._generator.save(self._generator.generate([]), save_path)
            return save_path

        all_strokes: list[Stroke] = []
        for i, page_placements in enumerate(pages, start=1):
            strokes = self.placements_to_strokes(page_placements)
            page_num_strokes = self._generate_page_number_strokes(i)
            all_strokes.extend(strokes)
            all_strokes.extend(page_num_strokes)

        # ストローク順序を保持（optimize_stroke_orderを使わない）
        gcode = self._generator.generate(all_strokes, vary_speed=True)
        self._generator.save(gcode, save_path)
        return save_path

    def generate_preview(
        self,
        text: str,
        save_path: str | Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[Path]:
        save_path = Path(save_path)

        if progress_callback:
            progress_callback(0.0, "\u7d44\u7248\u4e2d...")

        pages = self.text_to_placements(text)
        ruled_lines = self._typesetter._layout.ruled_line_strokes()

        if not pages or not pages[0]:
            self._preview_with_ruled_lines([], ruled_lines, save_path)
            if progress_callback:
                progress_callback(1.0, "\u5b8c\u4e86")
            return [save_path]

        if progress_callback:
            progress_callback(0.05, "\u30b9\u30c8\u30ed\u30fc\u30af\u751f\u6210\u4e2d...")

        n_pages = len(pages)
        stem = save_path.stem
        suffix = save_path.suffix
        parent = save_path.parent
        result: list[Path] = []

        for i, page_placements in enumerate(pages, start=1):
            page_base = (i - 1) / n_pages
            page_span = 1.0 / n_pages

            def _page_stroke_progress(
                frac: float, desc: str, _base=page_base, _span=page_span
            ) -> None:
                if progress_callback:
                    progress_callback(_base + frac * _span * 0.8, desc)

            page_path = save_path if n_pages == 1 else parent / f"{stem}_p{i}{suffix}"
            strokes = self.placements_to_strokes(
                page_placements, progress_callback=_page_stroke_progress
            )
            if progress_callback:
                progress_callback(
                    page_base + page_span * 0.85,
                    f"ストローク最適化中 ({i}/{n_pages})...",
                )
            optimized = optimize_stroke_order(strokes)
            if progress_callback:
                progress_callback(
                    page_base + page_span * 0.9,
                    f"プレビュー描画中 ({i}/{n_pages})...",
                )
            page_num_strokes = self._generate_page_number_strokes(i)
            self._preview_with_ruled_lines(
                optimized, ruled_lines, page_path,
                page_number=i, page_number_strokes=page_num_strokes,
            )
            result.append(page_path)

        if progress_callback:
            progress_callback(1.0, "完了")
        return result

    def generate_gcode_file(
        self,
        text: str,
        save_path: str | Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        save_path = Path(save_path)

        if progress_callback:
            progress_callback(0.0, "\u7d44\u7248\u4e2d...")

        pages = self.text_to_placements(text)
        if not pages or not pages[0]:
            gcode = self._generator.generate([])
            self._generator.save(gcode, save_path)
            if progress_callback:
                progress_callback(1.0, "\u5b8c\u4e86")
            return

        if progress_callback:
            progress_callback(0.1, "\u30b9\u30c8\u30ed\u30fc\u30af\u751f\u6210\u4e2d...")

        def _stroke_progress(frac: float, desc: str) -> None:
            if progress_callback:
                progress_callback(0.1 + frac * 0.6, desc)

        strokes = self.placements_to_strokes(pages[0], progress_callback=_stroke_progress)

        if progress_callback:
            progress_callback(0.7, "G-code \u5909\u63db\u4e2d...")

        gcode = self.strokes_to_gcode(strokes)
        self._generator.save(gcode, save_path)

        if progress_callback:
            progress_callback(1.0, "\u5b8c\u4e86")

    def create_app(self):
        try:
            import gradio as gr  # noqa: F401
        except ImportError:
            return None

        from src.ui.gradio_app import create_app

        return create_app(self)
