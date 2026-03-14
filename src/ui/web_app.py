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
from src.gcode.preview import preview_strokes
from src.layout.page_layout import PageConfig
from src.layout.typesetter import CharPlacement, Typesetter

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
    ) -> None:
        self._page_config = page_config or PageConfig(
            paper_size=(297.0, 210.0),
        )
        self._plotter_config = plotter_config or PlotterConfig()
        self._typesetter = Typesetter(self._page_config)
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
            try:
                import torch

                self._style_sample = torch.zeros(1, 10, 3)
            except ImportError:
                self._style_sample = None

        self._kanjivg_parser: KanjiVGParser | None = None
        self._kanjivg_dir: Path | None = None
        if kanjivg_dir is not None:
            d = Path(kanjivg_dir)
            if d.is_dir():
                self._kanjivg_parser = KanjiVGParser()
                self._kanjivg_dir = d

    def text_to_placements(self, text: str) -> list[list[CharPlacement]]:
        return self._typesetter.typeset(text)

    def placements_to_strokes(self, placements: list[CharPlacement]) -> list[Stroke]:
        """各文字のストロークを3段階フォールバックで生成。"""
        strokes: list[Stroke] = []
        for p in placements:
            char_strokes = self._generate_char_strokes(p)
            strokes.extend(char_strokes)
        return strokes

    def _generate_char_strokes(self, placement: CharPlacement) -> list[Stroke]:
        """Tier 1: ML推論 → Tier 2: KanjiVG → Tier 3: 矩形。"""
        if self._inference is not None:
            try:
                raw = self._inference.generate(
                    self._style_sample,
                    num_steps=50,
                    temperature=self._temperature,
                )
                return self._position_strokes(raw, placement)
            except Exception:
                logger.warning("ML inference failed for '%s'", placement.char, exc_info=True)

        if self._kanjivg_dir is not None:
            char_strokes = self._load_kanjivg_json(placement)
            if char_strokes is not None:
                return self._position_strokes(char_strokes, placement)

        return self._rect_fallback(placement)

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

    def _position_strokes(self, strokes: list[Stroke], placement: CharPlacement) -> list[Stroke]:
        """正規化ストロークをCharPlacementの位置・サイズに合わせる。"""
        if not strokes:
            return []

        all_pts = np.concatenate(strokes, axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        ranges = maxs - mins
        current_size = ranges.max() if ranges.max() > 0 else 1.0
        scale = placement.font_size / current_size

        half = placement.font_size / 2.0
        offset = np.array([placement.x, placement.y - half])

        return [(stroke - mins) * scale + offset for stroke in strokes]

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

    def generate_preview(self, text: str, save_path: str | Path) -> None:
        save_path = Path(save_path)
        pages = self.text_to_placements(text)
        if not pages or not pages[0]:
            preview_strokes([], config=self._plotter_config, save_path=save_path)
            return
        strokes = self.placements_to_strokes(pages[0])
        optimized = optimize_stroke_order(strokes)
        preview_strokes(optimized, config=self._plotter_config, save_path=save_path)

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
            self.generate_preview(text, save_path=path)
            return str(path)

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
            preview_img = gr.Image(label="Preview")
            gcode_file = gr.File(label="G-code")

            preview_btn.click(_on_preview, inputs=text_input, outputs=preview_img)
            gcode_btn.click(_on_generate, inputs=text_input, outputs=gcode_file)

        return app
