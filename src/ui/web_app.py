from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.gcode.config import PlotterConfig
from src.gcode.generator import GCodeGenerator
from src.gcode.optimizer import optimize_stroke_order
from src.gcode.preview import preview_strokes
from src.layout.page_layout import PageConfig
from src.layout.typesetter import CharPlacement, Typesetter

Stroke = npt.NDArray[np.float64]


class PlotterPipeline:
    def __init__(
        self,
        page_config: PageConfig | None = None,
        plotter_config: PlotterConfig | None = None,
    ) -> None:
        self._page_config = page_config or PageConfig()
        self._plotter_config = plotter_config or PlotterConfig()
        self._typesetter = Typesetter(self._page_config)
        self._generator = GCodeGenerator(self._plotter_config)

    def text_to_placements(self, text: str) -> list[list[CharPlacement]]:
        return self._typesetter.typeset(text)

    def placements_to_strokes(self, placements: list[CharPlacement]) -> list[Stroke]:
        """各文字を外接矩形ストロークに変換する仮実装。
        将来MLモデルの推論に差し替え予定。"""
        strokes: list[Stroke] = []
        for p in placements:
            half = p.font_size / 2.0
            x0, y0 = p.x, p.y - half
            x1, y1 = p.x + p.font_size, p.y + half
            rect = np.array([
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [x0, y0],
            ])
            strokes.append(rect)
        return strokes

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
