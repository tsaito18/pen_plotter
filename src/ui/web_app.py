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
from src.layout.line_breaking import is_halfwidth
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
        self._typesetter = Typesetter(self._page_config, font_size=5.0)
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

    _SKIP_RENDER = set(" \t　")

    _CHAR_SUBSTITUTIONS: dict[str, str] = {
        '（': '(',
        '）': ')',
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

    def _generate_char_strokes(self, placement: CharPlacement) -> list[Stroke]:
        """Tier 1: ML推論 → Tier 2: KanjiVG → Tier 3: 括弧等の幾何生成 → Tier 4: 矩形。"""
        if placement.char in self._SKIP_RENDER:
            return []

        original_char = placement.char
        lookup_char = self._CHAR_SUBSTITUTIONS.get(original_char, original_char)
        if lookup_char != original_char:
            placement = CharPlacement(
                char=lookup_char, x=placement.x, y=placement.y,
                font_size=placement.font_size, page=placement.page,
            )

        reference = self._load_reference_strokes(placement.char)

        if self._inference is not None:
            try:
                raw = self._inference.generate(
                    self._style_sample,
                    num_steps=50,
                    temperature=self._temperature,
                    reference_strokes=reference,
                )
                return self._position_strokes(raw, placement)
            except Exception:
                logger.warning("ML inference failed for '%s'", placement.char, exc_info=True)

        if self._kanjivg_dir is not None:
            char_strokes = self._load_kanjivg_json(placement)
            if char_strokes is not None:
                return self._position_strokes(char_strokes, placement)

        paren_strokes = self._simple_paren_strokes(original_char, placement)
        if paren_strokes is not None:
            return self._position_strokes(paren_strokes, placement)

        return self._rect_fallback(placement)

    def _simple_paren_strokes(self, char: str, placement: CharPlacement) -> list[Stroke] | None:
        """Generate normalized arc strokes for parentheses (0-1 range)."""
        if char in ('(', '（'):
            points = []
            for i in range(20):
                t = i / 19
                x = 0.3 * np.cos(np.pi * 0.6 * (t - 0.5))
                y = t
                points.append([x, y])
            return [np.array(points)]
        elif char in (')', '）'):
            points = []
            for i in range(20):
                t = i / 19
                x = 1.0 - 0.3 * np.cos(np.pi * 0.6 * (t - 0.5))
                y = t
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

    def _position_strokes(self, strokes: list[Stroke], placement: CharPlacement) -> list[Stroke]:
        """正規化ストロークをCharPlacementの位置・サイズに合わせる。

        アスペクト比を保持し、セル内で中央配置する。
        """
        if not strokes:
            return []

        all_pts = np.concatenate(strokes, axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        ranges = maxs - mins
        current_size = ranges.max() if ranges.max() > 0 else 1.0

        fs = placement.font_size

        if placement.char in self._SMALL_PUNCT:
            target_size = fs * 0.4
        elif placement.char in self._SMALL_KANA:
            target_size = fs * 0.6
        else:
            target_size = fs

        scale = target_size / current_size
        scaled = [(stroke - mins) * scale for stroke in strokes]

        rendered_w = ranges[0] * scale
        rendered_h = ranges[1] * scale

        cell_width = fs * 0.6 if is_halfwidth(placement.char) else fs
        x_offset = placement.x + (cell_width - rendered_w) / 2
        line_spacing = self._page_config.line_spacing
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
                ax.plot(stroke[:, 0], stroke[:, 1], "b-", linewidth=0.5)

        ax.set_xlim(-2, cfg.paper_width + 2)
        ax.set_ylim(-2, cfg.paper_height + 2)
        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=300)
        plt.close(fig)

    def generate_preview(self, text: str, save_path: str | Path) -> None:
        save_path = Path(save_path)
        pages = self.text_to_placements(text)
        ruled_lines = self._typesetter._layout.ruled_line_strokes()
        if not pages or not pages[0]:
            self._preview_with_ruled_lines([], ruled_lines, save_path)
            return
        strokes = self.placements_to_strokes(pages[0])
        optimized = optimize_stroke_order(strokes)
        self._preview_with_ruled_lines(
            optimized, ruled_lines, save_path,
        )

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
