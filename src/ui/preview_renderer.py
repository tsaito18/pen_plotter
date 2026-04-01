from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.gcode.config import PlotterConfig
from src.gcode.preview import _draw_stroke_with_width
from src.layout.page_layout import PageConfig

Stroke = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)

# ページ番号「P. N」の位置（用紙座標mm）
_PAGE_NUM_X = 22.0
_PAGE_NUM_Y = 8.5  # 下端からの距離


class PreviewRenderer:
    def __init__(
        self,
        *,
        plotter_config: PlotterConfig,
        page_config: PageConfig,
        report_bg_path: Path | str | None = None,
    ) -> None:
        self._plotter_config = plotter_config
        self._page_config = page_config
        self._report_bg_path: Path | None = Path(report_bg_path) if report_bg_path else None

    def preview_with_ruled_lines(
        self,
        strokes: list[Stroke],
        ruled_lines: list[Stroke],
        save_path: str | Path,
        page_number: int | None = None,
        page_number_strokes: list[Stroke] | None = None,
    ) -> None:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        cfg = self._plotter_config
        fig, ax = plt.subplots(1, 1, figsize=(10, 14))

        # 背景: スキャン画像 or 白
        bg_path = self._report_bg_path
        if bg_path and bg_path.exists():
            from PIL import Image

            bg_img = Image.open(bg_path)
            ax.imshow(
                bg_img,
                extent=[0, cfg.paper_width, 0, cfg.paper_height],
                aspect="auto",
                zorder=0,
            )
        else:
            paper_rect = patches.Rectangle(
                (cfg.paper_origin_x, cfg.paper_origin_y),
                cfg.paper_width,
                cfg.paper_height,
                linewidth=1,
                edgecolor="black",
                facecolor="white",
                linestyle="-",
            )
            ax.add_patch(paper_rect)

        # 紙の境界線（黒）
        paper_border = patches.Rectangle(
            (0, 0),
            cfg.paper_width,
            cfg.paper_height,
            linewidth=1.0,
            edgecolor="black",
            facecolor="none",
            linestyle="-",
            zorder=1,
        )
        ax.add_patch(paper_border)

        # 罫線（デバッグ: 赤で表示）
        for line in ruled_lines:
            if len(line) >= 2:
                ax.plot(line[:, 0], line[:, 1], color="#ff000044", linewidth=0.5)

        # 文字ストローク（黒）
        for stroke in strokes:
            if len(stroke) >= 2:
                _draw_stroke_with_width(ax, stroke, color="#1a1a1a")

        # ページ番号（手書きストローク）
        if page_number_strokes:
            for stroke in page_number_strokes:
                if len(stroke) >= 2:
                    _draw_stroke_with_width(ax, stroke, color="#1a1a1a")

        ax.set_xlim(-2, cfg.paper_width + 2)
        ax.set_ylim(-2, cfg.paper_height + 2)
        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=300)
        plt.close(fig)
