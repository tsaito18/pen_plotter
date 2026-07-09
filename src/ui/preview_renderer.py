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
        finishes: list[str] | None = None,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
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

        # 文字ストローク（黒）。finishes があれば対応 index の筆画タイプで
        # 太さプロファイルを切り替える。不足分は "none"（IndexError 回避）。
        pv = self._plotter_config.pressure_variation
        et = self._plotter_config.entry_taper
        for i, stroke in enumerate(strokes):
            if len(stroke) >= 2:
                finish = finishes[i] if finishes and i < len(finishes) else "none"
                _draw_stroke_with_width(
                    ax,
                    stroke,
                    color="#1a1a1a",
                    finish=finish,
                    pressure_variation=pv,
                    entry_taper=et,
                )
            elif len(stroke) == 1:
                # 単一点ストローク（中黒・/中点·）はペンを下ろすだけの点。小さな
                # 塗り円で描く（線にせず実機のペンダウン1点に対応）。
                ax.plot(
                    stroke[0, 0],
                    stroke[0, 1],
                    marker="o",
                    markersize=1.6,
                    markerfacecolor="#1a1a1a",
                    markeredgecolor="#1a1a1a",
                    linestyle="none",
                )

        # ページ番号（手書きストローク）は補助描画のため finish="none"
        if page_number_strokes:
            for stroke in page_number_strokes:
                if len(stroke) >= 2:
                    _draw_stroke_with_width(ax, stroke, color="#1a1a1a", finish="none")

        ax.set_xlim(-2, cfg.paper_width + 2)
        ax.set_ylim(-2, cfg.paper_height + 2)
        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=300)
        plt.close(fig)


# --- ProcessPoolExecutor 用ワーカー（モジュールレベル関数、pickle 可能） ---

# ワーカープロセスごとの PreviewRenderer キャッシュ。プロセスは
# ProcessPoolExecutor により使い回されるため、同一 config が続く限り
# 毎回の再構築を避ける。dataclass は非 hashable なので dict キーではなく
# 単純な == 比較で使い回し判定する。
_worker_renderer_state: dict[str, object] = {"key": None, "renderer": None}


def render_page_worker(
    strokes: list[Stroke],
    finishes: list[str],
    ruled_lines: list[Stroke],
    save_path: str | Path,
    page_number: int | None,
    page_number_strokes: list[Stroke] | None,
    plotter_config: PlotterConfig,
    page_config: PageConfig,
    report_bg_path: Path | None,
) -> None:
    """1ページ分のプレビュー描画をワーカープロセスで行う。

    ``ProcessPoolExecutor.submit`` の対象になるトップレベル関数（pickle 可能な
    引数のみ受け取る）。matplotlib はスレッドセーフでないため描画をプロセス
    並列にし、親プロセス（CUDA/モデル推論）とは別プロセスで実行する。
    """
    import matplotlib

    matplotlib.use("Agg")

    key = (plotter_config, page_config, report_bg_path)
    if _worker_renderer_state["key"] != key:
        _worker_renderer_state["renderer"] = PreviewRenderer(
            plotter_config=plotter_config,
            page_config=page_config,
            report_bg_path=report_bg_path,
        )
        _worker_renderer_state["key"] = key

    renderer: PreviewRenderer = _worker_renderer_state["renderer"]  # type: ignore[assignment]
    renderer.preview_with_ruled_lines(
        strokes,
        ruled_lines,
        save_path,
        page_number=page_number,
        page_number_strokes=page_number_strokes,
        finishes=finishes,
    )
