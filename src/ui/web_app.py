from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.gcode.config import PlotterConfig
from src.gcode.generator import GCodeGenerator
from src.gcode.optimizer import (
    optimize_stroke_order,
    optimize_stroke_order_with_finishes,
)
from src.layout.page_layout import PageConfig
from src.layout.typesetter import CharPlacement, Typesetter
from src.model.augmentation import AugmentConfig, HandwritingAugmenter
from src.model.stroke_finishing import insert_connections
from src.ui.stroke_renderer import CharCoverageReport  # noqa: F401 (re-export)

Stroke = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


@dataclass
class _DrawUnit:
    placement: CharPlacement
    strokes: list[Stroke]
    finishes: list[str]
    source_index: int
    serpentine: bool


def _scaled_augment_config(messiness: float) -> AugmentConfig:
    """汚さ倍率 messiness でレイアウト揺らぎ4項目をスケールした設定を返す。

    baseline_drift（行内上下）・spacing_variation（字間）・size_variation
    （字サイズ）・slant_variation（字の傾き）を一括倍率する。messiness=1.0 で
    AugmentConfig の素の値、0 で揺らぎなし（整った字）、2 で倍に乱れる。
    jitter/density 等は字形そのものの質感なので据え置く。
    """
    base = AugmentConfig()
    return replace(
        base,
        baseline_drift=base.baseline_drift * messiness,
        spacing_variation=base.spacing_variation * messiness,
        size_variation=base.size_variation * messiness,
        slant_variation=base.slant_variation * messiness,
    )


class PlotterPipeline:
    def __init__(
        self,
        page_config: PageConfig | None = None,
        plotter_config: PlotterConfig | None = None,
        checkpoint_path: Path | str | None = None,
        kanjivg_dir: Path | str | None = None,
        style_sample: object | None = None,
        temperature: float = 0.2,
        user_strokes_dir: Path | str | None = None,
        messiness: float = 0.4,
        instance_variation: float = 0.1,
        connection_strength: float = 0.0,
        skip_non_japanese: bool = False,
        seed: int | None = None,
        plot_page_numbers: bool = True,
    ) -> None:
        self._connection_strength = connection_strength
        self._plot_page_numbers = bool(plot_page_numbers)
        self._page_config = page_config or PageConfig(
            paper_size=(210.0, 297.0),
            margin_top=48.0,
            margin_bottom=34.0,
            margin_left=5.0,
            margin_right=5.0,
            line_spacing=7.14,
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
            self._page_config,
            font_size=4.5,
            # seed 指定時は augmenter の乱数を固定し、同一テキストで再現可能な
            # レイアウト揺らぎを得る（A/B目視比較の定点観測用）
            augmenter=HandwritingAugmenter(_scaled_augment_config(messiness), seed=seed),
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
            instance_variation=instance_variation,
            skip_non_japanese=skip_non_japanese,
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

    def _simple_punct_strokes(self, char: str) -> list[Stroke] | None:
        return self._stroke_renderer._simple_punct_strokes(char)

    def _simple_paren_strokes(self, char: str, placement: CharPlacement) -> list[Stroke] | None:
        return self._stroke_renderer._simple_paren_strokes(char, placement)

    def _load_reference_strokes(self, char: str) -> list[Stroke] | None:
        # renderer は (strokes, raw_types) を返すが、本シムは後方互換のため
        # strokes だけを返す（既存呼び出し元・テストの契約を維持）。
        strokes, _types = self._stroke_renderer._load_reference_strokes(char)
        return strokes

    def _load_kanjivg_json(self, placement: CharPlacement) -> list[Stroke] | None:
        strokes, _types = self._stroke_renderer._load_kanjivg_json(placement)
        return strokes

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
        finishes: list[str] | None = None,
    ) -> None:
        if self._REPORT_PAPER_BG:
            self._preview_renderer._report_bg_path = self._REPORT_PAPER_BG
        self._preview_renderer.preview_with_ruled_lines(
            strokes,
            ruled_lines,
            save_path,
            page_number=page_number,
            page_number_strokes=page_number_strokes,
            finishes=finishes,
        )

    # --- パイプライン本体 ---

    def text_to_placements(self, text: str) -> list[list[CharPlacement]]:
        return self._typesetter.typeset(text)

    def placements_to_strokes(
        self,
        placements: list[CharPlacement],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[Stroke]:
        """\u914d\u7f6e\u60c5\u5831\u304b\u3089\u30b9\u30c8\u30ed\u30fc\u30af\u5217\u3092\u751f\u6210\u3059\u308b\uff08\u5f8c\u65b9\u4e92\u63db\u30e9\u30c3\u30d1\u30fc\uff09\u3002

        ``placements_to_strokes_with_finishes`` \u306e strokes \u90e8\u5206\u306e\u307f\u3092\u8fd4\u3059\u3002
        G-code \u7d4c\u8def\u30fb\u65e2\u5b58\u30c6\u30b9\u30c8\u306f\u3053\u306e strokes \u3060\u3051\u3092\u4f7f\u3046\u3002

        Args:
            placements: 1\u30da\u30fc\u30b8\u5206\u306e\u6587\u5b57\u914d\u7f6e\u60c5\u5831\u3002
            progress_callback: \u9032\u6357\u901a\u77e5\u30b3\u30fc\u30eb\u30d0\u30c3\u30af\u3002

        Returns:
            \u751f\u6210\u3055\u308c\u305f\u5168\u30b9\u30c8\u30ed\u30fc\u30af\u5217\uff08\u66f8\u304d\u9806\u3092\u4fdd\u6301\uff09\u3002
        """
        return self.placements_to_strokes_with_finishes(placements, progress_callback)[0]

    def placements_to_strokes_with_finishes(
        self,
        placements: list[CharPlacement],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[list[Stroke], list[str]]:
        """\u914d\u7f6e\u60c5\u5831\u304b\u3089 strokes \u3068\u4e26\u8d70\u3059\u308b finishes \u3092\u751f\u6210\u3059\u308b\u3002

        \u5404\u6587\u5b57\u3092 ``generate_char_strokes_with_finishes`` \u3067\u63cf\u753b\u3057\u3001strokes \u3068
        \u7b46\u753b\u30bf\u30a4\u30d7\uff08``finish``\uff09\u3092\u9806\u5e8f\u4fdd\u6301\u3067\u4e26\u884c\u96c6\u7d04\u3059\u308b\u3002\u5206\u6570\u7dda\u306a\u3069\u5f8c\u304b\u3089
        ``append`` \u3059\u308b\u88dc\u52a9\u30b9\u30c8\u30ed\u30fc\u30af\u306b\u306f ``"none"`` \u3092\u5bfe\u5fdc\u4ed8\u3051\u3001strokes \u3068
        finishes \u306e\u9577\u3055\u3092\u5e38\u306b\u4e00\u81f4\u3055\u305b\u308b\u3002

        Args:
            placements: 1\u30da\u30fc\u30b8\u5206\u306e\u6587\u5b57\u914d\u7f6e\u60c5\u5831\u3002
            progress_callback: \u9032\u6357\u901a\u77e5\u30b3\u30fc\u30eb\u30d0\u30c3\u30af\u3002

        Returns:
            ``(strokes, finishes)``\u3002``len(strokes) == len(finishes)`` \u3092\u6e80\u305f\u3059\u3002
        """
        self._stroke_renderer._augmenter = self._typesetter.augmenter
        self._stroke_renderer._last_coverage = CharCoverageReport()
        units: list[_DrawUnit] = []
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
            char_strokes, char_finishes = self._stroke_renderer.generate_char_strokes_with_finishes(
                p
            )

            if (
                prev_end_x is not None
                and augmenter is not None
                and has_real_renderer
                and char_strokes
            ):
                shift = augmenter.random_uniform(0, 0.1)
                char_strokes = [s - np.array([shift, 0.0]) for s in char_strokes]

            # 連綿: 同じ字の近い画を確率的に薄いつなぎ画で結ぶ（近いほど高確率＋乱数）
            if self._connection_strength > 0 and augmenter is not None and len(char_strokes) > 1:
                char_strokes, char_finishes = insert_connections(
                    char_strokes,
                    char_finishes,
                    self._connection_strength,
                    p.font_size,
                    augmenter._rng,
                )

            if char_strokes:
                last_stroke = char_strokes[-1]
                prev_end_x = last_stroke[-1, 0]

            # $$ 数式は画像レンダリングに分数線が含まれるので、layout 側の分数線は引かない
            is_math_rendered = getattr(p, "math_source", None) or getattr(p, "math_skip", False)
            if getattr(p, "role", None) == "numerator" and not is_math_rendered:
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
                    char_strokes.append(np.array([[line_x0, line_y], [line_x1, line_y]]))
                    char_finishes.append("none")

            units.append(
                _DrawUnit(
                    placement=p,
                    strokes=char_strokes,
                    finishes=char_finishes,
                    source_index=i,
                    serpentine=self._can_serpentine_unit(p),
                )
            )

        strokes: list[Stroke] = []
        finishes: list[str] = []
        for unit in self._serpentine_draw_units(units):
            strokes.extend(unit.strokes)
            finishes.extend(unit.finishes)
        return strokes, finishes

    @staticmethod
    def _can_serpentine_unit(placement: CharPlacement) -> bool:
        """通常文字だけ蛇行対象にする。"""
        return not (
            placement.line_segment is not None
            or placement.math_source is not None
            or placement.math_skip
            or placement.role is not None
        )

    @staticmethod
    def _serpentine_draw_units(units: list[_DrawUnit]) -> list[_DrawUnit]:
        if not units:
            return []

        safe_units = [unit for unit in units if unit.serpentine]
        if not safe_units:
            return units

        safe_slots = [i for i, unit in enumerate(units) if unit.serpentine]
        ordered_safe = PlotterPipeline._order_units_by_serpentine_lines(safe_units)
        result = list(units)
        for slot, unit in zip(safe_slots, ordered_safe, strict=True):
            result[slot] = unit
        return result

    @staticmethod
    def _order_units_by_serpentine_lines(units: list[_DrawUnit]) -> list[_DrawUnit]:
        ordered = sorted(units, key=lambda unit: (-unit.placement.y, unit.placement.x))
        font_sizes = [unit.placement.font_size for unit in units]
        y_tolerance = max(min(font_sizes) * 0.35, 0.01)
        lines: list[list[_DrawUnit]] = []

        for unit in ordered:
            if not lines or abs(lines[-1][0].placement.y - unit.placement.y) > y_tolerance:
                lines.append([unit])
            else:
                lines[-1].append(unit)

        result: list[_DrawUnit] = []
        for line_index, line in enumerate(lines):
            if line_index % 2 == 1:
                line.sort(key=lambda unit: (-unit.placement.x, unit.source_index))
            else:
                line.sort(key=lambda unit: (unit.placement.x, unit.source_index))
            result.extend(line)
        return result

    def _generate_page_number_strokes(self, page_number: int) -> list[Stroke]:
        """ページ番号を手書きストロークで生成。用紙の「P.」印字の右横に配置。"""
        text = str(page_number)
        x = 30.0  # 「P.」印字の右、下線の上
        y = 7.0  # 下端からの距離（下線の上に載るように）
        font_size = 3.5
        all_strokes: list[Stroke] = []
        for ch in text:
            placement = CharPlacement(char=ch, x=x, y=y, font_size=font_size)
            char_strokes = self._stroke_renderer.generate_char_strokes(placement)
            all_strokes.extend(char_strokes)
            x += font_size * 0.5
        return all_strokes

    def _page_number_strokes_for(self, page_number: int) -> list[Stroke]:
        if not self._plot_page_numbers:
            return []
        return self._generate_page_number_strokes(page_number)

    def strokes_to_gcode(
        self, strokes: list[Stroke], finishes: list[str] | None = None
    ) -> list[str]:
        """ストローク列を G-code 化する。

        finishes を渡すと終端Zリフト（払い・はね）が G-code に乗る。順序最適化は
        finishes と同期する版を使い、対応関係を保つ。finishes 無しは従来挙動。
        """
        if finishes is None:
            optimized = optimize_stroke_order(strokes)
            return self._generator.generate(optimized)
        optimized, optimized_finishes = optimize_stroke_order_with_finishes(strokes, finishes)
        return self._generator.generate(optimized, finishes=optimized_finishes)

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
        all_finishes: list[str] = []
        for i, page_placements in enumerate(pages, start=1):
            strokes, finishes = self.placements_to_strokes_with_finishes(page_placements)
            page_num_strokes = self._page_number_strokes_for(i)
            all_strokes.extend(strokes)
            all_finishes.extend(finishes)
            all_strokes.extend(page_num_strokes)
            all_finishes.extend(["none"] * len(page_num_strokes))

        # ストローク順序を保持（optimize_stroke_orderを使わない）
        gcode = self._generator.generate(all_strokes, finishes=all_finishes, vary_speed=True)
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
            strokes, finishes = self.placements_to_strokes_with_finishes(
                page_placements, progress_callback=_page_stroke_progress
            )
            if progress_callback:
                progress_callback(
                    page_base + page_span * 0.85,
                    f"ストローク最適化中 ({i}/{n_pages})...",
                )
            optimized, optimized_finishes = optimize_stroke_order_with_finishes(strokes, finishes)
            if progress_callback:
                progress_callback(
                    page_base + page_span * 0.9,
                    f"プレビュー描画中 ({i}/{n_pages})...",
                )
            page_num_strokes = self._page_number_strokes_for(i)
            self._preview_with_ruled_lines(
                optimized,
                ruled_lines,
                page_path,
                page_number=i,
                page_number_strokes=page_num_strokes,
                finishes=optimized_finishes,
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
    ) -> list[Path]:
        """\u30c6\u30ad\u30b9\u30c8\u304b\u3089G-code\u3092\u751f\u6210\u3002\u8907\u6570\u30da\u30fc\u30b8\u306e\u5834\u5408\u306f\u30da\u30fc\u30b8\u3054\u3068\u306b\u5225\u30d5\u30a1\u30a4\u30eb\u3092\u51fa\u529b\u3059\u308b\u3002

        Returns:
            \u751f\u6210\u3055\u308c\u305fG-code\u30d5\u30a1\u30a4\u30eb\u30d1\u30b9\u306e\u30ea\u30b9\u30c8\uff08\u30da\u30fc\u30b8\u9806\uff09\u3002
        """
        save_path = Path(save_path)
        stem = save_path.stem
        suffix = save_path.suffix or ".gcode"
        parent = save_path.parent

        if progress_callback:
            progress_callback(0.0, "\u7d44\u7248\u4e2d...")

        pages = self.text_to_placements(text)
        if not pages or not pages[0]:
            gcode = self._generator.generate([])
            self._generator.save(gcode, save_path)
            if progress_callback:
                progress_callback(1.0, "\u5b8c\u4e86")
            return [save_path]

        n_pages = len(pages)
        result: list[Path] = []

        for i, page_placements in enumerate(pages, start=1):
            page_base = (i - 1) / n_pages
            page_span = 1.0 / n_pages

            def _stroke_progress(frac: float, desc: str, _base=page_base, _span=page_span) -> None:
                if progress_callback:
                    progress_callback(_base + frac * _span * 0.7, desc)

            strokes, finishes = self.placements_to_strokes_with_finishes(
                page_placements, progress_callback=_stroke_progress
            )
            page_num_strokes = self._page_number_strokes_for(i)
            all_strokes = strokes + page_num_strokes
            all_finishes = finishes + ["none"] * len(page_num_strokes)

            if progress_callback:
                progress_callback(
                    page_base + page_span * 0.85,
                    f"G-code \u5909\u63db\u4e2d ({i}/{n_pages})...",
                )
            gcode = self._generator.generate(all_strokes, finishes=all_finishes, vary_speed=True)

            page_path = save_path if n_pages == 1 else parent / f"{stem}_p{i}{suffix}"
            self._generator.save(gcode, page_path)
            result.append(page_path)

        if progress_callback:
            progress_callback(1.0, "\u5b8c\u4e86")

        return result

    def create_app(self):
        """既存テスト互換のため残す薄いラッパ。

        新 UI は副作用なし設計のため pipeline 自体は使わず、
        StrokeRenderer が知っている環境引数（checkpoint / kanjivg / user_strokes）
        を取り出して新シグネチャの create_app へ転送する。
        """
        try:
            import gradio as gr  # noqa: F401
        except ImportError:
            return None

        from src.ui.gradio_app import create_app

        renderer = self._stroke_renderer
        return create_app(
            checkpoint_path=getattr(renderer, "_checkpoint_path", None),
            kanjivg_dir=getattr(renderer, "_kanjivg_dir", None),
            user_strokes_dir=getattr(renderer, "_user_strokes_dir", None),
        )


def build_pipeline(
    settings,
    checkpoint_path: Path | str | None = None,
    kanjivg_dir: Path | str | None = None,
    style_sample: object | None = None,
    user_strokes_dir: Path | str | None = None,
    skip_non_japanese: bool = False,
) -> PlotterPipeline:
    """UISettings から副作用なしで PlotterPipeline を構築するファクトリ。

    UI 層は「現設定の snapshot」を保持し、操作のたびに本関数で新しい
    パイプラインを生成することで、_apply_settings 的な属性差し替えを排除する。

    Args:
        settings: UISettings インスタンス。
        checkpoint_path: ML モデルの checkpoint パス。
        kanjivg_dir: KanjiVG JSON ディレクトリ。
        style_sample: 明示的な style_sample（指定時は user_strokes より優先）。
        user_strokes_dir: ユーザーストローク JSON ディレクトリ。
        skip_non_japanese: 日本語文字以外の描画を一時的にスキップする。

    Returns:
        構築済みの PlotterPipeline。
    """
    from src.ui.settings import UISettings

    if not isinstance(settings, UISettings):
        raise TypeError(f"settings must be UISettings, got {type(settings).__name__}")

    page_config = PageConfig(
        paper_size=(settings.paper_width, settings.paper_height),
        margin_top=settings.margin_top,
        margin_bottom=settings.margin_bottom,
        margin_left=settings.margin_left,
        margin_right=settings.margin_right,
        line_spacing=settings.line_spacing,
    )
    plotter_config = PlotterConfig(
        work_area_width=220.0,
        work_area_height=310.0,
        paper_origin_x=0.0,
        paper_origin_y=0.0,
        paper_width=settings.paper_width,
        paper_height=settings.paper_height,
        pressure_variation=settings.pressure_variation,
        entry_taper=settings.entry_taper,
    )
    pipeline = PlotterPipeline(
        page_config=page_config,
        plotter_config=plotter_config,
        checkpoint_path=checkpoint_path,
        kanjivg_dir=kanjivg_dir,
        style_sample=style_sample,
        temperature=settings.temperature,
        user_strokes_dir=user_strokes_dir,
        messiness=settings.messiness,
        instance_variation=settings.instance_variation,
        connection_strength=settings.connection_strength,
        skip_non_japanese=skip_non_japanese,
        plot_page_numbers=settings.plot_page_numbers,
    )
    # PlotterPipeline.__init__ は font_size=4.5 ハードコーディングのため、
    # UISettings.font_size を反映するため Typesetter を再構築する
    if settings.font_size != 4.5:
        pipeline._typesetter = Typesetter(
            page_config,
            font_size=settings.font_size,
            augmenter=pipeline._typesetter.augmenter,
        )
    return pipeline
