"""テキスト → 組版 → 手書きストローク → プレビュー画像 / G-code の生成パイプライン。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.collector.profiles import resolve_character_root
from src.gcode.generator import GCodeGenerator
from src.geometry import Stroke
from src.handwriting.augmentation import AugmentConfig, HandwritingAugmenter
from src.handwriting.finishing import NONE, insert_connections
from src.layout.placement import CharPlacement
from src.layout.typesetter import Typesetter
from src.render.char_renderer import CharCoverageReport, CharRenderer
from src.render.preview import render_page_preview
from src.resources import report_paper_path
from src.settings import Settings

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

# 用紙の「P.」印字の右・下線の上に手書きでページ番号を入れる位置
_PAGE_NUMBER_X = 30.0
_PAGE_NUMBER_Y = 7.0
_PAGE_NUMBER_FONT_SIZE = 3.5
# 直前の字に続けて書くとき、字を少し左へ寄せる最大量(mm)
_CHAR_LEFT_SHIFT_MAX = 0.1


@lru_cache(maxsize=4)
def _load_inference(checkpoint: Path, user_strokes_dir: Path | None) -> object | None:
    """ML 推論器（重いので同じ組み合わせはキャッシュして使い回す）。失敗時は None。"""
    try:
        from src.model.inference import StrokeInference

        engine = StrokeInference.from_user_strokes(checkpoint, user_strokes_dir)
    except Exception:
        logger.warning("Failed to load ML checkpoint: %s", checkpoint, exc_info=True)
        return None
    logger.info("ML inference engine loaded from %s", checkpoint)
    return engine


@dataclass
class PageStrokes:
    """1 ページ分の描画結果（書き順どおり）。``finishes[i]`` が ``strokes[i]`` の筆法。"""

    strokes: list[Stroke]
    finishes: list[str]


class PlotterPipeline:
    """設定と字形ソースから、テキストの手書きプレビュー / G-code を生成する。

    Args:
        settings: 生成設定。
        checkpoint_path: V3 モデルのチェックポイント（無ければ ML 変形なし）。
        kanjivg_dir: KanjiVG 参照字形ディレクトリ。
        user_strokes_dir: ユーザー筆跡（文字ディレクトリ群の親、またはプロファイルのルート）。
        profile: ``user_strokes_dir`` がプロファイルのルートのときに使うプロファイル ID。
        japanese_only: かな・漢字・句読点・数字以外を描かない。
        seed: 揺らぎの乱数 seed（配置・字形の揺らぎを再現可能にする）。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        checkpoint_path: Path | str | None = None,
        kanjivg_dir: Path | str | None = None,
        user_strokes_dir: Path | str | None = None,
        profile: str | None = None,
        japanese_only: bool = False,
        seed: int | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.page_config = self.settings.page_config()
        self.plotter_config = self.settings.plotter_config()
        self.augmenter = HandwritingAugmenter(
            AugmentConfig().scaled(self.settings.messiness), seed=seed
        )
        self.typesetter = Typesetter(
            self.page_config, font_size=self.settings.font_size, augmenter=self.augmenter
        )
        user_dir = resolve_character_root(user_strokes_dir, profile)
        inference = None
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            inference = _load_inference(Path(checkpoint_path), user_dir)
        self.renderer = CharRenderer(
            kanjivg_dir=kanjivg_dir,
            user_strokes_dir=user_dir,
            inference=inference,
            augmenter=self.augmenter,
            line_spacing=self.page_config.line_spacing,
            temperature=self.settings.temperature,
            instance_variation=self.settings.instance_variation,
            japanese_only=japanese_only,
        )
        self.generator = GCodeGenerator(self.plotter_config)

    @property
    def coverage(self) -> CharCoverageReport:
        """直近の生成で各文字をどの経路で描いたか。"""
        return self.renderer.coverage

    # ------------------------------------------------------------------
    # 段階ごとの処理
    # ------------------------------------------------------------------

    def typeset(self, text: str) -> list[list[CharPlacement]]:
        """テキストをページごとの配置要素へ組版する。"""
        return self.typesetter.typeset(text)

    def render_page(
        self, placements: list[CharPlacement], progress: ProgressCallback | None = None
    ) -> PageStrokes:
        """1 ページ分の配置要素を手書きストロークへ変換する（書き順は下記）。

        本文の文字は行ごとに左右交互（蛇行）の順で書き、ペンの移動を減らす。罫線・
        数式はその場の順序を保つ。
        """
        aug = self.augmenter
        drew_any = False
        units: list[_DrawUnit] = []
        for i, p in enumerate(placements):
            if progress:
                progress(
                    i / max(len(placements), 1), f"ストローク生成中 ({i + 1}/{len(placements)})"
                )
            rendered = self.renderer.render(p)
            strokes, finishes = rendered.strokes, rendered.finishes
            if drew_any and self.renderer.has_reference_source and strokes:
                shift = float(aug.rng.uniform(0, _CHAR_LEFT_SHIFT_MAX))
                strokes = [s - np.array([shift, 0.0]) for s in strokes]
            if self.settings.connection_strength > 0 and len(strokes) > 1:
                strokes, finishes = insert_connections(
                    strokes, finishes, self.settings.connection_strength, p.font_size, aug.rng
                )
            drew_any = drew_any or bool(strokes)
            units.append(_DrawUnit(p, strokes, finishes, i, serpentine=p.is_text))

        page = PageStrokes([], [])
        for unit in _serpentine_order(units):
            page.strokes.extend(unit.strokes)
            page.finishes.extend(unit.finishes)
        return page

    def page_number_strokes(self, page_number: int) -> list[Stroke]:
        """ページ番号の手書きストローク（無効設定なら空）。"""
        if not self.settings.plot_page_numbers:
            return []
        strokes: list[Stroke] = []
        x = _PAGE_NUMBER_X
        for ch in str(page_number):
            placement = CharPlacement(ch, x, _PAGE_NUMBER_Y, _PAGE_NUMBER_FONT_SIZE)
            strokes.extend(self.renderer.render(placement).strokes)
            x += _PAGE_NUMBER_FONT_SIZE * 0.5
        return strokes

    def render_document(
        self, text: str, progress: ProgressCallback | None = None
    ) -> list[PageStrokes]:
        """全ページを描画する（ページ番号込み）。空テキストなら空ページ 1 枚。"""
        self.renderer.coverage = CharCoverageReport()
        if progress:
            progress(0.0, "組版中...")
        pages = self.typeset(text)
        result: list[PageStrokes] = []
        for i, placements in enumerate(pages, start=1):
            base, span = (i - 1) / len(pages), 1.0 / len(pages)

            def page_progress(frac: float, desc: str, _b: float = base, _s: float = span) -> None:
                if progress:
                    progress(_b + frac * _s * 0.8, desc)

            page = self.render_page(placements, page_progress)
            numbers = self.page_number_strokes(i) if placements else []
            page.strokes.extend(numbers)
            page.finishes.extend([NONE] * len(numbers))
            result.append(page)
        return result

    # ------------------------------------------------------------------
    # 出力
    # ------------------------------------------------------------------

    def generate_preview(
        self, text: str, save_path: str | Path, progress: ProgressCallback | None = None
    ) -> list[Path]:
        """ページごとのプレビュー PNG を保存してパスを返す（複数ページは ``_p<N>`` 付き）。"""
        pages = self.render_document(text, progress)
        paths = _page_paths(Path(save_path), len(pages))
        ruled = self.typesetter.layout.ruled_line_strokes()
        for i, (page, path) in enumerate(zip(pages, paths, strict=True), start=1):
            if progress:
                progress((i - 0.1) / len(pages), f"プレビュー描画中 ({i}/{len(pages)})...")
            render_page_preview(
                page.strokes,
                page.finishes,
                path,
                config=self.plotter_config,
                ruled_lines=ruled,
                background=report_paper_path(),
            )
        if progress:
            progress(1.0, "完了")
        return paths

    def generate_gcode(
        self, text: str, save_path: str | Path, progress: ProgressCallback | None = None
    ) -> list[Path]:
        """ページごとの G-code を保存してパスを返す（複数ページは ``_p<N>`` 付き）。"""
        save_path = Path(save_path)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".gcode")
        pages = self.render_document(text, progress)
        paths = _page_paths(save_path, len(pages))
        for i, (page, path) in enumerate(zip(pages, paths, strict=True), start=1):
            if progress:
                progress((i - 0.15) / len(pages), f"G-code 変換中 ({i}/{len(pages)})...")
            gcode = self.generator.generate(page.strokes, finishes=page.finishes, vary_speed=True)
            self.generator.save(gcode, path)
        if progress:
            progress(1.0, "完了")
        return paths


@dataclass
class _DrawUnit:
    placement: CharPlacement
    strokes: list[Stroke]
    finishes: list[str]
    index: int
    serpentine: bool


def _page_paths(save_path: Path, n_pages: int) -> list[Path]:
    if n_pages == 1:
        return [save_path]
    return [
        save_path.parent / f"{save_path.stem}_p{i}{save_path.suffix}" for i in range(1, n_pages + 1)
    ]


def _serpentine_order(units: list[_DrawUnit]) -> list[_DrawUnit]:
    """本文の文字だけを、行ごとに左→右 / 右→左 と交互に並べ替える。

    それ以外（罫線・数式）は元の位置に残し、文字だけが占めていた位置へ並べ直す。
    """
    slots = [i for i, u in enumerate(units) if u.serpentine]
    if not slots:
        return units
    chars = [units[i] for i in slots]
    y_tolerance = max(min(u.placement.font_size for u in chars) * 0.35, 0.01)
    lines: list[list[_DrawUnit]] = []
    for unit in sorted(chars, key=lambda u: (-u.placement.y, u.placement.x)):
        if not lines or abs(lines[-1][0].placement.y - unit.placement.y) > y_tolerance:
            lines.append([unit])
        else:
            lines[-1].append(unit)
    ordered: list[_DrawUnit] = []
    for n, line in enumerate(lines):
        reverse = n % 2 == 1
        line.sort(key=lambda u: (-u.placement.x if reverse else u.placement.x, u.index))
        ordered.extend(line)
    result = list(units)
    for slot, unit in zip(slots, ordered, strict=True):
        result[slot] = unit
    return result
