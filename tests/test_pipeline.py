"""パイプライン（テキスト → プレビュー / G-code）・設定・レイアウト診断。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.diagnostics import diagnose_layout, diagnose_placements
from src.handwriting.finishing import CONNECT
from src.layout.placement import CharPlacement, MathSpec
from src.pipeline import PlotterPipeline, _DrawUnit, _serpentine_order
from src.settings import Settings


@pytest.fixture
def pipeline(kanjivg_dir: Path, user_strokes_root: Path) -> PlotterPipeline:
    return PlotterPipeline(kanjivg_dir=kanjivg_dir, user_strokes_dir=user_strokes_root, seed=0)


def _pen_downs(gcode: list[str]) -> int:
    return sum(1 for line in gcode if line == Settings().plotter_config().pen_down_command)


# --- 設定 ---


def test_settings_defaults_are_valid_and_roundtrip():
    s = Settings()
    assert s.validate() == []
    assert Settings.from_dict(s.to_dict()) == s
    restored = Settings.from_dict({"font_size": "6", "plot_page_numbers": "false", "x": 1})
    assert restored.font_size == 6.0 and restored.plot_page_numbers is False
    assert Settings.from_dict({"font_size": "abc"}).font_size == s.font_size
    assert Settings.from_dict(None) == s


@pytest.mark.parametrize(
    "override",
    [
        {"margin_top": 200, "margin_bottom": 100},
        {"margin_left": 150, "margin_right": 70},
        {"font_size": 0},
        {"line_spacing": 1.0},
        {"messiness": -1},
    ],
)
def test_settings_validation_errors(override: dict):
    assert replace(Settings(), **override).validate()


# --- 生成 ---


@pytest.mark.slow
def test_generate_gcode_and_preview(pipeline: PlotterPipeline, tmp_path: Path):
    text = "# 十人\n人十a2 $x^2$ ，"
    (gcode_path,) = pipeline.generate_gcode(text, tmp_path / "out.gcode")
    gcode = gcode_path.read_text().splitlines()
    assert gcode[1] == "$H" and _pen_downs(gcode) > 5
    (png,) = pipeline.generate_preview(text, tmp_path / "out.png")
    assert png.stat().st_size > 1000
    cov = pipeline.coverage
    assert cov.user_strokes and cov.kanjivg and cov.geometric


def test_multiple_pages_get_numbered_files(pipeline: PlotterPipeline, tmp_path: Path):
    paths = pipeline.generate_gcode("十\n-----\n人\n-----\n一", tmp_path / "doc.gcode")
    assert [p.name for p in paths] == ["doc_p1.gcode", "doc_p2.gcode", "doc_p3.gcode"]


def test_page_numbers_can_be_disabled(kanjivg_dir: Path, tmp_path: Path):
    def pen_downs(plot_numbers: bool) -> int:
        p = PlotterPipeline(
            Settings(plot_page_numbers=plot_numbers), kanjivg_dir=kanjivg_dir, seed=0
        )
        return _pen_downs(p.generate_gcode("一", tmp_path / "x.gcode")[0].read_text().splitlines())

    assert pen_downs(True) > pen_downs(False)


def test_seed_alone_reproduces_all_randomness(
    kanjivg_dir: Path, user_strokes_root: Path, tiny_checkpoint: Path, tmp_path: Path
):
    """seed だけで配置・字形・ML 温度ノイズの全揺らぎが再現できる（グローバル乱数に非依存）。"""

    def run(seed: int) -> str:
        np.random.seed(None)  # グローバル乱数を毎回かき混ぜても結果が変わらないこと
        p = PlotterPipeline(
            Settings(messiness=1.0, instance_variation=0.5, temperature=1.0),
            checkpoint_path=tiny_checkpoint,
            kanjivg_dir=kanjivg_dir,
            user_strokes_dir=user_strokes_root,
            seed=seed,
        )
        return p.generate_gcode("十人一十人あ", tmp_path / f"{seed}.gcode")[0].read_text()

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_empty_text_gives_an_empty_page(pipeline: PlotterPipeline, tmp_path: Path):
    (path,) = pipeline.generate_gcode("", tmp_path / "empty.gcode")
    assert _pen_downs(path.read_text().splitlines()) == 0


def test_profile_selection(kanjivg_dir: Path, user_strokes_root: Path):
    p = PlotterPipeline(kanjivg_dir=kanjivg_dir, user_strokes_dir=user_strokes_root, profile="taro")
    p.render_page(p.typeset("十")[0])
    assert p.coverage.user_strokes == ["十"]


def test_connections_insert_pen_down_joins(kanjivg_dir: Path):
    p = PlotterPipeline(Settings(connection_strength=1.0), kanjivg_dir=kanjivg_dir, seed=0)
    page = p.render_page(p.typeset("口口口口口口")[0])
    assert CONNECT in page.finishes
    assert len(page.strokes) == len(page.finishes)


def test_serpentine_order_alternates_direction_per_line():
    def unit(x: float, y: float, i: int, serpentine: bool = True) -> _DrawUnit:
        return _DrawUnit(CharPlacement("a", x, y, 5.0), [], [], i, serpentine)

    units = [unit(0, 10, 0), unit(5, 10, 1), unit(0, 0, 2), unit(5, 0, 3), unit(0, -9, 4, False)]
    order = [u.index for u in _serpentine_order(units)]
    assert order == [0, 1, 3, 2, 4]  # 2 行目は右→左、非対象は元の位置


# --- 診断 ---


def test_diagnose_reports_missing_glyphs(pipeline: PlotterPipeline):
    report = diagnose_layout(pipeline, "十十無")
    assert [(m.char, m.count) for m in report.missing_glyphs] == [("無", 1)]
    assert not report.ok and "無" in report.summary()


def test_diagnose_detects_horizontal_math_overlap():
    math = CharPlacement("", 10, 50, 5, math=MathSpec("x", (10, 50, 20, 4), "baseline"))
    after = CharPlacement("字", 15, 50, 5)
    (overlap,) = diagnose_placements([[math, after]], line_spacing=8)
    assert overlap.kind == "horizontal" and overlap.amount_mm == pytest.approx(15)


def test_style_revision_reloads_the_ml_style(tiny_checkpoint: Path, kanjivg_dir, user_strokes_root):
    from src.pipeline import _load_inference

    _load_inference.cache_clear()
    kwargs = {"checkpoint_path": tiny_checkpoint, "kanjivg_dir": kanjivg_dir}
    kwargs["user_strokes_dir"] = user_strokes_root
    PlotterPipeline(**kwargs, style_revision=0)
    PlotterPipeline(**kwargs, style_revision=0)
    assert _load_inference.cache_info().misses == 1
    PlotterPipeline(**kwargs, style_revision=1)  # 筆跡が増えたらスタイルを推定し直す
    assert _load_inference.cache_info().misses == 2
