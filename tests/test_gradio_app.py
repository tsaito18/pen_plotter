"""Tests for src/ui/gradio_app module."""

from __future__ import annotations

import gradio as gr
import pytest

from src.ui.gradio_app import (
    _EXAMPLE_ESSAY,
    _EXAMPLE_MATH_REPORT,
    _EXAMPLE_REPORT_HEADER,
    _apply_settings,
    _format_coverage,
    create_app,
)
from src.ui.web_app import CharCoverageReport, PlotterPipeline


class TestCreateApp:
    @pytest.fixture
    def pipeline(self):
        return PlotterPipeline()

    def test_returns_blocks(self, pipeline):
        app = create_app(pipeline)
        assert isinstance(app, gr.Blocks)

    def test_has_tabs(self, pipeline):
        app = create_app(pipeline)
        blocks = app.blocks
        tab_labels = [b.label for b in blocks.values() if isinstance(b, gr.Tab)]
        assert "作成" in tab_labels
        assert "設定" in tab_labels
        assert "ヘルプ" in tab_labels


class TestFormatCoverage:
    def test_empty_report(self):
        report = CharCoverageReport()
        assert _format_coverage(report) == ""

    def test_user_strokes_only(self):
        report = CharCoverageReport(user_strokes=["あ", "い", "あ"])
        result = _format_coverage(report)
        assert "ユーザー筆跡" in result
        assert "3字" in result
        assert "2種" in result

    def test_rect_fallback_has_warning(self):
        report = CharCoverageReport(rect_fallback=["X", "Y"])
        result = _format_coverage(report)
        assert "矩形フォールバック" in result

    def test_multiple_tiers(self):
        report = CharCoverageReport(
            user_strokes=["あ"],
            kanjivg=["漢"],
            geometric=["、"],
        )
        result = _format_coverage(report)
        assert "ユーザー筆跡" in result
        assert "KanjiVG" in result
        assert "幾何生成" in result

    def test_summary_line(self):
        report = CharCoverageReport(
            user_strokes=["あ"],
            skipped=[" "],
        )
        result = _format_coverage(report)
        assert "全2文字" in result
        assert "描画: 1" in result
        assert "スキップ: 1" in result

    def test_deduplication(self):
        report = CharCoverageReport(user_strokes=["あ", "あ", "あ"])
        result = _format_coverage(report)
        assert "3字" in result
        assert "1種" in result


class TestApplySettings:
    @pytest.fixture
    def pipeline(self):
        return PlotterPipeline()

    def test_updates_page_config(self, pipeline):
        _apply_settings(pipeline, 7.0, 10.0, 20, 10, 30, 20, 1500, 4000, 0.20, 0.8)
        assert pipeline._page_config.margin_top == 20.0
        assert pipeline._page_config.margin_bottom == 10.0
        assert pipeline._page_config.margin_left == 30.0
        assert pipeline._page_config.margin_right == 20.0
        assert pipeline._page_config.line_spacing == 10.0

    def test_updates_typesetter_font_size(self, pipeline):
        _apply_settings(pipeline, 7.0, 10.0, 20, 10, 30, 20, 1500, 4000, 0.20, 0.8)
        assert pipeline._typesetter.font_size == 7.0

    def test_preserves_augmenter(self, pipeline):
        original_augmenter = pipeline._typesetter.augmenter
        _apply_settings(pipeline, 7.0, 10.0, 20, 10, 30, 20, 1500, 4000, 0.20, 0.8)
        assert pipeline._typesetter.augmenter is original_augmenter

    def test_updates_plotter_config(self, pipeline):
        _apply_settings(pipeline, 6.0, 8.0, 30, 15, 25, 15, 1500, 4000, 0.20, 0.8)
        assert pipeline._plotter_config.draw_speed == 1500.0
        assert pipeline._plotter_config.travel_speed == 4000.0
        assert pipeline._plotter_config.pen_delay == 0.20

    def test_updates_temperature(self, pipeline):
        _apply_settings(pipeline, 6.0, 8.0, 30, 15, 25, 15, 1000, 3000, 0.15, 1.5)
        assert pipeline._temperature == 1.5


class TestExamples:
    def test_examples_are_non_empty(self):
        assert len(_EXAMPLE_REPORT_HEADER.strip()) > 0
        assert len(_EXAMPLE_MATH_REPORT.strip()) > 0
        assert len(_EXAMPLE_ESSAY.strip()) > 0

    def test_report_header_has_math(self):
        assert "$" in _EXAMPLE_REPORT_HEADER

    def test_math_report_has_block_math(self):
        assert "$$" in _EXAMPLE_MATH_REPORT

    def test_essay_is_plain_text(self):
        assert "$" not in _EXAMPLE_ESSAY
        assert "#" not in _EXAMPLE_ESSAY
