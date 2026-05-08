"""Tests for src/ui/gradio_app module.

旧 _apply_settings / _format_coverage 直接テストは廃止。
理由: 新 UI は副作用ゼロ設計（毎回 build_pipeline で新規構築）に移行したため、
in-place 設定差し替えユーティリティは存在しない。
代わりに UISettings 経由のスモークテストへ集約する。
"""

from __future__ import annotations

import gradio as gr
import pytest

from src.ui.gradio_app import (
    _EXAMPLE_ESSAY,
    _EXAMPLE_MATH_REPORT,
    _EXAMPLE_REPORT_HEADER,
    _HELP_MARKDOWN,
    _format_coverage,
    create_app,
)
from src.ui.settings import UISettings
from src.ui.web_app import CharCoverageReport, PlotterPipeline, build_pipeline


class TestCreateApp:
    """新シグネチャ create_app(checkpoint_path, kanjivg_dir, user_strokes_dir) のテスト。"""

    def test_returns_blocks_with_no_args(self):
        app = create_app()
        assert isinstance(app, gr.Blocks)

    def test_returns_blocks_with_kwargs_none(self):
        app = create_app(checkpoint_path=None, kanjivg_dir=None, user_strokes_dir=None)
        assert isinstance(app, gr.Blocks)

    def test_returns_blocks_with_dirs(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        app = create_app(user_strokes_dir=empty)
        assert isinstance(app, gr.Blocks)

    def test_pipeline_builds_from_default_settings(self):
        # UI 内で行う build_pipeline 呼び出しが副作用なく成功することを担保。
        pipeline = build_pipeline(UISettings.default())
        assert isinstance(pipeline, PlotterPipeline)


class TestPlotterPipelineCreateApp:
    """既存 test_web_app の TestGradioGallery が利用する pipeline.create_app() の互換性。"""

    def test_pipeline_create_app_returns_blocks(self):
        pipeline = PlotterPipeline()
        app = pipeline.create_app()
        assert isinstance(app, gr.Blocks)


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

    def test_summary_line(self):
        report = CharCoverageReport(user_strokes=["あ"], skipped=[" "])
        result = _format_coverage(report)
        assert "全2文字" in result
        assert "描画: 1" in result
        assert "スキップ: 1" in result


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


class TestHelpMarkdown:
    def test_help_non_empty(self):
        assert len(_HELP_MARKDOWN.strip()) > 0

    def test_help_has_format_reference(self):
        assert "見出し" in _HELP_MARKDOWN or "書式" in _HELP_MARKDOWN


class TestAppSmoke:
    """Blocks 内に主要コンポーネントが存在することを担保するスモーク。"""

    @pytest.fixture
    def app(self):
        return create_app()

    def test_has_gallery(self, app):
        assert any(isinstance(b, gr.Gallery) for b in app.blocks.values())

    def test_has_textbox(self, app):
        assert any(isinstance(b, gr.Textbox) for b in app.blocks.values())

    def test_has_state(self, app):
        # UISettings 用と stale フラグ用に最低 2 つの State を期待する
        states = [b for b in app.blocks.values() if isinstance(b, gr.State)]
        assert len(states) >= 2
