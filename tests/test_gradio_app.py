"""Tests for src/ui/gradio_app module.

旧 _apply_settings / _format_coverage 直接テストは廃止。
理由: 新 UI は副作用ゼロ設計（毎回 build_pipeline で新規構築）に移行したため、
in-place 設定差し替えユーティリティは存在しない。
代わりに UISettings 経由のスモークテストへ集約する。
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pytest

from src.ui.gradio_app import (
    _APP_CSS,
    _APP_HEAD,
    _EXAMPLE_ESSAY,
    _EXAMPLE_MATH_REPORT,
    _EXAMPLE_REPORT_HEADER,
    _FONT_HEAD,
    _HELP_MARKDOWN,
    _WEBSERIAL_HEAD,
    _WEBSERIAL_SCRIPT,
    _format_coverage,
    _resolve_restored_profile,
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

    def test_has_browser_state(self, app):
        # 設定・プロファイルの localStorage 永続化に BrowserState を 2 つ期待する
        bs = [b for b in app.blocks.values() if isinstance(b, gr.BrowserState)]
        assert len(bs) >= 2

    def test_has_webserial_panel_components(self, app):
        elem_ids = {
            getattr(b, "elem_id", None)
            for b in app.blocks.values()
            if getattr(b, "elem_id", None)
        }
        assert "webserial-status" in elem_ids
        assert "webserial-log" in elem_ids
        assert "webserial-connect-btn" in elem_ids
        assert "webserial-disconnect-btn" in elem_ids
        assert "webserial-home-btn" in elem_ids
        assert "webserial-pen-up-btn" in elem_ids
        assert "webserial-pen-down-btn" in elem_ids
        assert "webserial-start-btn" in elem_ids
        assert "webserial-stop-btn" in elem_ids
        assert "webserial-emergency-btn" in elem_ids

    def test_generation_and_webserial_are_separate_tabs(self, app):
        tab_labels = {
            getattr(b, "label", None)
            for b in app.blocks.values()
            if type(b).__name__ == "Tab"
        }
        assert "生成" in tab_labels
        assert "プロッタ送信" in tab_labels

    def test_has_webserial_upload_file_input(self, app):
        files = [b for b in app.blocks.values() if isinstance(b, gr.Files)]
        labels = {getattr(b, "label", "") for b in files}
        assert "G-code ダウンロード" in labels
        assert "アップロード G-code (.gcode/.nc/.txt)" in labels


class TestWebSerialScript:
    def test_head_injects_webserial_script(self):
        assert "<script>" in _WEBSERIAL_HEAD
        assert "window.penPlotterWebSerial" in _WEBSERIAL_HEAD

    def test_script_has_webserial_contract(self):
        assert "navigator.serial.requestPort()" in _WEBSERIAL_SCRIPT
        assert "baudRate: BAUD_RATE" in _WEBSERIAL_SCRIPT
        assert "const BAUD_RATE = 115200" in _WEBSERIAL_SCRIPT
        assert '"$H", "G4 P1", "G92 X0 Y297 Z0", "G90"' in _WEBSERIAL_SCRIPT
        assert 'const PEN_UP_COMMAND = "G1G90 Z0.5 F5000"' in _WEBSERIAL_SCRIPT
        assert 'const PEN_DOWN_COMMAND = "G1G90 Z5 F5000"' in _WEBSERIAL_SCRIPT

    def test_script_has_stream_safety_guards(self):
        assert "window.confirm" in _WEBSERIAL_SCRIPT
        assert "state.cancelRequested" in _WEBSERIAL_SCRIPT
        assert "GRBL 応答タイムアウト" in _WEBSERIAL_SCRIPT
        assert "/^error:/i" in _WEBSERIAL_SCRIPT
        assert "/^ALARM:/i" in _WEBSERIAL_SCRIPT
        assert "new Uint8Array([0x21, 0x18])" in _WEBSERIAL_SCRIPT

    def test_script_exposes_normalize_helper(self):
        assert "normalizeGcode" in _WEBSERIAL_SCRIPT
        assert "replace(/;.*$/," in _WEBSERIAL_SCRIPT
        assert "replace(/\\([^)]*\\)/g," in _WEBSERIAL_SCRIPT


class TestAppBrandingAssets:
    def test_google_fonts_are_included_in_head(self):
        assert "fonts.googleapis.com" in _FONT_HEAD
        assert "fonts.gstatic.com" in _FONT_HEAD
        assert "family=Inter" in _FONT_HEAD
        assert "family=Noto+Sans+JP" in _FONT_HEAD

    def test_app_head_combines_fonts_and_webserial(self):
        assert "fonts.googleapis.com" in _APP_HEAD
        assert "window.penPlotterWebSerial" in _APP_HEAD

    def test_app_css_uses_inter_and_noto_sans_jp(self):
        assert 'font-family: "Inter", "Noto Sans JP"' in _APP_CSS
        assert "max-width: 1400px" in _APP_CSS

    def test_run_ui_enables_pwa_with_app_assets(self):
        run_ui = Path("scripts/run_ui.py").read_text(encoding="utf-8")
        assert "from src.ui.gradio_app import _APP_CSS, _APP_HEAD, create_app" in run_ui
        assert "css=_APP_CSS" in run_ui
        assert "head=_APP_HEAD" in run_ui
        assert "pwa=True" in run_ui


class TestResolveRestoredProfile:
    """_resolve_restored_profile() のテスト（永続プロファイルの照合解決）。"""

    def test_valid_stored_profile_kept(self):
        """選択肢に存在する保存値はそのまま採用する。"""
        assert _resolve_restored_profile("b", ["a", "b"], "a") == "b"

    def test_missing_stored_profile_falls_back(self):
        """選択肢に無い保存値（削除/リネーム後）はデフォルトへ。"""
        assert _resolve_restored_profile("z", ["a", "b"], "a") == "a"

    def test_none_stored_falls_back(self):
        """保存値 None はデフォルトへ。"""
        assert _resolve_restored_profile(None, ["a", "b"], "a") == "a"

    def test_empty_options_returns_default(self):
        """選択肢が空なら default（None）を返す。"""
        assert _resolve_restored_profile("x", [], None) is None
