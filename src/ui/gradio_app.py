"""Gradio Web UI（生成タブ＋ブラウザ WebSerial によるプロッタ送信タブ）。

UI は :class:`Settings` の snapshot を ``gr.State`` に持ち、プレビュー / G-code 生成の
たびにパイプラインを組み立て直す（副作用で設定を差し替えない）。設定は
ブラウザの localStorage に保存し、再訪時に復元する。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import shutil
import tempfile
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from pathlib import Path

import gradio as gr

from src.collector.profiles import list_profiles
from src.pipeline import PlotterPipeline
from src.render.char_renderer import CharCoverageReport
from src.resources import report_paper_path
from src.settings import Settings
from src.ui import content

logger = logging.getLogger(__name__)

_ASSETS = Path(__file__).with_name("assets")
APP_CSS = (_ASSETS / "app.css").read_text(encoding="utf-8")


@dataclass(frozen=True)
class _SliderSpec:
    field: str
    minimum: float
    maximum: float
    step: float
    label: str
    info: str | None = None


# 設定パネルのスライダー（セクション名 → スライダー）。Settings のフィールドと 1 対 1。
_SLIDER_SECTIONS: dict[str, list[_SliderSpec]] = {
    "レイアウト": [
        _SliderSpec("font_size", 3.0, 10.0, 0.1, "フォントサイズ (mm)"),
        _SliderSpec("line_spacing", 5.0, 15.0, 0.01, "行間隔 (mm)"),
    ],
    "余白 (mm)": [
        _SliderSpec("margin_top", 5, 60, 1, "上"),
        _SliderSpec("margin_bottom", 5, 50, 1, "下"),
        _SliderSpec("margin_left", 1, 50, 1, "左"),
        _SliderSpec("margin_right", 1, 50, 1, "右"),
    ],
    "手書き": [
        _SliderSpec("temperature", 0.0, 2.0, 0.1, "温度", "高いほど字形の揺らぎが大きくなります"),
        _SliderSpec(
            "messiness",
            0.0,
            2.0,
            0.1,
            "汚さ",
            "行内の上下動・字間・サイズ・傾きのばらつき。0=整った字、2=大きく乱れる",
        ),
    ],
    "人らしさ調整": [
        _SliderSpec(
            "pressure_variation",
            0.0,
            1.0,
            0.05,
            "筆圧変化",
            "画の中の濃淡（プレビュー演出用）。実機は描画中Zが振れて点線化するため0推奨",
        ),
        _SliderSpec(
            "instance_variation",
            0.0,
            1.0,
            0.05,
            "字のばらつき",
            "同じ字を毎回少し変える。0=毎回同じ形、大=書くたびに違う",
        ),
        _SliderSpec(
            "entry_taper",
            0.0,
            1.0,
            0.05,
            "入筆",
            "始筆を軽く入れて立ち上げる筆の入り。実機は始筆がかすれ得るため0推奨",
        ),
        _SliderSpec(
            "finish_strength",
            0.0,
            1.0,
            0.05,
            "払い・はねの抜き",
            "画の終わりを細く抜く。0=鉛筆の手書きどおり同じ太さ（推奨）、大=筆ペン風",
        ),
        _SliderSpec(
            "connection_strength",
            0.0,
            1.0,
            0.05,
            "連綿（続け字）",
            "同じ字の近い画を薄い線で続ける。近いほど高確率＋乱数。Z一定で点線化しない",
        ),
    ],
}
# 折りたたみ表示にするセクション
_COLLAPSED_SECTIONS = {"人らしさ調整"}
_SLIDERS = [spec for specs in _SLIDER_SECTIONS.values() for spec in specs]
_SLIDER_FIELDS = [spec.field for spec in _SLIDERS]
assert set(_SLIDER_FIELDS) | {"plot_page_numbers", "paper_width", "paper_height"} == {
    f.name for f in fields(Settings)
}


def _report_paper_data_uri() -> str:
    """レポート用紙画像を縮小した data URI（WebSerial プレビューの背景）。無ければ空。"""
    path = report_paper_path()
    if path is None:
        return ""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((1200, 1200))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
    except OSError:
        logger.exception("report paper background load failed")
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def app_head() -> str:
    """``<head>`` に注入する HTML（フォント・用紙背景・WebSerial スクリプト）。"""
    script = (_ASSETS / "webserial_sender.js").read_text(encoding="utf-8")
    # webserial_sender.js より前に背景画像を定義する
    paper = f"<script>window.__ppReportPaper={json.dumps(_report_paper_data_uri())};</script>"
    return content.FONT_HEAD + paper + f"<script>\n{script}\n</script>"


def format_coverage(report: CharCoverageReport) -> str:
    """描画経路ごとの文字数を Markdown で要約する（未収録には警告を付ける）。"""
    tiers_def = [
        ("ユーザー筆跡", report.user_strokes, ""),
        ("部品合成", report.composed, ""),
        ("ML推論", report.ml_inference, ""),
        ("KanjiVG", report.kanjivg, ""),
        ("幾何生成", report.geometric, ""),
        ("未収録（空白化）", report.missing_glyphs, "⚠️ "),
    ]
    total = sum(len(chars) for _, chars, _ in tiers_def) + len(report.skipped)
    if total == 0:
        return ""
    tiers: list[str] = []
    for label, chars, icon in tiers_def:
        if not chars:
            continue
        unique = sorted(set(chars))
        preview = "".join(unique[:60]) + ("..." if len(unique) > 60 else "")
        tiers.append(f"{icon}**{label}** ({len(chars)}字 / {len(unique)}種): {preview}")
    rendered = total - len(report.skipped) - len(report.missing_glyphs)
    summary = f"全{total}文字 (描画: {rendered}, スキップ: {len(report.skipped)})"
    return summary + "\n\n" + " | ".join(tiers) if tiers else summary


def _cleanup_paths(paths: Iterable[str | Path] | None) -> None:
    for p in paths or []:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("temp cleanup failed for %s: %s", p, exc)


def _validation_html(errors: list[str]) -> str:
    if not errors:
        return ""
    items = "".join(f"<li>{e}</li>" for e in errors)
    return (
        '<div style="padding:8px 12px;background:#fff1f0;border-left:4px solid #ff4d4f;'
        'border-radius:4px;color:#a8071a;"><strong>設定エラー</strong>'
        f'<ul style="margin:4px 0 0 16px;">{items}</ul></div>'
    )


def _estimate_pages(text: str, settings: Settings) -> str:
    n = len(text) if text else 0
    content_w = settings.paper_width - settings.margin_left - settings.margin_right
    content_h = settings.paper_height - settings.margin_top - settings.margin_bottom
    per_line = max(int(content_w / settings.font_size), 1)
    lines = max(int(content_h / settings.line_spacing), 1)
    pages = max(1, -(-n // max(per_line * lines, 1)))
    return f"文字数: {n} | 推定ページ数: {pages}"


def create_app(
    checkpoint_path: Path | str | None = None,
    kanjivg_dir: Path | str | None = None,
    user_strokes_dir: Path | str | None = None,
) -> gr.Blocks:
    """Gradio Blocks を構築する。

    Args:
        checkpoint_path: ML モデルのチェックポイント。
        kanjivg_dir: KanjiVG 参照字形ディレクトリ。
        user_strokes_dir: ユーザー筆跡（プロファイルのルート、または 1 人分）。
    """
    profile_ids: list[str] = []
    if user_strokes_dir is not None and Path(user_strokes_dir).is_dir():
        profile_ids = [p.id for p in list_profiles(Path(user_strokes_dir))]
    default_profile = profile_ids[0] if profile_ids else None

    def build(settings: Settings, profile: str | None, japanese_only: bool) -> PlotterPipeline:
        return PlotterPipeline(
            settings,
            checkpoint_path=checkpoint_path,
            kanjivg_dir=kanjivg_dir,
            user_strokes_dir=user_strokes_dir,
            profile=profile,
            japanese_only=japanese_only,
        )

    defaults = Settings()
    stale = gr.update(value=content.STALE_BANNER_HTML, visible=True)

    with gr.Blocks(title="Pen Plotter") as app:
        settings_state = gr.State(value=defaults)
        persisted_settings = gr.BrowserState(None, storage_key="pen_plotter_settings_v1")
        persisted_profile = gr.BrowserState(None, storage_key="pen_plotter_profile_v1")
        prev_preview_paths = gr.State(value=[])
        prev_gcode_tmpdir = gr.State(value=None)
        sliders: dict[str, gr.Slider] = {}

        gr.Markdown("# Pen Plotter")
        with gr.Tabs():
            with gr.Tab("生成"):
                with gr.Row(equal_height=False):
                    # ===== 左: 入力 =====
                    with gr.Column(scale=2, elem_classes=["pp-section", "pp-section--flush"]):
                        gr.HTML('<div class="pp-section-title">テキスト入力</div>')
                        text_input = gr.Textbox(
                            lines=18,
                            placeholder="テキストを入力...\n\n# 見出し\n$数式$ / $$ブロック数式$$",
                            show_label=False,
                            container=False,
                        )
                        char_count_md = gr.Markdown("文字数: 0")
                        with gr.Row():
                            preview_btn = gr.Button("プレビュー", variant="primary", scale=2)
                            gcode_btn = gr.Button("G-code 生成", variant="secondary", scale=2)
                            clear_btn = gr.Button(
                                "クリア", variant="secondary", scale=1, elem_classes=["pp-ghost"]
                            )
                        with gr.Accordion("例文を挿入", open=False), gr.Row():
                            example_btns = {
                                label: gr.Button(
                                    label, variant="secondary", size="sm", elem_classes=["pp-ghost"]
                                )
                                for label in content.EXAMPLES
                            }

                    # ===== 中央: プレビュー =====
                    with gr.Column(scale=3, elem_classes=["pp-section", "pp-section--flush"]):
                        gr.HTML('<div class="pp-section-title">プレビュー</div>')
                        stale_banner = gr.HTML(value="", visible=False)
                        status_md = gr.Markdown(visible=False)
                        preview_gallery = gr.Gallery(
                            label="プレビュー",
                            columns=1,
                            height=700,
                            show_label=False,
                            object_fit="contain",
                            preview=True,
                        )
                        with gr.Accordion("文字カバレッジ", open=False):
                            coverage_md = gr.Markdown("")
                        gcode_files = gr.Files(label="G-code ダウンロード", interactive=False)

                    # ===== 右: 設定 =====
                    with gr.Column(scale=2, elem_classes=["pp-section", "pp-section--flush"]):
                        for i, (section, specs) in enumerate(_SLIDER_SECTIONS.items()):
                            if section in _COLLAPSED_SECTIONS:
                                group = gr.Accordion(section, open=False)
                            else:
                                group = nullcontext()
                                rule = " pp-section-title--rule" if i else ""
                                gr.HTML(f'<div class="pp-section-title{rule}">{section}</div>')
                            if section == "手書き":
                                profile_select = gr.Dropdown(
                                    choices=profile_ids,
                                    value=default_profile,
                                    label="人物プロファイル",
                                    visible=bool(profile_ids),
                                    interactive=bool(profile_ids),
                                )
                                japanese_only = gr.Checkbox(
                                    value=False,
                                    label="日本語文字だけプロット（英数字・数式・記号をスキップ）",
                                )
                                plot_page_numbers = gr.Checkbox(
                                    value=defaults.plot_page_numbers, label="ページ番号をプロット"
                                )
                            with group:
                                for spec in specs:
                                    sliders[spec.field] = gr.Slider(
                                        spec.minimum,
                                        spec.maximum,
                                        value=getattr(defaults, spec.field),
                                        step=spec.step,
                                        label=spec.label,
                                        info=spec.info,
                                    )
                        reset_btn = gr.Button(
                            "デフォルトに戻す",
                            variant="secondary",
                            size="sm",
                            elem_classes=["pp-ghost"],
                        )
                        validation_md = gr.HTML(value="", visible=False)

                with gr.Accordion("ヘルプ", open=False):
                    gr.Markdown(content.HELP_MARKDOWN)

            with gr.Tab("プロッタ送信"):
                ws = _build_webserial_tab()

        slider_list = [sliders[f] for f in _SLIDER_FIELDS]

        # ===== 設定の変更 =====

        def settings_outputs(settings: Settings, *, mark_stale: bool = True) -> tuple:
            errors = settings.validate()
            ok = not errors
            return (
                settings,
                stale if mark_stale else gr.update(value="", visible=False),
                gr.update(value=_validation_html(errors), visible=not ok),
                gr.update(interactive=ok),
                gr.update(interactive=ok),
                settings.to_dict(),
            )

        settings_targets = [
            settings_state,
            stale_banner,
            validation_md,
            preview_btn,
            gcode_btn,
            persisted_settings,
        ]
        for field_name, slider in sliders.items():
            slider.change(
                lambda value, s, _f=field_name: settings_outputs(replace(s, **{_f: float(value)})),
                inputs=[slider, settings_state],
                outputs=settings_targets,
            )
        plot_page_numbers.change(
            lambda value, s: settings_outputs(replace(s, plot_page_numbers=bool(value))),
            inputs=[plot_page_numbers, settings_state],
            outputs=settings_targets,
        )
        profile_select.change(
            lambda pid: (stale, pid),
            inputs=[profile_select],
            outputs=[stale_banner, persisted_profile],
        )
        japanese_only.change(lambda _v: stale, inputs=[japanese_only], outputs=[stale_banner])
        text_input.change(
            _estimate_pages, inputs=[text_input, settings_state], outputs=[char_count_md]
        )

        def on_reset():
            return (*settings_outputs(Settings(), mark_stale=False), *_slider_values(Settings()),
                    False)  # fmt: skip

        reset_btn.click(
            on_reset, outputs=[*settings_targets, *slider_list, plot_page_numbers, japanese_only]
        )

        def on_load(stored_settings: dict | None, stored_profile: str | None):
            settings = Settings.from_dict(stored_settings)
            profile = stored_profile if stored_profile in profile_ids else default_profile
            return (
                *settings_outputs(settings, mark_stale=False),
                *_slider_values(settings),
                gr.update(value=profile) if profile_ids else gr.update(),
            )

        app.load(
            on_load,
            inputs=[persisted_settings, persisted_profile],
            outputs=[*settings_targets, *slider_list, plot_page_numbers, profile_select],
        )

        # ===== 生成 =====

        def on_preview(text, settings, profile, jp_only, old_paths, progress=gr.Progress()):  # noqa: B008
            _cleanup_paths(old_paths)
            if errors := settings.validate():
                return [], [], gr.update(value=_validation_html(errors), visible=True), "", stale
            if not text or not text.strip():
                msg = gr.update(value="**テキストを入力してください。**", visible=True)
                return [], [], msg, "", gr.update(visible=False)
            generated: list[Path] = []
            try:
                pipeline = build(settings, profile, jp_only)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
                    base_path = Path(fp.name)
                start = time.time()
                generated = pipeline.generate_preview(
                    text, base_path, lambda f, d: progress(min(max(f, 0.0), 1.0), desc=d)
                )
            except Exception as exc:
                logger.exception("Preview failed")
                _cleanup_paths(generated)
                return [], [], gr.update(value=f"**エラー:** {exc}", visible=True), "", stale
            paths = [str(p) for p in generated]
            status = f"**{len(paths)}ページ生成しました** ({time.time() - start:.1f}秒)"
            return (
                paths,
                paths,
                gr.update(value=status, visible=True),
                format_coverage(pipeline.coverage),
                gr.update(visible=False),
            )

        preview_btn.click(
            on_preview,
            inputs=[text_input, settings_state, profile_select, japanese_only, prev_preview_paths],
            outputs=[preview_gallery, prev_preview_paths, status_md, coverage_md, stale_banner],
        )

        def on_generate(text, settings, profile, jp_only, old_tmpdir, progress=gr.Progress()):  # noqa: B008
            if old_tmpdir:
                shutil.rmtree(old_tmpdir, ignore_errors=True)
            if errors := settings.validate():
                return (
                    None,
                    None,
                    gr.update(value=_validation_html(errors), visible=True),
                    gr.update(),
                )
            if not text or not text.strip():
                msg = gr.update(value="**テキストを入力してください。**", visible=True)
                return None, None, msg, gr.update()
            tmp_dir = Path(tempfile.mkdtemp(prefix="penplotter_"))
            try:
                start = time.time()
                paths = build(settings, profile, jp_only).generate_gcode(
                    text,
                    tmp_dir / "output.gcode",
                    lambda f, d: progress(min(max(f, 0.0), 1.0), desc=d),
                )
            except Exception as exc:
                logger.exception("G-code generation failed")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return None, None, gr.update(value=f"**エラー:** {exc}", visible=True), gr.update()
            status = (
                f"**G-code を {len(paths)} ページ生成しました** ({time.time() - start:.1f}秒) "
                "— 自動ダウンロードを開始します"
            )
            pages = list(range(1, len(paths) + 1))
            return (
                [str(p) for p in paths],
                str(tmp_dir),
                gr.update(value=status, visible=True),
                gr.update(choices=pages, value=pages),
            )

        preview_inputs = [ws["source"], gcode_files, ws["upload"], ws["pages"]]
        gcode_btn.click(
            on_generate,
            inputs=[text_input, settings_state, profile_select, japanese_only, prev_gcode_tmpdir],
            outputs=[gcode_files, prev_gcode_tmpdir, status_md, ws["pages"]],
        ).then(fn=None, inputs=[gcode_files], js=content.TRIGGER_MULTI_DOWNLOAD_JS).then(
            fn=None, inputs=preview_inputs, js=_ws_js("preview", with_files=True)
        )

        def on_clear(old_paths, old_tmpdir):
            _cleanup_paths(old_paths)
            if old_tmpdir:
                shutil.rmtree(old_tmpdir, ignore_errors=True)
            cleared_pages = gr.update(choices=[], value=[])
            return "", [], None, gr.update(visible=False), "文字数: 0", "", [], None, cleared_pages

        clear_btn.click(
            on_clear,
            inputs=[prev_preview_paths, prev_gcode_tmpdir],
            outputs=[
                text_input,
                preview_gallery,
                gcode_files,
                status_md,
                char_count_md,
                coverage_md,
                prev_preview_paths,
                prev_gcode_tmpdir,
                ws["pages"],
            ],
        )
        for label, btn in example_btns.items():
            btn.click(lambda _t=content.EXAMPLES[label]: _t, outputs=[text_input])

        # ===== WebSerial 送信（ブラウザ側 JS が処理する） =====

        ws["source"].change(
            lambda source: (
                gr.update(visible=source == "uploaded"),
                gr.update(visible=source != "uploaded"),
            ),
            inputs=[ws["source"]],
            outputs=[ws["upload_group"], ws["pages"]],
        ).then(fn=None, inputs=preview_inputs, js=_ws_js("preview", with_files=True))
        for comp in (ws["pages"], ws["upload"]):
            comp.change(fn=None, inputs=preview_inputs, js=_ws_js("preview", with_files=True))
        ws["start"].click(fn=None, inputs=preview_inputs, js=_ws_js("start", with_files=True))
        for key, method in [
            ("connect", "connect"),
            ("disconnect", "disconnect"),
            ("resume", "resume"),
            ("stop", "stop"),
            ("emergency", "emergencyStop"),
        ]:
            ws[key].click(fn=None, js=_ws_js(method))

    return app


def _slider_values(settings: Settings) -> list:
    return [getattr(settings, f) for f in _SLIDER_FIELDS] + [settings.plot_page_numbers]


def _ws_js(method: str, *, with_files: bool = False) -> str:
    if with_files:
        args = "source, generatedFiles, uploadedFiles, pages"
        return f"({args}) => window.penPlotterWebSerial.{method}({args})"
    return f"() => window.penPlotterWebSerial.{method}()"


def _build_webserial_tab() -> dict:
    """「プロッタ送信」タブの部品を作り、イベント配線用に名前付きで返す。"""
    c: dict = {}
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["pp-fieldset"]):
                gr.HTML('<div class="pp-legend">接続</div>')
                gr.HTML(value=content.WEBSERIAL_STATUS_HTML, elem_id="webserial-status")
                with gr.Row():
                    c["connect"] = gr.Button(
                        "接続", variant="primary", elem_id="webserial-connect-btn"
                    )
                    c["disconnect"] = gr.Button(
                        "切断", variant="secondary", elem_id="webserial-disconnect-btn"
                    )
            with gr.Group(elem_classes=["pp-fieldset"]):
                gr.HTML('<div class="pp-legend">送信データ選択</div>')
                gr.HTML(
                    '<div class="pp-fieldset-note">'
                    "「生成」タブで作った G-code は「生成済み G-code」で選べます。</div>"
                )
                c["source"] = gr.Radio(
                    choices=[("生成済み G-code", "generated"), ("アップロード G-code", "uploaded")],
                    value="generated",
                    label="送信元",
                )
                c["pages"] = gr.Dropdown(
                    choices=[],
                    value=[],
                    multiselect=True,
                    label="送信ページ",
                    info="送信するページ（既定: 全ページ）。1ページごとに用紙交換で停止します。",
                )
                # gr.Files を visible=False で初期化すると loading 表示が固着するため
                # 親 Group の表示で切り替える
                with gr.Group(visible=False) as c["upload_group"]:
                    c["upload"] = gr.Files(
                        label="アップロード G-code (.gcode/.nc/.txt)",
                        file_types=[".gcode", ".nc", ".txt"],
                    )
            with gr.Group(elem_classes=["pp-fieldset"]):
                gr.HTML('<div class="pp-legend">ジョブ送信</div>')
                with gr.Row():
                    c["start"] = gr.Button(
                        "送信開始", variant="primary", elem_id="webserial-start-btn"
                    )
                    c["stop"] = gr.Button("停止", variant="secondary", elem_id="webserial-stop-btn")
                    c["emergency"] = gr.Button(
                        "緊急停止", variant="stop", elem_id="webserial-emergency-btn"
                    )
                    c["resume"] = gr.Button(
                        "続行（用紙交換後）", variant="primary", elem_id="webserial-resume-btn"
                    )
                gr.HTML(value=content.WEBSERIAL_PROGRESS_HTML)
        with gr.Column(scale=2), gr.Group(elem_classes=["pp-fieldset"]):
            gr.HTML('<div class="pp-legend">プレビュー</div>')
            gr.HTML(value=content.WEBSERIAL_PREVIEW_HTML, elem_id="webserial-preview")
    with gr.Group(elem_classes=["pp-fieldset"]):
        gr.HTML('<div class="pp-legend">ログ</div>')
        gr.HTML(value=content.WEBSERIAL_LOG_HTML, elem_id="webserial-log")
    return c
