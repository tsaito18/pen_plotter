"""Gradio Web UI for pen plotter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import gradio as gr

from src.ui.web_app import PlotterPipeline

logger = logging.getLogger(__name__)

_EXAMPLE_REPORT_HEADER = """\
# 物理学実験レポート

## 実験目的

オームの法則 $V = IR$ を検証し、抵抗値の温度依存性について考察する。

## 実験方法

直流電源を用いて回路に電圧を印加し、電流計と電圧計の読み取り値を記録した。測定は室温 $T = 25$ ℃の環境で行った。
"""

_EXAMPLE_MATH_REPORT = """\
# 微分方程式の解法

## 問題

次の二階線形微分方程式を解け。

$$\\frac{d^2y}{dx^2} + 4y = 0$$

## 解法

特性方程式 $\\lambda^2 + 4 = 0$ より $\\lambda = \\pm 2i$ を得る。

したがって一般解は

$$y = C_1 \\cos 2x + C_2 \\sin 2x$$

ここで $C_1$, $C_2$ は任意定数である。
"""

_EXAMPLE_ESSAY = """\
近年、人工知能の発展は目覚ましく、私たちの生活に大きな変化をもたらしている。\
特に自然言語処理の分野では、大規模言語モデルの登場により、\
文章の生成や翻訳の精度が飛躍的に向上した。

しかしながら、技術の進歩には常に倫理的な課題が伴う。\
個人情報の保護やバイアスの問題など、解決すべき課題は多い。

今後は技術と倫理のバランスを取りながら、\
社会全体で議論を深めていく必要があるだろう。
"""

_HELP_MARKDOWN = """\
### 書式リファレンス

| 書式 | 入力例 | 説明 |
|------|--------|------|
| 見出し | `# 大見出し` / `## 中見出し` | 最大3段階 |
| インライン数式 | `$V = IR$` | 文中に数式を挿入 |
| ブロック数式 | `$$E = mc^2$$` | 独立行に数式を配置 |
| 分数 | `$\\frac{a}{b}$` | 分子/分母を上下に配置 |
| 上付き・下付き | `$x^2$` / `$f_0$` | 指数・添字 |
| ギリシャ文字 | `$\\alpha$` `$\\beta$` `$\\omega$` | 主要なギリシャ文字に対応 |
| 段落区切り | 空行 | 空行で段落を分割 |

### 対応文字

ひらがな・カタカナ・漢字（常用）・英数字・数式記号

手書きサンプルが収集済みの文字はユーザー筆跡で描画され、\
それ以外は KanjiVG データをベースに生成されます。\
プレビュー後「文字カバレッジ」で各文字の描画方式を確認できます。

### ヒント

- 設定タブでフォントサイズや余白を調整できます
- 温度を上げると文字の揺らぎが増し、下げると整った字になります
- G-code 生成は1ページ目のみ出力されます
"""


def create_app(pipeline: PlotterPipeline) -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(title="Pen Plotter") as app:
        prev_files = gr.State([])

        gr.Markdown("# Pen Plotter")

        with gr.Tabs():
            with gr.Tab("作成"):
                with gr.Row(equal_height=False):
                    # --- 左カラム: 入力 ---
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            lines=16,
                            placeholder=(
                                "テキストを入力...\n\n# 見出し\n$数式$ / $$ブロック数式$$"
                            ),
                            label="テキスト入力",
                        )
                        char_count_md = gr.Markdown("文字数: 0")

                        with gr.Row():
                            preview_btn = gr.Button("プレビュー", variant="primary", scale=2)
                            gcode_btn = gr.Button("G-code 生成", scale=2)
                            clear_btn = gr.Button("クリア", variant="secondary", scale=1)

                    # --- 右カラム: 出力 ---
                    with gr.Column(scale=3):
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
                        gcode_file = gr.File(label="G-code ダウンロード")

            with gr.Tab("設定"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### レイアウト")
                        font_size = gr.Slider(
                            3.0, 10.0, value=4.5, step=0.1, label="フォントサイズ (mm)"
                        )
                        line_spacing = gr.Slider(
                            5.0, 15.0, value=7.16, step=0.01, label="行間隔 (mm)"
                        )
                        gr.Markdown("### 余白 (mm)")
                        margin_top = gr.Slider(5, 60, value=48, step=1, label="上")
                        margin_bottom = gr.Slider(5, 50, value=34, step=1, label="下")
                        margin_left = gr.Slider(1, 50, value=5, step=1, label="左")
                        margin_right = gr.Slider(1, 50, value=5, step=1, label="右")

                    with gr.Column():
                        gr.Markdown("### プロッタ")
                        draw_speed = gr.Slider(
                            200, 3000, value=1000, step=50, label="描画速度 (mm/min)"
                        )
                        travel_speed = gr.Slider(
                            1000, 5000, value=3000, step=100, label="移動速度 (mm/min)"
                        )
                        pen_delay = gr.Slider(
                            0.05, 0.50, value=0.15, step=0.01, label="ペン遅延 (s)"
                        )
                        gr.Markdown("### ML モデル")
                        temperature = gr.Slider(
                            0.1,
                            2.0,
                            value=1.0,
                            step=0.1,
                            label="温度",
                            info="高いほど文字の揺らぎが大きくなります",
                        )

                reset_btn = gr.Button("デフォルトに戻す")

            with gr.Tab("ヘルプ"):
                gr.Markdown(_HELP_MARKDOWN)
                gr.Markdown("### 例文を試す")
                with gr.Row():
                    ex_report_btn = gr.Button("レポートヘッダー")
                    ex_math_btn = gr.Button("数式レポート")
                    ex_essay_btn = gr.Button("小論文")

        # --- Callbacks ---

        def _update_char_count(text: str) -> str:
            n = len(text) if text else 0
            page_config = pipeline._page_config
            chars_per_line = int(
                (page_config.paper_size[0] - page_config.margin_left - page_config.margin_right)
                / pipeline._typesetter.font_size
            )
            lines_per_page = int(
                (page_config.paper_size[1] - page_config.margin_top - page_config.margin_bottom)
                / page_config.line_spacing
            )
            chars_per_page = max(chars_per_line * lines_per_page, 1)
            est_pages = max(1, -(-n // chars_per_page))
            return f"文字数: {n} | 推定ページ数: {est_pages}"

        text_input.change(_update_char_count, inputs=[text_input], outputs=[char_count_md])

        all_settings = [
            font_size,
            line_spacing,
            margin_top,
            margin_bottom,
            margin_left,
            margin_right,
            draw_speed,
            travel_speed,
            pen_delay,
            temperature,
        ]

        def _on_preview(
            text: str,
            prev: list[str],
            font_sz: float,
            line_sp: float,
            m_top: float,
            m_bottom: float,
            m_left: float,
            m_right: float,
            draw_spd: float,
            travel_spd: float,
            pen_dl: float,
            temp: float,
            progress=gr.Progress(),
        ):
            for f in prev or []:
                Path(f).unlink(missing_ok=True)

            if not text or not text.strip():
                return (
                    [],
                    [],
                    gr.update(value="**テキストを入力してください。**", visible=True),
                    None,
                    "",
                )

            def _progress_cb(frac: float, desc: str) -> None:
                progress(frac, desc=desc)

            try:
                _apply_settings(
                    pipeline,
                    font_sz,
                    line_sp,
                    m_top,
                    m_bottom,
                    m_left,
                    m_right,
                    draw_spd,
                    travel_spd,
                    pen_dl,
                    temp,
                )

                start = time.time()
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    path = Path(f.name)
                paths = pipeline.generate_preview(
                    text, save_path=path, progress_callback=_progress_cb
                )
                elapsed = time.time() - start

                str_paths = [str(p) for p in paths]
                status = f"**{len(paths)}ページ生成しました** ({elapsed:.1f}秒)"
                coverage_text = _format_coverage(pipeline._last_coverage)
                return (
                    str_paths,
                    str_paths,
                    gr.update(value=status, visible=True),
                    None,
                    coverage_text,
                )
            except Exception as e:
                logger.exception("Preview failed")
                return (
                    [],
                    prev or [],
                    gr.update(value=f"**エラー:** {e}", visible=True),
                    None,
                    "",
                )

        preview_btn.click(
            _on_preview,
            inputs=[text_input, prev_files] + all_settings,
            outputs=[preview_gallery, prev_files, status_md, gcode_file, coverage_md],
        )

        def _on_generate(
            text: str,
            font_sz: float,
            line_sp: float,
            m_top: float,
            m_bottom: float,
            m_left: float,
            m_right: float,
            draw_spd: float,
            travel_spd: float,
            pen_dl: float,
            temp: float,
            progress=gr.Progress(),
        ):
            if not text or not text.strip():
                return None, gr.update(value="**テキストを入力してください。**", visible=True)

            def _progress_cb(frac: float, desc: str) -> None:
                progress(frac, desc=desc)

            try:
                _apply_settings(
                    pipeline,
                    font_sz,
                    line_sp,
                    m_top,
                    m_bottom,
                    m_left,
                    m_right,
                    draw_spd,
                    travel_spd,
                    pen_dl,
                    temp,
                )

                start = time.time()
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
                    gcode_path = Path(f.name)
                pipeline.generate_gcode_file(
                    text, save_path=gcode_path, progress_callback=_progress_cb
                )
                elapsed = time.time() - start

                status = f"**G-code を生成しました** ({elapsed:.1f}秒)"
                return str(gcode_path), gr.update(value=status, visible=True)
            except Exception as e:
                logger.exception("G-code generation failed")
                return None, gr.update(value=f"**エラー:** {e}", visible=True)

        gcode_btn.click(
            _on_generate,
            inputs=[text_input] + all_settings,
            outputs=[gcode_file, status_md],
        )

        clear_btn.click(
            lambda: (
                "",
                gr.update(value=[], visible=True),
                None,
                gr.update(visible=False),
                "文字数: 0",
                "",
            ),
            outputs=[
                text_input,
                preview_gallery,
                gcode_file,
                status_md,
                char_count_md,
                coverage_md,
            ],
        )

        def _reset_defaults() -> tuple:
            return 6.0, 8.0, 30, 15, 25, 15, 1000, 3000, 0.15, 1.0

        reset_btn.click(_reset_defaults, outputs=all_settings)

        ex_report_btn.click(lambda: _EXAMPLE_REPORT_HEADER, outputs=[text_input])
        ex_math_btn.click(lambda: _EXAMPLE_MATH_REPORT, outputs=[text_input])
        ex_essay_btn.click(lambda: _EXAMPLE_ESSAY, outputs=[text_input])

    return app


def _format_coverage(report: object) -> str:
    """CharCoverageReport をMarkdownにフォーマット。"""
    total = (
        len(report.user_strokes)
        + len(report.ml_inference)
        + len(report.kanjivg)
        + len(report.geometric)
        + len(report.rect_fallback)
        + len(report.skipped)
    )
    if total == 0:
        return ""

    tiers: list[str] = []

    def _tier(label: str, chars: list[str], icon: str = "") -> None:
        if not chars:
            return
        unique = sorted(set(chars))
        n_unique = len(unique)
        preview = "".join(unique[:60])
        suffix = "..." if n_unique > 60 else ""
        prefix = f"{icon} " if icon else ""
        tiers.append(f"{prefix}**{label}** ({len(chars)}字 / {n_unique}種): {preview}{suffix}")

    _tier("ユーザー筆跡", report.user_strokes)
    _tier("ML推論", report.ml_inference)
    _tier("KanjiVG", report.kanjivg)
    _tier("幾何生成", report.geometric)
    _tier("矩形フォールバック", report.rect_fallback, icon="\u26a0\ufe0f")

    rendered = total - len(report.skipped)
    summary = f"全{total}文字 (描画: {rendered}, スキップ: {len(report.skipped)})"

    return summary + "\n\n" + " | ".join(tiers) if tiers else summary


def _apply_settings(
    pipeline: PlotterPipeline,
    font_sz: float,
    line_sp: float,
    m_top: float,
    m_bottom: float,
    m_left: float,
    m_right: float,
    draw_spd: float,
    travel_spd: float,
    pen_dl: float,
    temp: float,
) -> None:
    """設定をパイプラインに反映する。"""
    from src.layout.page_layout import PageConfig
    from src.layout.typesetter import Typesetter

    page_config = PageConfig(
        line_spacing=float(line_sp),
        margin_top=float(m_top),
        margin_bottom=float(m_bottom),
        margin_left=float(m_left),
        margin_right=float(m_right),
    )
    pipeline._page_config = page_config
    pipeline._typesetter = Typesetter(
        page_config=page_config,
        font_size=float(font_sz),
        augmenter=pipeline._typesetter.augmenter,
    )
    pipeline._temperature = float(temp)
    pipeline._plotter_config.draw_speed = float(draw_spd)
    pipeline._plotter_config.travel_speed = float(travel_spd)
    pipeline._plotter_config.pen_delay = float(pen_dl)
