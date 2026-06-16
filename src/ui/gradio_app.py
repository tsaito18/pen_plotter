"""Pen Plotter Gradio Web UI（全面書き換え版）。

副作用ゼロを徹底するため、UI 層は UISettings の snapshot を gr.State に保持し、
プレビュー / G-code 生成のたびに build_pipeline で新規パイプラインを構築する。
旧版の _apply_settings によるパイプライン属性差し替えは廃止。

設計の核:
- gr.State(UISettings) と gr.State(stale: bool) を起点に状態を一元管理
- Slider の change で UISettings を不変的に更新（dataclass.replace）し、stale=True
- バリデーションは UISettings.validate() に委譲し、エラー時はボタン disable
- 例外時のテンポラリリークを try/finally で防ぐ
- ユーザー指示により旧 gradio_app.py は完全置き換え（保険ファイルなし）
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import gradio as gr

from src.ui.settings import UISettings
from src.ui.web_app import PlotterPipeline, build_pipeline
from src.collector.profiles import list_profiles

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
$$
\\frac{d^2y}{dx^2} + 4y = 0
$$
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

_EXAMPLE_TABLE = """\
# 引張試験結果
各供試材の機械的性質を表に示す。

: 表1 各材料の機械的性質
| 材料 | 降伏応力 | 引張強さ | 伸び |
|---|---|---|---|
| SS400 | 245 | 400 | 28 |
| S35C | 305 | 510 | 23 |
| SUS304 | 205 | 520 | 40 |

降伏応力は SUS304 が最も低く、引張強さは S35C と SUS304 が高い。
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
| 表 | `\\| 列1 \\| 列2 \\|`<br>`\\|---\\|---\\|`<br>`\\| a \\| b \\|` | パイプ表（2行目の区切り必須）。中央寄せで描画 |
| 表キャプション | 表の直後に `: 表1 タイトル` | 表の下に中央寄せで配置 |
| 段落区切り | 空行 | 空行で段落を分割 |
| 段落字下げなし | `\\noindent 本文` | 段落先頭の字下げを抑止 |

### 対応文字

ひらがな・カタカナ・漢字（常用）・英数字・数式記号

手書きサンプルが収集済みの文字はユーザー筆跡で描画され、\
それ以外は KanjiVG データをベースに生成されます。\
プレビュー後「文字カバレッジ」で各文字の描画方式を確認できます。

### ヒント

- 設定パネルでフォントサイズや余白を調整できます
- 温度を上げると文字の揺らぎが増し、下げると整った字になります
- G-code 生成は全ページ分が自動ダウンロードされます（初回はブラウザの許可ダイアログを承認してください）
- 設定を変更したらプレビューを再生成してください（黄色の警告が出ます）
"""


_STALE_BANNER_HTML = (
    '<div style="padding:8px 12px;background:#fff7e6;border-left:4px solid #faad14;'
    'border-radius:4px;color:#874d00;">'
    "設定が変更されました。プレビューを再生成してください。"
    "</div>"
)

_WEBSERIAL_SCRIPT = (Path(__file__).with_name("webserial_sender.js")).read_text(encoding="utf-8")
_WEBSERIAL_HEAD = f"<script>\n{_WEBSERIAL_SCRIPT}\n</script>"
_FONT_HEAD = """\
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;600;700&display=swap" rel="stylesheet">
"""


def _report_paper_data_uri() -> str:
    """レポート用紙背景を縮小して base64 data URI 化する。

    プレビュー Canvas の背景に敷くため、静的配信を使わず head 注入で渡す。
    画像が無い/読込失敗時は空文字を返し、起動を止めない。
    """
    import base64
    import io

    from PIL import Image

    try:
        path = Path("data/report_paper.jpg")
        if not path.exists():
            return ""
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((1200, 1200))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return "data:image/jpeg;base64," + encoded
    except Exception:
        logger.exception("report paper background load failed")
        return ""


# webserial_sender.js より前に window.__ppReportPaper を定義する順序にする。
_REPORT_PAPER_HEAD = (
    "<script>window.__ppReportPaper=" + json.dumps(_report_paper_data_uri()) + ";</script>"
)
_APP_HEAD = _FONT_HEAD + _REPORT_PAPER_HEAD + _WEBSERIAL_HEAD
_APP_CSS = """\
:root {
    --pp-primary: #2563eb;
    --pp-primary-hover: #1d4ed8;
    --pp-primary-soft: #eff4ff;
    --pp-success: #16a34a;
    --pp-success-soft: #ecfdf3;
    --pp-warning: #d97706;
    --pp-warning-soft: #fff7ed;
    --pp-danger: #dc2626;
    --pp-danger-soft: #fef2f2;
    --pp-neutral-50: #f8fafc;
    --pp-neutral-100: #f1f5f9;
    --pp-neutral-200: #e2e8f0;
    --pp-neutral-300: #cbd5e1;
    --pp-neutral-400: #94a3b8;
    --pp-neutral-500: #64748b;
    --pp-neutral-600: #475569;
    --pp-neutral-800: #1e293b;
    --pp-radius: 12px;
    --pp-radius-sm: 8px;
    --pp-shadow: 0 1px 2px rgba(15, 23, 42, 0.06), 0 4px 12px rgba(15, 23, 42, 0.06);
    --pp-space-1: 4px;
    --pp-space-2: 8px;
    --pp-space-3: 12px;
    --pp-space-4: 16px;
    --pp-space-5: 24px;
}

.gradio-container {
    max-width: 1400px !important;
    font-family: "Inter", "Noto Sans JP", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body,
button,
input,
textarea,
select,
label {
    font-family: "Inter", "Noto Sans JP", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-container .prose,
.gradio-container .markdown,
.gradio-container table {
    font-family: "Inter", "Noto Sans JP", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Technical Editorial（製図×ミニマル）。
   囲みボックス・影は使わず、極細 hairline と余白で構造化する。 */

/* セクション: 上部 hairline＋たっぷりの上下マージン。背景はごく薄い紙地。 */
.gradio-container .pp-section {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: var(--pp-space-4) 0 0;
    margin-top: var(--pp-space-5);
}

.gradio-container .pp-section--flush {
    margin-top: 0;
    padding-top: 0;
    border-top: none;
}

/* 極細罫線。セクション区切り・状態/ログの上下罫に共用。 */
.gradio-container .pp-hairline,
.gradio-container .pp-section:not(.pp-section--flush) {
    border-top: 1px solid var(--pp-neutral-200);
}

/* エディトリアルな小見出し: 小さめ・字間広め・淡色グレー。 */
.pp-section-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--pp-neutral-500);
    letter-spacing: 0.08em;
    margin-bottom: var(--pp-space-3);
}

.pp-section-title p {
    margin: 0;
}

/* カラム内で連続する小見出しの区切り: 上に hairline＋余白。 */
.pp-section-title--rule {
    border-top: 1px solid var(--pp-neutral-200);
    padding-top: var(--pp-space-4);
    margin-top: var(--pp-space-4);
}

/* ステータスバッジ: 小さな色ドット＋テキスト（枠なし）。 */
.pp-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0;
    background: transparent;
    font-size: 0.8rem;
    font-weight: 600;
    line-height: 1.4;
}

.pp-badge::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}

.pp-badge--idle {
    color: var(--pp-neutral-500);
}

.pp-badge--connected {
    color: var(--pp-success);
}

.pp-badge--streaming {
    color: var(--pp-primary);
}

.pp-badge--unsupported {
    color: var(--pp-danger);
}

/* 状態表示: fieldset 内に置くため枠なし。バッジ＋詳細テキストのみ。 */
.pp-status-card {
    display: flex;
    flex-direction: column;
    gap: var(--pp-space-2);
}

.pp-status-head {
    display: flex;
    align-items: center;
    gap: var(--pp-space-3);
    flex-wrap: wrap;
}

.pp-status-detail {
    font-size: 0.82rem;
    color: var(--pp-neutral-600);
    word-break: break-all;
}

.pp-progress-card {
    display: flex;
    flex-direction: column;
    gap: var(--pp-space-2);
    margin-top: var(--pp-space-3);
}

/* 進捗バー: fieldset 内の大型トラック（角丸）＋青 fill。 */
.pp-progress-track {
    height: 14px;
    background: var(--pp-neutral-100);
    border: 1px solid var(--pp-neutral-200);
    border-radius: 999px;
    overflow: hidden;
}

.pp-progress-fill {
    height: 100%;
    width: 0%;
    background: var(--pp-primary);
    border-radius: 999px;
    transition: width 0.2s ease;
}

.pp-progress-text {
    font-size: 0.78rem;
    color: var(--pp-neutral-500);
    font-variant-numeric: tabular-nums;
}

/* ログ: fieldset 内の等幅スクロール領域。枠は親 fieldset が持つ。 */
.pp-log {
    height: 200px;
    overflow: auto;
    background: var(--pp-neutral-50);
    border: 1px solid var(--pp-neutral-200);
    border-radius: 3px;
    padding: var(--pp-space-3);
    font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
    font-size: 0.78rem;
    line-height: 1.6;
}

/* 現在行表示（等幅）。Tkinter の TkFixedFont ラベル相当。 */
.pp-current-line {
    font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
    font-size: 0.78rem;
    color: var(--pp-neutral-600);
    word-break: break-all;
}

/* Instrument Panel: 白地＋細枠のクリーンな計器パネル（fieldset/LabelFrame 風）。
   背景はベタグレーを避け白に統一し、1px 実線で機械的に区切る。 */
.gradio-container .pp-fieldset {
    position: relative;
    background: #ffffff;
    border: 1px solid var(--pp-neutral-300);
    border-radius: 4px;
    box-shadow: none;
    padding: calc(var(--pp-space-5) + 2px) var(--pp-space-4) var(--pp-space-4);
    margin-bottom: var(--pp-space-4);
    overflow: visible;
}

/* Gradio が pp-fieldset 内の Group/Block に当てる地色・枠・影を打ち消し、
   入れ子の二重枠／のっぺりグレーを排す。 */
.gradio-container .pp-fieldset,
.gradio-container .pp-fieldset > div,
.gradio-container .pp-fieldset .gr-group,
.gradio-container .pp-fieldset .block,
.gradio-container .pp-fieldset .form {
    background: #ffffff;
}

.gradio-container .pp-fieldset .block,
.gradio-container .pp-fieldset .gr-group {
    border: none;
    box-shadow: none;
}

/* legend: 枠の上辺にラベルを重ね、白背景でボーダーを途切れさせる本物の fieldset 風。
   gr.HTML のラッパ階層に依存せず、.pp-legend 自身を .pp-fieldset 基準で絶対配置する。 */
.gradio-container .pp-fieldset .pp-legend {
    position: absolute;
    top: 0;
    left: var(--pp-space-3);
    transform: translateY(-50%);
    display: inline-block;
    background: #ffffff;
    padding: 0 var(--pp-space-2);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--pp-neutral-600);
    letter-spacing: 0.03em;
    line-height: 1.2;
    z-index: 1;
}

/* legend を内包する gr.HTML ブロックの余白を畳み、枠内コンテンツとの隙間を消す。 */
.gradio-container .pp-fieldset > div:has(.pp-legend) {
    margin: 0;
    padding: 0;
    min-height: 0;
}

.pp-legend p {
    margin: 0;
}

/* 注記: 小さく薄いキャプション。 */
.pp-fieldset-note {
    color: var(--pp-neutral-400);
    font-size: 0.74rem;
    line-height: 1.4;
    margin: 0 0 var(--pp-space-2);
}

/* 将来用プレビュープレースホルダ: 白地＋点線枠＋中央に淡色テキスト。 */
.pp-preview-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 360px;
    background: #ffffff;
    border: 1px dashed var(--pp-neutral-300);
    border-radius: 3px;
    color: var(--pp-neutral-400);
    font-size: 0.85rem;
}

/* プロッタ送信プレビュー: A4 比率の Canvas に G-code を直接描画する。 */
.pp-preview-canvas {
    width: 100%;
    aspect-ratio: 210/297;
    background: #ffffff;
    border: 1px solid var(--pp-neutral-300);
    border-radius: 3px;
    display: block;
}

.pp-preview-info {
    margin-top: 6px;
    font-size: 0.8rem;
    color: var(--pp-neutral-400);
}

/* primary ボタンを青系に統一（既定オレンジを上書き）。
   stop(緊急停止=赤) には適用しないよう :not(.stop) で除外する。 */
.gradio-container button.primary:not(.stop) {
    background: var(--pp-primary);
    border-color: var(--pp-primary);
    color: #ffffff;
}

.gradio-container button.primary:not(.stop):hover,
.gradio-container button.primary:not(.stop):focus {
    background: var(--pp-primary-hover);
    border-color: var(--pp-primary-hover);
    color: #ffffff;
}

/* secondary ボタン: 機械操作 / 切断 / 停止。背景に溶けないよう視認できる枠＋
   立体感を付与。primary(青)より控えめ、pp-ghost より一段はっきり。 */
.gradio-container button.secondary {
    background: #ffffff;
    border: 1px solid var(--pp-neutral-300);
    color: var(--pp-neutral-800);
    box-shadow: 0 1px 1px rgba(15, 23, 42, 0.04);
}

.gradio-container button.secondary:hover,
.gradio-container button.secondary:focus {
    background: var(--pp-neutral-100);
    border-color: var(--pp-neutral-400);
    color: var(--pp-neutral-800);
}

/* 控えめ ghost ボタン: クリア / リセット / 例文挿入。
   secondary より一段控えめに（枠細く・文字薄く・影なし）。 */
.gradio-container button.pp-ghost,
.gradio-container .pp-ghost button {
    background: transparent;
    border: 1px solid var(--pp-neutral-200);
    color: var(--pp-neutral-600);
    box-shadow: none;
}

.gradio-container button.pp-ghost:hover,
.gradio-container .pp-ghost button:hover {
    background: var(--pp-neutral-100);
    border-color: var(--pp-neutral-400);
    color: var(--pp-neutral-800);
}
"""

_WEBSERIAL_STATUS_HTML = """\
<div class="pp-status-card" id="webserial-status-panel">
  <div class="pp-status-head">
    <span id="webserial-status-badge" class="pp-badge pp-badge--idle">初期化中</span>
    <span id="webserial-status-value" class="pp-status-detail">初期化中</span>
  </div>
</div>
"""

_WEBSERIAL_PROGRESS_HTML = """\
<div class="pp-progress-card" id="webserial-progress-panel">
  <div class="pp-progress-track">
    <div id="webserial-progress-bar" class="pp-progress-fill"></div>
  </div>
  <div id="webserial-progress-text" class="pp-progress-text">0 / 0 行 (0%)</div>
  <div id="webserial-current-line" class="pp-current-line">現在行: -</div>
  <div id="webserial-paper-change"
       style="display:none; margin-top:8px; padding:10px 12px; border-radius:8px;
              background:#fff7e6; border:1px solid #ffd591; color:#874d00; font-weight:600;">
  </div>
</div>
"""

_WEBSERIAL_PREVIEW_HTML = """\
<canvas id="webserial-preview-canvas" class="pp-preview-canvas"></canvas>
<div id="webserial-preview-info" class="pp-preview-info">対象なし</div>
"""

_WEBSERIAL_LOG_HTML = """\
<div id="webserial-log-entries" class="pp-log">
</div>
"""


# Gradio 6.9 の gr.Files は JS callback に FileData[] を渡す。
# 各要素は {path, url, size, orig_name, mime_type, meta} 構造。
# Chrome の multi-download 許可ダイアログは初回のみ。400ms 間隔で順次クリックする。
_TRIGGER_MULTI_DOWNLOAD_JS = r"""
(files) => {
    if (!files) { return; }
    const list = Array.isArray(files) ? files : [files];
    list.forEach((f, i) => {
        if (!f) { return; }
        const url = f.url || f.path || (typeof f === 'string' ? f : null);
        if (!url) { return; }
        const name = f.orig_name || (typeof url === 'string' ? url.split('/').pop() : 'file');
        setTimeout(() => {
            const a = document.createElement('a');
            a.href = url;
            a.download = name;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }, i * 400);
    });
}
"""


def _format_coverage(report: object) -> str:
    """CharCoverageReport を Markdown サマリへ変換する。

    各カバレッジ階層（ユーザー筆跡 / ML / KanjiVG / 幾何 / 未収録 / 矩形）を
    descending priority で表示し、未収録・矩形フォールバックには警告アイコンを付ける。
    """
    total = (
        len(report.user_strokes)  # type: ignore[attr-defined]
        + len(report.ml_inference)  # type: ignore[attr-defined]
        + len(report.kanjivg)  # type: ignore[attr-defined]
        + len(report.geometric)  # type: ignore[attr-defined]
        + len(report.rect_fallback)  # type: ignore[attr-defined]
        + len(report.missing_glyphs)  # type: ignore[attr-defined]
        + len(report.skipped)  # type: ignore[attr-defined]
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

    _tier("ユーザー筆跡", report.user_strokes)  # type: ignore[attr-defined]
    _tier("ML推論", report.ml_inference)  # type: ignore[attr-defined]
    _tier("KanjiVG", report.kanjivg)  # type: ignore[attr-defined]
    _tier("幾何生成", report.geometric)  # type: ignore[attr-defined]
    _tier("未収録（空白化）", report.missing_glyphs, icon="⚠️")  # type: ignore[attr-defined]
    _tier("矩形フォールバック", report.rect_fallback, icon="⚠️")  # type: ignore[attr-defined]

    rendered = (
        total
        - len(report.skipped)  # type: ignore[attr-defined]
        - len(report.missing_glyphs)  # type: ignore[attr-defined]
    )
    summary = f"全{total}文字 (描画: {rendered}, スキップ: {len(report.skipped)})"  # type: ignore[attr-defined]

    return summary + "\n\n" + " | ".join(tiers) if tiers else summary


def _cleanup_paths(paths: Iterable[str | Path] | None) -> None:
    """テンポラリ画像のクリーンアップ。失敗してもログのみで握りつぶす。"""
    if not paths:
        return
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("temp cleanup failed for %s: %s", p, exc)


def _resolve_restored_profile(
    stored_profile: str | None,
    profile_ids: list[str],
    default_profile: str | None,
) -> str | None:
    """永続化された profile_id を現在の選択肢に照合して解決する。

    保存後にプロファイルが削除/リネームされていた場合に備え、
    現在の選択肢に存在しない値はデフォルトへフォールバックする。

    Args:
        stored_profile: localStorage から復元した profile_id。
        profile_ids: 現在利用可能な profile_id 一覧。
        default_profile: フォールバック先（通常は先頭プロファイル）。

    Returns:
        有効な profile_id、または default_profile。
    """
    if stored_profile is not None and stored_profile in profile_ids:
        return stored_profile
    return default_profile


def _validation_status(errors: list[str]) -> str:
    """validate() の戻り値を赤色 HTML で表示するためのフラグメント。"""
    if not errors:
        return ""
    items = "".join(f"<li>{e}</li>" for e in errors)
    return (
        '<div style="padding:8px 12px;background:#fff1f0;border-left:4px solid #ff4d4f;'
        'border-radius:4px;color:#a8071a;"><strong>設定エラー</strong>'
        f'<ul style="margin:4px 0 0 16px;">{items}</ul></div>'
    )


def create_app(
    checkpoint_path: Path | str | None = None,
    kanjivg_dir: Path | str | None = None,
    user_strokes_dir: Path | str | None = None,
) -> gr.Blocks:
    """Gradio Blocks を構築する。

    Args:
        checkpoint_path: ML モデルチェックポイント (.pt)。
        kanjivg_dir: KanjiVG JSON ディレクトリ。
        user_strokes_dir: ユーザー手書きストロークディレクトリ。

    Returns:
        構築済みの gr.Blocks。

    Note:
        環境引数（checkpoint_path 等）は closure に保持し、
        UI 操作のたびに UISettings + これらを束ねて build_pipeline を呼ぶ。
    """

    # 重い初期化（モデルロード/KanjiVG スキャン）を 1 度だけ走らせ、
    # 以降は UISettings 差分のみで PlotterPipeline を再生成する。
    # _stroke_renderer は環境引数依存なので毎回再構築でも OK。
    profile_options: list[tuple[str, str]] = []
    profile_root = Path(user_strokes_dir) if user_strokes_dir else None
    if profile_root is not None:
        profile_options = [(p.id, str(p.path)) for p in list_profiles(profile_root)]
        if not profile_options and profile_root.is_dir():
            profile_options = [(profile_root.name, str(profile_root))]

    default_profile = profile_options[0][0] if profile_options else None

    env_kwargs: dict[str, object] = {
        "checkpoint_path": checkpoint_path,
        "kanjivg_dir": kanjivg_dir,
    }

    def _profile_dir(profile_id: str | None) -> str | Path | None:
        if not profile_options:
            return user_strokes_dir
        for pid, path in profile_options:
            if pid == profile_id:
                return path
        return profile_options[0][1]

    def _build(
        settings: UISettings,
        profile_id: str | None = None,
        skip_non_japanese: bool = False,
    ) -> PlotterPipeline:
        return build_pipeline(
            settings,
            checkpoint_path=env_kwargs["checkpoint_path"],
            kanjivg_dir=env_kwargs["kanjivg_dir"],
            user_strokes_dir=_profile_dir(profile_id),
            skip_non_japanese=skip_non_japanese,
        )

    default_settings = UISettings.default()

    with gr.Blocks(title="Pen Plotter") as app:
        app._pen_plotter_webserial_head = _WEBSERIAL_HEAD  # type: ignore[attr-defined]
        settings_state = gr.State(value=default_settings)
        # ブラウザ localStorage への永続化。同一ブラウザで再訪時に設定を復元する。
        # storage_key にバージョン接尾辞を付け、将来の構造変更時に破棄しやすくする。
        persisted_settings = gr.BrowserState(None, storage_key="pen_plotter_settings_v1")
        persisted_profile = gr.BrowserState(None, storage_key="pen_plotter_profile_v1")
        # stale=True なら「設定が変わったがプレビュー未更新」状態
        preview_stale = gr.State(value=False)
        # 旧プレビューの一時パスを保持し、再生成時に確実にクリーンアップする
        prev_preview_paths = gr.State(value=[])
        # G-code 生成時の一時ディレクトリ（次回生成時に丸ごと削除）
        prev_gcode_tmpdir = gr.State(value=None)

        gr.Markdown("# Pen Plotter")

        with gr.Tabs():
            with gr.Tab("生成"):
                with gr.Row(equal_height=False):
                    # ========== 左カラム: 入力 ==========
                    with gr.Column(scale=2, elem_classes=["pp-section", "pp-section--flush"]):
                        gr.HTML('<div class="pp-section-title">テキスト入力</div>')
                        text_input = gr.Textbox(
                            lines=18,
                            placeholder=(
                                "テキストを入力...\n\n# 見出し\n$数式$ / $$ブロック数式$$"
                            ),
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

                        with gr.Accordion("例文を挿入", open=False):
                            with gr.Row():
                                ex_report_btn = gr.Button(
                                    "レポートヘッダー",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes=["pp-ghost"],
                                )
                                ex_math_btn = gr.Button(
                                    "数式レポート",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes=["pp-ghost"],
                                )
                                ex_essay_btn = gr.Button(
                                    "小論文",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes=["pp-ghost"],
                                )
                                ex_table_btn = gr.Button(
                                    "表サンプル",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes=["pp-ghost"],
                                )

                    # ========== 中央カラム: プレビュー ==========
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

                    # ========== 右カラム: 設定 ==========
                    with gr.Column(scale=2, elem_classes=["pp-section", "pp-section--flush"]):
                        gr.HTML('<div class="pp-section-title">レイアウト</div>')
                        font_size = gr.Slider(
                            3.0,
                            10.0,
                            value=default_settings.font_size,
                            step=0.1,
                            label="フォントサイズ (mm)",
                        )
                        line_spacing = gr.Slider(
                            5.0,
                            15.0,
                            value=default_settings.line_spacing,
                            step=0.01,
                            label="行間隔 (mm)",
                        )

                        gr.HTML(
                            '<div class="pp-section-title pp-section-title--rule">余白 (mm)</div>'
                        )
                        margin_top = gr.Slider(
                            5,
                            60,
                            value=default_settings.margin_top,
                            step=1,
                            label="上",
                        )
                        margin_bottom = gr.Slider(
                            5,
                            50,
                            value=default_settings.margin_bottom,
                            step=1,
                            label="下",
                        )
                        margin_left = gr.Slider(
                            1,
                            50,
                            value=default_settings.margin_left,
                            step=1,
                            label="左",
                        )
                        margin_right = gr.Slider(
                            1,
                            50,
                            value=default_settings.margin_right,
                            step=1,
                            label="右",
                        )

                        gr.HTML(
                            '<div class="pp-section-title pp-section-title--rule">ML モデル</div>'
                        )
                        profile_select = gr.Dropdown(
                            choices=[pid for pid, _ in profile_options],
                            value=default_profile,
                            label="人物プロファイル",
                            visible=bool(profile_options),
                            interactive=bool(profile_options),
                        )
                        skip_non_japanese = gr.Checkbox(
                            value=False,
                            label="日本語文字だけプロット（英数字・数式・記号をスキップ）",
                        )
                        plot_page_numbers = gr.Checkbox(
                            value=default_settings.plot_page_numbers,
                            label="ページ番号をプロット",
                        )
                        temperature = gr.Slider(
                            0.1,
                            2.0,
                            value=default_settings.temperature,
                            step=0.1,
                            label="温度",
                            info="高いほど文字の揺らぎが大きくなります",
                        )
                        messiness = gr.Slider(
                            0.0,
                            2.0,
                            value=default_settings.messiness,
                            step=0.1,
                            label="汚さ",
                            info="行内の上下動・字間・サイズ・傾きのばらつき。0=整った字、2=大きく乱れる",
                        )
                        with gr.Accordion("人らしさ調整", open=False):
                            pressure_variation = gr.Slider(
                                0.0,
                                1.0,
                                value=default_settings.pressure_variation,
                                step=0.05,
                                label="筆圧変化",
                                info="画の中の濃淡（プレビュー演出用）。実機は描画中Zが振れて点線化するため0推奨",
                            )
                            instance_variation = gr.Slider(
                                0.0,
                                1.0,
                                value=default_settings.instance_variation,
                                step=0.05,
                                label="字のばらつき",
                                info="同じ字を毎回少し変える。0=毎回同じ形、大=書くたびに違う",
                            )
                            entry_taper = gr.Slider(
                                0.0,
                                1.0,
                                value=default_settings.entry_taper,
                                step=0.05,
                                label="入筆",
                                info="始筆を軽く入れて立ち上げる筆の入り。実機は始筆がかすれ得るため0推奨",
                            )
                            connection_strength = gr.Slider(
                                0.0,
                                1.0,
                                value=default_settings.connection_strength,
                                step=0.05,
                                label="連綿（続け字）",
                                info="同じ字の近い画を薄い線で続ける。近いほど高確率＋乱数。Z一定で点線化しない",
                            )

                        reset_btn = gr.Button(
                            "デフォルトに戻す",
                            variant="secondary",
                            size="sm",
                            elem_classes=["pp-ghost"],
                        )
                        validation_md = gr.HTML(value="", visible=False)

                with gr.Accordion("ヘルプ", open=False):
                    gr.Markdown(_HELP_MARKDOWN)

            with gr.Tab("プロッタ送信"):
                with gr.Row(equal_height=False):
                    # ===== 左カラム: 操作系（接続 / 送信データ選択 / ジョブ送信）=====
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["pp-fieldset"]):
                            gr.HTML('<div class="pp-legend">接続</div>')
                            gr.HTML(
                                value=_WEBSERIAL_STATUS_HTML,
                                elem_id="webserial-status",
                            )
                            with gr.Row():
                                webserial_connect_btn = gr.Button(
                                    "接続", variant="primary", elem_id="webserial-connect-btn"
                                )
                                webserial_disconnect_btn = gr.Button(
                                    "切断",
                                    variant="secondary",
                                    elem_id="webserial-disconnect-btn",
                                )

                        with gr.Group(elem_classes=["pp-fieldset"]):
                            gr.HTML('<div class="pp-legend">送信データ選択</div>')
                            gr.HTML(
                                '<div class="pp-fieldset-note">'
                                "「生成」タブで作った G-code は「生成済み G-code」で選べます。"
                                "</div>"
                            )
                            webserial_source = gr.Radio(
                                choices=[
                                    ("生成済み G-code", "generated"),
                                    ("アップロード G-code", "uploaded"),
                                ],
                                value="generated",
                                label="送信元",
                            )
                            webserial_pages = gr.Dropdown(
                                choices=[],
                                value=[],
                                multiselect=True,
                                label="送信ページ",
                                info="送信するページを選択（既定: 全ページ）。1ページごとに用紙交換で停止します。",
                            )
                            # gr.Files を visible=False で初期化すると初回表示時に
                            # loading 表示が固着するため、Group 側の表示を切り替える。
                            with gr.Group(visible=False) as webserial_upload_group:
                                webserial_upload = gr.Files(
                                    label="アップロード G-code (.gcode/.nc/.txt)",
                                    file_types=[".gcode", ".nc", ".txt"],
                                )

                        with gr.Group(elem_classes=["pp-fieldset"]):
                            gr.HTML('<div class="pp-legend">ジョブ送信</div>')
                            with gr.Row():
                                webserial_start_btn = gr.Button(
                                    "送信開始",
                                    variant="primary",
                                    elem_id="webserial-start-btn",
                                )
                                webserial_stop_btn = gr.Button(
                                    "停止", variant="secondary", elem_id="webserial-stop-btn"
                                )
                                webserial_emergency_btn = gr.Button(
                                    "緊急停止",
                                    variant="stop",
                                    elem_id="webserial-emergency-btn",
                                )
                                webserial_resume_btn = gr.Button(
                                    "続行（用紙交換後）",
                                    variant="primary",
                                    elem_id="webserial-resume-btn",
                                )
                            gr.HTML(value=_WEBSERIAL_PROGRESS_HTML)

                    # ===== 右カラム: プレビュー =====
                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=["pp-fieldset"]):
                            gr.HTML('<div class="pp-legend">プレビュー</div>')
                            gr.HTML(
                                value=_WEBSERIAL_PREVIEW_HTML,
                                elem_id="webserial-preview",
                            )

                # ===== 最下部 全幅: ログ（常時表示）=====
                with gr.Group(elem_classes=["pp-fieldset"]):
                    gr.HTML('<div class="pp-legend">ログ</div>')
                    gr.HTML(value=_WEBSERIAL_LOG_HTML, elem_id="webserial-log")

        slider_components = [
            font_size,
            line_spacing,
            margin_top,
            margin_bottom,
            margin_left,
            margin_right,
            temperature,
            messiness,
            pressure_variation,
            instance_variation,
            entry_taper,
            connection_strength,
        ]

        # ===== コールバック =====

        def _update_char_count(text: str, settings: UISettings) -> str:
            n = len(text) if text else 0
            content_w = settings.paper_width - settings.margin_left - settings.margin_right
            content_h = settings.paper_height - settings.margin_top - settings.margin_bottom
            chars_per_line = max(int(content_w / settings.font_size), 1)
            lines_per_page = max(int(content_h / settings.line_spacing), 1)
            chars_per_page = max(chars_per_line * lines_per_page, 1)
            est_pages = max(1, -(-n // chars_per_page))
            return f"文字数: {n} | 推定ページ数: {est_pages}"

        text_input.change(
            _update_char_count,
            inputs=[text_input, settings_state],
            outputs=[char_count_md],
        )

        # 各 Slider の change で UISettings を不変更新し、stale=True にする。
        # validate() のエラーをボタン disable とエラー HTML 表示に紐付ける。
        def _make_setting_updater(field: str):
            def _update(value, settings: UISettings):
                # frozen=True なので dataclasses.replace で新インスタンスを作る
                new_settings = replace(settings, **{field: float(value)})
                errors = new_settings.validate()
                has_err = bool(errors)
                return (
                    new_settings,
                    True,  # stale
                    gr.update(value=_STALE_BANNER_HTML, visible=True),
                    gr.update(value=_validation_status(errors), visible=has_err),
                    gr.update(interactive=not has_err),
                    gr.update(interactive=not has_err),
                    new_settings.to_dict(),  # ブラウザ永続化
                )

            return _update

        slider_field_map = {
            font_size: "font_size",
            line_spacing: "line_spacing",
            margin_top: "margin_top",
            margin_bottom: "margin_bottom",
            margin_left: "margin_left",
            margin_right: "margin_right",
            temperature: "temperature",
            messiness: "messiness",
            pressure_variation: "pressure_variation",
            instance_variation: "instance_variation",
            entry_taper: "entry_taper",
            connection_strength: "connection_strength",
        }
        for slider, field in slider_field_map.items():
            slider.change(
                _make_setting_updater(field),
                inputs=[slider, settings_state],
                outputs=[
                    settings_state,
                    preview_stale,
                    stale_banner,
                    validation_md,
                    preview_btn,
                    gcode_btn,
                    persisted_settings,
                ],
            )

        def _on_profile_change(profile_id: str | None):
            return (
                True,
                gr.update(value=_STALE_BANNER_HTML, visible=True),
                profile_id,  # ブラウザ永続化
            )

        profile_select.change(
            _on_profile_change,
            inputs=[profile_select],
            outputs=[preview_stale, stale_banner, persisted_profile],
        )

        skip_non_japanese.change(
            lambda _value: (True, gr.update(value=_STALE_BANNER_HTML, visible=True)),
            inputs=[skip_non_japanese],
            outputs=[preview_stale, stale_banner],
        )

        def _on_plot_page_numbers_change(value: bool, settings: UISettings):
            new_settings = replace(settings, plot_page_numbers=bool(value))
            errors = new_settings.validate()
            has_err = bool(errors)
            return (
                new_settings,
                True,
                gr.update(value=_STALE_BANNER_HTML, visible=True),
                gr.update(value=_validation_status(errors), visible=has_err),
                gr.update(interactive=not has_err),
                gr.update(interactive=not has_err),
                new_settings.to_dict(),
            )

        plot_page_numbers.change(
            _on_plot_page_numbers_change,
            inputs=[plot_page_numbers, settings_state],
            outputs=[
                settings_state,
                preview_stale,
                stale_banner,
                validation_md,
                preview_btn,
                gcode_btn,
                persisted_settings,
            ],
        )

        def _on_preview(
            text: str,
            settings: UISettings,
            profile_id: str | None,
            skip_non_japanese: bool,
            old_paths: list[str],
            progress=gr.Progress(),
        ):
            # 旧プレビュー画像は確実に消す（リセット時にも再利用される）
            _cleanup_paths(old_paths)

            errors = settings.validate()
            if errors:
                return (
                    [],
                    [],
                    gr.update(value=_validation_status(errors), visible=True),
                    "",
                    False,
                    gr.update(visible=False),
                )

            if not text or not text.strip():
                return (
                    [],
                    [],
                    gr.update(value="**テキストを入力してください。**", visible=True),
                    "",
                    False,
                    gr.update(visible=False),
                )

            def _progress_cb(frac: float, desc: str) -> None:
                # web_app 側で page_base + frac*span 等の累積計算を行うため、
                # ここでは念のため [0,1] にクランプして UI 表示を安定させる
                clamped = max(0.0, min(1.0, float(frac)))
                progress(clamped, desc=desc)

            generated: list[Path] = []
            try:
                pipeline = _build(settings, profile_id, skip_non_japanese)

                start = time.time()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
                    base_path = Path(fp.name)
                # NamedTemporaryFile の context で作られる空ファイルは
                # generate_preview が上書きするので問題なし。
                generated = pipeline.generate_preview(
                    text, save_path=base_path, progress_callback=_progress_cb
                )
                elapsed = time.time() - start

                str_paths = [str(p) for p in generated]
                status = f"**{len(generated)}ページ生成しました** ({elapsed:.1f}秒)"
                coverage_text = _format_coverage(pipeline._last_coverage)
                return (
                    str_paths,
                    str_paths,
                    gr.update(value=status, visible=True),
                    coverage_text,
                    False,  # stale クリア
                    gr.update(visible=False),  # stale バナー隠す
                )
            except Exception as exc:
                logger.exception("Preview failed")
                # 例外時は中途半端に作られた画像も全削除（リーク防止）
                _cleanup_paths([str(p) for p in generated])
                return (
                    [],
                    [],
                    gr.update(value=f"**エラー:** {exc}", visible=True),
                    "",
                    True,  # 失敗時は stale 維持（再試行を促す）
                    gr.update(value=_STALE_BANNER_HTML, visible=True),
                )

        preview_btn.click(
            _on_preview,
            inputs=[
                text_input,
                settings_state,
                profile_select,
                skip_non_japanese,
                prev_preview_paths,
            ],
            outputs=[
                preview_gallery,
                prev_preview_paths,
                status_md,
                coverage_md,
                preview_stale,
                stale_banner,
            ],
        )

        def _on_generate(
            text: str,
            settings: UISettings,
            profile_id: str | None,
            skip_non_japanese: bool,
            old_tmpdir: str | None,
            progress=gr.Progress(),
        ):
            # 前回の G-code テンポラリを丸ごと掃除
            if old_tmpdir:
                shutil.rmtree(old_tmpdir, ignore_errors=True)

            errors = settings.validate()
            if errors:
                return (
                    None,
                    None,
                    gr.update(value=_validation_status(errors), visible=True),
                    gr.update(),
                )
            if not text or not text.strip():
                return (
                    None,
                    None,
                    gr.update(value="**テキストを入力してください。**", visible=True),
                    gr.update(),
                )

            def _progress_cb(frac: float, desc: str) -> None:
                clamped = max(0.0, min(1.0, float(frac)))
                progress(clamped, desc=desc)

            tmp_dir: Path | None = None
            try:
                pipeline = _build(settings, profile_id, skip_non_japanese)
                tmp_dir = Path(tempfile.mkdtemp(prefix="penplotter_"))
                base_path = tmp_dir / "output.gcode"

                start = time.time()
                paths = pipeline.generate_gcode_file(
                    text, save_path=base_path, progress_callback=_progress_cb
                )
                elapsed = time.time() - start

                str_paths = [str(p) for p in paths]
                status = (
                    f"**G-code を {len(paths)} ページ生成しました** ({elapsed:.1f}秒) "
                    "— 自動ダウンロードを開始します"
                )
                page_choices = list(range(1, len(paths) + 1))
                return (
                    str_paths,
                    str(tmp_dir),
                    gr.update(value=status, visible=True),
                    gr.update(choices=page_choices, value=page_choices),
                )
            except Exception as exc:
                logger.exception("G-code generation failed")
                if tmp_dir is not None:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                return (
                    None,
                    None,
                    gr.update(value=f"**エラー:** {exc}", visible=True),
                    gr.update(),
                )

        gcode_btn.click(
            _on_generate,
            inputs=[
                text_input,
                settings_state,
                profile_select,
                skip_non_japanese,
                prev_gcode_tmpdir,
            ],
            outputs=[gcode_files, prev_gcode_tmpdir, status_md, webserial_pages],
        ).then(
            fn=None,
            inputs=[gcode_files],
            outputs=None,
            js=_TRIGGER_MULTI_DOWNLOAD_JS,
        ).then(
            # gcode_files / webserial_pages 更新後に送信対象プレビューを描く。
            fn=None,
            inputs=[webserial_source, gcode_files, webserial_upload, webserial_pages],
            outputs=None,
            js=(
                "(source, generatedFiles, uploadedFiles, pages) => "
                "window.penPlotterWebSerial.preview(source, generatedFiles, uploadedFiles, pages)"
            ),
        )

        def _on_webserial_source_change(source: str):
            # アップロード選択時はファイル欄、生成済み選択時はページ選択を表示する
            is_uploaded = source == "uploaded"
            return (
                gr.update(visible=is_uploaded),  # webserial_upload_group
                gr.update(visible=not is_uploaded),  # webserial_pages
            )

        # 対象確認用プレビュー: start() と同じ引数順で先頭ページを薄灰描画する。
        _webserial_preview_inputs = [
            webserial_source,
            gcode_files,
            webserial_upload,
            webserial_pages,
        ]
        _webserial_preview_js = (
            "(source, generatedFiles, uploadedFiles, pages) => "
            "window.penPlotterWebSerial.preview(source, generatedFiles, uploadedFiles, pages)"
        )

        webserial_source.change(
            _on_webserial_source_change,
            inputs=[webserial_source],
            outputs=[webserial_upload_group, webserial_pages],
        ).then(
            fn=None,
            inputs=_webserial_preview_inputs,
            outputs=None,
            js=_webserial_preview_js,
        )

        webserial_pages.change(
            fn=None,
            inputs=_webserial_preview_inputs,
            outputs=None,
            js=_webserial_preview_js,
        )

        webserial_upload.change(
            fn=None,
            inputs=_webserial_preview_inputs,
            outputs=None,
            js=_webserial_preview_js,
        )

        webserial_connect_btn.click(
            fn=None,
            outputs=None,
            js="() => window.penPlotterWebSerial.connect()",
        )
        webserial_disconnect_btn.click(
            fn=None,
            outputs=None,
            js="() => window.penPlotterWebSerial.disconnect()",
        )
        webserial_start_btn.click(
            fn=None,
            inputs=[webserial_source, gcode_files, webserial_upload, webserial_pages],
            outputs=None,
            js="(source, generatedFiles, uploadedFiles, pages) => window.penPlotterWebSerial.start(source, generatedFiles, uploadedFiles, pages)",
        )
        webserial_resume_btn.click(
            fn=None,
            outputs=None,
            js="() => window.penPlotterWebSerial.resume()",
        )
        webserial_stop_btn.click(
            fn=None,
            outputs=None,
            js="() => window.penPlotterWebSerial.stop()",
        )
        webserial_emergency_btn.click(
            fn=None,
            outputs=None,
            js="() => window.penPlotterWebSerial.emergencyStop()",
        )

        def _on_clear(old_paths: list[str], old_tmpdir: str | None):
            _cleanup_paths(old_paths)
            if old_tmpdir:
                shutil.rmtree(old_tmpdir, ignore_errors=True)
            return (
                "",  # text_input
                [],  # preview_gallery
                None,  # gcode_files
                gr.update(visible=False),  # status_md
                "文字数: 0",  # char_count_md
                "",  # coverage_md
                [],  # prev_preview_paths
                None,  # prev_gcode_tmpdir
                gr.update(choices=[], value=[]),  # webserial_pages
            )

        clear_btn.click(
            _on_clear,
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
                webserial_pages,
            ],
        )

        def _on_reset():
            d = UISettings.default()
            return (
                d,  # settings_state
                False,  # stale
                gr.update(value="", visible=False),  # stale_banner
                gr.update(value="", visible=False),  # validation_md
                gr.update(interactive=True),  # preview_btn
                gr.update(interactive=True),  # gcode_btn
                d.font_size,
                d.line_spacing,
                d.margin_top,
                d.margin_bottom,
                d.margin_left,
                d.margin_right,
                d.temperature,
                d.messiness,
                d.pressure_variation,
                d.instance_variation,
                d.entry_taper,
                d.connection_strength,
                False,
                d.plot_page_numbers,
                d.to_dict(),  # ブラウザ永続化もデフォルトへ
            )

        reset_btn.click(
            _on_reset,
            outputs=[
                settings_state,
                preview_stale,
                stale_banner,
                validation_md,
                preview_btn,
                gcode_btn,
                *slider_components,
                skip_non_japanese,
                plot_page_numbers,
                persisted_settings,
            ],
        )

        ex_report_btn.click(lambda: _EXAMPLE_REPORT_HEADER, outputs=[text_input])
        ex_math_btn.click(lambda: _EXAMPLE_MATH_REPORT, outputs=[text_input])
        ex_essay_btn.click(lambda: _EXAMPLE_ESSAY, outputs=[text_input])
        ex_table_btn.click(lambda: _EXAMPLE_TABLE, outputs=[text_input])

        # ページロード時にブラウザ localStorage から設定を復元する。
        # 永続値が無ければ UISettings.from_dict(None) が default() を返すため、
        # 初回訪問時は通常のデフォルト表示になる。
        profile_ids = [pid for pid, _ in profile_options]

        def _on_load(stored_settings: dict | None, stored_profile: str | None):
            settings = UISettings.from_dict(stored_settings)
            errors = settings.validate()
            has_err = bool(errors)
            prof = _resolve_restored_profile(stored_profile, profile_ids, default_profile)
            return (
                settings,
                settings.font_size,
                settings.line_spacing,
                settings.margin_top,
                settings.margin_bottom,
                settings.margin_left,
                settings.margin_right,
                settings.temperature,
                settings.messiness,
                settings.pressure_variation,
                settings.instance_variation,
                settings.entry_taper,
                settings.connection_strength,
                settings.plot_page_numbers,
                gr.update(value=prof) if profile_options else gr.update(),
                gr.update(value=_validation_status(errors), visible=has_err),
                gr.update(interactive=not has_err),
                gr.update(interactive=not has_err),
            )

        app.load(
            _on_load,
            inputs=[persisted_settings, persisted_profile],
            outputs=[
                settings_state,
                *slider_components,
                plot_page_numbers,
                profile_select,
                validation_md,
                preview_btn,
                gcode_btn,
            ],
        )

    return app
