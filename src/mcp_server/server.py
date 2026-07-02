"""Pen Plotter MCP server (stdio, local).

3 ツール + 1 リソース:
- render_preview: md → 各ページ PNG（base64）
- generate_gcode: md → 各ページ G-code テキスト
- get_syntax_help: docs/書式リファレンス.md
"""

from __future__ import annotations

import base64
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.ui.settings import UISettings
from src.ui.web_app import build_pipeline

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_KANJIVG = _REPO_ROOT / "data" / "strokes"
_DEFAULT_USER_STROKES = _REPO_ROOT / "data_examples" / "user_strokes" / "yamataku_v1"
_DEFAULT_CHECKPOINT = _REPO_ROOT / "data_examples" / "models_yamataku" / "finetuned.pt"


def _env_path(key: str, default: Path) -> Path | None:
    val = os.environ.get(key)
    p = Path(val) if val else default
    return p if p.exists() else None


_CHECKPOINT = _env_path("PEN_PLOTTER_CHECKPOINT", _DEFAULT_CHECKPOINT)
_KANJIVG = _env_path("PEN_PLOTTER_KANJIVG_DIR", _DEFAULT_KANJIVG)
_USER_STROKES = _env_path("PEN_PLOTTER_USER_STROKES_DIR", _DEFAULT_USER_STROKES)


def _make_settings(overrides: dict[str, Any] | None) -> UISettings:
    s = UISettings.default()
    if not overrides:
        return s
    allowed = {
        "font_size",
        "line_spacing",
        "margin_top",
        "margin_bottom",
        "margin_left",
        "margin_right",
        "temperature",
        "messiness",
        "pressure_variation",
        "instance_variation",
        "entry_taper",
        "connection_strength",
        "plot_page_numbers",
        "paper_width",
        "paper_height",
    }
    kw = {k: v for k, v in overrides.items() if k in allowed}
    return replace(s, **kw)


mcp = FastMCP("pen-plotter")


@mcp.tool()
def render_preview(
    md_text: str,
    pages: str | None = None,
    settings: dict | None = None,
) -> list[dict]:
    """MD テキストをレンダしてページごとのプレビュー PNG を返す。

    Args:
        md_text: レポート本文（Markdown）。書式は get_syntax_help 参照。
        pages: 返却ページ指定 ("1,3-5")。省略で全ページ。全ページは重い。
        settings: UISettings 上書き辞書（font_size, temperature, messiness,
            pressure_variation, instance_variation, connection_strength,
            entry_taper, plot_page_numbers, margin_*, line_spacing 等）。
    """
    ui = _make_settings(settings)
    pipe = build_pipeline(
        ui,
        checkpoint_path=_CHECKPOINT,
        kanjivg_dir=_KANJIVG,
        user_strokes_dir=_USER_STROKES,
    )
    with tempfile.TemporaryDirectory() as tmp:
        save = Path(tmp) / "preview.png"
        paths = pipe.generate_preview(md_text, save)
        selected = _select_pages(paths, pages)
        return [
            {"page": i, "png_base64": base64.b64encode(p.read_bytes()).decode()}
            for i, p in selected
        ]


@mcp.tool()
def generate_gcode(
    md_text: str,
    pages: str | None = None,
    settings: dict | None = None,
) -> list[dict]:
    """MD テキストから G-code を生成してページごとに返す。

    Args:
        md_text: レポート本文（Markdown）。
        pages: 返却ページ指定 ("1,3-5")。省略で全ページ。
        settings: UISettings 上書き辞書。
    """
    ui = _make_settings(settings)
    pipe = build_pipeline(
        ui,
        checkpoint_path=_CHECKPOINT,
        kanjivg_dir=_KANJIVG,
        user_strokes_dir=_USER_STROKES,
    )
    with tempfile.TemporaryDirectory() as tmp:
        save = Path(tmp) / "out.gcode"
        paths = pipe.generate_gcode_file(md_text, save)
        selected = _select_pages(paths, pages)
        return [{"page": i, "gcode": p.read_text(encoding="utf-8")} for i, p in selected]


_MCP_SYNTAX_HELP = """\
# レポート MD 書式（手書きレンダ用）

## 見出し
`# 大` `## 中` `### 小`（3段階、階層でインデントが深くなる）

## 段落
空行で区切る。自動折り返し（禁則処理あり）。`\\noindent 本文` で字下げなし。

## 改ページ
`---`（ハイフン3個以上のみの行）で強制改ページ。**章ごとに `---` を入れると章別ページになる**。

## 数式
- インライン: `$V = IR$`（文中）
- ブロック: `$$E = mc^2 \\tag{1}$$`（独立行・中央。\\tag で式番号）
- 分数: `\\frac{a}{b}`。**ブロック内の入れ子分数は `\\dfrac`**（縮小防止）
- 平方根: `\\sqrt{2gh}`（入れ子OK）
- 上付き/下付き: `x^2` `p_1` `x^{10}`（複数文字はブレース）
- ギリシャ: `\\alpha \\beta \\Delta \\sigma \\rho \\nu \\lambda \\pi \\mu` または Unicode 直書き（Δ σ ν）
- 演算子: `\\cos \\sin \\log \\sum \\int \\cdot`
- 数式内の `.` は小数点のまま。本文の `.` は句点 `。` に自動変換

## 表
```
| 列1 | 列2 |
|---|---|
| a | b |
: 表1 タイトル
```
- 区切り行 `|---|---|` 必須
- 表直後の `: タイトル` 行がキャプション（省略可）

## 使えない記法
画像 `![]()`、リンク、箇条書き `- `、太字/斜体、コードブロックは非対応（そのまま文字として描かれるか無視）。

## settings（render_preview / generate_gcode の引数）
既定＝レポート提出向け。変える場合の目安:
- きれいめ: `{"messiness": 0.3, "temperature": 0.15, "instance_variation": 0.05}`
- 汚し: `{"messiness": 0.8, "temperature": 0.35, "instance_variation": 0.3}`
"""


@mcp.tool()
def get_syntax_help() -> str:
    """レポート MD の書き方（見出し・数式・表・settings）を返す。md 作成前に必ず読む。"""
    return _MCP_SYNTAX_HELP


def _select_pages(paths: list[Path], spec: str | None) -> list[tuple[int, Path]]:
    numbered = list(enumerate(paths, start=1))
    if not spec:
        return numbered
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.update(range(int(a), int(b) + 1))
        else:
            wanted.add(int(part))
    return [(i, p) for i, p in numbered if i in wanted]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
