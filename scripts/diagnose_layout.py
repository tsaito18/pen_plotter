"""レイアウト診断 CLI: md/テキストの □（字形なし）と文字かぶりを検出する。

使い方:
    uv run python scripts/diagnose_layout.py <text.md> \
        [--kanjivg-dir data/strokes] [--user-strokes-dir ...] [--checkpoint ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ui.layout_diagnostics import diagnose_layout
from src.ui.web_app import PlotterPipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="レイアウト診断（□・かぶり検出）")
    ap.add_argument("path", type=Path, help="診断するテキスト/Markdown ファイル")
    ap.add_argument("--kanjivg-dir", type=Path, default=Path("data/strokes"))
    ap.add_argument("--user-strokes-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args()

    text = args.path.read_text(encoding="utf-8")
    pipeline = PlotterPipeline(
        checkpoint_path=args.checkpoint,
        kanjivg_dir=args.kanjivg_dir,
        user_strokes_dir=args.user_strokes_dir,
    )
    report = diagnose_layout(pipeline, text)
    print(f"=== レイアウト診断: {args.path.name} ===")
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
