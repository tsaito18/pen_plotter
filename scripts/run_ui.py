"""Gradio Web UI を起動する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui.gradio_app import APP_CSS, app_head, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Pen Plotter Web UI")
    parser.add_argument("--checkpoint", type=Path, default=None, help="V3 チェックポイント (.pt)")
    parser.add_argument("--kanjivg-dir", type=Path, default=Path("data/strokes"))
    parser.add_argument(
        "--user-strokes-dir",
        type=Path,
        default=Path("data/user_strokes"),
        help="ユーザー筆跡（人物プロファイルのルート）",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="公開リンクを生成")
    args = parser.parse_args()

    checkpoint = args.checkpoint if args.checkpoint and args.checkpoint.exists() else None
    kanjivg_dir = args.kanjivg_dir if args.kanjivg_dir.exists() else None
    user_dir = args.user_strokes_dir if args.user_strokes_dir.exists() else None
    sources = [
        f"ML推論 ({checkpoint})" if checkpoint else "",
        f"ユーザー筆跡 ({user_dir})" if user_dir else "",
        f"KanjiVG ({kanjivg_dir})" if kanjivg_dir else "",
    ]
    print("字形ソース:", ", ".join(s for s in sources if s) or "幾何字形のみ")

    app = create_app(checkpoint_path=checkpoint, kanjivg_dir=kanjivg_dir, user_strokes_dir=user_dir)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
        head=app_head(),
        pwa=True,
    )


if __name__ == "__main__":
    main()
