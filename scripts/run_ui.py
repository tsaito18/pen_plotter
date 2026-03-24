"""Gradio Web UIを起動するスクリプト。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui.web_app import PlotterPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Pen Plotter Web UI")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="訓練済みチェックポイント (.pt)"
    )
    parser.add_argument(
        "--kanjivg-dir",
        type=Path,
        default=Path("data/strokes"),
        help="KanjiVGストロークデータのディレクトリ (default: data/strokes)",
    )
    parser.add_argument("--port", type=int, default=7860, help="ポート番号 (default: 7860)")
    parser.add_argument("--share", action="store_true", help="公開リンクを生成")
    args = parser.parse_args()

    kanjivg_dir = args.kanjivg_dir if args.kanjivg_dir.exists() else None
    checkpoint = args.checkpoint if args.checkpoint and args.checkpoint.exists() else None

    pipeline = PlotterPipeline(
        checkpoint_path=checkpoint,
        kanjivg_dir=kanjivg_dir,
    )

    app = pipeline.create_app()
    if app is None:
        print("Error: gradio がインストールされていません。")
        print("  pip install gradio")
        sys.exit(1)

    mode = []
    if checkpoint:
        mode.append(f"ML推論 ({checkpoint})")
    if kanjivg_dir:
        mode.append(f"KanjiVGフォールバック ({kanjivg_dir})")
    if not mode:
        mode.append("矩形フォールバック")
    print(f"描画モード: {', '.join(mode)}")

    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
