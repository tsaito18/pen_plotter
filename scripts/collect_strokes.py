"""筆跡の収集画面を起動する（Web UI の ``/collect``。スタジオと同じサーバー）。

旧来のオプション（--output-dir / --person-id / --port / --kanjivg-dir）はそのまま使える。
プロファイルは画面で選ぶ（--person-id は最初に選ばれるプロファイルの作成だけ行う）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_ui import serve
from src.collector.profiles import ensure_profile


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="筆跡の収集（Web UI の /collect）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/user_strokes"),
        help="プロファイルのルート (default: data/user_strokes)",
    )
    parser.add_argument("--person-id", default="taiga", help="作成しておくプロファイル ID")
    parser.add_argument("--port", type=int, default=7860, help="ポート番号 (default: 7860)")
    parser.add_argument("--kanjivg-dir", type=Path, default=Path("data/strokes"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/models/finetuned.pt"))
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--open", action="store_true", help="起動後にブラウザを開く")
    args = parser.parse_args(argv)
    ensure_profile(args.output_dir, args.person_id)
    args.user_strokes_dir = args.output_dir
    args.page = "/collect"
    serve(args)


if __name__ == "__main__":
    main()
