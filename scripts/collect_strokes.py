"""手書きストローク収集サーバーを起動するスクリプト。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector.ipad_sync import StrokeCollectorApp


def main() -> None:
    parser = argparse.ArgumentParser(description="手書きストローク収集 Web UI")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/user_strokes"),
        help="保存先ディレクトリ (default: data/user_strokes)",
    )
    parser.add_argument("--port", type=int, default=8080, help="ポート番号 (default: 8080)")
    args = parser.parse_args()

    app = StrokeCollectorApp(output_dir=args.output_dir, port=args.port)
    print(f"収集サーバー起動: http://localhost:{args.port}/")
    print(f"保存先: {args.output_dir}")
    print("iPad/PCのブラウザでアクセスし、文字を書いて送信してください。")
    print("Ctrl+C で終了")
    app.serve()


if __name__ == "__main__":
    main()
