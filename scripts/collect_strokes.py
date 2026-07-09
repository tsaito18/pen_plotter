"""手書きストローク収集サーバーを起動するスクリプト。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector import ipad_sync
from src.collector.ipad_sync import StrokeCollectorApp


def main() -> None:
    parser = argparse.ArgumentParser(description="手書きストローク収集 Web UI")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/user_strokes"),
        help="プロファイルを格納するルートディレクトリ (default: data/user_strokes)",
    )
    parser.add_argument(
        "--person-id",
        type=str,
        default="taiga",
        help="起動時に選択する人物プロファイル ID (default: taiga)",
    )
    parser.add_argument("--port", type=int, default=8080, help="ポート番号 (default: 8080)")
    parser.add_argument(
        "--kanjivg-dir",
        type=Path,
        default=Path("data/strokes"),
        help="KanjiVGストロークデータのディレクトリ (default: data/strokes)",
    )
    parser.add_argument(
        "--chars",
        type=str,
        default=None,
        help=(
            "収集対象文字を直接指定して既定のガイドセット(GUIDED_CHARS)を上書きする。"
            "指定文字のみをガイド対象にする（重複は除去、順序は保持）。"
            "例: --chars 'abcσδΩがず' / 未指定なら既定の常用セット"
        ),
    )
    args = parser.parse_args()

    if args.chars:
        # 既定ガイドセットを差し替える。ipad_sync 内の全参照はモジュール
        # グローバル名 GUIDED_CHARS の実行時ルックアップなので、ここで再代入
        # すれば select_next_char / 進捗集計など全経路に反映される。
        custom = list(dict.fromkeys(args.chars))
        ipad_sync.GUIDED_CHARS = custom
        print(f"ガイド文字を上書き: {len(custom)}字 -> {''.join(custom)}")

    kanjivg_dir = args.kanjivg_dir if args.kanjivg_dir.exists() else None
    app = StrokeCollectorApp(
        output_dir=args.output_dir,
        port=args.port,
        kanjivg_dir=kanjivg_dir,
        person_id=args.person_id,
    )
    print(f"収集サーバー起動: http://localhost:{args.port}/")
    print(f"保存先: {args.output_dir / args.person_id}")
    print("iPad/PCのブラウザでアクセスし、文字を書いて送信してください。")
    print("Ctrl+C で終了")
    app.serve()


if __name__ == "__main__":
    main()
