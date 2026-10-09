"""Web UI を起動する（スタジオ ``/`` と筆跡 ``/collect`` を 1 つのサーバーで配る）。

プロッタへの送信はブラウザの WebSerial で行うため、プロッタをつないだ PC の
Chrome / Edge で ``http://localhost:<port>`` を開く。筆跡の収集は iPad などから
``http://<この PC の IP>:<port>/collect`` を開く。
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui.server import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pen Plotter Web UI")
    parser.add_argument("--checkpoint", type=Path, default=None, help="V3 チェックポイント (.pt)")
    parser.add_argument("--kanjivg-dir", type=Path, default=Path("data/strokes"))
    parser.add_argument(
        "--user-strokes-dir",
        type=Path,
        default=Path("data/user_strokes"),
        help="ユーザー筆跡（人物プロファイルのルート）",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("data/models"),
        help="モデルの置き場（選択できるモデルと学習の保存先）",
    )
    parser.add_argument("--host", default="0.0.0.0", help="待ち受けアドレス（既定: LAN にも公開）")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="起動後にブラウザを開く")
    parser.add_argument("--page", default="/", help="--open で開くページ（/ または /collect）")
    args = parser.parse_args(argv)
    serve(args)


def serve(args: argparse.Namespace) -> None:
    """解析済みの引数でサーバーを起動する（``collect_strokes.py`` からも使う）。"""

    checkpoint = args.checkpoint if args.checkpoint and args.checkpoint.exists() else None
    kanjivg_dir = args.kanjivg_dir if args.kanjivg_dir.exists() else None
    # 筆跡はここへ保存するので、無ければ作る
    args.user_strokes_dir.mkdir(parents=True, exist_ok=True)
    user_dir = args.user_strokes_dir
    sources = [
        f"ML推論 ({checkpoint})" if checkpoint else "",
        f"ユーザー筆跡 ({user_dir})" if user_dir else "",
        f"KanjiVG ({kanjivg_dir})" if kanjivg_dir else "",
    ]
    print("字形ソース:", ", ".join(s for s in sources if s) or "幾何字形のみ")

    import uvicorn

    app = create_app(
        checkpoint_path=checkpoint,
        kanjivg_dir=kanjivg_dir,
        user_strokes_dir=user_dir,
        models_dir=args.models_dir,
    )
    url = f"http://localhost:{args.port}{args.page}"
    print(f"スタジオ: http://localhost:{args.port}/  （プロッタ送信はこの PC の Chrome / Edge で）")
    print(f"筆跡:     http://<この PC の IP>:{args.port}/collect  （iPad の Safari などで）")
    if args.open:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
