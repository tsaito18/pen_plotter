"""xDraw A4 G-code 送信 GUI のエントリポイント。

Windows でデスクトップショートカットや「python scripts/run_plotter_gui.py」
として起動するための薄いラッパー。pythonpath にプロジェクトルートを追加して
src.plotter_gui.app.MainWindow.main() を呼ぶだけ。
"""

from __future__ import annotations

import datetime
import sys
import traceback
from pathlib import Path

# scripts/ から実行されたとき、プロジェクトルートを sys.path に追加して
# `from src.plotter_gui...` を解決可能にする。これがないと
# ショートカット起動時に ModuleNotFoundError が出る。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _install_crash_logger() -> None:
    """未捕捉例外を実行ファイル横の error.log に書き出すフックを仕込む。

    --windowed ビルド時は stderr が無効化されてエラーが見えないため、
    配布先でもクラッシュ原因を追えるよう、exe 横にテキスト形式で残す。
    """
    if getattr(sys, "frozen", False):
        # PyInstaller bundle 時: exe と同じディレクトリにログを出す。
        log_path = Path(sys.executable).parent / "plotter_gui_error.log"
    else:
        log_path = _PROJECT_ROOT / "plotter_gui_error.log"

    def _hook(exc_type, exc, tb):
        with log_path.open("a", encoding="utf-8") as f:
            ts = datetime.datetime.now().isoformat()
            f.write(f"\n=== {ts} ===\n")
            traceback.print_exception(exc_type, exc, tb, file=f)
        # コンソールが残っていれば標準出力にも出す (console=True ビルド向け)。
        traceback.print_exception(exc_type, exc, tb)

    sys.excepthook = _hook


def main() -> None:
    _install_crash_logger()
    from src.plotter_gui.app import MainWindow

    MainWindow.main()


if __name__ == "__main__":
    main()
