"""xDraw A4 G-code 送信 GUI のエントリポイント。

Windows でデスクトップショートカットや「python scripts/run_plotter_gui.py」
として起動するための薄いラッパー。pythonpath にプロジェクトルートを追加して
src.plotter_gui.app.MainWindow.main() を呼ぶだけ。
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ から実行されたとき、プロジェクトルートを sys.path に追加して
# `from src.plotter_gui...` を解決可能にする。これがないと
# ショートカット起動時に ModuleNotFoundError が出る。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    from src.plotter_gui.app import MainWindow

    MainWindow.main()


if __name__ == "__main__":
    main()
