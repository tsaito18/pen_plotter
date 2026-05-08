"""``python -m src.plotter_gui`` で GUI を起動するためのエントリポイント。"""

from __future__ import annotations

from src.plotter_gui.app import MainWindow

if __name__ == "__main__":
    MainWindow.main()
