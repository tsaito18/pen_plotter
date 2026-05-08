"""Tkinter GUI 全体の smoke テスト。

Tkinter は ``Tk()`` 呼び出し時に X サーバ接続を要求するため、
WSL/CI では DISPLAY が無いと _tkinter.TclError を上げる。
``DISPLAY`` 環境変数が無い実行環境では skip する設計とし、
Windows ネイティブ実行時のみ実行されるようにする。

また Tkinter 非依存の import 検証は別テスト関数で行い、
DISPLAY 無しでも import エラーを検出できるようにする。
"""

from __future__ import annotations

import os

import pytest

# DISPLAY が無い場合のガード。tkinter 自体は import 可能なため、
# import チェックは別 (display 不要) のテストで行う。
requires_display = pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="Tkinter requires a display (set DISPLAY env var)",
)


def test_imports_work_without_display() -> None:
    """DISPLAY が無くても import だけは通ること (Tk() を呼ばない経路)。

    各 widget モジュールが import 時に Tk() を呼んでいると WSL で
    落ちるため、import 副作用が無いことを保証する。
    """
    from src.plotter_gui.app import MainWindow  # noqa: F401
    from src.plotter_gui.preview import parse_gcode, render_strokes  # noqa: F401
    from src.plotter_gui.widgets.control_panel import ControlPanelWidget  # noqa: F401
    from src.plotter_gui.widgets.file_picker import FilePickerWidget  # noqa: F401
    from src.plotter_gui.widgets.job_panel import JobPanelWidget  # noqa: F401
    from src.plotter_gui.widgets.log_view import LogViewWidget  # noqa: F401
    from src.plotter_gui.widgets.port_panel import PortPanelWidget  # noqa: F401


@requires_display
def test_main_window_instantiates() -> None:
    """Tk() を立ち上げて MainWindow を構築 → 即破棄。

    レイアウト矛盾や widget 構築時の例外が無いことを保証する。
    Step 5 で Worker と接続する前段の最小確認。
    """
    import tkinter as tk

    from src.plotter_gui.app import MainWindow

    root = tk.Tk()
    root.withdraw()  # 画面に出さない (CI 上で headless 実行用の保険)
    try:
        MainWindow(root)
    finally:
        root.destroy()


@requires_display
def test_widgets_instantiate_with_none_callbacks() -> None:
    """各 widget が callback=None でも構築できること。

    Step 5 で Worker と接続する前は callback を未配線で配置するため、
    None 許容 (Optional[Callable]) のシグネチャを要求する。
    """
    import tkinter as tk

    from src.plotter_gui.widgets.control_panel import ControlPanelWidget
    from src.plotter_gui.widgets.file_picker import FilePickerWidget
    from src.plotter_gui.widgets.job_panel import JobPanelWidget
    from src.plotter_gui.widgets.log_view import LogViewWidget
    from src.plotter_gui.widgets.port_panel import PortPanelWidget

    root = tk.Tk()
    root.withdraw()
    try:
        FilePickerWidget(root)
        PortPanelWidget(root)
        ControlPanelWidget(root)
        JobPanelWidget(root)
        LogViewWidget(root)
    finally:
        root.destroy()


@requires_display
def test_main_window_starts_and_stops_worker() -> None:
    """MainWindow を起動して即座に閉じても Worker thread が join されること。

    Step 5 で MainWindow が PlotterWorker を保持する設計に変わるため、
    閉じ忘れによる thread leak が起きないことを保証する。
    """
    import queue
    import tkinter as tk

    from src.plotter_gui.app import MainWindow
    from src.plotter_gui.worker import PlotterWorker

    # 実シリアル接続が走らないよう、外部から PlotterWorker を注入する。
    # boot_wait_sec=0 で connect 時の sleep もスキップ。
    event_queue: queue.Queue = queue.Queue()
    worker = PlotterWorker(event_queue=event_queue, boot_wait_sec=0.0)

    root = tk.Tk()
    root.withdraw()
    try:
        window = MainWindow(root, worker=worker)
        # _on_close は worker.stop() + root.destroy() を行う。
        window._on_close()
        # Worker thread が join 済みであること。
        assert worker._thread is None or not worker._thread.is_alive()
    except Exception:
        # 異常時のクリーンアップ (root が破棄されていても exception 安全)
        try:
            root.destroy()
        except Exception:
            pass
        worker.stop()
        raise
