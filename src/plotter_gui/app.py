"""xDraw A4 GUI のメインウィンドウ。

Step 6: 送信フローを実装。``UiState`` で接続中・送信中の状態を保持し、
``dispatch_event`` がそれを破壊的に更新する。MainWindow は薄いラッパーで、
送信開始/緊急停止/ファイル選択のロジックは ``request_stream`` /
``handle_file_selected`` としてモジュールトップに切り出してテスト可能にする。

レイアウト (grid 上):
    +---------------------+----------------------+
    | row=0 PortPanel     | row=0..2             |
    +---------------------+ FilePickerWidget     |
    | row=1 ControlPanel  | (右カラムを縦結合)     |
    +---------------------+                      |
    | row=2 JobPanel      |                      |
    +---------------------+----------------------+
    | row=3 LogView (col=0..1 結合で横長)         |
    +-----------------------------------------------+

Threading:
    - GUI 操作は全て Tk メインスレッドから実行する (Tk オブジェクトは
      他スレッドから触ると不定な挙動になる)。
    - Worker thread が put したイベントを ``_poll_events`` で吸い上げて
      ``dispatch_event`` で widget に反映する。これが Tk 呼び出しの
      唯一のスレッド境界。
"""

from __future__ import annotations

import queue
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any

from src.plotter_gui.events import (
    Connected,
    Disconnected,
    JobFinished,
    JobStarted,
    LogEvent,
    Progress,
)
from src.plotter_gui.widgets.control_panel import ControlPanelWidget
from src.plotter_gui.widgets.file_picker import FilePickerWidget
from src.plotter_gui.widgets.job_panel import JobPanelWidget
from src.plotter_gui.widgets.log_view import LogViewWidget
from src.plotter_gui.widgets.port_panel import PortPanelWidget
from src.plotter_gui.worker import PlotterWorker

# event_queue ポーリング間隔 (ミリ秒)。短すぎると CPU を食い、長すぎると
# 進捗反映が体感的にカクつく。50ms = 20Hz は GUI 更新として十分な滑らかさ。
_POLL_INTERVAL_MS = 50


@dataclass
class UiState:
    """MainWindow の永続 UI 状態。``dispatch_event`` が破壊的に更新する。

    is_connected と is_running の組合せで「機械操作可能か」を決める:
    is_connected = True かつ is_running = False のときのみ control_panel の
    ホーミング/ペンテストボタンを有効化する (送信中の機械操作を物理的に防ぐ)。
    """

    is_connected: bool = False
    is_running: bool = False


def dispatch_event(
    event: Any,
    *,
    state: UiState,
    port_panel: Any,
    control_panel: Any,
    job_panel: Any,
    log_view: Any,
) -> None:
    """Worker からのイベントを各 widget の状態更新メソッドに振り分ける。

    Tk 非依存にするため、widget は duck-typed で受け取る (テスト時は MagicMock 可)。
    ``state`` は本関数が破壊的に更新するため、呼び出し側 (MainWindow) は
    同一インスタンスを使い回す前提。
    """
    if isinstance(event, Connected):
        state.is_connected = True
        port_panel.set_connected(True)
        # 送信中なら機械操作を有効化しない (送信ストリーム中の意図せぬ介入を防止)。
        control_panel.set_enabled(not state.is_running)
        log_view.add_log("info", f"Connected: {event.port_name}")
        return

    if isinstance(event, Disconnected):
        state.is_connected = False
        port_panel.set_connected(False)
        control_panel.set_enabled(False)
        log_view.add_log("info", "Disconnected")
        return

    if isinstance(event, LogEvent):
        log_view.add_log(event.level, event.message)
        return

    if isinstance(event, JobStarted):
        # stream のみ UI ロック対象。home/pen_up/pen_down は数秒で完了するため
        # 毎回 UI 切替するとちらつくのでログのみ出す。
        if event.kind == "stream":
            state.is_running = True
            job_panel.reset_progress()
            job_panel.set_running(True)
            control_panel.set_enabled(False)
        log_view.add_log("info", f"[start] {event.kind}")
        return

    if isinstance(event, JobFinished):
        if event.kind == "stream":
            state.is_running = False
            job_panel.set_running(False)
            # 接続が維持されていれば機械操作を再開可能に戻す。送信中に
            # 切断された (cancelled 等) ケースでは disabled のまま保つ。
            control_panel.set_enabled(state.is_connected)
        if event.success:
            log_view.add_log("info", f"[done] {event.kind}")
        else:
            # エラー詳細を error メッセージに連結し、原因追跡可能にする。
            log_view.add_log("error", f"[fail] {event.kind}: {event.error}")
        return

    if isinstance(event, Progress):
        job_panel.update_progress(event.idx, event.total, event.line)
        return

    # 未知イベントは黙って無視。将来の拡張で新イベントを足したときに
    # 強制例外で UI が落ちるのを避ける (ログには出る運用が望ましい)。


def request_stream(
    *,
    state: UiState,
    selected_lines: list[str] | None,
    worker: Any,
    log_view: Any,
) -> None:
    """送信開始ロジック。``MainWindow._on_start`` から呼ばれる。

    Tk 非依存ロジックとして切り出すことで、DISPLAY 無しでも分岐網羅可能。
    条件不足時 (未接続 or ファイル未選択) は warn ログのみ残し、worker は呼ばない。
    """
    if not state.is_connected:
        log_view.add_log("warn", "未接続です。先に接続してください。")
        return
    if selected_lines is None:
        log_view.add_log("warn", "G-code ファイルが選択されていません。")
        return
    worker.submit_stream(selected_lines)


def handle_file_selected(path: Path, *, log_view: Any) -> list[str] | None:
    """ファイル選択 callback のロジック部分。

    UTF-8 で読み込んで splitlines する。読み込み失敗時は None 返却 +
    error ログで UI を落とさず継続。コメント・空行は worker 側 (SerialSender)
    で除外されるため、ここでは行加工しない。
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log_view.add_log("error", f"failed to read file: {exc}")
        return None

    log_view.add_log("info", f"file selected: {path.name} ({len(lines)} lines)")
    return lines


class MainWindow:
    """xDraw A4 G-code 送信用メインウィンドウ。

    Step 6: 送信フロー (start/emergency_stop/file_selected) を完成させる。
    ロジックは ``request_stream`` / ``handle_file_selected`` に切り出し、
    本クラスは Tk widget 配線と event_queue ポーリングに専念する。
    """

    def __init__(self, root: tk.Tk, worker: PlotterWorker | None = None) -> None:
        self._root = root
        # 後方互換のため self.root も残す (既存テスト参照)。
        self.root = root

        # UI 永続状態。dispatch_event が破壊的に更新し、callback が読み出す。
        self._state: UiState = UiState()
        # 選択済み G-code 内容 (送信開始時に worker へ渡す)。
        self._selected_lines: list[str] | None = None
        self._selected_path: Path | None = None

        # Worker は DI 可能。テスト時は MockSerial を仕込んだ Worker を渡す。
        # 未指定時は既定 (実シリアル接続) の Worker を生成。
        self._event_queue: queue.Queue = queue.Queue()
        if worker is None:
            self._worker = PlotterWorker(event_queue=self._event_queue)
        else:
            # 既存の event_queue を共有させる: Worker 側に新規キューを
            # 注入し直すと「テストから注入した worker」と「MainWindow 内部の queue」
            # の参照が乖離してイベントが届かなくなる。
            self._worker = worker
            self._event_queue = worker._event_queue
        self._worker.start()

        root.title("xDraw A4 G-code Sender")
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=0)
        root.rowconfigure(2, weight=0)
        root.rowconfigure(3, weight=1)

        # ---------- 左カラム ----------
        self.port_panel = PortPanelWidget(
            root,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
        )
        self.port_panel.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self.control_panel = ControlPanelWidget(
            root,
            on_home=self._on_home,
            on_pen_up=self._on_pen_up,
            on_pen_down=self._on_pen_down,
        )
        self.control_panel.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        # 接続前は機械操作を無効化 (Connected 受信時に有効化する)。
        self.control_panel.set_enabled(False)

        self.job_panel = JobPanelWidget(
            root,
            on_start=self._on_start,
            on_emergency_stop=self._on_emergency_stop,
        )
        self.job_panel.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        # 起動時は停止状態 → 緊急停止ボタンは無効、送信開始ボタンは有効。
        # (送信開始は条件不足時に request_stream が warn ログを出す。)
        self.job_panel.set_running(False)

        # ---------- 右カラム ----------
        right = ttk.LabelFrame(root, text="プレビュー")
        right.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=4, pady=4)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.file_picker = FilePickerWidget(right, on_file_selected=self._on_file_selected)
        self.file_picker.grid(row=0, column=0, sticky="nsew")

        # ---------- 下段 ----------
        self.log_view = LogViewWidget(root)
        self.log_view.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)

        # ウィンドウ X ボタンで Worker thread を確実に join する。
        # Tk.destroy() を直接呼ぶと daemon thread が中途半端に残る可能性があるため、
        # protocol ハンドラ経由で stop() → destroy() の順を強制する。
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # event_queue ポーリング開始。以降 50ms 周期で再帰スケジュールされる。
        self._poll_events()

    # -----------------------------------------------------------------
    # callback (widget → worker)
    # -----------------------------------------------------------------

    def _on_connect(self, port_name: str) -> None:
        self._worker.submit_connect(port_name)

    def _on_disconnect(self) -> None:
        self._worker.submit_disconnect()

    def _on_home(self) -> None:
        self._worker.submit_home()

    def _on_pen_up(self) -> None:
        self._worker.submit_pen_up()

    def _on_pen_down(self) -> None:
        self._worker.submit_pen_down()

    def _on_start(self) -> None:
        """送信開始ボタンの callback。

        条件チェック (接続済み + ファイル選択済み) は ``request_stream``
        に委譲する。本メソッドは Tk widget の参照を集約するだけ。
        """
        request_stream(
            state=self._state,
            selected_lines=self._selected_lines,
            worker=self._worker,
            log_view=self.log_view,
        )

    def _on_emergency_stop(self) -> None:
        """緊急停止ボタンの callback。

        Worker thread が send_line で待機していても、メインスレッドから
        直接 port.write(b"!") + soft reset を送るため即応する。
        worker 側で port=None ガード済みのため、送信中以外でも安全に呼べる。
        """
        self._worker.emergency_stop()

    def _on_file_selected(self, path: Path) -> None:
        """ファイル選択時の callback。

        プレビュー描画は FilePickerWidget 内で完結している。MainWindow では
        送信用の行リストを読み込んで保持する役割のみ。
        """
        self._selected_path = path
        self._selected_lines = handle_file_selected(path, log_view=self.log_view)

    # -----------------------------------------------------------------
    # ライフサイクル
    # -----------------------------------------------------------------

    def _on_close(self) -> None:
        """ウィンドウクローズ時の片付け。

        Worker thread を停止 → Tk を破棄の順序が重要:
        Tk 破棄が先だと after コールバック中の Tk 呼び出しが
        TclError を起こす可能性がある。
        """
        self._worker.stop()
        try:
            self._root.destroy()
        except tk.TclError:
            # 既に destroy 済みのケースは握りつぶす (二重クローズ対策)。
            pass

    # -----------------------------------------------------------------
    # event_queue ポーリング
    # -----------------------------------------------------------------

    def _poll_events(self) -> None:
        """50ms 周期で event_queue を吸い上げ、各イベントを widget に反映する。

        Tk オブジェクトはメインスレッドからしか触れないため、Worker thread が
        put したイベントを必ずこのメソッドで取り出してから widget を操作する。
        """
        try:
            while True:
                event = self._event_queue.get_nowait()
                dispatch_event(
                    event,
                    state=self._state,
                    port_panel=self.port_panel,
                    control_panel=self.control_panel,
                    job_panel=self.job_panel,
                    log_view=self.log_view,
                )
        except queue.Empty:
            pass

        # 自己再スケジュール。root が destroy 済みだと TclError が出るため
        # try/except でガード (テスト中の即終了経路でも安全)。
        try:
            self._root.after(_POLL_INTERVAL_MS, self._poll_events)
        except tk.TclError:
            pass

    # -----------------------------------------------------------------
    # エントリポイント
    # -----------------------------------------------------------------

    @staticmethod
    def main() -> None:
        root = tk.Tk()
        # 初期サイズ: A4 縦置きプレビュー + 左カラム + ログが入る程度。
        root.geometry("1100x800")
        MainWindow(root)
        root.mainloop()
