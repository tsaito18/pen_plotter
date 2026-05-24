"""GUI から駆動するプロッタ通信ワーカー。

設計の要点:
- 通信処理は専用 thread 1 本に閉じ込め、`command_queue` 経由で逐次処理する。
  これにより GUI(Tkinter) スレッドは pyserial の I/O でブロックされない。
- ワーカーから GUI への通知は `event_queue` (queue.Queue) に
  frozen dataclass を投入する形に統一し、スレッド間の共有可変状態を持たない。
- `serial_factory` は DI ポイント。テスト時は MockSerial を返す関数を渡す。
- `emergency_stop()` だけはメインスレッドから直接呼ぶ。Worker thread が
  send_line で待機中にキャンセルしたいケースでは、command_queue を通して
  「停止コマンド」を投入しても処理されないため、port への直接書き込みが必要。
  そのため `_port_lock` で書き込み排他を取る。
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from src.comm.grbl_controller import GrblController
from src.comm.serial_sender import SerialPort, SerialSender, StreamCancelled
from src.gcode.config import PlotterConfig
from src.plotter_gui.events import (
    Connected,
    Disconnected,
    JobFinished,
    JobStarted,
    LogEvent,
    Progress,
)

# Worker thread 終了用 sentinel。tuple で比較せず、識別子で判別する。
_SHUTDOWN: object = object()

# xDraw A4 のホーミング後初期化シーケンス。`$H` 単発では紙座標系が
# 設定されないため、左上を (0, 297) に固定する G92 と絶対座標 G90 を続ける。
# (src/gcode/generator.py のヘッダ生成と同一仕様)
_HOMING_SEQUENCE: tuple[str, ...] = (
    "$H",
    "G4 P1",
    "G92 X0 Y297 Z0",
    "G90",
)


def _default_serial_factory(port_name: str) -> SerialPort:
    """既定のシリアルポート生成関数 (pyserial 利用)。

    DI で差し替え可能にしたいテスト時はこの関数を使わず、
    Worker のコンストラクタに MockSerial を返す factory を渡す。
    """
    import serial as pyserial  # 局所 import: テスト時は dispatch されないため未使用

    return pyserial.Serial(port_name, 115200, timeout=10)  # type: ignore[return-value]


class PlotterWorker:
    """GUI からのコマンドを順次実行するワーカー。

    Threading model:
        - 1 producer (GUI thread) → submit_*() で command_queue に投入
        - 1 consumer (worker thread) → command_queue.get() でブロック取り出し
        - emergency_stop() のみ producer thread から直接 port を書く例外経路
    """

    # GRBL 起動メッセージ受信待ち時間 (秒)。実機では 2 秒程度必要だが、
    # テスト時は serial_factory に MockSerial を渡すと同時に boot_wait_sec=0 で
    # 上書きする想定。
    def __init__(
        self,
        event_queue: queue.Queue,
        serial_factory: Callable[[str], SerialPort] | None = None,
        boot_wait_sec: float = 2.0,
    ) -> None:
        self._event_queue = event_queue
        self._serial_factory = serial_factory or _default_serial_factory
        self._boot_wait_sec = boot_wait_sec

        self._command_queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

        # 接続状態 (worker thread から書き込み、emergency_stop からも読む)
        self._port: SerialPort | None = None
        self._sender: SerialSender | None = None
        self._controller: GrblController | None = None

        # emergency_stop ↔ worker thread の port 書き込み排他用ロック。
        # 通常の send_line は worker thread に閉じているため不要だが、
        # emergency_stop はメインスレッドから直接 port.write を呼ぶため
        # 競合可能性がある。
        self._port_lock = threading.Lock()

        # ストリーミング途中停止用
        self._cancel_event = threading.Event()

    # -----------------------------------------------------------------
    # ライフサイクル
    # -----------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="PlotterWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Worker thread に停止 sentinel を投げて join。

        timeout 内に終わらない場合は daemon thread のため放置するが、
        通常はキューが空になり次第 sentinel が処理されて終了する。
        """
        if self._thread is None:
            return
        self._command_queue.put(_SHUTDOWN)
        self._thread.join(timeout=timeout)

    # -----------------------------------------------------------------
    # コマンド投入 (GUI スレッドから呼ぶ)
    # -----------------------------------------------------------------

    def submit_connect(self, port_name: str) -> None:
        self._command_queue.put(("connect", {"port_name": port_name}))

    def submit_disconnect(self) -> None:
        self._command_queue.put(("disconnect", {}))

    def submit_home(self) -> None:
        self._command_queue.put(("home", {}))

    def submit_pen_up(self) -> None:
        self._command_queue.put(("pen_up", {}))

    def submit_pen_down(self) -> None:
        self._command_queue.put(("pen_down", {}))

    def submit_stream(self, gcode_lines: list[str]) -> None:
        self._command_queue.put(("stream", {"gcode_lines": gcode_lines}))

    # -----------------------------------------------------------------
    # 緊急停止 (メインスレッドから直接呼ぶ)
    # -----------------------------------------------------------------

    def emergency_stop(self) -> None:
        """送信中ストリームを即座に止める。

        GRBL リアルタイムコマンドは "ok" 応答を返さず、
        通常の send_line フローでは送れないため、port に直接バイト書き込みする。
        2 段階処理:
          1) cancel_event.set() で次行送信を止める (バッファに溜めない)
          2) "!" (Feed Hold) で進行中モーションを保留
          3) "\\x18" (Soft Reset) で完全リセット → アラーム状態
        順序を守らないと、リセット後に送信途中の行が走り出す危険がある。
        """
        self._cancel_event.set()

        with self._port_lock:
            port = self._port
            if port is None:
                # 未接続時は no-op。GUI からの誤呼び出しでも安全側に倒す。
                self._emit(LogEvent(level="warn", message="emergency stop (no port)"))
                return
            try:
                port.write(b"!")
                port.write(b"\x18")
            except Exception as exc:  # noqa: BLE001 — port 状態によらず必ず通知
                self._emit(LogEvent(level="error", message=f"emergency stop failed: {exc}"))
                return

        self._emit(LogEvent(level="warn", message="emergency stop"))

    # -----------------------------------------------------------------
    # 内部: イベント送出ヘルパ
    # -----------------------------------------------------------------

    def _emit(self, event: Any) -> None:
        self._event_queue.put(event)

    # -----------------------------------------------------------------
    # 内部: thread メインループ
    # -----------------------------------------------------------------

    def _run(self) -> None:
        while True:
            cmd = self._command_queue.get()
            if cmd is _SHUTDOWN:
                return

            kind, kwargs = cmd
            self._dispatch(kind, kwargs)

    def _dispatch(self, kind: str, kwargs: dict) -> None:
        """単一コマンドを実行。Started/Finished と例外処理をここに集約する。

        例外: kind == "stream" は _do_stream 内で Started/Finished を自前で発行する
        (テストで _do_stream を直接呼ぶため)。dispatch 経路では二重発行を避けるべく
        Started/Finished をここでは出さない。
        """
        if kind == "stream":
            # _do_stream 内で例外も含めて完結させる
            try:
                self._do_stream(**kwargs)
            except Exception:  # noqa: BLE001 — _do_stream 内で必ず通知済
                pass
            return

        self._emit(JobStarted(kind=kind))
        try:
            if kind == "connect":
                self._do_connect(**kwargs)
            elif kind == "disconnect":
                self._do_disconnect()
            elif kind == "home":
                self._do_home()
            elif kind == "pen_up":
                self._do_pen_up()
            elif kind == "pen_down":
                self._do_pen_down()
            else:
                raise ValueError(f"unknown command: {kind}")
        except Exception as exc:  # noqa: BLE001 — UI に必ず通知するため握りつぶす
            self._emit(LogEvent(level="error", message=f"{kind} failed: {exc}"))
            self._emit(JobFinished(kind=kind, success=False, error=str(exc)))
            return

        self._emit(JobFinished(kind=kind, success=True))

    # -----------------------------------------------------------------
    # 個別アクション (テストから直接呼ぶ)
    #
    # _do_* は JobStarted/JobFinished を出さない。これらは _dispatch 側の責務。
    # ただし _do_stream は JobStarted/JobFinished を二重発行されないよう、
    # _dispatch を介さない経路でも一貫性を保つよう自己完結させてある。
    # -----------------------------------------------------------------

    def _do_connect(self, port_name: str) -> None:
        port = self._serial_factory(port_name)

        # GRBL 起動メッセージ ("Grbl 1.1f ['$' for help]") は数百ms～2秒程度
        # 遅れて来る。実機で確実に読み捨てるため固定 sleep を入れる。
        # テスト時は boot_wait_sec=0 に上書きする。
        if self._boot_wait_sec > 0:
            time.sleep(self._boot_wait_sec)

        # 起動メッセージ読み捨て: pyserial の Serial にのみ in_waiting がある。
        # MockSerial にはそもそも属性が無いため hasattr で分岐し、
        # テスト時は readline ループに入らない (応答キューを使い切ると無限化するため)。
        if hasattr(port, "in_waiting"):
            try:
                while getattr(port, "in_waiting", 0) > 0:
                    port.readline()
            except Exception:  # noqa: BLE001 — 起動メッセージ読み捨ての失敗は致命傷ではない
                pass

        self._port = port
        self._sender = SerialSender(port)
        self._controller = GrblController(port)
        self._emit(Connected(port_name=port_name))

    def _do_disconnect(self) -> None:
        port = self._port
        if port is not None:
            close_fn = getattr(port, "close", None)
            if callable(close_fn):
                close_fn()
        self._port = None
        self._sender = None
        self._controller = None
        self._emit(Disconnected())

    def _do_home(self) -> None:
        sender = self._require_sender()
        # 4 行を 1 行ずつ ok を待ちながら送る。$H は完了まで数秒かかるが
        # SerialSender.send_line は readline で blocking 待ちするため自然と同期取れる。
        for line in _HOMING_SEQUENCE:
            self._emit(LogEvent(level="sent", message=line))
            resp = sender.send_line(line)
            self._emit(LogEvent(level="recv", message=resp.raw))

    def _do_pen_up(self) -> None:
        sender = self._require_sender()
        cmd = PlotterConfig().pen_up_command
        self._emit(LogEvent(level="sent", message=cmd))
        resp = sender.send_line(cmd)
        self._emit(LogEvent(level="recv", message=resp.raw))

    def _do_pen_down(self) -> None:
        sender = self._require_sender()
        cmd = PlotterConfig().pen_down_command
        self._emit(LogEvent(level="sent", message=cmd))
        resp = sender.send_line(cmd)
        self._emit(LogEvent(level="recv", message=resp.raw))

    def _do_stream(self, gcode_lines: list[str]) -> None:
        """ストリーム送信。Started/Finished も自前で発行する。

        前回の emergency_stop で残った cancel flag は、新規 stream 開始時に
        stale な状態として破棄する。進行中 stream への停止は stream 開始後に
        emergency_stop() が再度 set する。
        """
        try:
            sender = self._require_sender()
        except RuntimeError as exc:
            # 接続前の呼び出しは _dispatch 側に任せたいので例外を再 raise する。
            # Started/Finished は出さない (まだ Started を出していない)。
            raise exc

        self._cancel_event.clear()
        self._emit(JobStarted(kind="stream"))

        def progress_cb(idx: int, total: int, line: str, _resp: Any) -> None:
            self._emit(Progress(idx=idx, total=total, line=line))

        try:
            sender.stream(
                gcode_lines,
                progress_callback=progress_cb,
                cancel_event=self._cancel_event,
            )
        except StreamCancelled as exc:
            # キャンセルは「失敗」ではなく通常終了系。error == "cancelled" で識別。
            self._emit(JobFinished(kind="stream", success=False, error="cancelled"))
            self._emit(LogEvent(level="info", message=f"cancelled: {exc}"))
            return
        except Exception as exc:  # noqa: BLE001 — UI に必ず通知
            self._emit(LogEvent(level="error", message=f"stream failed: {exc}"))
            self._emit(JobFinished(kind="stream", success=False, error=str(exc)))
            return

        self._emit(JobFinished(kind="stream", success=True))

    # -----------------------------------------------------------------
    # 内部: ガード
    # -----------------------------------------------------------------

    def _require_sender(self) -> SerialSender:
        if self._sender is None:
            raise RuntimeError("not connected")
        return self._sender
