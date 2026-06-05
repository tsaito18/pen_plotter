"""PlotterWorker / events 層の単体テスト。

threading.Thread 経由のテストはタイミング依存で脆くなるため、
- `_do_*` 系の private メソッドを直接呼ぶ単体テストを中心
- thread ライフサイクルは start/stop 1 ケースだけ
の方針で構成する。
"""

from __future__ import annotations

import queue
import time
from typing import Any

import pytest
from src.comm.serial_sender import GrblResponse, StreamCancelled
from src.gcode.config import PlotterConfig
from src.plotter_gui.events import (
    Connected,
    Disconnected,
    JobFinished,
    JobStarted,
    LogEvent,
    Progress,
)
from src.plotter_gui.worker import PlotterWorker
from tests.comm_mocks import MockSerial


def _drain(q: queue.Queue) -> list:
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def _make_worker(mock: MockSerial) -> tuple[PlotterWorker, queue.Queue]:
    eq: queue.Queue = queue.Queue()

    def factory(_port_name: str) -> MockSerial:
        return mock

    # boot_wait_sec=0 でテストを 2 秒のスリープから解放
    worker = PlotterWorker(event_queue=eq, serial_factory=factory, boot_wait_sec=0)
    return worker, eq


class TestConnect:
    def test_emit_connected_event_after_connect(self) -> None:
        mock = MockSerial()
        worker, eq = _make_worker(mock)

        worker._do_connect("COM_FAKE")

        events = _drain(eq)
        kinds = [type(e) for e in events]
        assert Connected in kinds
        connected = next(e for e in events if isinstance(e, Connected))
        assert connected.port_name == "COM_FAKE"


class TestHome:
    def test_home_sends_4_line_sequence(self) -> None:
        mock = MockSerial()
        # $H, G4 P1, G92 X0 Y297 Z0, G90 の 4 行に対する応答
        for _ in range(4):
            mock.queue_response("ok")
        worker, _eq = _make_worker(mock)
        worker._do_connect("COM_FAKE")

        worker._do_home()

        # 接続時の起動メッセージ読み捨ては MockSerial に in_waiting が無いのでスキップされる前提
        assert mock.written == [
            b"$H\n",
            b"G4 P1\n",
            b"G92 X0 Y297 Z0\n",
            b"G90\n",
        ]

    def test_home_without_connection_raises(self) -> None:
        mock = MockSerial()
        worker, _eq = _make_worker(mock)
        with pytest.raises(RuntimeError):
            worker._do_home()


class TestPenControl:
    def test_pen_up_uses_config_command(self) -> None:
        mock = MockSerial()
        mock.queue_response("ok")
        worker, _eq = _make_worker(mock)
        worker._do_connect("COM_FAKE")

        worker._do_pen_up()

        expected = (PlotterConfig().pen_up_command + "\n").encode()
        assert mock.written[-1] == expected

    def test_pen_down_uses_config_command(self) -> None:
        mock = MockSerial()
        mock.queue_response("ok")
        worker, _eq = _make_worker(mock)
        worker._do_connect("COM_FAKE")

        worker._do_pen_down()

        expected = (PlotterConfig().pen_down_command + "\n").encode()
        assert mock.written[-1] == expected


class TestStream:
    def test_stream_progress_events(self) -> None:
        mock = MockSerial()
        for _ in range(3):
            mock.queue_response("ok")
        worker, eq = _make_worker(mock)
        worker._do_connect("COM_FAKE")

        worker._do_stream(["G90", "G21", "G0 X10"])

        events = _drain(eq)
        progress_events = [e for e in events if isinstance(e, Progress)]
        assert len(progress_events) == 3
        assert progress_events[0].idx == 1 and progress_events[0].total == 3
        assert progress_events[1].idx == 2 and progress_events[1].total == 3
        assert progress_events[2].idx == 3 and progress_events[2].total == 3
        assert progress_events[0].line == "G90"
        assert progress_events[2].line == "G0 X10"

    def test_stream_cancelled_emits_finished_with_cancelled(self) -> None:
        class CancellingSender:
            def stream(
                self,
                gcode_lines: list[str],
                progress_callback: Any = None,
                cancel_event: Any = None,
            ) -> None:
                if progress_callback is not None:
                    progress_callback(1, len(gcode_lines), gcode_lines[0], GrblResponse.parse("ok"))
                if cancel_event is not None:
                    cancel_event.set()
                raise StreamCancelled("cancelled at line 2/2")

        mock = MockSerial()
        worker, eq = _make_worker(mock)
        worker._do_connect("COM_FAKE")
        worker._sender = CancellingSender()  # type: ignore[assignment]
        worker._do_stream(["G90", "G21"])

        events = _drain(eq)
        finished = [e for e in events if isinstance(e, JobFinished)]
        assert any(
            f.kind == "stream" and not f.success and f.error == "cancelled" for f in finished
        )

    def test_stream_can_start_after_emergency_stop_cancel_flag(self) -> None:
        mock = MockSerial()
        mock.queue_response("ok")
        worker, eq = _make_worker(mock)
        worker._do_connect("COM_FAKE")

        worker.emergency_stop()
        _drain(eq)

        worker._do_stream(["G90"])

        events = _drain(eq)
        assert any(isinstance(e, Progress) and e.idx == 1 and e.line == "G90" for e in events)
        assert any(isinstance(e, JobFinished) and e.kind == "stream" and e.success for e in events)
        assert mock.written[-1] == b"G90\n"

    def test_stream_emits_started_and_finished_on_success(self) -> None:
        mock = MockSerial()
        for _ in range(2):
            mock.queue_response("ok")
        worker, eq = _make_worker(mock)
        worker._do_connect("COM_FAKE")

        worker._do_stream(["G90", "G21"])

        events = _drain(eq)
        started = [e for e in events if isinstance(e, JobStarted) and e.kind == "stream"]
        finished = [e for e in events if isinstance(e, JobFinished) and e.kind == "stream"]
        assert len(started) == 1
        assert len(finished) == 1
        assert finished[0].success is True

    def test_stream_without_connection_raises(self) -> None:
        mock = MockSerial()
        worker, _eq = _make_worker(mock)
        with pytest.raises(RuntimeError):
            worker._do_stream(["G90"])


class TestEmergencyStop:
    def test_emergency_stop_writes_realtime_commands(self) -> None:
        mock = MockSerial()
        worker, eq = _make_worker(mock)
        worker._do_connect("COM_FAKE")

        # 接続時の write は無い (MockSerial に in_waiting なし)。念のため記録長を保持。
        baseline_len = len(mock.written)
        worker.emergency_stop()

        # Feed Hold(!) と Soft Reset(0x18) の 2 段階処理。
        # この順序は GRBL リアルタイムコマンド仕様に従う:
        #  1) "!" で送り中のモーションを即座に保留 (バッファは保持)
        #  2) "\x18" で全リセット (アラーム状態へ遷移)
        # この順を逆にすると、リセット後にバッファ残骸が走る危険がある。
        added = mock.written[baseline_len:]
        assert b"!" in added
        assert b"\x18" in added
        assert added.index(b"!") < added.index(b"\x18")

        events = _drain(eq)
        assert any(isinstance(e, LogEvent) and e.level == "warn" for e in events)

    def test_emergency_stop_without_connection_is_safe(self) -> None:
        mock = MockSerial()
        worker, _eq = _make_worker(mock)
        # 未接続状態で呼んでも例外が出ないこと
        worker.emergency_stop()


class TestThreadLifecycle:
    def test_thread_lifecycle_start_stop(self) -> None:
        mock = MockSerial()
        worker, eq = _make_worker(mock)

        worker.start()
        try:
            worker.submit_connect("COM_FAKE")
            # connect/disconnect は MockSerial 同期処理なので短時間で完了
            time.sleep(0.05)
            worker.submit_disconnect()
            time.sleep(0.05)
        finally:
            worker.stop()

        # 内部 thread が join 済みであること (stop はタイムアウト付きで join)
        assert worker._thread is not None
        assert not worker._thread.is_alive()

        events = _drain(eq)
        assert any(isinstance(e, Connected) for e in events)
        assert any(isinstance(e, Disconnected) for e in events)

    def test_stop_without_start_is_safe(self) -> None:
        mock = MockSerial()
        worker, _eq = _make_worker(mock)
        # start していなくても stop が例外で落ちないこと
        worker.stop()


class TestDisconnect:
    def test_disconnect_closes_port_and_emits_event(self) -> None:
        mock = MockSerial()
        worker, eq = _make_worker(mock)
        worker._do_connect("COM_FAKE")
        _drain(eq)  # Connected を排出

        worker._do_disconnect()

        assert mock.closed is True
        events = _drain(eq)
        assert any(isinstance(e, Disconnected) for e in events)


class TestSubmitFlow:
    """submit_* 経由でも _do_* と同じイベントが出ることを 1 ケースだけ検証。"""

    def test_submit_pen_up_through_thread(self) -> None:
        mock = MockSerial()
        # connect 後 pen_up
        mock.queue_response("ok")
        worker, eq = _make_worker(mock)

        worker.start()
        try:
            worker.submit_connect("COM_FAKE")
            worker.submit_pen_up()
            # 同期完了待ち。タイミング依存だが余裕を持たせる。
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if mock.written:
                    break
                time.sleep(0.02)
        finally:
            worker.stop(timeout=2.0)

        # PlotterConfig.pen_up_command が送信されていること
        assert any(w == (PlotterConfig().pen_up_command + "\n").encode() for w in mock.written)
        # JobStarted("pen_up") と JobFinished("pen_up", success=True) が流れていること
        events = _drain(eq)
        assert any(isinstance(e, JobStarted) and e.kind == "pen_up" for e in events)
        assert any(isinstance(e, JobFinished) and e.kind == "pen_up" and e.success for e in events)
