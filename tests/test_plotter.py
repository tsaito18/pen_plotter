"""実機送信: GRBL シリアル通信・ポート検出・Tkinter 送信 GUI（ロジック部分）。"""

from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.comm import port_finder
from src.comm.serial_sender import GrblResponse, SerialSender, StreamCancelled
from src.gcode.config import PlotterConfig
from tests.comm_mocks import MockSerial

tk = pytest.importorskip("tkinter")

from src.plotter_gui.app import (  # noqa: E402
    UiState,
    dispatch_event,
    handle_file_selected,
    request_stream,
)
from src.plotter_gui.events import (  # noqa: E402
    Connected,
    Disconnected,
    JobFinished,
    JobStarted,
    Progress,
)
from src.plotter_gui.preview import parse_gcode  # noqa: E402
from src.plotter_gui.worker import PlotterWorker  # noqa: E402

# --- GRBL 通信 ---


def test_grbl_response_parsing():
    assert GrblResponse.parse("ok").is_ok
    assert GrblResponse.parse("error:20").error_code == 20
    assert GrblResponse.parse("ALARM:1").alarm_code == 1


def test_stream_skips_comments_and_reports_progress():
    port = MockSerial()
    for _ in range(2):
        port.queue_response("ok")
    progress: list[tuple[int, int, str]] = []
    lines = ["; header", "G90", "", "G0 X1 ; move"]
    SerialSender(port).stream(lines, lambda i, n, line, _r: progress.append((i, n, line)))
    assert port.written == [b"G90\n", b"G0 X1\n"]
    assert progress == [(1, 2, "G90"), (2, 2, "G0 X1")]


def test_stream_stops_on_error_and_on_cancel():
    port = MockSerial()
    port.queue_response("error:9")
    with pytest.raises(RuntimeError, match="error 9"):
        SerialSender(port).stream(["G1 X1", "G1 X2"])
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(StreamCancelled):
        SerialSender(MockSerial()).stream(["G1 X1"], cancel_event=cancel)


def test_xdraw_port_detection(monkeypatch):
    def fake(vid: int | None, device: str) -> MagicMock:
        return MagicMock(vid=vid, pid=0x7523, device=device)

    ports = [fake(None, "COM1"), fake(0x1A86, "COM3"), fake(0x1A86, "COM4")]
    monkeypatch.setattr(port_finder, "comports", lambda: ports)
    assert port_finder.find_xdraw_port() == "COM3"
    assert len(port_finder.list_candidate_ports()) == 3
    monkeypatch.setattr(port_finder, "comports", lambda: [fake(None, "COM1")])
    assert port_finder.find_xdraw_port() is None


# --- Worker ---


def _drain(q: queue.Queue) -> list:
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def _worker(port: MockSerial) -> tuple[PlotterWorker, queue.Queue]:
    events: queue.Queue = queue.Queue()
    return PlotterWorker(events, serial_factory=lambda _name: port, boot_wait_sec=0), events


def test_worker_connect_home_and_pen_commands():
    port = MockSerial()
    for _ in range(6):
        port.queue_response("ok")
    worker, events = _worker(port)
    worker._do_connect("COM9")
    worker._do_home()
    worker._do_pen_up()
    worker._do_pen_down()
    sent = [b.decode().strip() for b in port.written]
    assert sent[:4] == ["$H", "G4 P1", "G92 X0 Y297 Z0", "G90"]
    assert sent[4:] == [PlotterConfig().pen_up_command, PlotterConfig().pen_down_command]
    assert any(isinstance(e, Connected) for e in _drain(events))
    worker._do_disconnect()
    assert port.closed and isinstance(_drain(events)[-1], Disconnected)
    with pytest.raises(RuntimeError, match="not connected"):
        worker._do_home()


def test_worker_stream_emits_progress_and_can_be_cancelled():
    port = MockSerial()
    port.queue_response("ok")
    worker, events = _worker(port)
    worker._do_connect("COM9")
    worker._do_stream(["G1 X1"])
    got = _drain(events)
    assert any(isinstance(e, Progress) for e in got)
    assert JobFinished(kind="stream", success=True) in got

    worker.emergency_stop()  # 送信中でなくても安全、次の送信は新たに始められる
    assert port.written[-2:] == [b"!", b"\x18"]
    port.queue_response("ok")
    worker._do_stream(["G1 X2"])
    assert JobFinished(kind="stream", success=True) in _drain(events)


def test_worker_thread_processes_submitted_commands():
    port = MockSerial()
    port.queue_response("ok")
    worker, events = _worker(port)
    worker.start()
    worker.submit_connect("COM9")
    worker.submit_pen_up()
    worker.stop()
    assert JobFinished(kind="pen_up", success=True) in _drain(events)


# --- GUI のロジック部分 ---


def _widgets() -> dict[str, MagicMock]:
    return {k: MagicMock() for k in ("port_panel", "control_panel", "job_panel", "log_view")}


def test_ui_locks_machine_controls_while_streaming():
    state, w = UiState(), _widgets()
    dispatch_event(Connected("COM3"), state=state, **w)
    w["control_panel"].set_enabled.assert_called_with(True)
    dispatch_event(JobStarted("stream"), state=state, **w)
    w["control_panel"].set_enabled.assert_called_with(False)
    w["job_panel"].set_running.assert_called_with(True)
    dispatch_event(Progress(3, 10, "G1"), state=state, **w)
    w["job_panel"].update_progress.assert_called_with(3, 10, "G1")
    dispatch_event(JobFinished("stream", success=False, error="x"), state=state, **w)
    w["control_panel"].set_enabled.assert_called_with(True)
    assert w["log_view"].add_log.call_args.args[0] == "error"
    dispatch_event(Disconnected(), state=state, **w)
    assert not state.is_connected


def test_request_stream_requires_connection_and_file(tmp_path: Path):
    worker, log = MagicMock(), MagicMock()
    request_stream(state=UiState(), selected_lines=["G1"], worker=worker, log_view=log)
    request_stream(state=UiState(True), selected_lines=None, worker=worker, log_view=log)
    worker.submit_stream.assert_not_called()
    request_stream(state=UiState(True), selected_lines=["G1"], worker=worker, log_view=log)
    worker.submit_stream.assert_called_once_with(["G1"])

    path = tmp_path / "a.gcode"
    path.write_text("G90\nG1 X1\n")
    assert handle_file_selected(path, log_view=log) == ["G90", "G1 X1"]
    assert handle_file_selected(tmp_path / "missing", log_view=log) is None


def test_parse_gcode_into_pen_down_strokes():
    header = ["; c", "$H", "G92 X0 Y297 Z0", "G1G90 Z0.5 F5000"]
    body = ["G0 X1 Y1", "G1G90 Z5.0 F5000", "G1 X2 Y1", "G1 Y3", "G0 X9 Y9", "G0 X5 Y5"]
    gcode = "\n".join([*header, *body, "G1 X6 Y5 Z3.0"])
    strokes = parse_gcode(gcode)
    assert [s.points for s in strokes] == [[(1.0, 1.0), (2.0, 1.0), (2.0, 3.0)]]


def test_crash_logger_writes_traceback(monkeypatch, tmp_path: Path):
    import scripts.run_plotter_gui as rpg

    monkeypatch.setattr(rpg, "_PROJECT_ROOT", tmp_path)
    original = sys.excepthook
    try:
        rpg._install_crash_logger()
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
        assert "boom" in (tmp_path / "plotter_gui_error.log").read_text(encoding="utf-8")
    finally:
        sys.excepthook = original


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Tk にはディスプレイが必要")
def test_main_window_starts_and_closes():
    from src.plotter_gui.app import MainWindow

    root = tk.Tk()
    try:
        window = MainWindow(root, worker=_worker(MockSerial())[0])
        root.update()
        window._on_close()
    except tk.TclError:
        root.destroy()
