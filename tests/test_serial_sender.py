import threading

import pytest
from src.comm.serial_sender import (
    GrblResponse,
    SerialSender,
    StreamCancelled,
)
from tests.comm_mocks import MockSerial


class TestGrblResponse:
    def test_parse_ok(self):
        resp = GrblResponse.parse("ok")
        assert resp.is_ok

    def test_parse_error(self):
        resp = GrblResponse.parse("error:20")
        assert resp.is_error
        assert resp.error_code == 20

    def test_parse_alarm(self):
        resp = GrblResponse.parse("ALARM:1")
        assert resp.is_alarm
        assert resp.alarm_code == 1


class TestSerialSender:
    def test_send_single_line(self):
        mock = MockSerial()
        mock.queue_response("ok")
        sender = SerialSender(mock)
        result = sender.send_line("G0 X10 Y10")
        assert result.is_ok
        assert mock.written[0] == b"G0 X10 Y10\n"

    def test_send_strips_comments(self):
        mock = MockSerial()
        mock.queue_response("ok")
        sender = SerialSender(mock)
        sender.send_line("G0 X10 ; comment")
        assert b";" not in mock.written[0]

    def test_stream_gcode(self):
        mock = MockSerial()
        lines = ["G90", "G21", "G0 X10 Y10", "M2"]
        for _ in lines:
            mock.queue_response("ok")
        sender = SerialSender(mock)
        results = sender.stream(lines)
        assert len(results) == 4
        assert all(r.is_ok for r in results)

    def test_stream_stops_on_error(self):
        mock = MockSerial()
        mock.queue_response("ok")
        mock.queue_response("error:20")
        sender = SerialSender(mock)
        with pytest.raises(RuntimeError):
            sender.stream(["G90", "G0 X999 Y999", "M2"])

    def test_send_empty_lines_skipped(self):
        mock = MockSerial()
        mock.queue_response("ok")
        sender = SerialSender(mock)
        sender.stream(["", "; comment only", "G90"])
        assert len(mock.written) == 1  # G90のみ送信

    def test_progress_callback_invoked_for_each_line(self):
        mock = MockSerial()
        lines = ["G90", "G21", "G0 X10"]
        for _ in lines:
            mock.queue_response("ok")
        sender = SerialSender(mock)

        calls: list[tuple[int, int, str, GrblResponse]] = []

        def cb(idx: int, total: int, line: str, resp: GrblResponse) -> None:
            calls.append((idx, total, line, resp))

        sender.stream(lines, progress_callback=cb)

        assert len(calls) == 3
        # idx は 1 始まり、total は有効行数 (=3)
        assert calls[0][0] == 1 and calls[0][1] == 3
        assert calls[1][0] == 2 and calls[1][1] == 3
        assert calls[2][0] == 3 and calls[2][1] == 3
        assert calls[0][2] == "G90"
        assert calls[1][2] == "G21"
        assert calls[2][2] == "G0 X10"
        assert all(c[3].is_ok for c in calls)

    def test_progress_callback_total_excludes_empty_lines(self):
        mock = MockSerial()
        lines = ["", "; comment", "G90", "G21", "G0 X10"]
        for _ in range(3):
            mock.queue_response("ok")
        sender = SerialSender(mock)

        totals: list[int] = []

        def cb(idx: int, total: int, line: str, resp: GrblResponse) -> None:
            totals.append(total)

        sender.stream(lines, progress_callback=cb)

        # 空行・コメント行を除いた有効行数 (=3) が total になるべき
        assert totals == [3, 3, 3]

    def test_stream_cancelled_raises_before_next_line(self):
        mock = MockSerial()
        mock.queue_response("ok")
        mock.queue_response("ok")  # 2行目以降は届かない想定だが安全側で用意
        sender = SerialSender(mock)

        cancel_event = threading.Event()
        seen: list[int] = []

        # callback 内で cancel をセットし、次の行送信前に StreamCancelled が上がるか検証
        def cb(idx: int, total: int, line: str, resp: GrblResponse) -> None:
            seen.append(idx)
            if idx == 1:
                cancel_event.set()

        with pytest.raises(StreamCancelled):
            sender.stream(
                ["G90", "G21", "G0 X10"],
                progress_callback=cb,
                cancel_event=cancel_event,
            )

        # 1行目は送信済み、2行目以降は送信されないこと
        assert mock.written == [b"G90\n"]
        assert seen == [1]

    def test_stream_no_callback_works_without_args(self):
        # 既存 API（progress_callback / cancel_event を渡さない）の後方互換確認
        mock = MockSerial()
        for _ in range(2):
            mock.queue_response("ok")
        sender = SerialSender(mock)
        results = sender.stream(["G90", "G21"])
        assert len(results) == 2
        assert all(r.is_ok for r in results)
