import pytest
from unittest.mock import MagicMock
from src.comm.serial_sender import SerialSender, SerialPort, GrblResponse


class MockSerial:
    """テスト用モックシリアルポート"""
    def __init__(self):
        self.written: list[bytes] = []
        self.responses: list[bytes] = []
        self._response_idx = 0

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def readline(self, timeout: float = 1.0) -> bytes:
        if self._response_idx < len(self.responses):
            resp = self.responses[self._response_idx]
            self._response_idx += 1
            return resp
        return b""

    def queue_response(self, response: str) -> None:
        self.responses.append((response + "\r\n").encode())


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
        results = sender.stream(["", "; comment only", "G90"])
        assert len(mock.written) == 1  # G90のみ送信
