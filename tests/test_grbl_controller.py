import pytest
from src.comm.grbl_controller import GrblController, GrblStatus, GrblSettings


class MockSerial:
    def __init__(self):
        self.written: list[bytes] = []
        self.responses: list[bytes] = []
        self._idx = 0

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def readline(self, timeout: float = 1.0) -> bytes:
        if self._idx < len(self.responses):
            r = self.responses[self._idx]
            self._idx += 1
            return r
        return b""

    def queue(self, *responses: str) -> None:
        for r in responses:
            self.responses.append((r + "\r\n").encode())


class TestGrblStatus:
    def test_parse_idle(self):
        status = GrblStatus.parse("<Idle|MPos:0.000,0.000,0.000|WPos:0.000,0.000,0.000>")
        assert status.state == "Idle"
        assert status.mpos == (0.0, 0.0, 0.0)
        assert status.wpos == (0.0, 0.0, 0.0)

    def test_parse_run(self):
        status = GrblStatus.parse("<Run|MPos:10.500,20.300,0.000|WPos:10.500,20.300,0.000>")
        assert status.state == "Run"
        assert status.mpos[0] == pytest.approx(10.5)
        assert status.mpos[1] == pytest.approx(20.3)

    def test_parse_alarm(self):
        status = GrblStatus.parse("<Alarm|MPos:0.000,0.000,0.000|WPos:0.000,0.000,0.000>")
        assert status.state == "Alarm"


class TestGrblSettings:
    def test_parse_settings(self):
        lines = [
            "$0=10",
            "$1=25",
            "$100=80.000",
            "$110=3000.000",
        ]
        settings = GrblSettings.parse(lines)
        assert settings[0] == "10"
        assert settings[100] == "80.000"
        assert settings[110] == "3000.000"


class TestGrblController:
    def test_home(self):
        mock = MockSerial()
        mock.queue("ok")
        ctrl = GrblController(mock)
        ctrl.home()
        assert b"$H\n" in mock.written

    def test_get_status(self):
        mock = MockSerial()
        mock.queue("<Idle|MPos:1.000,2.000,0.000|WPos:1.000,2.000,0.000>")
        ctrl = GrblController(mock)
        status = ctrl.get_status()
        assert status.state == "Idle"
        assert status.mpos[0] == pytest.approx(1.0)

    def test_get_settings(self):
        mock = MockSerial()
        mock.queue("$0=10", "$1=25", "$100=80.000", "ok")
        ctrl = GrblController(mock)
        settings = ctrl.get_settings()
        assert settings[0] == "10"
        assert settings[100] == "80.000"

    def test_set_setting(self):
        mock = MockSerial()
        mock.queue("ok")
        ctrl = GrblController(mock)
        ctrl.set_setting(100, "80.000")
        assert b"$100=80.000\n" in mock.written

    def test_reset(self):
        mock = MockSerial()
        ctrl = GrblController(mock)
        ctrl.reset()
        assert b"\x18" in mock.written
