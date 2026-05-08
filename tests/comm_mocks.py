"""通信層テスト用の共通モック。

`tests/test_serial_sender.py` と `tests/test_plotter_worker.py` で
同一仕様の MockSerial を使うため、共有モジュールに切り出している。
"""

from __future__ import annotations


class MockSerial:
    """テスト用モックシリアルポート。

    pyserial の Serial 互換最低限のメソッドのみ提供。
    Worker 経由のテストでは `close()` も呼ばれるため、
    呼び出し回数を `closed` で観測できるようにしている。
    """

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.responses: list[bytes] = []
        self._response_idx = 0
        self.closed = False

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

    def close(self) -> None:
        self.closed = True
