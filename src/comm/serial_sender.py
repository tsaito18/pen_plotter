from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class StreamCancelled(Exception):
    """``stream()`` が ``cancel_event`` によって途中停止したことを示す。"""


@runtime_checkable
class SerialPort(Protocol):
    def write(self, data: bytes) -> int: ...
    def readline(self, timeout: float = 1.0) -> bytes: ...


@dataclass(frozen=True)
class GrblResponse:
    raw: str
    is_ok: bool = False
    is_error: bool = False
    is_alarm: bool = False
    error_code: int | None = None
    alarm_code: int | None = None

    @classmethod
    def parse(cls, line: str) -> GrblResponse:
        stripped = line.strip()

        if stripped == "ok":
            return cls(raw=stripped, is_ok=True)

        if stripped.startswith("error:"):
            code = int(stripped.split(":")[1])
            return cls(raw=stripped, is_error=True, error_code=code)

        if stripped.startswith("ALARM:"):
            code = int(stripped.split(":")[1])
            return cls(raw=stripped, is_alarm=True, alarm_code=code)

        return cls(raw=stripped)


class SerialSender:
    def __init__(self, port: SerialPort) -> None:
        self._port = port

    @staticmethod
    def _clean_line(line: str) -> str:
        """コメント除去・前後空白トリム"""
        if ";" in line:
            line = line[: line.index(";")]
        return line.strip()

    def send_line(self, line: str) -> GrblResponse:
        cleaned = self._clean_line(line)
        if not cleaned:
            return GrblResponse.parse("ok")

        self._port.write((cleaned + "\n").encode())
        raw_response = self._port.readline().decode().strip()
        return GrblResponse.parse(raw_response)

    def stream(
        self,
        gcode_lines: list[str],
        progress_callback: Callable[[int, int, str, GrblResponse], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[GrblResponse]:
        # 有効行を先に確定させて total を固定する。これにより GUI 側の進捗バーが
        # コメント・空行で揺れずに済む（test_progress_callback_total_excludes_empty_lines）。
        effective: list[str] = []
        for line in gcode_lines:
            cleaned = self._clean_line(line)
            if cleaned:
                effective.append(cleaned)

        total = len(effective)
        results: list[GrblResponse] = []

        for i, cleaned in enumerate(effective):
            idx = i + 1  # 1始まり: GUI 表示 (n/total) との直感的整合のため

            # キャンセルは「次行の送信開始前」にチェック。送信途中での中断は
            # GRBL のバッファ状態を壊しうるため、行境界でのみ停止する。
            if cancel_event is not None and cancel_event.is_set():
                raise StreamCancelled(f"cancelled at line {idx}/{total}")

            resp = self.send_line(cleaned)
            results.append(resp)

            # 進捗通知は error/alarm 判定より前に呼ぶ。GUI が「失敗した行」も
            # 進捗ログに含めて表示できるようにするため。
            if progress_callback is not None:
                progress_callback(idx, total, cleaned, resp)

            if resp.is_error:
                raise RuntimeError(f"Grbl error {resp.error_code} on line: {cleaned}")
            if resp.is_alarm:
                raise RuntimeError(f"Grbl alarm {resp.alarm_code} on line: {cleaned}")

        return results
