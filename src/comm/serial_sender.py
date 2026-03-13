from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


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

    def stream(self, gcode_lines: list[str]) -> list[GrblResponse]:
        results: list[GrblResponse] = []
        for line in gcode_lines:
            cleaned = self._clean_line(line)
            if not cleaned:
                continue

            resp = self.send_line(cleaned)
            results.append(resp)

            if resp.is_error:
                raise RuntimeError(
                    f"Grbl error {resp.error_code} on line: {cleaned}"
                )
            if resp.is_alarm:
                raise RuntimeError(
                    f"Grbl alarm {resp.alarm_code} on line: {cleaned}"
                )

        return results
