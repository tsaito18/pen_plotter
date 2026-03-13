from __future__ import annotations

import re
from dataclasses import dataclass

from src.comm.serial_sender import SerialPort


@dataclass(frozen=True)
class GrblStatus:
    state: str
    mpos: tuple[float, float, float]
    wpos: tuple[float, float, float]

    _PATTERN = re.compile(
        r"<(\w+)"
        r"\|MPos:([-\d.]+),([-\d.]+),([-\d.]+)"
        r"\|WPos:([-\d.]+),([-\d.]+),([-\d.]+)>"
    )

    @classmethod
    def parse(cls, raw: str) -> GrblStatus:
        m = cls._PATTERN.match(raw.strip())
        if not m:
            raise ValueError(f"Cannot parse status: {raw!r}")
        return cls(
            state=m.group(1),
            mpos=(float(m.group(2)), float(m.group(3)), float(m.group(4))),
            wpos=(float(m.group(5)), float(m.group(6)), float(m.group(7))),
        )


class GrblSettings:
    _PATTERN = re.compile(r"^\$(\d+)=(.+)$")

    @classmethod
    def parse(cls, lines: list[str]) -> dict[int, str]:
        result: dict[int, str] = {}
        for line in lines:
            m = cls._PATTERN.match(line.strip())
            if m:
                result[int(m.group(1))] = m.group(2)
        return result


class GrblController:
    def __init__(self, port: SerialPort) -> None:
        self._port = port

    def home(self) -> None:
        self._port.write(b"$H\n")
        self._port.readline()

    def get_status(self) -> GrblStatus:
        self._port.write(b"?\n")
        raw = self._port.readline().decode().strip()
        return GrblStatus.parse(raw)

    def get_settings(self) -> dict[int, str]:
        self._port.write(b"$$\n")
        lines: list[str] = []
        while True:
            raw = self._port.readline().decode().strip()
            if raw == "ok" or raw == "":
                break
            lines.append(raw)
        return GrblSettings.parse(lines)

    def set_setting(self, key: int, value: str) -> None:
        self._port.write(f"${key}={value}\n".encode())
        self._port.readline()

    def reset(self) -> None:
        self._port.write(b"\x18")
