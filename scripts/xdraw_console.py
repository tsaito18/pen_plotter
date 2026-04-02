"""xDraw A4 対話コンソール。コマンドを手入力して応答を確認する。

使い方:
  python xdraw_console.py
  python xdraw_console.py --port COM3
"""

from __future__ import annotations

import argparse
import sys
import time

import serial
from serial.tools.list_ports import comports

XDRAW_VIDS = ["1A86:7523", "1A86:8040"]
BAUDRATE = 115200


def find_xdraw_port() -> str | None:
    for port in comports():
        for vid in XDRAW_VIDS:
            if vid in port.hwid.upper():
                return port.device
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="xDraw A4 interactive console")
    parser.add_argument("--port", default=None)
    args = parser.parse_args()

    port = args.port or find_xdraw_port()
    if not port:
        print("xDraw not found. Available ports:")
        for p in comports():
            print(f"  {p.device}: {p.description} [{p.hwid}]")
        sys.exit(1)

    ser = serial.Serial(port, BAUDRATE, timeout=3)
    time.sleep(2)

    # 起動メッセージを読む
    while ser.in_waiting:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line:
            print(f"[startup] {line}")

    print(f"Connected to {port} @ {BAUDRATE}")
    print("Type G-code commands (e.g. $H, G1G90 Z5 F5000, $$)")
    print("Type 'quit' to exit\n")

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd.lower() in ("quit", "exit", "q"):
            break
        if not cmd:
            continue

        ser.write((cmd + "\r").encode("ascii"))

        # 応答を複数行読む（タイムアウトまで）
        time.sleep(0.1)
        while True:
            resp = ser.readline().decode("ascii", errors="replace").strip()
            if resp:
                print(f"  <- {resp}")
            else:
                break

    ser.close()
    print("Disconnected.")


if __name__ == "__main__":
    main()
