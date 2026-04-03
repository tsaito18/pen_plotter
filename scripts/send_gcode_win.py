"""xDraw A4 へG-codeを送信するスクリプト（Windows側で実行）。

使い方:
  python send_gcode_win.py test.gcode
  python send_gcode_win.py test.gcode --port COM3
  python send_gcode_win.py test.gcode --from-server 192.168.86.100
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
    """xDraw のシリアルポートを自動検出。"""
    for port in comports():
        for vid in XDRAW_VIDS:
            if vid in port.hwid.upper():
                return port.device
    return None


def send_gcode(port_name: str, gcode_path: str, dry_run: bool = False) -> None:
    """G-codeファイルを1行ずつ送信。"""
    with open(gcode_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith(";")]

    print(f"Port: {port_name}")
    print(f"File: {gcode_path} ({len(lines)} commands)")

    if dry_run:
        for line in lines:
            print(f"  [DRY] {line}")
        return

    ser = serial.Serial(port_name, BAUDRATE, timeout=10)
    time.sleep(2)
    # 起動メッセージを読み捨て
    while ser.in_waiting:
        ser.readline()

    print("Sending...")
    for i, line in enumerate(lines):
        ser.write((line + "\r").encode("ascii"))
        resp = ser.readline().decode("ascii", errors="replace").strip()
        status = "ok" if resp.startswith("ok") else f"[{resp}]"
        print(f"  [{i+1}/{len(lines)}] {line} -> {status}")
        if not resp.startswith("ok"):
            print(f"  WARNING: unexpected response: {resp}")

    ser.close()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="xDraw A4 G-code sender")
    parser.add_argument("gcode_file", help="G-codeファイルパス")
    parser.add_argument("--port", default=None, help="シリアルポート (例: COM3)。省略時は自動検出")
    parser.add_argument("--dry-run", action="store_true", help="送信せずにコマンドを表示")
    parser.add_argument(
        "--from-server",
        default=None,
        help="サーバーからscpでファイル取得 (例: 192.168.86.100)",
    )
    args = parser.parse_args()

    # サーバーからファイル取得
    if args.from_server:
        import subprocess

        remote_path = args.gcode_file
        local_path = remote_path.split("/")[-1]
        key_path = Path.home() / ".ssh" / "id_tsaito18"
        key_opt = f"-i {key_path}" if key_path.exists() else ""
        cmd = f"scp {key_opt} taiga@{args.from_server}:{remote_path} {local_path}"
        print(f"Downloading: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        args.gcode_file = local_path

    # ポート検出
    port = args.port
    if port is None:
        port = find_xdraw_port()
        if port is None:
            print("Error: xDraw が見つかりません。--port でCOMポートを指定してください。")
            print("接続されているポート:")
            for p in comports():
                print(f"  {p.device}: {p.description} [{p.hwid}]")
            sys.exit(1)

    send_gcode(port, args.gcode_file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
