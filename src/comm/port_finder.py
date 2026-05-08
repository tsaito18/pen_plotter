"""xDraw A4 (CH340 USB-シリアル) のポート検出ユーティリティ。

GUI / CLI 双方から再利用するための純粋関数モジュール。
ハードウェア依存ロジックを 1 箇所に集約することで、テスト時は
``comports`` を monkeypatch するだけで挙動検証が可能になる。
"""

from __future__ import annotations

from serial.tools.list_ports import comports
from serial.tools.list_ports_common import ListPortInfo

# CH340 chip ID。xDraw A4 で確認されている VID:PID バリアント。
# 文字列 hwid マッチではなく int 比較にすることで、
# (vid, pid) が None のポートも自然に除外され、誤検出を防げる。
XDRAW_VIDS: set[tuple[int, int]] = {(0x1A86, 0x7523), (0x1A86, 0x8040)}


def find_xdraw_port() -> str | None:
    """xDraw A4 (CH340) のシリアルポートを自動検出する。

    Returns:
        該当ポートのデバイス名（例: ``"COM3"`` / ``"/dev/ttyUSB0"``）。
        該当ポートがない場合は ``None``。複数該当時は最初の 1 つを返す。
    """
    for port in comports():
        if (port.vid, port.pid) in XDRAW_VIDS:
            return port.device
    return None


def list_candidate_ports() -> list[ListPortInfo]:
    """USB シリアルポート全候補を返す。

    GUI で「自動検出に失敗した場合の手動選択肢」を提示するために、
    CH340 以外も含めて全ポートを返す。
    """
    return list(comports())
