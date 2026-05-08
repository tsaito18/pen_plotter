"""xDraw A4 (CH340) ポート検出ロジックのテスト。

WSL 環境では実シリアルデバイスが見えないため、`comports()` を monkeypatch で
差し替えてフェイクポート集合に対する検出挙動のみを検証する。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.comm import port_finder
from src.comm.port_finder import (
    XDRAW_VIDS,
    find_xdraw_port,
    list_candidate_ports,
)


def _fake_port(vid: int | None, pid: int | None, device: str) -> MagicMock:
    """ListPortInfo 互換のフェイクオブジェクトを生成。"""
    port = MagicMock()
    port.vid = vid
    port.pid = pid
    port.device = device
    return port


class TestFindXdrawPort:
    def test_single_match(self, monkeypatch):
        ports = [_fake_port(0x1A86, 0x7523, "COM3")]
        monkeypatch.setattr(port_finder, "comports", lambda: ports)
        assert find_xdraw_port() == "COM3"

    def test_second_vid_match(self, monkeypatch):
        # 8040 バリアント（同じ CH340 系列の別 PID）も検出対象
        ports = [_fake_port(0x1A86, 0x8040, "COM7")]
        monkeypatch.setattr(port_finder, "comports", lambda: ports)
        assert find_xdraw_port() == "COM7"

    def test_no_match(self, monkeypatch):
        # 関係ない VID（FTDI 0x0403 等）と vid/pid 不明ポート（None）
        ports = [
            _fake_port(0x0403, 0x6001, "COM1"),
            _fake_port(None, None, "COM99"),
        ]
        monkeypatch.setattr(port_finder, "comports", lambda: ports)
        assert find_xdraw_port() is None

    def test_multiple_match_returns_first(self, monkeypatch):
        ports = [
            _fake_port(0x1A86, 0x7523, "COM3"),
            _fake_port(0x1A86, 0x8040, "COM5"),
        ]
        monkeypatch.setattr(port_finder, "comports", lambda: ports)
        assert find_xdraw_port() == "COM3"


class TestListCandidatePorts:
    def test_returns_all_ports_including_non_ch340(self, monkeypatch):
        ports = [
            _fake_port(0x1A86, 0x7523, "COM3"),
            _fake_port(0x0403, 0x6001, "COM1"),
            _fake_port(None, None, "COM99"),
        ]
        monkeypatch.setattr(port_finder, "comports", lambda: ports)
        result = list_candidate_ports()
        assert len(result) == 3
        assert [p.device for p in result] == ["COM3", "COM1", "COM99"]


class TestXdrawVids:
    def test_contains_known_ch340_variants(self):
        # CH340 chip ID。複数バリアント(7523/8040)に対応していることを保証
        assert (0x1A86, 0x7523) in XDRAW_VIDS
        assert (0x1A86, 0x8040) in XDRAW_VIDS
