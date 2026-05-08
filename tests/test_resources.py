"""src/plotter_gui/_resources.py の単体テスト。

PyInstaller --onefile bundle 実行時 (sys._MEIPASS 設定時) と通常実行時の
両方でリソースパスが正しく解決されることを検証する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.plotter_gui._resources import resource_path


def test_resource_path_in_normal_execution() -> None:
    """通常実行時は repository root 基準で解決される。"""
    p = resource_path("data/report_paper.jpg")
    # _resources.py から見て parents[2] が repo root
    assert p.is_absolute()
    assert p.name == "report_paper.jpg"
    assert p.parents[0].name == "data"


def test_resource_path_with_meipass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PyInstaller bundle 実行を模した場合、sys._MEIPASS 配下に解決される。"""
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    p = resource_path("data/report_paper.jpg")
    assert p == tmp_path / "data" / "report_paper.jpg"


def test_resource_path_accepts_path_object() -> None:
    """Path オブジェクトも受け付けることを確認する。"""
    p = resource_path(Path("data/report_paper.jpg"))
    assert p.name == "report_paper.jpg"
