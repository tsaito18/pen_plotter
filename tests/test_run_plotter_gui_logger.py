"""run_plotter_gui の crash logger フックの単体テスト。

--windowed ビルド時 (sys.stderr 無効) でもクラッシュ原因が exe 横の
plotter_gui_error.log に残ることが運用上の前提なので、最低限
- sys.excepthook が差し替わること
- フックが呼ばれた時にログファイルへ追記されること
を確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import scripts.run_plotter_gui as rpg


def test_install_crash_logger_replaces_excepthook(monkeypatch, tmp_path: Path) -> None:
    """`_install_crash_logger` 呼び出しで sys.excepthook が変更されること。"""
    # 非 frozen 経路では _PROJECT_ROOT 直下にログを書く設計だが、
    # テストでは tmp_path に向かわせて実プロジェクトを汚染しない。
    monkeypatch.setattr(rpg, "_PROJECT_ROOT", tmp_path)

    original = sys.excepthook
    try:
        rpg._install_crash_logger()
        assert sys.excepthook is not original
    finally:
        sys.excepthook = original


def test_crash_logger_writes_to_logfile(monkeypatch, tmp_path: Path) -> None:
    """フック発火でログファイルに traceback が追記されること。"""
    monkeypatch.setattr(rpg, "_PROJECT_ROOT", tmp_path)

    original = sys.excepthook
    try:
        rpg._install_crash_logger()
        try:
            raise RuntimeError("boom-for-test")
        except RuntimeError:
            exc_type, exc, tb = sys.exc_info()
            sys.excepthook(exc_type, exc, tb)

        log_path = tmp_path / "plotter_gui_error.log"
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "boom-for-test" in content
        assert "RuntimeError" in content
    finally:
        sys.excepthook = original
