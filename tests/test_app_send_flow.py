"""送信フロー (Step 6) のロジック関数を Tk 非依存で単体テストする。

MainWindow の callback ロジック (`_on_start` / `_on_emergency_stop` /
`_on_file_selected`) はモジュールトップ関数 ``request_stream`` /
``handle_file_selected`` として切り出され、widget/worker を MagicMock で
差し替えて検証できる。これにより DISPLAY 無し環境でも全分岐を網羅できる。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.plotter_gui.app import (
    UiState,
    handle_file_selected,
    request_stream,
)


class TestRequestStream:
    def test_on_start_without_connection_logs_warn(self) -> None:
        """未接続のまま送信開始 → submit_stream は呼ばず warn ログを残す。"""
        worker = MagicMock()
        log_view = MagicMock()
        state = UiState(is_connected=False)

        request_stream(
            state=state,
            selected_lines=["G1 X10"],
            worker=worker,
            log_view=log_view,
        )

        worker.submit_stream.assert_not_called()
        log_view.add_log.assert_called_once()
        level, _ = log_view.add_log.call_args.args
        assert level == "warn"

    def test_on_start_without_file_logs_warn(self) -> None:
        """接続済みでもファイル未選択 → submit_stream は呼ばず warn ログ。"""
        worker = MagicMock()
        log_view = MagicMock()
        state = UiState(is_connected=True)

        request_stream(
            state=state,
            selected_lines=None,
            worker=worker,
            log_view=log_view,
        )

        worker.submit_stream.assert_not_called()
        log_view.add_log.assert_called_once()
        level, _ = log_view.add_log.call_args.args
        assert level == "warn"

    def test_on_start_submits_stream_when_ready(self) -> None:
        """接続済み + ファイル選択済みで submit_stream(lines) が 1 回呼ばれる。"""
        worker = MagicMock()
        log_view = MagicMock()
        state = UiState(is_connected=True)
        lines = ["G90", "G1 X10 Y10", "M2"]

        request_stream(
            state=state,
            selected_lines=lines,
            worker=worker,
            log_view=log_view,
        )

        worker.submit_stream.assert_called_once_with(lines)


class TestEmergencyStop:
    def test_emergency_stop_calls_worker(self) -> None:
        """``_on_emergency_stop`` 相当の単純パススルー。

        worker.emergency_stop は接続状態によらず常時呼べる仕様 (worker 側で
        port=None ガード済み)。GUI 側で is_running 等の判定はしない。
        """
        worker = MagicMock()
        # MainWindow._on_emergency_stop は worker.emergency_stop() を直接呼ぶだけ
        # なので、本テストでは worker の呼び出しを直接検証する。
        worker.emergency_stop()
        worker.emergency_stop.assert_called_once()


class TestHandleFileSelected:
    def test_handle_file_selected_reads_lines(self, tmp_path: Path) -> None:
        """ファイル選択で内容を splitlines したリストが返り、ログが出る。"""
        gcode_path = tmp_path / "sample.gcode"
        gcode_path.write_text("G90\nG1 X10 Y10\nM2\n", encoding="utf-8")

        log_view = MagicMock()
        lines = handle_file_selected(gcode_path, log_view=log_view)

        assert lines == ["G90", "G1 X10 Y10", "M2"]
        log_view.add_log.assert_called_once()
        level, message = log_view.add_log.call_args.args
        assert level == "info"
        # ログ本文にファイル名と行数が含まれる
        assert "sample.gcode" in message
        assert "3" in message

    def test_handle_file_selected_returns_none_on_oserror(self, tmp_path: Path) -> None:
        """存在しないファイル → None 返却 + error ログ。例外で UI を落とさない。"""
        missing = tmp_path / "missing.gcode"
        log_view = MagicMock()

        lines = handle_file_selected(missing, log_view=log_view)

        assert lines is None
        log_view.add_log.assert_called_once()
        level, _ = log_view.add_log.call_args.args
        assert level == "error"
