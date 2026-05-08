"""``MainWindow._dispatch_event`` を Tk 非依存ロジックとして単体テストする。

Tkinter は DISPLAY 必須のため、CI/WSL では起動できない。MainWindow の
イベント分配ロジックを ``dispatch_event`` 関数として切り出してあるため、
widget を MagicMock に差し替えるだけで全ロジックを検証できる。

Step 6 で ``UiState`` を引数に追加し、接続中/送信中の組合せで
control_panel の有効/無効を決める設計に変更している。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.plotter_gui.app import UiState, dispatch_event
from src.plotter_gui.events import (
    Connected,
    Disconnected,
    JobFinished,
    JobStarted,
    LogEvent,
    Progress,
)


def _make_mocks() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """各 widget の MagicMock を生成して返す。"""
    return MagicMock(), MagicMock(), MagicMock(), MagicMock()


class TestConnected:
    def test_connected_enables_control_panel(self) -> None:
        """Connected で port_panel/control_panel が有効化されログが出る。"""
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState()
        dispatch_event(
            Connected(port_name="COM3"),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        port_panel.set_connected.assert_called_once_with(True)
        control_panel.set_enabled.assert_called_once_with(True)
        # メッセージ本文にポート名が含まれていること (人間が読んで判別可能)
        log_view.add_log.assert_called_once()
        level, message = log_view.add_log.call_args.args
        assert level == "info"
        assert "COM3" in message
        # state も破壊的に更新されていること
        assert state.is_connected is True

    def test_connected_does_not_enable_control_panel_during_running(self) -> None:
        """送信中に Connected が来ても control_panel は無効のまま保たれる。

        実運用では「送信中に再接続」は起きないが、テストで状態遷移の
        独立性を保証する。is_running 優先で control_panel を抑止する。
        """
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState(is_running=True)
        dispatch_event(
            Connected(port_name="COM3"),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        port_panel.set_connected.assert_called_once_with(True)
        # 送信中は機械操作ボタンを有効化しない
        control_panel.set_enabled.assert_called_once_with(False)
        assert state.is_connected is True
        assert state.is_running is True


class TestDisconnected:
    def test_disconnected_disables_control_panel(self) -> None:
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState(is_connected=True)
        dispatch_event(
            Disconnected(),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        port_panel.set_connected.assert_called_once_with(False)
        control_panel.set_enabled.assert_called_once_with(False)
        log_view.add_log.assert_called_once()
        level, _ = log_view.add_log.call_args.args
        assert level == "info"

    def test_disconnected_resets_state_and_disables_panel(self) -> None:
        """Disconnected で state.is_connected が False に戻り、control_panel が無効化される。"""
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState(is_connected=True, is_running=False)
        dispatch_event(
            Disconnected(),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        assert state.is_connected is False
        control_panel.set_enabled.assert_called_once_with(False)


class TestLogEvent:
    def test_log_event_forwards_to_log_view(self) -> None:
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState()
        dispatch_event(
            LogEvent(level="warn", message="something happened"),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        log_view.add_log.assert_called_once_with("warn", "something happened")


class TestJobStarted:
    def test_job_started_logs_info(self) -> None:
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState()
        dispatch_event(
            JobStarted(kind="home"),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        log_view.add_log.assert_called_once()
        level, message = log_view.add_log.call_args.args
        assert level == "info"
        assert "home" in message

    def test_job_started_stream_locks_ui(self) -> None:
        """``JobStarted("stream")`` で UI が送信中ロック状態に切り替わる。

        - state.is_running = True に更新
        - job_panel: reset_progress() → set_running(True)
        - control_panel: set_enabled(False) で機械操作ボタンを抑止
        """
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState(is_connected=True)
        dispatch_event(
            JobStarted(kind="stream"),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        assert state.is_running is True
        job_panel.reset_progress.assert_called_once()
        job_panel.set_running.assert_called_once_with(True)
        control_panel.set_enabled.assert_called_once_with(False)
        log_view.add_log.assert_called_once()
        level, message = log_view.add_log.call_args.args
        assert level == "info"
        assert "stream" in message

    def test_job_started_home_does_not_lock_ui(self) -> None:
        """``JobStarted("home")`` (短時間ジョブ) では UI ロックしない。

        ホーミング/ペン制御は数秒で完了するため、毎回 UI ロックすると
        体感がうるさくなる。stream のみロックするポリシー。
        """
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState(is_connected=True)
        dispatch_event(
            JobStarted(kind="home"),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        # state.is_running は変化しない
        assert state.is_running is False
        # job_panel の running 切替も呼ばれない
        job_panel.set_running.assert_not_called()
        job_panel.reset_progress.assert_not_called()
        # control_panel も触らない
        control_panel.set_enabled.assert_not_called()
        # ログのみ出る
        log_view.add_log.assert_called_once()


class TestJobFinished:
    def test_job_finished_success_logs_info(self) -> None:
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState()
        dispatch_event(
            JobFinished(kind="home", success=True),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        log_view.add_log.assert_called_once()
        level, message = log_view.add_log.call_args.args
        assert level == "info"
        assert "home" in message

    def test_job_finished_failure_logs_error(self) -> None:
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState()
        dispatch_event(
            JobFinished(kind="home", success=False, error="serial port not open"),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        log_view.add_log.assert_called_once()
        level, message = log_view.add_log.call_args.args
        assert level == "error"
        assert "home" in message
        assert "serial port not open" in message

    def test_job_finished_stream_unlocks_ui_when_connected(self) -> None:
        """送信完了 (success) で UI ロックが解除され、機械操作も復帰する。"""
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState(is_connected=True, is_running=True)
        dispatch_event(
            JobFinished(kind="stream", success=True),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        assert state.is_running is False
        job_panel.set_running.assert_called_once_with(False)
        # 接続中なので機械操作ボタンを再度有効化
        control_panel.set_enabled.assert_called_once_with(True)
        log_view.add_log.assert_called_once()
        level, _ = log_view.add_log.call_args.args
        assert level == "info"

    def test_job_finished_stream_keeps_disabled_when_not_connected(self) -> None:
        """送信中に切断された場合の終了 → control_panel は無効のまま。

        cancelled error など失敗終了経路でも state リセットは確実に行う。
        """
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState(is_connected=False, is_running=True)
        dispatch_event(
            JobFinished(kind="stream", success=False, error="cancelled"),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        assert state.is_running is False
        job_panel.set_running.assert_called_once_with(False)
        control_panel.set_enabled.assert_called_once_with(False)
        # 失敗終了は error ログ
        log_view.add_log.assert_called_once()
        level, message = log_view.add_log.call_args.args
        assert level == "error"
        assert "cancelled" in message


class TestProgress:
    def test_progress_event_updates_job_panel(self) -> None:
        """Progress イベントは job_panel.update_progress に流す。"""
        port_panel, control_panel, job_panel, log_view = _make_mocks()
        state = UiState(is_connected=True, is_running=True)
        dispatch_event(
            Progress(idx=3, total=10, line="G1 X10"),
            state=state,
            port_panel=port_panel,
            control_panel=control_panel,
            job_panel=job_panel,
            log_view=log_view,
        )
        job_panel.update_progress.assert_called_once_with(3, 10, "G1 X10")
        # 他の widget は触らない
        log_view.add_log.assert_not_called()
        port_panel.set_connected.assert_not_called()
        control_panel.set_enabled.assert_not_called()
