"""UI スレッド ↔ Worker スレッド間で受け渡すイベント定義。

GUI(Tkinter) は queue.Queue を `after()` で定期 poll する設計のため、
Worker からのイベントは frozen dataclass にして不変・スレッドセーフに保つ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class JobStarted:
    """ジョブの開始を示すイベント。

    kind: "connect" | "disconnect" | "home" | "pen_up" | "pen_down" | "stream"
    """

    kind: str


@dataclass(frozen=True)
class JobFinished:
    """ジョブの完了 (成功/失敗いずれも) を示すイベント。"""

    kind: str
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class Progress:
    """ストリーミング送信の進捗。

    GUI 進捗バーは `idx / total` を表示する。idx は 1 始まり。
    """

    idx: int
    total: int
    line: str


@dataclass(frozen=True)
class LogEvent:
    """ログ表示用の汎用イベント。

    level: "info" | "sent" | "recv" | "warn" | "error"
    """

    level: str
    message: str


@dataclass(frozen=True)
class Connected:
    """ポート接続完了。"""

    port_name: str


@dataclass(frozen=True)
class Disconnected:
    """ポート切断完了。"""


WorkerEvent: TypeAlias = JobStarted | JobFinished | Progress | LogEvent | Connected | Disconnected
