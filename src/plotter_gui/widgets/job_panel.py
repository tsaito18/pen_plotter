"""送信開始 / 緊急停止 / 進捗バー widget。

GUI スレッドからしか呼ばれない前提のため Lock は不要。Worker からの
Progress イベントを MainWindow が受け取って ``update_progress`` を
呼ぶ流れになる (Step 5)。
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk


class JobPanelWidget(ttk.LabelFrame):
    """送信開始 / 緊急停止ボタンと進捗表示。

    Public API:
        - ``set_running(running: bool)``: 送信中/停止中の UI 切り替え
        - ``update_progress(idx, total, line)``: 進捗バー + ラベル更新
        - ``reset_progress()``: 進捗を 0 に戻す (新規ジョブ開始時)
    """

    def __init__(
        self,
        parent: tk.Misc,
        on_start: Callable[[], None] | None = None,
        on_emergency_stop: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, text="ジョブ送信")
        self._on_start = on_start
        self._on_emergency_stop = on_emergency_stop

        # 上段: 送信開始 + 緊急停止ボタン
        row1 = ttk.Frame(self)
        row1.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        self._start_btn = ttk.Button(row1, text="送信開始", command=self._click_start)
        self._start_btn.pack(side=tk.LEFT, padx=2)
        self._stop_btn = ttk.Button(
            row1,
            text="緊急停止",
            command=self._click_stop,
            state=tk.DISABLED,
        )
        self._stop_btn.pack(side=tk.LEFT, padx=2)

        # 中段: 進捗バー (パーセント / determinate モード)
        row2 = ttk.Frame(self)
        row2.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        # maximum=100 で「百分率」を直接 value に入れる。idx/total から計算する。
        self._progress = ttk.Progressbar(row2, mode="determinate", maximum=100, length=300)
        self._progress.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # 下段: 進捗ラベル ("idx / total 行 (XX%)") + 現在送信行
        self._progress_var = tk.StringVar(value="0 / 0 行 (0%)")
        ttk.Label(self, textvariable=self._progress_var).pack(side=tk.TOP, anchor=tk.W, padx=4)
        self._line_var = tk.StringVar(value="")
        # 行ラベルは G-code 1 行をそのまま表示するため等幅で見せる。
        ttk.Label(self, textvariable=self._line_var, font=("TkFixedFont", 9)).pack(
            side=tk.TOP, anchor=tk.W, padx=4, pady=(0, 4)
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        if running:
            self._start_btn.configure(state=tk.DISABLED)
            self._stop_btn.configure(state=tk.NORMAL)
        else:
            self._start_btn.configure(state=tk.NORMAL)
            self._stop_btn.configure(state=tk.DISABLED)

    def update_progress(self, idx: int, total: int, line: str) -> None:
        # total=0 のガード: 0 除算を避ける。実用上は worker から
        # total>=1 が来る想定だが UI 側でも防御する。
        percent = int((idx / total) * 100) if total > 0 else 0
        self._progress["value"] = percent
        self._progress_var.set(f"{idx} / {total} 行 ({percent}%)")
        self._line_var.set(line)

    def reset_progress(self) -> None:
        self._progress["value"] = 0
        self._progress_var.set("0 / 0 行 (0%)")
        self._line_var.set("")

    # -----------------------------------------------------------------
    # 内部: callback
    # -----------------------------------------------------------------

    def _click_start(self) -> None:
        if self._on_start is not None:
            self._on_start()

    def _click_stop(self) -> None:
        if self._on_emergency_stop is not None:
            self._on_emergency_stop()
