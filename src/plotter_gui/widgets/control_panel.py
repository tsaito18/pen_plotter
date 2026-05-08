"""ホーミング / ペン Up / ペン Down テスト用コントロールパネル widget。

実機接続前は無効化、接続後・送信中ではない時のみ有効になる UI 制約を
``set_enabled`` で外部から切り替えられるようにする。
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk


class ControlPanelWidget(ttk.LabelFrame):
    """機械操作 (ホーミング・ペン制御) 用ボタン群。

    Public API:
        - ``set_enabled(enabled: bool)``: 全ボタンの有効/無効を一括で切り替え
    """

    def __init__(
        self,
        parent: tk.Misc,
        on_home: Callable[[], None] | None = None,
        on_pen_up: Callable[[], None] | None = None,
        on_pen_down: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, text="機械操作")
        self._on_home = on_home
        self._on_pen_up = on_pen_up
        self._on_pen_down = on_pen_down

        # 横並びで 3 ボタン。アイコン文字 (・) は使わず日本語ラベルに統一する。
        self._home_btn = ttk.Button(self, text="ホーミング ($H)", command=self._click_home)
        self._home_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self._pen_up_btn = ttk.Button(self, text="ペン Up", command=self._click_pen_up)
        self._pen_up_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self._pen_down_btn = ttk.Button(self, text="ペン Down", command=self._click_pen_down)
        self._pen_down_btn.pack(side=tk.LEFT, padx=4, pady=4)

        # 初期状態は無効 (未接続)。Step 5 で Worker 接続時に enable する。
        self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self._home_btn.configure(state=state)
        self._pen_up_btn.configure(state=state)
        self._pen_down_btn.configure(state=state)

    # 各ボタンの callback。None 配線時は no-op で済ませる。
    def _click_home(self) -> None:
        if self._on_home is not None:
            self._on_home()

    def _click_pen_up(self) -> None:
        if self._on_pen_up is not None:
            self._on_pen_up()

    def _click_pen_down(self) -> None:
        if self._on_pen_down is not None:
            self._on_pen_down()
