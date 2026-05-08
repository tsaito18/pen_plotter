"""スクロール対応ログ表示 widget。

Worker からの LogEvent を level (info/sent/recv/warn/error) で色分けし、
時刻を行頭に付けて追記表示する。Text は wrap=NONE にして長い G-code 行
でも改行しない (横スクロールでスキャン可)。
"""

from __future__ import annotations

import datetime
import tkinter as tk
from tkinter import ttk


# レベル別の前景色マッピング。
# - info: 黒 (通常)
# - sent: 青 (送信した行を識別)
# - recv: 緑 ("ok" 等の受信)
# - warn: 橙 (警告)
# - error: 赤 (失敗)
_LEVEL_COLORS: dict[str, str] = {
    "info": "#000000",
    "sent": "#1a73e8",
    "recv": "#1e8e3e",
    "warn": "#e8710a",
    "error": "#d93025",
}


class LogViewWidget(ttk.LabelFrame):
    """ログ追記表示 widget。

    Public API:
        - ``add_log(level, message)``: 1 行追加 (時刻自動付与、自動スクロール)
        - ``clear()``: ログ全消去
    """

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, text="ログ")

        # Text + Scrollbar を 1 つの Frame にまとめる。grid だと
        # Scrollbar の伸縮が綺麗に決まるが、ここではシンプルに pack で十分。
        self._text = tk.Text(
            self, height=10, wrap=tk.NONE, state=tk.DISABLED, font=("TkFixedFont", 9)
        )
        self._scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._text.yview)
        self._text.configure(yscrollcommand=self._scroll.set)

        self._scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # レベルごとの色タグを事前登録。add_log 時に tag 名を指定するだけで
        # 該当行に色を当てられる。
        for level, color in _LEVEL_COLORS.items():
            self._text.tag_configure(level, foreground=color)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def add_log(self, level: str, message: str) -> None:
        # 未知の level は info 扱い (色は黒) にしておく。
        tag = level if level in _LEVEL_COLORS else "info"
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{tag.upper():5}] {message}\n"

        # Text は state=DISABLED のままだと insert 不可。書き込み中だけ
        # NORMAL に切り替え、即 DISABLED に戻すことでユーザ編集を防ぐ。
        self._text.configure(state=tk.NORMAL)
        self._text.insert(tk.END, line, tag)
        self._text.see(tk.END)  # 末尾自動スクロール
        self._text.configure(state=tk.DISABLED)

    def clear(self) -> None:
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.configure(state=tk.DISABLED)
