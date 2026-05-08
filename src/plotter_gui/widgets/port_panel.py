"""ポート選択 + 接続/切断ウィジェット。

xDraw A4 (CH340) の自動検出を試みつつ、見つからない場合の
手動選択肢も提示するハイブリッド設計。GUI 上で Worker と
直接やりとりせず、callback で疎結合に保つ。
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from src.comm.port_finder import find_xdraw_port, list_candidate_ports


class PortPanelWidget(ttk.LabelFrame):
    """ポート選択 + 接続/切断パネル。

    Public API:
        - ``set_connected(state: bool)``: 接続状態に応じてボタン/ラベルを更新
        - ``selected_port_device`` (property 相当): Combobox の選択値から
          デバイス名 (例: "COM3") を取り出す
    """

    def __init__(
        self,
        parent: tk.Misc,
        on_connect: Callable[[str], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, text="ポート")
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect

        # 上段: 自動検出 + 再スキャンボタン
        row1 = ttk.Frame(self)
        row1.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        ttk.Button(row1, text="自動検出", command=self._on_auto_detect).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="再スキャン", command=self._refresh_ports).pack(side=tk.LEFT, padx=2)

        # 中段: ドロップダウン (候補ポート)
        row2 = ttk.Frame(self)
        row2.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        self._port_var = tk.StringVar()
        # Combobox: width はだいたい "/dev/ttyUSB0 (CH340 USB-Serial)" が
        # 収まる程度に取る。実機の description は OS により長くなるため余裕を持つ。
        self._combo = ttk.Combobox(row2, textvariable=self._port_var, state="readonly", width=42)
        self._combo.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # 下段: 接続/切断ボタン + 状態ラベル
        row3 = ttk.Frame(self)
        row3.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        self._connect_btn = ttk.Button(row3, text="接続", command=self._on_connect_click)
        self._connect_btn.pack(side=tk.LEFT, padx=2)
        self._disconnect_btn = ttk.Button(
            row3, text="切断", command=self._on_disconnect_click, state=tk.DISABLED
        )
        self._disconnect_btn.pack(side=tk.LEFT, padx=2)

        self._status_var = tk.StringVar(value="未接続")
        # 接続状態は色で一目瞭然にする。tk.Label は foreground 直接指定が
        # ttk より素直なため、ここだけ素の Label を使う。
        self._status_label = tk.Label(row3, textvariable=self._status_var, foreground="gray")
        self._status_label.pack(side=tk.LEFT, padx=8)

        # 初回は候補ポートを列挙しておく (ユーザが最初の操作前に確認できる)
        self._refresh_ports()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_connected(self, state: bool) -> None:
        """接続状態に応じて UI を更新。

        Worker 側からの Connected/Disconnected イベントに反応して
        Step 5 で MainWindow が呼び出す想定。
        """
        if state:
            self._connect_btn.configure(state=tk.DISABLED)
            self._disconnect_btn.configure(state=tk.NORMAL)
            self._status_var.set("接続中")
            self._status_label.configure(foreground="green")
        else:
            self._connect_btn.configure(state=tk.NORMAL)
            self._disconnect_btn.configure(state=tk.DISABLED)
            self._status_var.set("未接続")
            self._status_label.configure(foreground="gray")

    def selected_port_device(self) -> str | None:
        """Combobox の選択値からデバイス名 (先頭トークン) を取り出す。

        表示書式は ``"COM3 (USB-SERIAL CH340)"`` のため、最初の空白で
        split した先頭部分がデバイス名となる。
        """
        value = self._port_var.get()
        if not value:
            return None
        return value.split(" ", 1)[0]

    # -----------------------------------------------------------------
    # 内部: callback
    # -----------------------------------------------------------------

    def _refresh_ports(self) -> None:
        ports = list_candidate_ports()
        # 表示文字列は「device (description)」。ttk.Combobox は表示と値を
        # 別々に持てないので、選択値からデバイス名を抽出するヘルパで対応する。
        values = [f"{p.device} ({p.description})" for p in ports]
        self._combo["values"] = values
        # 何も選んでいない状態で空文字を残すと自動検出時の上書きが
        # 自然に動く (set() で常に上書きする方針なので問題なし)。

    def _on_auto_detect(self) -> None:
        device = find_xdraw_port()
        if device is None:
            # 候補が見つからなくても候補一覧は出しておく。
            self._refresh_ports()
            return
        # 既存候補リストに含まれていなければ強制的に追加表示。
        existing = list(self._combo["values"])
        match = next((v for v in existing if v.startswith(device + " ")), None)
        if match is None:
            existing.append(f"{device} (auto-detected)")
            self._combo["values"] = existing
            match = existing[-1]
        self._port_var.set(match)

    def _on_connect_click(self) -> None:
        device = self.selected_port_device()
        if device is None or self._on_connect is None:
            return
        self._on_connect(device)

    def _on_disconnect_click(self) -> None:
        if self._on_disconnect is None:
            return
        self._on_disconnect()
