"""G-code ファイル選択 + 静的プレビュー widget。

ファイル選択ダイアログで .gcode / .nc / .txt を選び、自動で
``preview.parse_gcode`` → ``preview.render_strokes`` を呼んでプレビュー表示する。

Step 4 では callback を未配線のままでも構築できるよう
``on_file_selected`` を Optional にする (Step 5 で Worker と接続)。
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, ttk

# matplotlib は描画バックエンドとして TkAgg を要求する。
# ヘッドレス環境で import 自体は通るよう、グローバル import のみ
# 行いバックエンド指定はモジュール初期化時には行わない。
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.plotter_gui.preview import Stroke, parse_gcode, render_strokes


class FilePickerWidget(ttk.Frame):
    """G-code ファイル選択 + プレビュー Canvas を内包する複合 widget。

    Public API:
        - ``selected_path``: 現在選択中のファイルパス (None or Path)
        - ``set_strokes(strokes)``: 外部から渡されたストロークで再描画
    """

    def __init__(
        self,
        parent: tk.Misc,
        on_file_selected: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_file_selected = on_file_selected
        self.selected_path: Path | None = None

        # 上段: ファイル選択ボタン + パス表示
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="ファイル選択", command=self._on_pick).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        self._path_var = tk.StringVar(value="(未選択)")
        ttk.Label(bar, textvariable=self._path_var).pack(
            side=tk.LEFT, padx=4, pady=4, fill=tk.X, expand=True
        )

        # 下段: matplotlib プレビュー Canvas。
        # A4 縦置きを意識して縦長にする。dpi はデフォルトのまま (100)。
        self._figure = Figure(figsize=(4.2, 6.0))
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 起動時は空の枠だけ表示。
        render_strokes(self._ax, [])
        self._canvas.draw_idle()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_strokes(self, strokes: list[Stroke]) -> None:
        """外部から渡されたストロークを再描画する (テスト/再利用用)。"""
        render_strokes(self._ax, strokes)
        self._canvas.draw_idle()

    # -----------------------------------------------------------------
    # 内部: ボタン callback
    # -----------------------------------------------------------------

    def _on_pick(self) -> None:
        # ダイアログの拡張子フィルタ。Windows/macOS でも一貫した挙動になるよう
        # 拡張子の組み合わせ単位で指定する。
        filename = filedialog.askopenfilename(
            title="G-code ファイルを選択",
            filetypes=[
                ("G-code", "*.gcode *.nc *.txt"),
                ("All files", "*.*"),
            ],
        )
        if not filename:
            return  # ダイアログキャンセル
        path = Path(filename)
        self.selected_path = path
        self._path_var.set(str(path))

        # パース失敗してもアプリを落とさない: 例外は握りつぶしてログに任せる。
        # 上位 (MainWindow) で LogView へ書き出す Step 5 まではコンソールに print。
        try:
            strokes = parse_gcode(path)
        except Exception as exc:  # noqa: BLE001 — UI が落ちないことを優先
            print(f"[FilePickerWidget] parse failed: {exc}")
            strokes = []
        self.set_strokes(strokes)

        if self._on_file_selected is not None:
            self._on_file_selected(path)
