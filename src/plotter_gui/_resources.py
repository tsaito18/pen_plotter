"""GUI から参照する外部リソース (画像など) のパス解決ヘルパ。

PyInstaller --onefile でビルドされた exe を実行すると、同梱データは実行時に
``sys._MEIPASS`` で示される一時ディレクトリへ展開される。通常の
``python -m src.plotter_gui`` 実行時は repository root 基準でデータを参照する。
両者を 1 関数で透過的に扱うために本モジュールを置く。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 通常実行時のベース: src/plotter_gui/_resources.py から見て parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resource_path(relative: str | Path) -> Path:
    """データファイルの絶対パスを返す。

    PyInstaller bundle 実行時は ``sys._MEIPASS`` 配下、それ以外は repository root
    配下から解決する。``relative`` は repo root から見た相対パス
    (例: ``"data/report_paper.jpg"``)。
    """
    bundle_dir = getattr(sys, "_MEIPASS", None)
    base = Path(bundle_dir) if bundle_dir else _REPO_ROOT
    return base / Path(relative)
