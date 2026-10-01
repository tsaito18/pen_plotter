"""同梱データ（レポート用紙画像など）のパス解決。

通常実行ではリポジトリルート基準、PyInstaller ``--onefile`` の exe 実行時は
``sys._MEIPASS`` 配下（同梱データの展開先）から解決する。CWD には依存しない。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resource_path(relative: str | Path) -> Path:
    """データファイルの絶対パスを返す。

    PyInstaller bundle 実行時は ``sys._MEIPASS`` 配下、それ以外は repository root
    配下から解決する。``relative`` は repo root から見た相対パス
    (例: ``"data/report_paper.jpg"``)。
    """
    bundle_dir = getattr(sys, "_MEIPASS", None)
    base = Path(bundle_dir) if bundle_dir else REPO_ROOT
    return base / Path(relative)


def report_paper_path() -> Path | None:
    """レポート用紙のスキャン画像（プレビュー背景）。無ければ None。"""
    path = resource_path("data/report_paper.jpg")
    return path if path.exists() else None
