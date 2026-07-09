"""Pen Plotter MCP サーバー起動スクリプト (stdio)。

環境変数:
- PEN_PLOTTER_CHECKPOINT  : 訓練済みモデル .pt（既定: data_examples/models_yamataku/finetuned.pt）
- PEN_PLOTTER_KANJIVG_DIR : KanjiVG ディレクトリ（既定: data/strokes）
- PEN_PLOTTER_USER_STROKES_DIR: ユーザーストローク（既定: data/user_strokes）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mcp_server.server import main

if __name__ == "__main__":
    main()
