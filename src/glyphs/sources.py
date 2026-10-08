"""ディスク上の字形ソース: KanjiVG 参照字形とユーザー筆跡サンプル。

どちらも ``<root>/<文字>/<文字>_*.json``（:class:`StrokeSample` 形式）で保存される。
KanjiVG は Y-UP、ユーザー筆跡は iPad Canvas そのままの Y-DOWN。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.collector.data_format import StrokeSample
from src.geometry import Stroke

logger = logging.getLogger(__name__)


def sample_to_arrays(sample: StrokeSample) -> tuple[list[Stroke], list[str]]:
    """2 点以上のストロークだけを配列化し、筆画タイプ（kvg:type）も同じ条件で間引く。

    ``types[j]`` が ``strokes[j]`` に 1 対 1 で対応する（不足分は ``""``）。
    """
    strokes: list[Stroke] = []
    types: list[str] = []
    for i, points in enumerate(sample.strokes):
        arr = np.array([[p.x, p.y] for p in points], dtype=np.float64)
        if len(arr) >= 2:
            strokes.append(arr)
            types.append(sample.stroke_types[i] if i < len(sample.stroke_types) else "")
    return strokes, types


class KanjiVGStore:
    """KanjiVG 参照字形（Y-UP）の読み込み。存在しないディレクトリなら常に None を返す。"""

    def __init__(self, root: Path | str | None) -> None:
        d = Path(root) if root is not None else None
        self.root: Path | None = d if d is not None and d.is_dir() else None
        self._load = lru_cache(maxsize=4096)(self._load_uncached)

    @property
    def available(self) -> bool:
        return self.root is not None

    def load(self, char: str) -> tuple[list[Stroke] | None, list[str]]:
        """``(strokes, kvg_types)`` を返す。字形が無ければ ``(None, [])``。"""
        strokes, types = self._load(char)
        if strokes is None:
            return None, []
        return list(strokes), list(types)

    def _load_uncached(self, char: str) -> tuple[tuple[Stroke, ...] | None, tuple[str, ...]]:
        # 「/」「..」等はパスとして解釈されてしまうので引かない
        if self.root is None or char in ("", ".", "..") or "/" in char or "\\" in char:
            return None, ()
        char_dir = self.root / char
        if not char_dir.is_dir():
            return None, ()
        files = sorted(char_dir.glob(f"{char}_*.json"))
        if not files:
            return None, ()
        try:
            strokes, types = sample_to_arrays(StrokeSample.load(files[0]))
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("KanjiVG JSON load failed for %r", char, exc_info=True)
            return None, ()
        return (tuple(strokes) if strokes else None), tuple(types)


class UserStrokeDB:
    """ユーザー筆跡サンプル（Y-DOWN）。文字ごとに「最も丁寧な 1 サンプル」を返す。

    同じ字には常に同じベース字形を使う（隣接する同一字でサンプルが入れ替わると
    品質が極端に振れるため）。丁寧さは総点数で近似する。書くたびの多様性は
    描画側の微小バリエーションが与える。
    """

    def __init__(self, char_dir_root: Path | str | None) -> None:
        self._samples: dict[str, list[list[Stroke]]] = {}
        root = Path(char_dir_root) if char_dir_root is not None else None
        if root is None or not root.is_dir():
            return
        for char_dir in sorted(root.iterdir()):
            if not char_dir.is_dir():
                continue
            for json_file in sorted(char_dir.glob("*.json")):
                try:
                    sample = StrokeSample.load(json_file)
                except (OSError, ValueError, KeyError, TypeError):
                    logger.warning("Failed to load user stroke: %s", json_file)
                    continue
                strokes = [
                    np.array([[p.x, p.y] for p in stroke], dtype=np.float64)
                    for stroke in sample.strokes
                ]
                self._samples.setdefault(char_dir.name, []).append(strokes)
        logger.info("Loaded user stroke DB: %d chars", len(self._samples))

    def __contains__(self, char: str) -> bool:
        return char in self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def items(self) -> Iterator[tuple[str, list[list[Stroke]]]]:
        """``(文字, その字の全サンプル)`` を返す。"""
        return iter(self._samples.items())

    def best_sample(self, char: str) -> list[Stroke] | None:
        samples = self._samples.get(char)
        if not samples:
            return None
        return max(samples, key=lambda s: sum(len(stroke) for stroke in s))
