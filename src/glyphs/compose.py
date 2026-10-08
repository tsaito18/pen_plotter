"""部品合成: 本人が書いた字から部品（部首など）を取り出し、まだ書いていない字を組み立てる。

本人サンプルの無い字を KanjiVG（教科書体の骨格）で描くと、周りの本人の字の中で
浮いてしまう。KanjiVG の部品表（``scripts/prepare_kanjivg.py`` が出力する
``components.json``）を使い、例えば「遮」なら「過」の「辶」・別の字の「庶」のように、
同じ部品を含む本人の字からその部品の画を切り出して、KanjiVG の部品の位置・大きさへ
はめ込む。部品が見つからない画だけ KanjiVG の画を使う。

本人がお手本と違う書き順・字体で書いた部品を混ぜると字が崩れるため、はめ込んだ
画が KanjiVG の対応する画と形・向きで近い部品だけを使う。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.geometry import Stroke, resample_stroke
from src.glyphs.sources import KanjiVGStore, UserStrokeDB
from src.glyphs.style import WritingStyle

logger = logging.getLogger(__name__)

COMPONENTS_FILE = "components.json"

# はめ込んだ画と KanjiVG の画の点間距離の上限（字の長辺比）。画の平均・最悪の画の両方で見る
_MAX_SHAPE_ERROR = 0.06
_MAX_STROKE_ERROR = 0.1
# 部品が字に占める大きさが、元の字と組み立てる字でこれ以上違う部品は使わない
# （大きく書いた「口」を「器」の小さな口へ縮めると、癖が強調されて崩れる）
_MAX_RELATIVE_SIZE_RATIO = 2.0
_MIN_PART_STROKES = 3  # 2 画以下の小部品は寄せ集めると継ぎはぎに見えるので使わない
# 本人の部品で描ける画がこの割合未満なら組み立てない（ほぼ KanjiVG のままになるため）
_MIN_USER_RATIO = 0.5
# 縦横の一方がこの比より細い部品（一・丨 など）は縦横比を保って拡縮する
_THIN_RATIO = 0.15
_COMPARE_POINTS = 16


@dataclass(frozen=True)
class Component:
    """字の中の部品 1 つ（``strokes`` はその字の画番号）。"""

    element: str
    position: str
    strokes: tuple[int, ...]


@dataclass(frozen=True)
class _Donor:
    char: str
    component: Component
    user_strokes: tuple[Stroke, ...]  # 本人の字の全画（Y-UP）
    relative_size: float  # 部品の大きさ / 字の大きさ（本人の字の中で）


def load_components(kanjivg_dir: Path | str | None) -> dict[str, tuple[int, list[Component]]]:
    """``{字: (画数, [部品...])}``。部品表が無ければ空（部品合成は使われない）。"""
    if kanjivg_dir is None:
        return {}
    path = Path(kanjivg_dir) / COMPONENTS_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.info("component table not found (%s); composition disabled", path)
        return {}
    return {
        char: (
            int(entry["strokes"]),
            [Component(e, pos, tuple(idx)) for e, pos, idx in entry["parts"]],
        )
        for char, entry in raw.items()
    }


class ComponentComposer:
    """本人の部品で字を組み立てる。結果は字ごとにキャッシュする（同じ字は同じ組み立て）。"""

    def __init__(
        self,
        components: dict[str, tuple[int, list[Component]]],
        user_db: UserStrokeDB,
        kanjivg: KanjiVGStore,
    ) -> None:
        self.components = components
        self.kanjivg = kanjivg
        self._donors: dict[str, list[_Donor]] = {}
        self._cache: dict[tuple[str, WritingStyle | None], list[Stroke] | None] = {}
        if components:
            self._index_donors(user_db)

    def _index_donors(self, user_db: UserStrokeDB) -> None:
        """部品 → その部品を含む本人の字（KanjiVG と画数が同じサンプル）。"""
        for char, samples in user_db.items():
            entry = self.components.get(char)
            if entry is None:
                continue
            n_strokes, parts = entry
            matching = [s for s in samples if len(s) == n_strokes]
            if not matching:
                continue
            sample = max(matching, key=lambda s: sum(len(stroke) for stroke in s))
            user = tuple(s * np.array([1.0, -1.0]) for s in sample)  # Y-DOWN → Y-UP
            span = _span(user)
            for comp in parts:
                if len(comp.strokes) < _MIN_PART_STROKES:
                    continue
                relative = _span([user[i] for i in comp.strokes]) / span
                self._donors.setdefault(comp.element, []).append(_Donor(char, comp, user, relative))

    def compose(self, char: str, style: WritingStyle | None = None) -> list[Stroke] | None:
        """``char`` を本人の部品で組み立てた字形（KanjiVG と同じ座標系・画順）。

        Args:
            char: 組み立てる字。
            style: KanjiVG の骨格（部品の配置と、部品が無い画）に掛ける書き癖。本人の部品は
                すでに本人の傾きで書かれているので掛けない。

        Returns:
            組み立てた字形。部品表に無い字・本人の部品で描ける画が半分未満なら None。
        """
        key = (char, style)
        if key not in self._cache:
            self._cache[key] = self._compose(char, style)
        return self._cache[key]

    def _compose(self, char: str, style: WritingStyle | None) -> list[Stroke] | None:
        entry = self.components.get(char)
        reference, _ = self.kanjivg.load(char)
        if entry is None or reference is None or len(reference) != entry[0]:
            return None
        target = style.apply(reference) if style is not None else reference
        span = _span(target)
        out: list[Stroke] = list(target)
        covered: set[int] = set()
        # 大きい部品から。はめ込めた部品の内側の小部品は飛ばす
        for comp in sorted(entry[1], key=lambda c: -len(c.strokes)):
            if covered.intersection(comp.strokes):
                continue
            placed = self._best_fit(char, comp, target, span)
            if placed is None:
                continue
            for i, stroke in zip(comp.strokes, placed, strict=True):
                out[i] = stroke
            covered.update(comp.strokes)
        if len(covered) < _MIN_USER_RATIO * len(target):
            return None
        return out

    def _best_fit(
        self, char: str, comp: Component, target: list[Stroke], span: float
    ) -> list[Stroke] | None:
        """同じ部品を持つ本人の字のうち、はめ込んだ形が KanjiVG に最も近いもの。"""
        goal = [target[i] for i in comp.strokes]
        relative = _span(goal) / max(span, 1e-9)
        best: tuple[float, list[Stroke]] | None = None
        for donor in self._donors.get(comp.element, []):
            if donor.char == char or len(donor.component.strokes) != len(comp.strokes):
                continue
            size_ratio = abs(np.log(relative / max(donor.relative_size, 1e-9)))
            if size_ratio > np.log(_MAX_RELATIVE_SIZE_RATIO):
                continue
            part = [donor.user_strokes[i] for i in donor.component.strokes]
            if any(len(s) < 2 for s in part):
                continue
            placed = _fit_into(part, goal)
            errors = _stroke_errors(placed, goal) / max(span, 1e-9)
            if errors.mean() > _MAX_SHAPE_ERROR or errors.max() > _MAX_STROKE_ERROR:
                continue
            # 形の近さに加え、同じ位置（へん・つくり等）・同じくらいの大きさの部品を優先
            score = errors.mean() + 0.02 * size_ratio
            if donor.component.position != comp.position:
                score += 0.01
            if best is None or score < best[0]:
                best = (score, placed)
        return best[1] if best is not None else None


def _fit_into(part: list[Stroke], goal: list[Stroke]) -> list[Stroke]:
    """部品の bbox を目標の bbox に合わせる（細い部品は縦横比を保ち、中央に置く）。"""
    src = np.concatenate(part)
    dst = np.concatenate(goal)
    s0, s1 = src.min(axis=0), src.max(axis=0)
    d0, d1 = dst.min(axis=0), dst.max(axis=0)
    src_span = np.maximum(s1 - s0, 1e-9)
    dst_span = d1 - d0
    scale = np.maximum(dst_span, 1e-9) / src_span
    if dst_span.min() < _THIN_RATIO * dst_span.max():
        scale[:] = scale[int(np.argmax(dst_span))]
    offset = d0 + (dst_span - src_span * scale) / 2
    return [(s - s0) * scale + offset for s in part]


def _span(strokes: list[Stroke] | tuple[Stroke, ...]) -> float:
    return float(np.ptp(np.concatenate(strokes), axis=0).max())


def _stroke_errors(placed: list[Stroke], goal: list[Stroke]) -> np.ndarray:
    """対応する画ごとの平均点間距離（向きが逆・書き順違いなら大きくなる）。"""
    return np.array(
        [
            np.linalg.norm(
                resample_stroke(np.asarray(p, dtype=np.float32), _COMPARE_POINTS)
                - resample_stroke(np.asarray(g, dtype=np.float32), _COMPARE_POINTS),
                axis=1,
            ).mean()
            for p, g in zip(placed, goal, strict=True)
        ]
    )
