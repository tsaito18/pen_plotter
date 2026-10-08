"""文字サイズの統一API（字種 × 画数・線の長さ）と字送り。

layout(typesetter) と render(positioning) はサイズ倍率をここから引く（二重管理しない）。

サイズ（本文フォントサイズ比）は字種と複雑さで決まる。本人の手書きレポートの実測で、
同じ人の字は「漢字は大きく画数が多いほど大きい・かなは小さく単純な形ほど小さい」:

- 漢字: 画数で 0.9（1〜3 画）→ 1.1（15 画以上）
- かな: 字形の総線長で 0.6（ン・く・し）→ 0.88（あ・ぬ・わ）。どのかなも漢字より小さい
- 英数字 0.8、小書き 0.55、句読点 0.35（字種の代表値 :func:`char_type_scale`）

画数・線長は KanjiVG から求めた複雑度マップ（``data/char_complexity.json``、
``scripts/compute_char_complexity.py`` で生成）を lazy にロードして引く。
マップに無い字は字種の代表値。
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.layout.line_breaking import is_halfwidth

logger = logging.getLogger(__name__)

# 点列の要素は (x, y) タプル/リスト、または {"x":.., "y":..} 辞書のいずれも許容
Point = Sequence[float] | Mapping[str, float]
Stroke = Sequence[Point]

# --- 字サイズの写像パラメータ（実物の手書きレポートの実測に合わせた値）----------

# 漢字: 画数 → サイズ
KANJI_SIZE_MIN = 0.9
KANJI_SIZE_MAX = 1.1
_KANJI_STROKES_LOW = 3
_KANJI_STROKES_HIGH = 15
# かな: 字形の総線長（KanjiVG 正規化座標）→ サイズ
KANA_SIZE_MIN = 0.6
KANA_SIZE_MAX = 0.88
_KANA_INK_LOW = 15.0
_KANA_INK_HIGH = 38.0

# 複雑度の合成重み（画数とink長の寄与）。書き味の調整で動かす想定の名前付き定数。
COMPLEXITY_WEIGHT_STROKE = 0.5
COMPLEXITY_WEIGHT_INK = 0.5

# robust 正規化に使う percentile 境界。外れ値（極端に画数の多い字など）で
# 大多数の字が 0/1 に潰れるのを防ぐ。
COMPLEXITY_PCT_LOW = 5.0
COMPLEXITY_PCT_HIGH = 95.0

# 複雑度マップの所在。CWD に依存せず引けるよう __file__ 起点で解決する。
# src/layout/char_metrics.py → parents[2] がプロジェクトルート。
_COMPLEXITY_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "char_complexity.json"


# --- ink 長計算（マップ生成スクリプトと共有する純粋関数）-------------------


def _point_xy(point: Point) -> tuple[float, float]:
    """点を (x, y) に正規化する。タプル/リストと JSON 辞書の両形式を許容。"""
    if isinstance(point, Mapping):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def stroke_ink_length(stroke: Stroke) -> float:
    """1ストローク点列の隣接点ユークリッド距離総和を返す。

    Args:
        stroke: 点列。各点は (x, y) または {"x":.., "y":..}。

    Returns:
        点列に沿った実線長。点が1個以下なら 0.0。
    """
    total = 0.0
    prev: tuple[float, float] | None = None
    for raw in stroke:
        cur = _point_xy(raw)
        if prev is not None:
            total += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        prev = cur
    return total


def char_ink_length(strokes: Sequence[Stroke]) -> float:
    """全ストロークの ink 長合計を返す。

    Args:
        strokes: ストロークのリスト。

    Returns:
        文字全体の実線長合計。
    """
    return sum(stroke_ink_length(stroke) for stroke in strokes)


def percentile(sorted_values: list[float], pct: float) -> float:
    """昇順済みリストに対する線形補間 percentile。

    numpy 依存を避けるため自前実装。マップ生成と正規化の両方で使う。
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[int(rank)]
    frac = rank - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def normalize_robust(
    values: Sequence[float],
    *,
    low_pct: float = COMPLEXITY_PCT_LOW,
    high_pct: float = COMPLEXITY_PCT_HIGH,
) -> list[float]:
    """percentile 境界で clamp してから min-max 正規化し 0-1 に写す。

    外れ値で大多数が潰れないよう、生の min/max ではなく percentile を境界にする。

    Args:
        values: 正規化対象の数値列（画数や ink 長）。
        low_pct: 下側境界の percentile。これ以下は 0 にクランプ。
        high_pct: 上側境界の percentile。これ以上は 1 にクランプ。

    Returns:
        入力と同順・同長の 0-1 値リスト。分散ゼロ時は全て 0.0。
    """
    if not values:
        return []
    ordered = sorted(values)
    lo = percentile(ordered, low_pct)
    hi = percentile(ordered, high_pct)
    span = hi - lo
    # 定数列でも percentile 補間の丸めで ~1e-16 の差が出るため、相対的に 0 とみなす
    if span <= 1e-12 * max(1.0, abs(hi)):
        return [0.0] * len(values)
    return [min(1.0, max(0.0, (v - lo) / span)) for v in values]


def compute_complexity(
    *,
    stroke_norm: float,
    ink_norm: float,
    w_stroke: float = COMPLEXITY_WEIGHT_STROKE,
    w_ink: float = COMPLEXITY_WEIGHT_INK,
) -> float:
    """正規化済み画数・ink長から複雑度(0-1)を合成する。

    Args:
        stroke_norm: 正規化画数 (0-1)。
        ink_norm: 正規化 ink 長 (0-1)。
        w_stroke: 画数の重み。
        w_ink: ink 長の重み。

    Returns:
        重み付き平均による複雑度。重み合計が1なら 0-1。
    """
    return w_stroke * stroke_norm + w_ink * ink_norm


# --- 種別スケール（typesetter.py の値が「正」）------------------------------

_SMALL_KANA = set("っゃゅょぁぃぅぇぉァィゥェォッャュョヵヶ")
_SMALL_PUNCT = set("・、。，．")

# 字種の代表値（複雑度マップに無い字に使う）
_KANJI_SCALE = 1.0
_KANA_SCALE = 0.8
_HALFWIDTH_SCALE = 0.8
_SMALL_KANA_SCALE = 0.55
_SMALL_PUNCT_SCALE = 0.35


# 半角文字の字送り（本文フォントサイズ比、字間は別途加算）。本人の手書きレポートの
# スキャン実測（Cursor / Measure: 小文字の字送り平均 ≈0.45、大文字のインク幅 ≈0.6〜0.67）に合わせた
# 字ごとの幅。等幅だと i・l の両側が空き、m・w が詰まって活字のように見える。
_HALFWIDTH_ADVANCE_DEFAULT = 0.55
_HALFWIDTH_ADVANCES: dict[str, float] = {
    **dict.fromkeys("il.,:;'!|", 0.22),
    **dict.fromkeys("jftrI()[]{}", 0.3),
    **dict.fromkeys(" J", 0.4),
    **dict.fromkeys("abcdeghknopqsuvxyz", 0.42),
    **dict.fromkeys("0123456789", 0.5),
    **dict.fromkeys("mw", 0.6),
    **dict.fromkeys("ABCDEFGHKLNOPQRSTUVXYZ", 0.68),
    **dict.fromkeys("MW", 0.8),
    **dict.fromkeys("αβγδεζηθικλνξοπρστυχϵς", 0.45),
    **dict.fromkeys("μφψωϕ", 0.52),
}


def halfwidth_advance(ch: str) -> float:
    """半角 1 文字の字送り（本文フォントサイズ比、字間を除く）。"""
    return _HALFWIDTH_ADVANCES.get(ch, _HALFWIDTH_ADVANCE_DEFAULT)


def _is_kana(ch: str) -> bool:
    return 0x3040 <= ord(ch) <= 0x30FF


def char_type_scale(ch: str) -> float:
    """字種の代表サイズ倍率（漢字 1.0 / かな・半角 0.8 / 小書き 0.55 / 句読点 0.35）。"""
    if ch in _SMALL_KANA:
        return _SMALL_KANA_SCALE
    if ch in _SMALL_PUNCT:
        return _SMALL_PUNCT_SCALE
    if is_halfwidth(ch):
        return _HALFWIDTH_SCALE
    if _is_kana(ch):
        return _KANA_SCALE
    return _KANJI_SCALE


# --- 複雑度マップ（lazy ロード）-------------------------------------------------

_complexity_map: dict[str, dict] | None = None
_complexity_map_loaded = False


def _load_complexity_map() -> dict[str, dict]:
    """複雑度マップを初回呼び出し時に1度だけロードしキャッシュする。

    マップ未生成（ファイル不在）や破損時も例外を投げず空 dict を返し、
    字サイズを字種の代表値へフォールバックさせる（マップが無くても動く堅牢性）。
    """
    global _complexity_map, _complexity_map_loaded
    if _complexity_map_loaded:
        return _complexity_map or {}
    _complexity_map_loaded = True
    try:
        raw = json.loads(_COMPLEXITY_MAP_PATH.read_text(encoding="utf-8"))
        # 統計メタ等の非文字キーが混ざっても引き時に無視されるので保持する
        _complexity_map = raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        logger.warning(
            "char complexity map not loaded (%s); char sizes fall back to the type defaults",
            _COMPLEXITY_MAP_PATH,
        )
        _complexity_map = {}
    return _complexity_map


def _lerp_clamped(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    t = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
    return y0 + (y1 - y0) * t


def effective_char_scale(ch: str) -> float:
    """字種と複雑さで決まる文字サイズ倍率（本文フォントサイズ比）。

    漢字は画数、かなは字形の総線長で連続的に変える（モジュール docstring 参照）。
    小書き・句読点・半角・複雑度マップに無い字は :func:`char_type_scale`。
    """
    base = char_type_scale(ch)
    if ch in _SMALL_KANA or ch in _SMALL_PUNCT or is_halfwidth(ch):
        return base
    entry = _load_complexity_map().get(ch)
    if not isinstance(entry, dict):
        return base
    if _is_kana(ch):
        ink = float(entry["ink_len"])
        return _lerp_clamped(ink, _KANA_INK_LOW, _KANA_INK_HIGH, KANA_SIZE_MIN, KANA_SIZE_MAX)
    strokes = float(entry["strokes"])
    return _lerp_clamped(
        strokes, _KANJI_STROKES_LOW, _KANJI_STROKES_HIGH, KANJI_SIZE_MIN, KANJI_SIZE_MAX
    )
