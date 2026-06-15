"""文字単位サイズの統一API（種別スケール × 密度補正）。

これまで layout(typesetter) と ui(stroke_renderer) でサイズ倍率が二重管理され
値も食い違っていた（ひらがな 0.85 と 0.88）。本モジュールを唯一の真実源にして
両者から import させ、整合性を担保する。

サイズは2軸の積で決まる:

- 種別スケール (`char_type_scale`): 漢字/かな/半角/小書き/句読点の字種ごとの
  視覚バランス補正。typesetter.py の値を「正」とする。
- 密度補正 (`density_scale`): 画数と正規化字形の ink 長から求めた複雑度
  (`data/char_complexity.json`) を、画数の少ない簡単な字は小さく・多い複雑な字は
  大きく見せる連続倍率に写す。マップに無い字は 1.0（無補正）。

複雑度マップは layout 配下に置く純粋関数(`stroke_ink_length` 等)で生成され
(`scripts/compute_char_complexity.py`)、ここではそれを lazy にロードして引くだけ。
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Mapping, Sequence

from src.layout.line_breaking import is_halfwidth

logger = logging.getLogger(__name__)

# 点列の要素は (x, y) タプル/リスト、または {"x":.., "y":..} 辞書のいずれも許容
Point = Sequence[float] | Mapping[str, float]
Stroke = Sequence[Point]

# --- 複雑度 → 密度補正の写像パラメータ -------------------------------------

# complexity(0-1) を密度補正へ写す線形係数。complexity 中央 0.5 付近で約 1.0、
# 簡単字(0)で 0.88、複雑字(1)で 1.12 になるよう設定。両端は下の clamp で頭打ち。
DENSITY_SCALE_MIN = 0.88
DENSITY_SCALE_MAX = 1.12
_DENSITY_SLOPE = DENSITY_SCALE_MAX - DENSITY_SCALE_MIN  # = 0.24

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


def _percentile(sorted_values: list[float], pct: float) -> float:
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
    lo = _percentile(ordered, low_pct)
    hi = _percentile(ordered, high_pct)
    span = hi - lo
    if span <= 0.0:
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

# 手書きバランス調整: 画数が少なく視覚的に軽い文字は小さめにする。
# 旧 typesetter._KANA_SIZE_OVERRIDES をここへ移植（唯一の真実源化）。
_KANA_SIZE_OVERRIDES: dict[str, float] = {
    # カタカナ: 画数少・形が小さい文字 → 小さめ
    "ロ": 0.68,
    "ハ": 0.78,
    "ニ": 0.78,
    "ノ": 0.75,
    "ヘ": 0.78,
    "フ": 0.80,
    "ク": 0.80,
    "ワ": 0.80,
    "カ": 0.82,
    "コ": 0.80,
    "ン": 0.78,
    "ソ": 0.78,
    "リ": 0.78,
    "ル": 0.80,
    "レ": 0.78,
    "イ": 0.80,
    "ト": 0.78,
    "チ": 0.82,
    "ラ": 0.82,
    # カタカナ: 画数多・形が大きい文字 → やや大きめ
    "ス": 0.85,
    "テ": 0.85,
    "セ": 0.85,
    "サ": 0.85,
    "タ": 0.85,
    "ナ": 0.85,
    "マ": 0.85,
    "ミ": 0.82,
    "ム": 0.85,
    "メ": 0.82,
    "モ": 0.85,
    "ヤ": 0.85,
    "ユ": 0.82,
    "ヨ": 0.82,
    "キ": 0.85,
    "ケ": 0.82,
    "シ": 0.82,
    "ネ": 0.85,
    "ヌ": 0.85,
    "オ": 0.85,
    "エ": 0.82,
    "ア": 0.85,
    "ウ": 0.85,
    "ダ": 0.88,
    "デ": 0.88,
    "ド": 0.88,
    "バ": 0.85,
    "パ": 0.85,
    "ガ": 0.88,
    "ギ": 0.88,
    "グ": 0.85,
    "ゲ": 0.85,
    "ゴ": 0.85,
    "ザ": 0.88,
    "ジ": 0.85,
    "ズ": 0.88,
    "ゼ": 0.88,
    "ゾ": 0.85,
    "ビ": 0.85,
    "ブ": 0.85,
    "ベ": 0.82,
    "ボ": 0.88,
    "ピ": 0.85,
    "プ": 0.82,
    "ペ": 0.82,
    "ポ": 0.85,
    "ヒ": 0.78,
    "ホ": 0.85,
    # ひらがな: 画数少・形が小さい文字 → 小さめ
    "の": 0.78,
    "く": 0.75,
    "し": 0.78,
    "へ": 0.78,
    "つ": 0.80,
    "り": 0.78,
    "い": 0.80,
    "こ": 0.78,
    "て": 0.72,
    "に": 0.80,
    "と": 0.80,
    "う": 0.80,
    "か": 0.82,
    "る": 0.80,
    "を": 0.80,
    # ひらがな: 標準〜やや大きめ
    "あ": 0.85,
    "お": 0.85,
    "き": 0.85,
    "け": 0.82,
    "さ": 0.82,
    "す": 0.82,
    "せ": 0.85,
    "そ": 0.82,
    "た": 0.85,
    "ち": 0.82,
    "な": 0.85,
    "ぬ": 0.85,
    "ね": 0.85,
    "は": 0.85,
    "ひ": 0.78,
    "ふ": 0.85,
    "ほ": 0.85,
    "ま": 0.85,
    "み": 0.82,
    "む": 0.85,
    "め": 0.82,
    "も": 0.82,
    "や": 0.85,
    "ゆ": 0.85,
    "よ": 0.82,
    "ら": 0.82,
    "れ": 0.82,
    "ろ": 0.80,
    "わ": 0.82,
    "ん": 0.80,
    "え": 0.82,
    # 濁音ひらがな
    "が": 0.88,
    "ぎ": 0.88,
    "ぐ": 0.85,
    "げ": 0.85,
    "ご": 0.85,
    "ざ": 0.88,
    "じ": 0.85,
    "ず": 0.85,
    "ぜ": 0.88,
    "ぞ": 0.85,
    "だ": 0.88,
    "ぢ": 0.85,
    "づ": 0.85,
    "で": 0.85,
    "ど": 0.88,
    "ば": 0.88,
    "び": 0.85,
    "ぶ": 0.88,
    "べ": 0.85,
    "ぼ": 0.88,
    "ぱ": 0.88,
    "ぴ": 0.85,
    "ぷ": 0.85,
    "ぺ": 0.85,
    "ぽ": 0.88,
}

# 種別デフォルト倍率（個別テーブルに無い字に適用）
_KANJI_SCALE = 1.0
_HIRAGANA_SCALE = 0.85
_KATAKANA_SCALE = 0.85
_HALFWIDTH_SCALE = 0.8
_SMALL_KANA_SCALE = 0.55
_SMALL_PUNCT_SCALE = 0.35


def char_type_scale(ch: str) -> float:
    """字種に応じたサイズ倍率を返す（個別調整テーブル優先）。

    優先順位: 小書き → 句読点 → 個別テーブル → 字種デフォルト。

    Args:
        ch: 対象文字（1文字）。

    Returns:
        サイズ倍率。漢字 1.0 / かな 0.85 / 半角 0.8 / 小書き 0.55 / 句読点 0.35。
    """
    if ch in _SMALL_KANA:
        return _SMALL_KANA_SCALE
    if ch in _SMALL_PUNCT:
        return _SMALL_PUNCT_SCALE
    if ch in _KANA_SIZE_OVERRIDES:
        return _KANA_SIZE_OVERRIDES[ch]
    cp = ord(ch)
    if 0x3040 <= cp <= 0x309F:
        return _HIRAGANA_SCALE
    if 0x30A0 <= cp <= 0x30FF:
        return _KATAKANA_SCALE
    if is_halfwidth(ch):
        return _HALFWIDTH_SCALE
    return _KANJI_SCALE


# --- 密度補正（複雑度マップに依存、lazy ロード）----------------------------

_complexity_map: dict[str, dict] | None = None
_complexity_map_loaded = False


def _load_complexity_map() -> dict[str, dict]:
    """複雑度マップを初回呼び出し時に1度だけロードしキャッシュする。

    マップ未生成（ファイル不在）や破損時も例外を投げず空 dict を返し、
    密度補正を 1.0 フォールバックさせる（マップが無くても動く堅牢性）。
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
            "char complexity map not loaded (%s); density_scale falls back to 1.0",
            _COMPLEXITY_MAP_PATH,
        )
        _complexity_map = {}
    return _complexity_map


def density_scale(ch: str) -> float:
    """複雑度に応じた連続サイズ補正を返す（マップ無し字は 1.0）。

    線形写像 `DENSITY_SCALE_MIN + slope*complexity` を `[MIN, MAX]` でクランプ。
    複雑度 0.5 付近で約 1.0、簡単字ほど小さく・複雑字ほど大きくなる。

    Args:
        ch: 対象文字（1文字）。

    Returns:
        密度補正倍率。clamp(0.88, 1.12)。マップに無い文字は 1.0。
    """
    entry = _load_complexity_map().get(ch)
    if not isinstance(entry, dict) or "complexity" not in entry:
        return 1.0
    complexity = float(entry["complexity"])
    scaled = DENSITY_SCALE_MIN + _DENSITY_SLOPE * complexity
    return min(DENSITY_SCALE_MAX, max(DENSITY_SCALE_MIN, scaled))


def effective_char_scale(ch: str) -> float:
    """種別スケール × 密度補正の最終サイズ倍率を返す。

    typesetter/renderer はこの単一関数を呼ぶことで二重管理を解消する。

    Args:
        ch: 対象文字（1文字）。

    Returns:
        `char_type_scale(ch) * density_scale(ch)`。
    """
    return char_type_scale(ch) * density_scale(ch)
