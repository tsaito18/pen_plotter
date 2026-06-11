"""KanjiVG 全字の複雑度マップ (data/char_complexity.json) を生成する。

各字の画数と正規化字形の ink 長を集計し、全字分布で robust 正規化してから
複雑度を合成する。char_metrics.py の純粋関数を再利用しロジックを二重化しない。

出力 JSON 形式::

    {"口": {"strokes": 3, "ink_len": 64.2, "complexity": 0.31}, ...}

正規化に使った percentile 境界などの統計は `__meta__` キーと stderr に出して
後から検証できるようにする（density_scale は文字キーのみ引くのでメタは無害）。

実行例::

    uv run python scripts/compute_char_complexity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from src.layout.char_metrics import (
    COMPLEXITY_PCT_HIGH,
    COMPLEXITY_PCT_LOW,
    COMPLEXITY_WEIGHT_INK,
    COMPLEXITY_WEIGHT_STROKE,
    _percentile,
    char_ink_length,
    compute_complexity,
    normalize_robust,
)

# src/layout/char_metrics.py から見たプロジェクトルート（このスクリプトと同基準）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STROKES_DIR = _PROJECT_ROOT / "data" / "strokes"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "char_complexity.json"


@dataclass
class CharRaw:
    """1字分の生メトリック（正規化前）。"""

    char: str
    strokes: int
    ink_len: float


def _strokes_to_point_lists(strokes: list) -> list[list[dict]]:
    """JSON の strokes（点辞書のリストのリスト）をそのまま返す薄いアダプタ。

    char_ink_length は辞書点を受け付けるため変換不要だが、空ストローク混入時に
    画数の数え方を一箇所に固定するため経由させる。
    """
    return [stroke for stroke in strokes if stroke]


def _collect_raw_metrics(strokes_dir: Path) -> list[CharRaw]:
    """data/strokes 配下の全 JSON を走査し生メトリックを集める。

    Args:
        strokes_dir: 1字1ディレクトリ・`{char}_0.json` を含むルート。

    Returns:
        各字の (char, 画数, ink長)。読めない/空の JSON はスキップ。
    """
    results: list[CharRaw] = []
    # macOS 由来の ._ AppleDouble や隠しファイルを除外しつつ全 json を走査
    for json_path in sorted(strokes_dir.rglob("*.json")):
        if json_path.name.startswith("._"):
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        char = data.get("character")
        raw_strokes = data.get("strokes")
        if not char or not raw_strokes:
            continue
        non_empty = _strokes_to_point_lists(raw_strokes)
        if not non_empty:
            continue
        results.append(
            CharRaw(
                char=char,
                strokes=len(non_empty),
                ink_len=char_ink_length(non_empty),
            )
        )
    return results


def build_complexity_map(strokes_dir: Path) -> tuple[dict, dict]:
    """全字を走査し複雑度マップと正規化統計を構築する。

    Args:
        strokes_dir: data/strokes ルート。

    Returns:
        (char→{strokes,ink_len,complexity} の dict, 統計 dict) のタプル。
    """
    raw = _collect_raw_metrics(strokes_dir)
    if not raw:
        raise RuntimeError(f"no stroke JSON found under {strokes_dir}")

    stroke_counts = [float(r.strokes) for r in raw]
    ink_lens = [r.ink_len for r in raw]
    stroke_norms = normalize_robust(stroke_counts)
    ink_norms = normalize_robust(ink_lens)

    char_map: dict = {}
    for r, s_norm, i_norm in zip(raw, stroke_norms, ink_norms):
        complexity = compute_complexity(
            stroke_norm=s_norm,
            ink_norm=i_norm,
            w_stroke=COMPLEXITY_WEIGHT_STROKE,
            w_ink=COMPLEXITY_WEIGHT_INK,
        )
        # 後勝ちで重複字（同一字の別ファイル）を1エントリに集約
        char_map[r.char] = {
            "strokes": r.strokes,
            "ink_len": round(r.ink_len, 4),
            "complexity": round(complexity, 4),
        }

    sorted_strokes = sorted(stroke_counts)
    sorted_ink = sorted(ink_lens)
    stats = {
        "char_count": len(char_map),
        "weights": {"stroke": COMPLEXITY_WEIGHT_STROKE, "ink": COMPLEXITY_WEIGHT_INK},
        "percentiles": {"low": COMPLEXITY_PCT_LOW, "high": COMPLEXITY_PCT_HIGH},
        "stroke_count": {
            "low_bound": _percentile(sorted_strokes, COMPLEXITY_PCT_LOW),
            "high_bound": _percentile(sorted_strokes, COMPLEXITY_PCT_HIGH),
            "min": sorted_strokes[0],
            "max": sorted_strokes[-1],
        },
        "ink_len": {
            "low_bound": round(_percentile(sorted_ink, COMPLEXITY_PCT_LOW), 4),
            "high_bound": round(_percentile(sorted_ink, COMPLEXITY_PCT_HIGH), 4),
            "min": round(sorted_ink[0], 4),
            "max": round(sorted_ink[-1], 4),
        },
    }
    return char_map, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strokes-dir",
        type=Path,
        default=_DEFAULT_STROKES_DIR,
        help="KanjiVG ストロークJSONのルート (default: data/strokes)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="出力JSONパス (default: data/char_complexity.json)",
    )
    args = parser.parse_args()

    char_map, stats = build_complexity_map(args.strokes_dir)

    # 統計は文字キーと衝突しない __meta__ に格納（density_scale は文字キーのみ引く）
    output = {"__meta__": stats, **char_map}
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=None),
        encoding="utf-8",
    )

    print(f"wrote {args.output} ({stats['char_count']} chars)", file=sys.stderr)
    print(f"normalization stats: {json.dumps(stats, ensure_ascii=False)}", file=sys.stderr)


if __name__ == "__main__":
    main()
