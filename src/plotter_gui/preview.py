"""G-code を 2D プレビュー用ストローク列に分解する純粋関数群。

Tkinter には依存しない。matplotlib の Axes を引数に取る描画関数と、
G-code テキスト/ファイルを Stroke リストに変換する関数のみを提供する。

GUI (`widgets/file_picker.py`) からは:
    strokes = parse_gcode(path)
    render_strokes(canvas_axes, strokes)
の 2 段階で利用する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# X / Y / Z をそれぞれ独立に抽出する。GRBL は同一行に複数軸が混じるため、
# 1 個の正規表現で全部取り出すよりも軸単位の re.search で扱うのが素直。
# 値は浮動小数 (符号付き) を許容する。
_X_RE = re.compile(r"[Xx](-?\d+(?:\.\d+)?)")
_Y_RE = re.compile(r"[Yy](-?\d+(?:\.\d+)?)")
_Z_RE = re.compile(r"[Zz](-?\d+(?:\.\d+)?)")

# 行頭が G0/G1 で始まるかを判定。"G1G90 Z..." のような連結記法も
# 先頭 G1/G0 として認識できるようにする (これは Z 軸行で別途除外)。
_GCMD_RE = re.compile(r"^\s*G(?P<num>\d+)")


@dataclass(frozen=True)
class Stroke:
    """1 本のストローク (連続した描画線)。

    points は ``[(x, y), ...]`` の形でペン下げ中の通過点列を保持する。
    起点 (G0 で打たれたペンダウン位置) も先頭に含めて、render 側で
    そのまま線分連結すれば見た目どおりに描画できる構造にする。
    """

    points: list[tuple[float, float]]


def parse_gcode(source: str | Path) -> list[Stroke]:
    """G-code テキスト or ファイルパスを解析してストローク列に分解する。

    Args:
        source: G-code 文字列、または .gcode/.nc/.txt ファイルへのパス。

    Returns:
        ストローク列。点が 1 個以下のストロークは除外する
        (= 移動だけで描画しない G0 単独行は無視する)。

    分解規則:
        - ``G0 X.. Y..`` → 新しいストロークの開始位置 (起点として保持)
        - ``G1 X.. Y..`` → 現在のストロークの続き
        - Z 軸を含む行 (例: ``G1G90 Z3.5 F5000``) → ペン制御。座標として無視
        - ``;`` 始まりのコメント / ``$H`` / ``G4`` / ``G92`` / ``G90`` → スキップ
        - X や Y のどちらか一方しか出ない行は前回値を継承
          (GRBL の標準モーダル動作)。
    """
    text = _read_source(source)

    strokes: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] | None = None
    # GRBL のモーダル仕様: 軸の値が省略された行は前回値を保持する。
    last_x: float | None = None
    last_y: float | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        # ホーミング ($H) / ドウェル (G4) / 座標系設定 (G92) / 絶対モード (G90)
        # は座標を持たない。かつ G92 は X/Y を含むため、明示的に除外する。
        if line.startswith("$") or line.startswith("G92") or line.startswith("G90"):
            continue
        if line.startswith("G4"):
            continue

        # Z を含む行はペン制御コマンド。X/Y が同時に書かれることは実機の
        # G-code には無いが、混在しても Z 行は座標として扱わない方針。
        if _Z_RE.search(line):
            continue

        m = _GCMD_RE.match(line)
        if m is None:
            # G0/G1 以外の M3/M5 等が混じった場合は無視 (xDraw では使わないが安全側)
            continue
        gnum = int(m.group("num"))
        if gnum not in (0, 1):
            continue

        x_match = _X_RE.search(line)
        y_match = _Y_RE.search(line)
        # X/Y が両方とも無い行 (例: 単独 F のみ) は座標更新なし → スキップ。
        if x_match is None and y_match is None:
            continue

        x = float(x_match.group(1)) if x_match else last_x
        y = float(y_match.group(1)) if y_match else last_y
        # モーダル継承の最初に X/Y がまだ未確定 (= None) のまま到達したら、
        # その行は座標として成立しないので無視する。
        if x is None or y is None:
            continue

        if gnum == 0:
            # 新規ストローク開始。直前のストロークが点 1 個以下なら破棄。
            if current is not None and len(current) >= 2:
                strokes.append(current)
            current = [(x, y)]
        else:  # gnum == 1
            if current is None:
                # G0 を経由せず G1 から始まる異常系。座標未確定のため
                # その行を起点にして暗黙のストロークを開始する。
                current = [(x, y)]
            else:
                current.append((x, y))

        last_x = x
        last_y = y

    # 末尾ストロークの flush (点 2 個以上の場合のみ保持)
    if current is not None and len(current) >= 2:
        strokes.append(current)

    return [Stroke(points=pts) for pts in strokes]


def render_strokes(ax: "Axes", strokes: list[Stroke]) -> None:
    """matplotlib Axes へストロークを描画する。

    A4 サイズ枠 (0,0)-(210,297) を境界に表示し、ストロークは黒線で描く。
    Y-UP 座標系 (matplotlib デフォルト) でそのまま描画するため
    ``invert_yaxis()`` は呼ばない。

    Args:
        ax: matplotlib の Axes オブジェクト。
        strokes: 描画対象ストローク列 (空リストの場合は枠だけ表示)。
    """
    ax.clear()
    # A4 紙の外形を薄いグレーで枠表示。座標感覚をユーザに与えるため、
    # ストロークが空でも常に描画する。
    ax.plot(
        [0, 210, 210, 0, 0],
        [0, 0, 297, 297, 0],
        color="#cccccc",
        linewidth=0.8,
    )

    for stroke in strokes:
        if len(stroke.points) < 2:
            continue
        xs = [p[0] for p in stroke.points]
        ys = [p[1] for p in stroke.points]
        ax.plot(xs, ys, color="black", linewidth=0.6)

    ax.set_xlim(-5, 215)
    ax.set_ylim(-5, 302)
    ax.set_aspect("equal", adjustable="box")
    # 紙面感を出すため tick はそのまま、grid は描かない。


def _read_source(source: str | Path) -> str:
    """``source`` が Path or 既存ファイルなら読み込み、文字列ならそのまま返す。

    str を渡されたとき「ファイルパスかも」と早合点すると、改行を含む
    G-code 文字列に対しても OS 依存のチェックが走って遅くなる。
    そのため明示的に Path 型または「改行を含まない既存ファイル名」だけを
    ファイルとみなす。
    """
    if isinstance(source, Path):
        return source.read_text()
    # str: 改行を含むなら確実に G-code 本文。
    if "\n" in source:
        return source
    # 改行なしの短い文字列: ファイルパスとして開けるか試す。
    # 実用上 G-code 1 行だけが渡るケースは無いが、念のため open 失敗時は
    # その文字列を本文として扱う。
    try:
        p = Path(source)
        if p.is_file():
            return p.read_text()
    except OSError:
        pass
    return source
