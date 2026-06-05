"""KanjiVG の ``kvg:type`` を筆法（とめ・はね・払い）へ分類し、ストローク終端を
その筆法に応じて加工するモジュール。

ペンプロッタは単線でペン太さを出せないため、毛筆/ペン字らしさは**軌跡形状**で
表現する。本モジュールは KanjiVG 参照経路で生成された配置後ストローク（mm 座標）
の終端を、払いなら接線方向に流し、はねなら逆向きの小フックを付け、とめは止める
（初期は無加工）ことで筆遣いを与える。

純粋関数のみ（numpy 依存、torch 不要）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 筆法カテゴリ
TOME = "tome"
HANE = "hane"
HARAI = "harai"
NONE = "none"

# CJK Strokes (kvg:type) → 筆法。終筆の性質で分類する:
#   鉤(hook)を含む          → はね
#   撇/捺/提（流して抜ける） → 払い
#   横/縦/点/折れ（鉤なし）  → とめ
# 実 KanjiVG の値分布は未検証のため初期マップは暫定。再変換後に集計して調整する。
KVG_TYPE_TO_FINISH: dict[str, str] = {
    # 払い（撇・捺・提）
    "㇒": HARAI,  # P   撇 / 左払い
    "㇓": HARAI,  # SP  竖撇
    "㇏": HARAI,  # N   捺 / 右払い
    "㇇": HARAI,  # HP  横撇（終端が左払い）
    "㇋": HARAI,  # HZZP 横折折撇
    "㇀": HARAI,  # T   提 / 右上はらい上げ
    "㇊": HARAI,  # HZT 横折提
    "㇙": HARAI,  # ST  竖提
    # はね（鉤 hook を含む）
    "㇁": HANE,  # WG    弯钩
    "㇂": HANE,  # XG    斜钩
    "㇃": HANE,  # BXG   扁斜钩
    "㇆": HANE,  # HZG   横折钩
    "㇈": HANE,  # HZWG  横折弯钩
    "㇉": HANE,  # SZWG  竖折弯钩
    "㇌": HANE,  # HPWG  横撇弯钩
    "㇖": HANE,  # HG    横钩
    "㇚": HANE,  # SG    竖钩 / 縦はね
    "㇟": HANE,  # SWG   竖弯钩
    "㇠": HANE,  # HXWG  横斜弯钩
    "㇡": HANE,  # HZZZG 横折折折钩
    "㇢": HANE,  # PG    撇钩
    # とめ（横・縦・点・折れ、鉤なし）
    "㇐": TOME,  # H   横
    "㇑": TOME,  # S   竖
    "㇔": TOME,  # D   点
    "㇕": TOME,  # HZ  横折
    "㇗": TOME,  # SZ  竖折
    "㇅": TOME,  # HZZ 横折折
    "㇍": TOME,  # HZW 横折弯
    "㇎": TOME,  # HZZZ 横折折折
    "㇄": TOME,  # SW  竖弯（鉤なしの折れ。実データで頻出するため明示マップ）
    "㇘": TOME,  # SWZ 竖弯
    "㇞": TOME,  # SZZ 竖折折
    "㇛": TOME,  # PD  撇点（終端が点）
    "㇜": TOME,  # PZ  撇折（終端が折れ）
    "㇝": TOME,  # TN
}


def classify_finish(kvg_type: str) -> str:
    """単一の ``kvg:type`` 文字列を筆法カテゴリへ分類する。

    variant 表記（``"㇒/a"`` のスラッシュ付き、``"㇑a"`` の接尾辞文字付き）は
    base となる CJK Stroke 1 文字に正規化してから引く。未知・空文字列は
    :data:`NONE`（無加工）を返す。

    Args:
        kvg_type: KanjiVG の raw ``kvg:type`` 文字列。

    Returns:
        :data:`TOME` / :data:`HANE` / :data:`HARAI` / :data:`NONE` のいずれか。
    """
    if not kvg_type:
        return NONE
    base = kvg_type.split("/")[0].strip()
    if base and base[0] in KVG_TYPE_TO_FINISH:
        # "㇑a" のような接尾辞付きは先頭の CJK Stroke 1 文字で引く
        return KVG_TYPE_TO_FINISH[base[0]]
    return KVG_TYPE_TO_FINISH.get(base, NONE)


def classify_finishes(kvg_types: list[str]) -> list[str]:
    """``kvg:type`` のリストを筆法カテゴリのリストへ一括変換する。"""
    return [classify_finish(t) for t in kvg_types]


# 軌跡からの筆法推定（かな等 kvg:type が無い字向け）の閾値。Y-UP 前提。
_INFER_HANE_UP = 0.45  # 終端接線の上向き成分がこれ超で「跳ね上げ」=はね
_INFER_HARAI_TURN_DEG = 35.0  # 終端の曲がりがこれ未満で「滑らか」
_INFER_HARAI_DIAG = 0.40  # 斜め度(min/max成分比)がこれ超で「斜めに流れる」


def _unit(v: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else None


def infer_finish_from_stroke(stroke: np.ndarray) -> str:
    """ストロークの軌跡形状から筆法（とめ/はね/払い）を推定する。

    KanjiVG の ``kvg:type`` を持たない字（ひらがな・カタカナ）向けのロジック判定。
    終端の接線・曲がり方から、跳ね上げ＝はね、斜めに滑らかに流れる＝払い、
    それ以外＝とめ とみなす。Y-UP 座標前提（上が +Y）。

    Args:
        stroke: ``(N, 2)`` の点列（mm 座標、Y-UP）。

    Returns:
        :data:`HANE` / :data:`HARAI` / :data:`TOME` のいずれか。判定不能は TOME。
    """
    pts = np.asarray(stroke, dtype=float)
    n = len(pts)
    if n < 3:
        return TOME
    v_end = _unit(pts[-1] - pts[-3])
    if v_end is None:
        return TOME
    # 終端より少し手前の進行方向（曲がり量の算出用）
    back = min(6, n - 1)
    v_pre = _unit(pts[-3] - pts[-1 - back])
    if v_pre is None:
        v_pre = v_end
    cos_turn = float(np.clip(np.dot(v_pre, v_end), -1.0, 1.0))
    turn_deg = float(np.degrees(np.arccos(cos_turn)))

    # はね: 終端が上向きに跳ねる（縦画下端からの跳ね上げ等）
    if v_end[1] > _INFER_HANE_UP:
        return HANE
    # 払い: 斜め下方向へ滑らかに流れて抜ける
    diag = min(abs(v_end[0]), abs(v_end[1])) / max(abs(v_end[0]), abs(v_end[1]), 1e-9)
    if turn_deg < _INFER_HARAI_TURN_DEG and v_end[1] < -0.1 and diag > _INFER_HARAI_DIAG:
        return HARAI
    return TOME


def infer_finishes(strokes: list[np.ndarray]) -> list[str]:
    """各ストロークの筆法を軌跡から推定する（かな等向け）。"""
    return [infer_finish_from_stroke(s) for s in strokes]


def arc_length_from_end(points: np.ndarray) -> np.ndarray:
    """各点の「終端までの弧長(mm)」配列を返す。

    終端（末尾点）が ``0``、始点が全長。終端へ向かって単調減少する。Zリフトを
    点数ではなく実距離(mm)で設計し、文字サイズ非依存にするための基礎量。

    Args:
        points: ``(N, 2)`` の点列（mm 座標）。

    Returns:
        長さ ``N`` の弧長配列（終端 0）。``N < 2`` なら ``[0.0]*N``。
    """
    n = len(points)
    if n < 2:
        return np.zeros(n, dtype=float)
    diffs = np.diff(points, axis=0)
    seg = np.sqrt((diffs**2).sum(axis=1))
    cum_from_start = np.concatenate([[0.0], np.cumsum(seg)])
    return cum_from_start[-1] - cum_from_start


def contact_profile(
    finish: str,
    arc_from_end: np.ndarray,
    lift_length: float,
    harai_min: float = 0.0,
    hane_min: float = 0.0,
) -> np.ndarray:
    """各点の接触率 ``contact ∈ [0, 1]`` 列を距離(mm)ベースで返す。

    実機の終端Zリフト（芯の接触圧抜き）とプレビューの線幅を同一ソースから導く
    無次元プロファイル。``1.0`` が完全接触（太く/濃く）、``0.0`` が完全リフト
    （細く/消える）。終端から ``lift_length`` mm 以内をリフト区間とし、終端で 0、
    境界で 1.0 へ。点数でなく**距離**で決めるため文字サイズに依らず同じ抜けになる。

    払い＝線形、はね＝二乗カーブ（終端付近で急峻）。とめ・none は全点 ``1.0``。

    Args:
        finish: :data:`TOME` / :data:`HANE` / :data:`HARAI` / :data:`NONE`。
        arc_from_end: 各点の終端までの弧長(mm)。:func:`arc_length_from_end` の出力。
        lift_length: 終端リフト区間の長さ(mm)。``<=0`` でリフトなし。
        harai_min: 払い終端の最小接触率（``0`` で完全リフト）。
        hane_min: はね終端の最小接触率（``0`` で完全リフト）。

    Returns:
        ``arc_from_end`` と同長の接触率配列。先頭1.0寄り、終端へ単調非増加。
    """
    arc = np.asarray(arc_from_end, dtype=float)
    contact = np.ones(len(arc), dtype=float)
    if len(arc) < 2 or finish not in (HARAI, HANE) or lift_length <= 0:
        return contact
    # 終端からの正規化位置 t: 終端0 → 境界1.0（境界より手前は1.0で頭打ち）
    t = np.clip(arc / lift_length, 0.0, 1.0)
    if finish == HARAI:
        contact = harai_min + (1.0 - harai_min) * t
    else:  # HANE: 終端付近で急峻
        contact = hane_min + (1.0 - hane_min) * t**2
    return contact


@dataclass
class FinishingConfig:
    """終端加工のパラメータ設定。

    長さ系の比率はすべて ``scale``（配置時の ``font_size`` mm）に対する割合で、
    非一様スケールの影響を受けない mm 量として加工に使う。

    Attributes:
        enabled: ``False`` で :func:`apply_finishing` を完全バイパス（A/B・回帰用）。
        harai_ext_ratio: 払いの延長長さ / ``scale``。
        harai_points: 払いで末尾に追加する点数。
        hane_hook_ratio: はねの延長長さ / ``scale``。
        hane_points: はねで末尾に追加する点数。
        tangent_window: 終端接線の算出に使う終端点数。
        tome_setback_ratio: とめ打ち込み量 / ``scale``。初期 0 は恒等。
    """

    enabled: bool = True
    harai_ext_ratio: float = 0.15
    harai_points: int = 5
    hane_hook_ratio: float = 0.12
    hane_points: int = 4
    tangent_window: int = 3
    tome_setback_ratio: float = 0.0


def _terminal_tangent(stroke: np.ndarray, k: int) -> np.ndarray | None:
    """終端の進行方向単位ベクトルを返す。

    末尾点から ``k`` 点さかのぼった点との差分を正規化する。ゼロ長（重複点で
    方向が定義できない）の場合は ``None`` を返す。

    Args:
        stroke: ``(N, 2)`` の点列。
        k: 接線算出に使う終端点数（さかのぼり量）。

    Returns:
        単位ベクトル ``(2,)``。ノルムが ``1e-9`` 未満なら ``None``。
    """
    n = len(stroke)
    if n < 2:
        return None
    v = stroke[-1] - stroke[-1 - min(k, n - 1)]
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return None
    return v / norm


def apply_harai(stroke: np.ndarray, scale: float, config: FinishingConfig) -> np.ndarray:
    """払い: 終端を接線方向へ細く流すように点を延長する。

    終端の進行方向へ ``harai_points`` 点を等間隔（``i = 1..n`` で
    ``ext_len * i / n``、``ext_len = scale * harai_ext_ratio``）に生成し末尾へ
    追加する。接線が定義できない場合は ``stroke`` をそのまま返す。

    Args:
        stroke: ``(N, 2)`` の配置後点列。
        scale: 配置時の ``font_size``（mm）。延長長さの基準。
        config: 加工パラメータ。

    Returns:
        ``(N + harai_points, 2)`` の点列（恒等時は入力そのまま）。
    """
    tangent = _terminal_tangent(stroke, config.tangent_window)
    if tangent is None:
        return stroke
    n = config.harai_points
    ext_len = scale * config.harai_ext_ratio
    end = stroke[-1]
    ext = np.array(
        [end + tangent * (ext_len * i / n) for i in range(1, n + 1)],
        dtype=float,
    )
    return np.vstack([stroke, ext])


def apply_hane(stroke: np.ndarray, scale: float, config: FinishingConfig) -> np.ndarray:
    """はね: 既存フック方向（終端接線）へ短く延長して跳ねを強調する。

    KanjiVG の鈎（はね）ストロークは**既にフック形状を経路に含む**ため、終端接線は
    そのままフックの向き（縦画なら左上など）を指す。ここで接線を回転させると二重
    フックになって逆向きに飛ぶため、回転はせず終端接線方向へ ``hook_len = scale *
    hane_hook_ratio`` だけ ``hane_points`` 点を延長する。実機ではこの延長区間で Z を
    持ち上げ（:func:`contact_profile`）、跳ねが細く抜ける。接線が定義できない場合は
    ``stroke`` をそのまま返す。

    Args:
        stroke: ``(N, 2)`` の配置後点列。
        scale: 配置時の ``font_size``（mm）。延長長さの基準。
        config: 加工パラメータ。

    Returns:
        ``(N + hane_points, 2)`` の点列（恒等時は入力そのまま）。
    """
    tangent = _terminal_tangent(stroke, config.tangent_window)
    if tangent is None:
        return stroke
    n = config.hane_points
    hook_len = scale * config.hane_hook_ratio
    end = stroke[-1]
    ext = np.array(
        [end + tangent * (hook_len * i / n) for i in range(1, n + 1)],
        dtype=float,
    )
    return np.vstack([stroke, ext])


def apply_tome(stroke: np.ndarray, scale: float, config: FinishingConfig) -> np.ndarray:
    """とめ: 終端を進行方向の逆へ微小に打ち込む。

    ``tome_setback_ratio <= 0`` の場合は恒等（入力そのまま）。``> 0`` の場合は
    終端を進行方向の逆へ ``scale * tome_setback_ratio`` だけ戻した点を 1 つ追加する。

    Args:
        stroke: ``(N, 2)`` の配置後点列。
        scale: 配置時の ``font_size``（mm）。打ち込み量の基準。
        config: 加工パラメータ。

    Returns:
        恒等時は入力そのまま、加工時は ``(N + 1, 2)`` の点列。
    """
    if config.tome_setback_ratio <= 0:
        return stroke
    tangent = _terminal_tangent(stroke, config.tangent_window)
    if tangent is None:
        return stroke
    setback = scale * config.tome_setback_ratio
    pt = stroke[-1] - tangent * setback
    return np.vstack([stroke, pt[np.newaxis, :]])


def apply_finishing(
    strokes: list[np.ndarray],
    finishes: list[str],
    scale: float,
    config: FinishingConfig | None = None,
) -> list[np.ndarray]:
    """各ストロークに index 対応の筆法で終端加工をディスパッチする。

    ``config.enabled`` が ``False`` なら ``strokes`` をそのまま返す。``finishes``
    が ``strokes`` より短い場合、不足分は :data:`NONE`（恒等）扱い。点数が 2 未満の
    ストロークは加工せず恒等。

    Args:
        strokes: 配置後ストローク ``(N, 2)`` のリスト。
        finishes: 各ストロークの筆法カテゴリ（:func:`classify_finishes` の出力）。
        scale: 配置時の ``font_size``（mm）。
        config: 加工パラメータ。``None`` なら既定の :class:`FinishingConfig`。

    Returns:
        加工後ストロークのリスト（入力と同数・同順）。
    """
    if config is None:
        config = FinishingConfig()
    if not config.enabled:
        return strokes
    out: list[np.ndarray] = []
    for i, stroke in enumerate(strokes):
        finish = finishes[i] if i < len(finishes) else NONE
        if len(stroke) < 2:
            out.append(stroke)
        elif finish == HARAI:
            out.append(apply_harai(stroke, scale, config))
        elif finish == HANE:
            out.append(apply_hane(stroke, scale, config))
        elif finish == TOME:
            out.append(apply_tome(stroke, scale, config))
        else:
            out.append(stroke)
    return out
