"""LaTeX 数式 → matplotlib レンダリング → スケルトン化 → ストローク変換。"""

from __future__ import annotations

import io
import logging
import re
from functools import lru_cache

import matplotlib
import numpy as np
from skimage import filters, morphology

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

Stroke = np.ndarray  # (N, 2) float64

_RENDER_DPI = 300
_FONT_SIZE_PT = 28
_PAD_INCHES = 0.05
_MIN_STROKE_PX = 2
_MIN_STROKE_MM = 0.5  # 0.5mm 未満のストロークは除去
_MIN_STROKE_UNIT = 0.02  # unit square の 2% 未満は除去（per-char 用）
_MATH_LIFT_FRACTION = 0.2  # 数式を bbox 高さの何割上にずらすか（行内でやや上寄せ）
_MATH_HEIGHT_STRETCH = 1.1  # 縦方向の引き伸ばし（matplotlib の分数が横に潰れて見えるのを補正）
# 下端 = y0 + h_mm*(0.5 + LIFT - STRETCH/2) を正に保ち、分母が下の行へはみ出さないこと


@lru_cache(maxsize=64)
def _render_formula_unit_strokes(
    math_src: str,
) -> tuple[float, tuple[np.ndarray, ...]] | None:
    """数式全体を unit square [0,1]×[0,1] Y-UP ストロークに変換（キャッシュ付き）。

    Returns:
        (aspect, strokes) — aspect = 描画画像の幅/高さ（mm 変換時の歪み補正用）。
        墨が無い等で失敗したら None。
    """
    gray = _render_to_gray(math_src)
    if gray is None:
        return None

    binary = _binarize(gray)
    if not binary.any():
        return None

    binary = _crop(binary)
    skeleton = morphology.skeletonize(binary)
    pixel_strokes = _trace_skeleton(skeleton)

    h_px, w_px = skeleton.shape
    aspect = w_px / max(h_px, 1)
    result: list[np.ndarray] = []
    for ps in pixel_strokes:
        if len(ps) < _MIN_STROKE_PX:
            continue
        col = ps[:, 0] / max(w_px - 1, 1)
        row = 1.0 - ps[:, 1] / max(h_px - 1, 1)  # Y-UP
        stroke = np.stack([col, row], axis=1).astype(np.float64)
        diffs = np.diff(stroke, axis=0)
        length = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        if length >= _MIN_STROKE_UNIT:
            result.append(stroke)

    return (aspect, tuple(result)) if result else None


def formula_draw_width_mm(math_src: str, h_mm: float) -> float:
    """数式を高さ h_mm で描いたときの実際の描画幅(mm)。式番号の配置に使う。

    render_latex_to_strokes と同じ draw_w = h_mm * aspect を返す。
    """
    rendered = _render_formula_unit_strokes(math_src)
    if not rendered:
        return 0.0
    aspect, _ = rendered
    return h_mm * aspect


@lru_cache(maxsize=128)
def _baseline_frac_from_top(math_src: str) -> float:
    """数式画像で、ベースラインが上端から何割の位置かを返す（TextPath の baseline=0 基準）。

    F→≈1.0（ベースラインに座る）、A_0→≈0.8（0 が下付きで下に出る）、分数→≈0.6。
    """
    from matplotlib.textpath import TextPath

    safe = re.sub(r"(?<!\\)%", r"\\%", math_src)
    try:
        with plt.rc_context({"mathtext.fontset": "cm"}):
            tp = TextPath((0, 0), f"${safe}$", size=_FONT_SIZE_PT)
        v = tp.vertices
        ymin, ymax = float(v[:, 1].min()), float(v[:, 1].max())
        span = ymax - ymin
        return ymax / span if span > 0 else 0.5
    except Exception:
        return 0.5


def render_latex_to_strokes(
    math_src: str,
    bbox_mm: tuple[float, float, float, float],
    align: str = "center",
) -> list[Stroke]:
    """LaTeX 数式をレンダリング → スケルトン化 → mm 座標ストロークに変換。

    Args:
        math_src: LaTeX ソース（$ なし。例: r'\\frac{F}{A_0}'）
        bbox_mm: (x_left_mm, y_bbox_mm, width_mm, height_mm) — Y-UP
        align: "center"=bbox 中心に上寄せ配置（ブロック数式用）。
            "baseline"=bbox_mm[1] を本文ベースライン y とみなし、数式のベースラインを
            そこに揃える（インライン数式用。LIFT/STRETCH を無効化し歪みなし）。

    Returns:
        ストローク列（各要素は (N,2) float64, mm, Y-UP）
    """
    rendered = _render_formula_unit_strokes(math_src)
    if not rendered:
        return []
    aspect, unit_strokes = rendered

    x0, y0, w_mm, h_mm = bbox_mm
    if align == "baseline":
        # インライン: 等倍・歪みなしで、数式ベースラインを本文ベースライン(y0)へ揃える。
        draw_w = h_mm * aspect
        draw_h = h_mm
        x_left = x0
        bf = _baseline_frac_from_top(math_src)
        y_bottom = y0 - (1.0 - bf) * draw_h  # baseline(Y-UP) = y_bottom + (1-bf)*draw_h = y0
    else:
        # ブロック: 高さを信頼し aspect から幅を算出（中心に上寄せ、分数の潰れを軽く補正）。
        # 幅基準にすると \qquad(番号) の空白がアスペクトを水増しし数式が縦に潰れて小さくなる。
        draw_w = h_mm * aspect
        draw_h = h_mm * _MATH_HEIGHT_STRETCH
        cx = x0 + w_mm / 2
        cy = y0 + h_mm / 2 + h_mm * _MATH_LIFT_FRACTION
        x_left = cx - draw_w / 2
        y_bottom = cy - draw_h / 2

    result: list[Stroke] = []
    for s in unit_strokes:
        x_out = x_left + s[:, 0] * draw_w
        y_out = y_bottom + s[:, 1] * draw_h
        stroke = np.stack([x_out, y_out], axis=1).astype(np.float64)
        diffs = np.diff(stroke, axis=0)
        length = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        if length >= _MIN_STROKE_MM:
            result.append(stroke)

    return result


def _render_to_gray(math_src: str) -> np.ndarray | None:
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed; math skeletonize unavailable")
        return None

    try:
        # Computer Modern（LaTeX 標準書体）でレンダリング。LaTeX 本体は不要。
        with plt.rc_context({"mathtext.fontset": "cm"}):
            fig = plt.figure(figsize=(8, 2), dpi=_RENDER_DPI)
            fig.patch.set_facecolor("white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_axis_off()
            # 未エスケープの % だけ \% にする（既に \% になってるものは二重化しない）
            safe_src = re.sub(r"(?<!\\)%", r"\\%", math_src)
            ax.text(
                0.5,
                0.5,
                f"${safe_src}$",
                fontsize=_FONT_SIZE_PT,
                ha="center",
                va="center",
                color="black",
            )
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=_RENDER_DPI,
                bbox_inches="tight",
                pad_inches=_PAD_INCHES,
            )
            plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf).convert("L"))
    except Exception:
        logger.exception("math render failed: %r", math_src)
        return None


def _binarize(gray: np.ndarray) -> np.ndarray:
    thresh = filters.threshold_otsu(gray)
    return gray < thresh  # True = ink


def _crop(binary: np.ndarray) -> np.ndarray:
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any():
        return binary
    r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return binary[r0 : r1 + 1, c0 : c1 + 1]


def _trace_skeleton(skeleton: np.ndarray) -> list[np.ndarray]:
    """スケルトン画像からストローク列（[(N,2) float64 [col,row], ...]）を抽出。

    DP 簡略化（tolerance=1.5px）でステアケースを除去したあと返す。
    """
    from skimage.measure import approximate_polygon

    coords = np.argwhere(skeleton)  # (K, 2): [row, col]
    if len(coords) == 0:
        return []

    pixel_set: set[tuple[int, int]] = {(int(c), int(r)) for r, c in coords}

    def nbrs(col: int, row: int) -> list[tuple[int, int]]:
        return [
            p
            for p in [
                (col - 1, row - 1),
                (col, row - 1),
                (col + 1, row - 1),
                (col - 1, row),
                (col + 1, row),
                (col - 1, row + 1),
                (col, row + 1),
                (col + 1, row + 1),
            ]
            if p in pixel_set
        ]

    deg: dict[tuple[int, int], int] = {p: len(nbrs(*p)) for p in pixel_set}

    # ---- 1. ノード（端点・分岐点）間のセグメントを収集 ----
    # deg=2 の点は通過点、deg≠2 の点がノード
    special: set[tuple[int, int]] = {p for p in pixel_set if deg[p] != 2}
    if not special:
        # 全点 deg=2 → 孤立ループ: 任意の1点をノードとして扱う
        special = {next(iter(pixel_set))}

    # 有向エッジ訪問管理（セグメント収集用）
    seg_visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    # segments: (node_a, node_b, pixel_path)
    segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []

    for s in special:
        for nb in nbrs(*s):
            if (s, nb) in seg_visited:
                continue
            seg_visited.add((s, nb))
            seg_visited.add((nb, s))

            path: list[tuple[int, int]] = [s, nb]
            prev, cur = s, nb
            while cur not in special:
                nxts = [n for n in nbrs(*cur) if n != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                seg_visited.add((cur, nxt))
                seg_visited.add((nxt, cur))
                path.append(nxt)
                prev, cur = cur, nxt

            segments.append((s, cur, path))

    # ---- 1b. 孤立ループ（specialノードに接続されていないdeg=2連結成分）を追加 ----
    # %の丸など、閉じた輪だけの連結成分は special に含まれないのでここで補完する。
    seg_pixels: set[tuple[int, int]] = {p for _, _, path in segments for p in path}
    unvisited = pixel_set - seg_pixels
    while unvisited:
        loop_start = next(iter(unvisited))
        # 連結成分を幅優先で収集
        comp: set[tuple[int, int]] = set()
        queue = [loop_start]
        while queue:
            p = queue.pop()
            if p in comp:
                continue
            comp.add(p)
            for n in nbrs(*p):
                if n in unvisited and n not in comp:
                    queue.append(n)
        unvisited -= comp
        # 1方向に辿ってループを閉じる
        first_nb = nbrs(*loop_start)[0] if nbrs(*loop_start) else None
        if first_nb is None:
            continue
        loop_path: list[tuple[int, int]] = [loop_start, first_nb]
        loop_seen: set[tuple[int, int]] = {loop_start, first_nb}
        prev_p, cur_p = loop_start, first_nb
        while True:
            candidates = [n for n in nbrs(*cur_p) if n != prev_p]
            back = [n for n in candidates if n == loop_start and len(loop_path) > 2]
            fresh = [n for n in candidates if n not in loop_seen]
            if back:
                loop_path.append(loop_start)
                break
            if not fresh:
                break
            nxt_p = fresh[0]
            loop_path.append(nxt_p)
            loop_seen.add(nxt_p)
            prev_p, cur_p = cur_p, nxt_p
        segments.append((loop_start, loop_start, loop_path))

    # ---- 2. ノードごとの隣接セグメントリストを構築 ----
    from collections import defaultdict

    adj: dict[tuple[int, int], list[tuple[int, int, bool]]] = defaultdict(list)
    # (other_node, seg_index, forward)
    for i, (a, b, _) in enumerate(segments):
        adj[a].append((b, i, True))
        if a != b:
            adj[b].append((a, i, False))

    # ---- 3. グリーディー最小ペンリフト経路 ----
    used: set[int] = set()
    raw_strokes: list[list[tuple[int, int]]] = []

    def seg_pts(idx: int, forward: bool) -> list[tuple[int, int]]:
        pts = segments[idx][2]
        return pts if forward else pts[::-1]

    def _dot(
        cur: tuple[int, int],
        prev: tuple[int, int],
        nxt: tuple[int, int],
    ) -> float:
        dx, dy = cur[0] - prev[0], cur[1] - prev[1]
        ex, ey = nxt[0] - cur[0], nxt[1] - cur[1]
        return float(dx * ex + dy * ey)

    # 端点 → 分岐点 → その他 の順でスタート
    endpoints = [p for p in special if deg[p] == 1]
    junctions = [p for p in special if deg[p] >= 3]
    start_order = (
        endpoints + junctions + [p for p in special if p not in set(endpoints + junctions)]
    )

    for start_node in start_order:
        while any(i not in used for _, i, _ in adj[start_node]):
            # 未使用セグメントの中で最初のものを選ぶ
            init = next((e for e in adj[start_node] if e[1] not in used), None)
            if init is None:
                break
            other, si, fwd = init
            used.add(si)
            path = list(seg_pts(si, fwd))
            cur_node = other

            # 続けられる限りセグメントをつなぐ（最直進を選択）
            while True:
                avail = [(o, i, f) for o, i, f in adj[cur_node] if i not in used]
                if not avail:
                    break
                # 直前2点から進行方向を計算し内積最大の枝へ
                if len(path) >= 2:
                    prev_pt = path[-2]
                    cur_pt = path[-1]
                    best = max(avail, key=lambda e: _dot(cur_pt, prev_pt, e[0]))
                else:
                    best = avail[0]
                other2, si2, fwd2 = best
                used.add(si2)
                path.extend(seg_pts(si2, fwd2)[1:])
                cur_node = other2

            raw_strokes.append(path)

    # 未使用セグメント（孤立連結成分）を追加
    for i, (_, _, pts) in enumerate(segments):
        if i not in used:
            raw_strokes.append(list(pts))

    # ---- 4. Douglas-Peucker 簡略化でステアケース除去 ----
    result: list[np.ndarray] = []
    for path in raw_strokes:
        arr = np.array(path, dtype=np.float64)
        simplified = approximate_polygon(arr[:, ::-1], tolerance=1.5)
        if len(simplified) >= 2:
            result.append(simplified[:, ::-1].astype(np.float64))

    return result


@lru_cache(maxsize=256)
def render_math_char_to_unit_strokes(text: str) -> tuple[np.ndarray, ...] | None:
    """1 文字／1 トークンを unit square [0,1]×[0,1] Y-UP のストロークに変換（キャッシュ付き）。

    `_position_strokes` と組み合わせて使う。返り値は tuple（lru_cache のため）。
    None のとき既存 fallback へ。
    """
    gray = _render_to_gray(text)
    if gray is None:
        return None

    binary = _binarize(gray)
    if not binary.any():
        return None

    binary = _crop(binary)
    skeleton = morphology.skeletonize(binary)
    pixel_strokes = _trace_skeleton(skeleton)

    h_px, w_px = skeleton.shape
    result: list[np.ndarray] = []
    for ps in pixel_strokes:
        if len(ps) < _MIN_STROKE_PX:
            continue
        col = ps[:, 0] / max(w_px - 1, 1)
        row = 1.0 - ps[:, 1] / max(h_px - 1, 1)  # Y-UP
        stroke = np.stack([col, row], axis=1).astype(np.float64)
        diffs = np.diff(stroke, axis=0)
        length = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        if length >= _MIN_STROKE_UNIT:
            result.append(stroke)

    return tuple(result) if result else None
