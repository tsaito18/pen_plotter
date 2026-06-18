"""LaTeX 数式 → matplotlib レンダリング → スケルトン化 → ストローク変換。"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
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


@lru_cache(maxsize=128)
def formula_ink_em(math_src: str) -> float:
    """数式のインク高さが nominal em の何倍かを返す（小文字≈0.45, 大文字≈0.7）。

    matplotlib は ``bbox_inches="tight"`` + crop でインク範囲に切り詰めるため、
    そのまま論理高(font_size)へスケールすると小文字 u 等が em いっぱいまで拡大され
    「でかすぎ」になる。インクの em 比を返し、描画高 = ink_em * font_size とすれば
    本文 em と同じ縮尺で揃う。墨なし／失敗時は 1.0（従来挙動相当）。

    Args:
        math_src: LaTeX ソース（$ なし）。

    Returns:
        インク高 / nominal em。
    """
    gray = _render_to_gray(math_src)
    if gray is None:
        return 1.0
    binary = _binarize(gray)
    if not binary.any():
        return 1.0
    binary = _crop(binary)
    h_px = binary.shape[0]
    em_px = _FONT_SIZE_PT * _RENDER_DPI / 72.0
    return h_px / em_px if em_px > 0 else 1.0


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


@lru_cache(maxsize=128)
def formula_aspect(math_src: str) -> float:
    """数式を matplotlib mathtext で描いたときのアスペクト比（幅/高さ）を返す。

    予約幅（折り返し・カーソル前進・中央寄せ）を実描画幅に一致させるための軽量関数。
    render_latex_to_strokes は draw_w = h_mm * aspect で描くので、予約側も同じ aspect を
    使えば論理幅（_CHAR_WIDTH_RATIO 等幅）との不一致による行はみ出しが解消する。
    skeletonize は描画結果に影響しないため、幅予約のたびに呼ぶこの関数では省略する。

    Args:
        math_src: LaTeX ソース（$ なし）。

    Returns:
        aspect = 描画画像の幅/高さ。墨なし／描画失敗時は 0.0。
    """
    gray = _render_to_gray(math_src)
    if gray is None:
        return 0.0
    binary = _binarize(gray)
    if not binary.any():
        return 0.0
    binary = _crop(binary)
    h_px, w_px = binary.shape
    return w_px / max(h_px, 1)


def formula_draw_width_mm(math_src: str, h_mm: float) -> float:
    """数式を高さ h_mm で描いたときの実際の描画幅(mm)。式番号の配置に使う。

    render_latex_to_strokes と同じ draw_w = h_mm * aspect を返す。
    """
    return h_mm * formula_aspect(math_src)


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


# ---- matplotlib mathtext 実配置の抽出（手書き差し替え用） ------------------
# matplotlib に LaTeX 数式を正しい配置でレイアウトさせ、各グリフ／罫線の位置・
# サイズを取り出す。通常グリフ（変数・数字・演算子）は手書きストロークへ差し替え、
# 大型構造記号（√・大括弧・∑・∫）はそのグリフだけ skeletonize し、罫線（分数線・
# 根号の横棒）は直線ストロークにする。座標系は baseline 原点・上向き正の pt（dpi=72
# で pt=px）。oy/y はグリフ／矩形の baseline（描画原点）の y 座標。

# 大型記号に使われる別フォント family の判定基準。DejaVu Sans 以外（STIXSize* 等）は
# √・大括弧・∑・∫ など手書き字形を持たない構造記号なので skeleton で描く。
_HANDWRITE_FONT_FAMILY = "DejaVu Sans"
# 単一グリフ skeleton 化のレンダリング pt（_render_to_gray の _FONT_SIZE_PT に依らず、
# 大型記号の縦横比を保つため十分大きい固定値で描く）。
_GLYPH_RENDER_PT = 120


# 数式の基準グリフ（上付き・添字でない通常文字）の標準サイズ pt。
# extract_math_layout が _FONT_SIZE_PT 基準でレイアウトするため、それと一致させる。
_REF_GLYPH_PT = _FONT_SIZE_PT


@lru_cache(maxsize=1)
def ref_cap_height_pt() -> float:
    """基準サイズ(``_REF_GLYPH_PT`` pt)で描いた大文字のインク高(pt)を返す。

    数式の手書き差し替えでスケールを取るときの基準。式全体の墨高(上付き・分数で
    背が高い)で割ると基準文字が縮むため、代わりに「通常サイズの大文字インク高」を
    本文の大文字インク高(≈0.7*font_size)へ揃える縮尺を取る。フォント不変なので
    キャッシュする。

    Returns:
        基準大文字のインク高(pt)。測定失敗時は em の概算値(0.7*_REF_GLYPH_PT)。
    """
    ink = glyph_ink_bbox("M", float(_REF_GLYPH_PT))
    if ink is None:
        return 0.7 * _REF_GLYPH_PT
    return ink[3]


@dataclass(frozen=True)
class MathGlyph:
    """matplotlib mathtext がレイアウトした 1 グリフ。座標は pt（baseline 原点・上向き正）。"""

    char: str  # 描画文字（chr(num)）
    x: float  # 左 x（pt）
    baseline_y: float  # baseline の y（pt, 上向き正）
    fontsize: float  # そのグリフの実 pt（下付き・分数で縮小される）
    is_large: bool  # True=大型構造記号（√・大括弧・∑・∫ 等。手書きにできない）


@dataclass(frozen=True)
class MathRect:
    """分数線・根号の横棒など。座標は pt（baseline 原点・上向き正、y=下端）。"""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class MathLayout:
    """matplotlib mathtext のレイアウト結果（pt 単位、baseline 原点・上向き正）。"""

    width: float  # 全体幅（pt）
    height: float  # baseline から上の高さ（pt）
    depth: float  # baseline から下の深さ（pt）
    glyphs: tuple[MathGlyph, ...]
    rects: tuple[MathRect, ...]


def extract_math_layout(math_src: str) -> MathLayout | None:
    """LaTeX 数式を matplotlib mathtext でレイアウトし、グリフ／罫線の配置を抽出する。

    各グリフは ``(x, baseline_y, fontsize)`` を原点に当該 fontsize で描いた位置に座る
    （baseline 原点・上向き正の pt）。手書き差し替えではこの配置に合わせて unit 字形を
    スケール・移動する。

    Args:
        math_src: LaTeX ソース（``$`` なし。``\\tag{}`` は呼び出し側で除去済みのこと）。

    Returns:
        ``MathLayout``。レイアウト失敗時は ``None``。
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser

    safe = re.sub(r"(?<!\\)%", r"\\%", math_src)
    try:
        # fontset は既定(dejavusans)のまま使う。cm を rc_context 内で MathTextParser に
        # 適用すると本物の Computer Modern フォント(cmmi10/cmr10/cmsy10)が選ばれ、通常
        # グリフの family_name が "DejaVu Sans" でなくなり is_large 判定が壊れる。手書き
        # 差し替えでは通常グリフの見た目は使わず配置と is_large 判定だけ要るので、判定が
        # 安定する dejavusans でレイアウトを取る（大型記号は STIXSize* で出る）。
        mtp = MathTextParser("path")
        prop = FontProperties(size=_FONT_SIZE_PT)
        vp = mtp.parse(f"${safe}$", dpi=72, prop=prop)
    except Exception:
        logger.exception("math layout extract failed: %r", math_src)
        return None

    glyphs: list[MathGlyph] = []
    for font, fontsize, num, ox, oy in vp.glyphs:
        glyphs.append(
            MathGlyph(
                char=chr(num),
                x=float(ox),
                baseline_y=float(oy),
                fontsize=float(fontsize),
                is_large=font.family_name != _HANDWRITE_FONT_FAMILY,
            )
        )
    rects = tuple(
        MathRect(x=float(x), y=float(y), width=float(w), height=float(h)) for x, y, w, h in vp.rects
    )
    return MathLayout(
        width=float(vp.width),
        height=float(vp.height),
        depth=float(vp.depth),
        glyphs=tuple(glyphs),
        rects=rects,
    )


@lru_cache(maxsize=512)
def glyph_ink_bbox(char: str, fontsize: float) -> tuple[float, float, float, float] | None:
    """1 グリフを baseline 原点・``fontsize`` pt で描いた実インク bbox を返す。

    matplotlib の ``vp.glyphs`` の ``(ox, oy)`` は glyph の baseline（描画原点）であり
    インク範囲ではない。手書き／skeleton の unit 字形を「そのグリフが画面で占める矩形」
    へ正確に貼り込むため、当該 fontsize で TextPath を描いたときの実インク範囲（baseline
    原点・上向き正の pt）を返す。空白等で墨が無ければ None。

    Args:
        char: 1 文字。
        fontsize: pt。

    Returns:
        ``(x_min, y_min, width, height)``（baseline 原点・上向き正の pt）。墨なしは None。
    """
    from matplotlib.textpath import TextPath

    try:
        # extract_math_layout と同じく既定 fontset(dejavusans)で取り、baseline・サイズの
        # 基準を一致させる。$...$ で囲み mathtext 経由で描くことで √・大括弧など別フォントの
        # 記号も正しいグリフが選ばれる（math 字形に統一）。
        safe = re.sub(r"(?<!\\)%", r"\\%", char)
        tp = TextPath((0, 0), f"${safe}$", size=fontsize)
        v = tp.vertices
        if len(v) == 0:
            return None
        x_min, y_min = float(v[:, 0].min()), float(v[:, 1].min())
        x_max, y_max = float(v[:, 0].max()), float(v[:, 1].max())
    except Exception:
        return None
    w, h = x_max - x_min, y_max - y_min
    if w <= 0 or h <= 0:
        return None
    return (x_min, y_min, w, h)


@lru_cache(maxsize=256)
def render_glyph_unit_strokes(char: str) -> tuple[np.ndarray, ...] | None:
    """1 つの記号（大型構造記号も含む）を unit square [0,1]×[0,1] Y-UP に skeleton 化する。

    ``√`` ``∑`` ``∫`` や大括弧など手書き字形を持たない構造記号を、matplotlib（cm
    fontset）の数式書体で描いてスケルトン化する。``render_math_char_to_unit_strokes``
    は本文文字向けに ``$...$`` でラップして描くが、こちらは大型記号用に十分大きい pt で
    描いて縦横比を保つ。返り値は tuple（lru_cache のため）。失敗時は None。

    Args:
        char: 描画する 1 文字（``√`` ``(`` ``∑`` 等）。

    Returns:
        unit square Y-UP のストローク列（tuple）。失敗時は None。
    """
    gray = _render_to_gray(char)
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
