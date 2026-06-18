from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.collector.data_format import StrokeSample
from src.layout.char_metrics import char_type_scale
from src.layout.line_breaking import is_halfwidth
from src.layout.page_layout import PageConfig
from src.layout.typesetter import CharPlacement
from src.model.augmentation import HandwritingAugmenter
from src.ui.math_skeletonize import (
    extract_math_layout,
    glyph_ink_bbox,
    ref_cap_height_pt,
    render_glyph_unit_strokes,
    render_latex_to_strokes,
)
from src.model.stroke_finishing import (
    FinishingConfig,
    apply_finishing,
    classify_finishes,
    infer_finishes,
)
from dataclasses import dataclass, field

Stroke = npt.NDArray[np.float64]


@dataclass
class CharCoverageReport:
    user_strokes: list[str] = field(default_factory=list)
    ml_inference: list[str] = field(default_factory=list)
    kanjivg: list[str] = field(default_factory=list)
    geometric: list[str] = field(default_factory=list)
    rect_fallback: list[str] = field(default_factory=list)
    missing_glyphs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


logger = logging.getLogger(__name__)


class StrokeRenderer:
    _SKIP_RENDER = set(" \t\u3000")
    _NON_JAPANESE_MODE_ALLOWED_CHARS = set("、。，．,.")

    _CHAR_SUBSTITUTIONS: dict[str, str] = {
        "\uff5b": "{",
        "\uff5d": "}",
        "\uff3b": "[",
        "\uff3d": "]",
        "\uff01": "!",
        "\uff1f": "?",
        "\uff1a": ":",
        "\uff1b": ";",
        "\uff1d": "=",
        "\uff0b": "+",
        "\uff0d": "-",
        "\uff0f": "/",
        "\u2212": "-",  # \u2212 \u6570\u5b66\u30de\u30a4\u30ca\u30b9 \u2192 ASCII -
        "\u301c": "~",  # \u301c \u6ce2\u30c0\u30c3\u30b7\u30e5 \u2192 ~
        "\uff5e": "~",  # \uff5e \u5168\u89d2\u30c1\u30eb\u30c0 \u2192 ~
    }

    _SMOOTH_CHARS = set(
        "\u3001\u3002\uff0c\uff0e\u30fb\u30fc\uff5e\u2014\u2015()\uff08\uff09\u300c\u300d\u300e\u300f\u3010\u3011\u3008\u3009\u300a\u300b\u3014\u3015"
    )

    _NOISE_SCALE = 0.15

    # 描画経路ごとの揺らぎ強度。経路間で質感(きれい↔汚い)が大きくばらつくと
    # 「同じページに定規直線と乱れた字が混在」する違和感が出るため、経路差を
    # 縮めた共通の基準値を一箇所に集約する。
    # matplotlib 数式画像は元から平滑で粗いので幾何字形より上に残すが、
    # 旧値(6.0)は本文から浮きすぎたため本文寄りに引き下げる。
    _WAVER_MATH_IMAGE = 2.5
    # 記号・句読点・括弧・ギリシャ文字など従来 distortion 無し(=定規直線)の経路に
    # 乗せる微量の揺らぎ。ツルツル感だけ消し、字形は崩さない控えめ値。
    _WAVER_SYMBOL = 0.4
    # 記号経路の微量揺らぎを当てる最小ストローク長。句点「。」・ピリオドの点など
    # 極短ストロークは揺らぎで破綻するため素のままにする。字形 bbox 最大辺に対する
    # 相対比と、点(約0.5mm)を確実に除外する絶対長(mm)の両方で判定する。
    _SYMBOL_WAVER_MIN_SPAN = 0.08
    _SYMBOL_WAVER_MIN_LEN_MM = 1.2

    def __init__(
        self,
        *,
        checkpoint_path: Path | str | None = None,
        kanjivg_dir: Path | str | None = None,
        style_sample: object | None = None,
        temperature: float = 1.0,
        user_strokes_dir: Path | str | None = None,
        augmenter: HandwritingAugmenter | None = None,
        page_config: PageConfig | None = None,
        finishing_config: FinishingConfig | None = None,
        enable_finishing: bool = True,
        instance_variation: float = 0.5,
        skip_non_japanese: bool = False,
    ) -> None:
        self._page_config = page_config or PageConfig()
        self._temperature = temperature
        self._augmenter = augmenter
        self._skip_non_japanese = skip_non_japanese
        # 同一字の繰り返しで形を変える per-stroke ランダムaffineの強度（0=無効）
        self._instance_variation = instance_variation

        # KanjiVG 参照経路（ML 推論 / safety-net）の終端加工（とめ・はね・払い）。
        # _direct_stroke（人の実筆跡）・幾何描画経路には適用しない。
        if finishing_config is not None:
            self._finishing_config = finishing_config
        elif enable_finishing:
            self._finishing_config = FinishingConfig()
        else:
            self._finishing_config = FinishingConfig(enabled=False)

        # 既存呼び出し元（pipeline.create_app() 等）が UI 構築時に環境引数を
        # 復元できるよう、入力時のパスを保持しておく
        self._checkpoint_path = checkpoint_path
        self._user_strokes_dir = user_strokes_dir

        self._inference = None
        if checkpoint_path is not None:
            cp = Path(checkpoint_path)
            if cp.exists():
                try:
                    from src.model.inference import StrokeInference

                    self._inference = StrokeInference(cp)
                    logger.info("ML inference engine loaded from %s", cp)
                except Exception:
                    logger.warning("Failed to load ML checkpoint: %s", cp, exc_info=True)

        if style_sample is not None:
            self._style_sample = style_sample
        else:
            self._style_sample = self._load_style_from_user_strokes(user_strokes_dir)

        self._user_stroke_db = self._load_user_stroke_db(user_strokes_dir)
        # 文字ごとに「最初に選んだサンプル」のインデックスを記憶し、同じ字には常に
        # 同じベース字形を使う。等確率ランダムだと隣接する同一字でベース字形が
        # 丸ごと入れ替わり「片方きれい・片方汚い」と極端に振れるため固定する。
        # 字ごとの多様性は instance_variation / 温度ノイズが別途与える。
        self._direct_choice_cache: dict[str, int] = {}
        self._last_coverage = CharCoverageReport()

        self._kanjivg_dir: Path | None = None
        if kanjivg_dir is not None:
            d = Path(kanjivg_dir)
            if d.is_dir():
                self._kanjivg_dir = d

    @staticmethod
    def _load_user_stroke_db(
        user_strokes_dir: Path | str | None,
    ) -> dict[str, list[list[Stroke]]]:
        db: dict[str, list[list[Stroke]]] = {}
        if user_strokes_dir is None:
            return db
        user_dir = Path(user_strokes_dir)
        if not user_dir.is_dir():
            return db
        for char_dir in sorted(user_dir.iterdir()):
            if not char_dir.is_dir():
                continue
            char = char_dir.name
            for json_file in sorted(char_dir.glob("*.json")):
                try:
                    sample = StrokeSample.load(json_file)
                    strokes = [
                        np.array([[p.x, p.y] for p in stroke], dtype=np.float64)
                        for stroke in sample.strokes
                    ]
                    db.setdefault(char, []).append(strokes)
                except Exception:
                    logger.warning("Failed to load user stroke: %s", json_file)
        logger.info("Loaded user stroke DB: %d chars", len(db))
        return db

    @staticmethod
    def _load_style_from_user_strokes(
        user_strokes_dir: Path | str | None,
    ) -> object:
        try:
            import torch
        except ImportError:
            return None

        if user_strokes_dir is not None:
            user_dir = Path(user_strokes_dir)
            if user_dir.is_dir():
                import json

                from src.model.data_utils import strokes_to_deltas

                json_files: list[Path] = []
                for char_dir in sorted(user_dir.iterdir()):
                    if char_dir.is_dir():
                        json_files.extend(sorted(char_dir.glob("*.json")))

                if json_files:
                    all_strokes: list[list[dict[str, float]]] = []
                    for jf in json_files:
                        try:
                            data = json.loads(jf.read_text(encoding="utf-8"))
                            all_strokes.extend(data["strokes"])
                        except (json.JSONDecodeError, KeyError):
                            continue

                    if all_strokes:
                        deltas = strokes_to_deltas(all_strokes)
                        style_sample = deltas.unsqueeze(0)
                        logger.info(
                            "Loaded style sample from %d files (%d points)",
                            len(json_files),
                            deltas.shape[0],
                        )
                        return style_sample

        return torch.zeros(1, 10, 3)

    def generate_char_strokes(self, placement: CharPlacement) -> list[Stroke]:
        """Tier 0-4 フォールバックでストローク生成（strokes のみ返す後方互換ラッパー）。

        Args:
            placement: 描画対象文字の配置情報。

        Returns:
            生成された各画のストローク列。
        """
        return self.generate_char_strokes_with_finishes(placement)[0]

    @staticmethod
    def _is_japanese_text_char(char: str) -> bool:
        """一時オプション用の描画許可判定。日本語文で必要な字だけ通す。"""
        if len(char) != 1:
            return False
        code = ord(char)
        return (
            0x3040 <= code <= 0x309F
            or 0x30A0 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0xFF66 <= code <= 0xFF9D
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0x2CEB0 <= code <= 0x2EBEF
            or 0x30000 <= code <= 0x3134F
        )

    @classmethod
    def _is_allowed_non_japanese_mode_char(cls, char: str) -> bool:
        if len(char) != 1:
            return False
        code = ord(char)
        return (
            char in cls._NON_JAPANESE_MODE_ALLOWED_CHARS
            or 0x30 <= code <= 0x39
            or 0xFF10 <= code <= 0xFF19
        )

    @classmethod
    def _should_skip_non_japanese(cls, placement: CharPlacement) -> bool:
        if placement.line_segment is not None:
            return getattr(placement, "role", None) is not None
        if getattr(placement, "math_source", None) or getattr(placement, "math_skip", False):
            return True
        return not (
            cls._is_japanese_text_char(placement.char)
            or cls._is_allowed_non_japanese_mode_char(placement.char)
        )

    def _resolve_finishes(self, raw_types: list[str], positioned: list[Stroke]) -> list[str]:
        """筆画タイプを決める。kvg:type があれば分類、無ければ軌跡から推定。

        漢字は KanjiVG の ``kvg:type`` を分類（正確）。かな等 ``kvg:type`` を持た
        ない字（raw_types が全空 → 全 none）は軌跡形状から推定して筆遣いを与える。
        """
        finishes = classify_finishes(raw_types)
        if all(f == "none" for f in finishes):
            return infer_finishes(positioned)
        return finishes

    def generate_char_strokes_with_finishes(
        self, placement: CharPlacement
    ) -> tuple[list[Stroke], list[str]]:
        """Tier 0-4 フォールバックで strokes と並走する finishes を生成する。

        各画の筆画タイプ（``finish``: ``"tome"``/``"hane"``/``"harai"``/
        ``"none"``）を strokes と同じ本数の ``list[str]`` として返す。幾何描画・
        人の実筆跡（``_direct_stroke``）経路には終端加工を入れない方針のため
        ``"none"`` を並べるだけで、ML 推論・KanjiVG safety-net 経路のみ
        ``classify_finishes`` で求めた筆画タイプを返す（``apply_finishing`` 後も
        strokes 本数は不変なので len は一致する）。

        Args:
            placement: 描画対象文字の配置情報。

        Returns:
            ``(strokes, finishes)``。``len(strokes) == len(finishes)`` を満たす。
            ``_SKIP_RENDER`` 該当字は ``([], [])``。
        """
        cov = self._last_coverage

        if self._skip_non_japanese and self._should_skip_non_japanese(placement):
            cov.skipped.append(placement.char)
            return [], []

        # math_skip=True: 先頭の数式 CharPlacement がまとめて描画済みなのでスキップ
        if getattr(placement, "math_skip", False):
            return [], []

        # 分数線・根号の屋根線などの補助線分は文字ではなく単一ストロークとして描画する
        if placement.line_segment is not None:
            x1, y1, x2, y2 = placement.line_segment
            return [np.array([[x1, y1], [x2, y2]], dtype=np.float64)], ["none"]

        if placement.char in self._SKIP_RENDER:
            cov.skipped.append(placement.char)
            return [], []

        original_char = placement.char
        lookup_char = self._CHAR_SUBSTITUTIONS.get(original_char, original_char)
        if lookup_char != original_char:
            placement = CharPlacement(
                char=lookup_char,
                x=placement.x,
                y=placement.y,
                font_size=placement.font_size,
                page=placement.page,
            )

        is_smooth = original_char in self._SMOOTH_CHARS or lookup_char in self._SMOOTH_CHARS

        # 数式ブロック: $$単位でレンダリング（math_source + math_bbox で直接 mm 座標を返す）
        if getattr(placement, "math_source", None) and getattr(placement, "math_bbox", None):
            align = getattr(placement, "math_align", "center")
            bbox = placement.math_bbox
            if align == "baseline":
                # 本文文字は placement.y を行ボックス下端とみなし line_spacing 内で縦中央
                # 配置する(_position_strokes)。インライン数式のベースラインも同じ下端ライン
                # へ揃えないと、行高と数式インク高の差の半分だけ下へずれる(issue #24)。
                x0, y0, w, h = bbox
                baseline_y = y0 + (self._page_config.line_spacing - placement.font_size) / 2
                bbox = (x0, baseline_y, w, h)
            # handwrite_math: matplotlib 配置で並べ、通常グリフを手書きストロークに差し替える。
            # skeletonize 経路(False)は従来どおり数式全体を画像→スケルトン化する。
            # render_math_handwritten が None（√ 等で手書き不可）のときは skeletonize へ落ちる。
            if getattr(placement, "math_handwrite", False):
                strokes = self.render_math_handwritten(
                    placement.math_source, bbox, align, font_size=placement.font_size
                )
                if strokes is not None:
                    cov.geometric.append(original_char)
                    return strokes, ["none"] * len(strokes)
            strokes = render_latex_to_strokes(placement.math_source, bbox, align)
            if strokes:
                cov.geometric.append(original_char)
                # matplotlib フォントのままだと整いすぎるので手書き揺らぎを乗せる。
                # 数式画像は元から平滑なので本文より強めに残すが、経路間の質感を
                # 揃えるため旧値(6.0)から引き下げる。
                strokes = self._apply_distortion(strokes, waver_scale=self._WAVER_MATH_IMAGE)
                return strokes, ["none"] * len(strokes)
            return [], []

        punct_strokes = self._simple_punct_strokes(lookup_char)
        if punct_strokes is not None:
            cov.geometric.append(original_char)
            positioned = self._position_strokes(punct_strokes, placement)
            positioned = self._apply_symbol_distortion(positioned)
            if lookup_char in ("、", ",", "，"):
                return positioned, ["harai"] * len(positioned)
            return positioned, ["none"] * len(positioned)

        paren_strokes = self._simple_paren_strokes(original_char, placement)
        if paren_strokes is not None:
            cov.geometric.append(original_char)
            positioned = self._position_strokes(paren_strokes, placement)
            positioned = self._apply_symbol_distortion(positioned)
            return positioned, ["none"] * len(positioned)

        slash_strokes = self._slash_strokes(lookup_char)
        if slash_strokes is not None:
            cov.geometric.append(original_char)
            positioned = self._position_strokes(slash_strokes, placement)
            positioned = self._apply_symbol_distortion(positioned)
            return positioned, ["none"] * len(positioned)

        # 英字はユーザーの実筆跡サンプルがあれば最優先（自然・本人の字）。書いた英字は
        # 直接ストロークで本人の手書きにする。サンプルが無い英字は後段の KanjiVG 参照
        # 経路で描画される（a-zA-Z は全字 KanjiVG に字形 JSON あり）。
        if lookup_char.isascii() and lookup_char.isalpha():
            direct_letter = self._direct_stroke(lookup_char)
            if direct_letter is not None:
                cov.user_strokes.append(original_char)
                positioned = self._position_strokes(direct_letter, placement)
                waver = self._waver_scale(len(positioned))
                positioned = positioned if is_smooth else self._apply_distortion(positioned, waver)
                return positioned, ["none"] * len(positioned)

        direct = self._direct_stroke(placement.char)
        if direct is not None:
            cov.user_strokes.append(original_char)
            positioned = self._position_strokes(direct, placement)
            waver = self._waver_scale(len(positioned))
            positioned = positioned if is_smooth else self._apply_distortion(positioned, waver)
            return positioned, ["none"] * len(positioned)

        reference, ref_types = self._load_reference_strokes(placement.char)

        if (
            self._inference is not None
            and reference is not None
            and self._is_ml_deformable(placement.char)
        ):
            try:
                waver = self._waver_scale(len(reference))
                raw = self._inference.generate(
                    self._style_sample,
                    num_steps=50,
                    temperature=self._temperature,
                    reference_strokes=reference,
                    deform_scale=waver,
                )
                cov.ml_inference.append(original_char)
                positioned = self._position_strokes(raw, placement)
                finishes = self._resolve_finishes(ref_types, positioned)
                positioned = apply_finishing(
                    positioned,
                    finishes,
                    scale=placement.font_size,
                    config=self._finishing_config,
                )
                positioned = self._apply_instance_variation(positioned, waver)
                positioned = positioned if is_smooth else self._apply_distortion(positioned, waver)
                # ML経路は deformable な CJK のみ通る(L407 ガード)。横棒を右上がりに。
                positioned = self._enforce_horizontal_rise(positioned, placement.font_size)
                return positioned, finishes
            except Exception:
                logger.warning("ML inference failed for '%s'", placement.char, exc_info=True)

        if self._kanjivg_dir is not None:
            char_strokes, char_types = self._load_kanjivg_json(placement)
            if char_strokes is not None:
                cov.kanjivg.append(original_char)
                positioned = self._position_strokes(char_strokes, placement)
                finishes = self._resolve_finishes(char_types, positioned)
                # 数字は終端リフト（はらい/はね）を当てると下線等が歪むため無効化
                if not self._is_ml_deformable(placement.char):
                    finishes = ["none"] * len(positioned)
                positioned = apply_finishing(
                    positioned,
                    finishes,
                    scale=placement.font_size,
                    config=self._finishing_config,
                )
                waver = self._waver_scale(len(char_strokes))
                positioned = self._apply_instance_variation(positioned, waver)
                positioned = positioned if is_smooth else self._apply_distortion(positioned, waver)
                # 数字等(非deformable)は下線・等号が崩れるため右上がり矯正を除外する
                if self._is_ml_deformable(placement.char):
                    positioned = self._enforce_horizontal_rise(positioned, placement.font_size)
                return positioned, finishes

        cov.missing_glyphs.append(original_char)
        return [], []

    # 根号は √記号グリフと屋根の横棒(rect)が matplotlib の path 抽出では分離して出る
    # （√記号は中身を覆うほど縦に伸びず、屋根 rect が宙に浮く）。手書き差し替えでは
    # 正しい根号を組めないので、√ を含む式は skeletonize 経路へフォールバックする。
    # 手書き経路で組めず式全体を skeletonize へフォールバックさせる文字。√ は大型記号
    # として個別 skeleton（記号）＋ rect（屋根線）で組めるため空にした（フォールバック無し）。
    _MATH_HANDWRITE_FALLBACK_CHARS = ""

    # インライン数式の基準グリフ（上付き・添字でない通常文字）高さを本文 cap height へ
    # 揃えるための比。本文英大文字のインク高 ≈ font_size * この値。DejaVu Sans 系の
    # cap height(≈0.7em)に合わせる。式全体の墨高(上付き・分数で背が高い)で割ると基準
    # 文字が本文より小さく縮むため、代わりに通常文字高を本文へ直接揃える。
    _MATH_INLINE_CAP_RATIO = 0.70

    def render_math_handwritten(
        self,
        math_src: str,
        bbox_mm: tuple[float, float, float, float],
        align: str = "center",
        font_size: float | None = None,
    ) -> list[Stroke] | None:
        """数式を matplotlib(LaTeX)配置で並べ、各グリフを手書きストロークに差し替える。

        matplotlib mathtext に正しい配置（分数・添字・上付き）をさせ、その pt 座標を
        mm bbox へ写す。通常グリフ（変数・数字・ギリシャ・演算子＝DejaVu Sans）は
        ユーザー筆跡 / KanjiVG / matplotlib skeleton から得た unit 字形を当該位置・サイズに
        貼る。大型構造記号（大括弧 ( ) ・∑・∫＝別フォント）は当該グリフを skeleton 化して
        貼る。分数線は ``vp.rects`` を直線ストロークにする。

        根号 √ は matplotlib の path 抽出で記号と屋根が分離するため、含む式は ``None`` を
        返して呼び出し側で skeletonize 経路へフォールバックさせる。

        座標系: matplotlib は baseline 原点・上向き正の pt（dpi=72）。``glyph_ink_bbox`` で
        各グリフの実インク bbox を pt で得て、pt→mm の一様スケール ``s`` と原点
        ``(x_left, baseline_mm)`` で mm(Y-UP) へ写す。縦位置（align）の取り方は
        ``render_latex_to_strokes`` と揃える。

        Args:
            math_src: LaTeX ソース（``$`` なし。``\\tag{}`` 除去済み想定）。
            bbox_mm: ``(x_left_mm, y_bbox_mm, width_mm, height_mm)``（Y-UP）。
            align: ``"center"``=ブロック中央寄せ / ``"baseline"``=インライン本文ベース揃え。
            font_size: 本文の論理 em（mm）。インライン時に基準グリフ高を本文 cap height
                （``font_size * _MATH_INLINE_CAP_RATIO``）へ揃える縮尺に使う。``None`` の
                ときは従来どおり bbox 高（``h_mm``）基準。

        Returns:
            mm 座標 Y-UP のストローク列。描くものが無ければ空リスト。手書きで組めない
            （√ を含む）式は ``None``（呼び出し側で skeletonize へフォールバック）。
        """
        layout = extract_math_layout(math_src)
        if layout is None:
            return None
        if any(g.char in self._MATH_HANDWRITE_FALLBACK_CHARS for g in layout.glyphs):
            return None
        ink_h = layout.height + layout.depth
        if ink_h <= 0 or layout.width <= 0:
            return None

        x0, y0, w_mm, h_mm = bbox_mm
        # pt→mm 一様スケール。基準グリフ高（通常文字）を本文 cap height へ揃える縮尺で取る。
        # 式全体高（layout.height+depth）→ bbox高 で割ると、上付き・分数で背が高い式ほど
        # 基準文字(I, M, r, g, L)が本文より縮む。基準サイズ大文字インク高(ref_cap_height_pt)を
        # 本文 cap height へ写す一定縮尺なら、どの式でも基準文字が本文と同大になる
        # （インライン・ブロック共通。font_size 未指定時のみ従来の bbox 高基準へフォールバック）。
        if font_size is not None:
            s = (font_size * self._MATH_INLINE_CAP_RATIO) / ref_cap_height_pt()
        else:
            s = h_mm / ink_h
        draw_w = layout.width * s
        if align == "baseline":
            # インライン: 数式 baseline(pt y=0) を本文ベースライン(y0)へ。歪みなし等倍。
            x_left = x0
            baseline_mm = y0  # 本文ベースライン
        else:
            # ブロック: bbox 中央へ上寄せ（render_latex_to_strokes center と同等の縦位置）。
            # 縦の中央化は cap 縮尺での実描画高(ink_h*s)で行う（bbox高 h_mm で中央化すると
            # cap 縮尺で式が縮んだ分だけ中心がずれる）。横は実描画幅で bbox 中央へ。
            draw_ink_h = ink_h * s
            cx = x0 + w_mm / 2
            cy = y0 + h_mm / 2 + h_mm * 0.2  # _MATH_LIFT_FRACTION 相当（行内でやや上寄せ）
            x_left = cx - draw_w / 2
            # 墨域 [-depth, +height] を実描画高 draw_ink_h で中央化し、baseline は下端 + depth*s。
            bottom_mm = cy - draw_ink_h / 2
            baseline_mm = bottom_mm + layout.depth * s

        def to_mm(px: float, py_baseline: float) -> tuple[float, float]:
            """pt(baseline 原点・上向き正) → mm(Y-UP)。py_baseline は baseline からの上向き pt。"""
            return (x_left + (px - 0.0) * s, baseline_mm + py_baseline * s)

        result: list[Stroke] = []

        # ---- 根号 √: 「入り→谷→屋根左端→屋根右端」の連結ポリラインで描く ----
        # matplotlib では √記号グリフと屋根(rect)が別要素で、記号頂点が屋根より上に
        # 突き出て"繋がってない"ように見える。対応する屋根 rect を x 近接で見つけ、
        # チェックマークの上がりを屋根左端へ直結し、そのまま屋根右端まで1本で描く。
        # 使った屋根 rect は下の rect ループで二重描画しないよう除外する。
        consumed_rects: set[int] = set()
        for g in layout.glyphs:
            if g.char != "√":
                continue
            ink = glyph_ink_bbox(g.char, g.fontsize)
            if ink is None:
                continue
            gx, gy, gw, gh = ink
            left = g.x + gx  # √ インク左端
            right = left + gw
            bottom = g.baseline_y + gy  # √ インク下端＝谷の最下点
            # 屋根 rect は「√ の右側にあって最も左の上部横棒」を選ぶ。√記号の advance が
            # ink より広く・背の高い根号では√グリフ上端が屋根に届かないため、ink 右端での
            # 厳密 x 一致ではなく『√右側で最左・かつ √下端より上』の rect を採る。
            # （√(L/g) 等で内側の分数線も候補に入るが、屋根の方が左なので最左選択で分離）
            best = None
            for ri, r in enumerate(layout.rects):
                if ri in consumed_rects:
                    continue
                roof_y = r.y + r.height / 2.0
                in_x = (left - gw * 0.3) <= r.x <= (right + gw * 2.5)
                above = roof_y >= bottom + gh * 0.3
                if in_x and above and (best is None or r.x < layout.rects[best].x):
                    best = ri
            if best is None:
                continue
            r = layout.rects[best]
            consumed_rects.add(best)
            roof_x = r.x
            roof_y = r.y + r.height / 2.0
            span = max(roof_x - left, gw)  # チェックマーク横幅（√左端→屋根左端）
            rise = roof_y - bottom  # 谷→屋根の高さ（中身の高さに追従）
            pts_pt = [
                (left, bottom + 0.55 * rise),  # 入り（左・中ほど）
                (left + 0.15 * span, bottom + 0.33 * rise),  # 谷の手前
                (left + 0.45 * span, bottom),  # 谷（最下点）
                (roof_x, roof_y),  # 上がって屋根の左端へ直結
                (roof_x + r.width, roof_y),  # 屋根右端まで
            ]
            poly = np.array([to_mm(px, py) for px, py in pts_pt], dtype=np.float64)
            result.append(poly)

        # ---- グリフ（通常＝手書き / 大型＝skeleton）----
        # 大型括弧の縦範囲を「中身（囲まれたグリフ/罫線）の上下端」に合わせる。
        # matplotlib の \left( グリフ高は控えめで分数全高に届かないため、括弧ペアを
        # 対応付け、間にある内容の実 top/bottom を測って括弧をその高さで描く。
        def _gtb(gg):
            ik = glyph_ink_bbox(gg.char, gg.fontsize)
            if ik is None:
                return None
            b = gg.baseline_y + ik[1]
            return (b, b + ik[3])

        paren_span: dict[int, tuple[float, float]] = {}
        stack: list[int] = []
        for gi, g in enumerate(layout.glyphs):
            if not (g.is_large and g.char in "()"):
                continue
            if g.char == "(":
                stack.append(gi)
            elif stack:
                oi = stack.pop()
                ox = layout.glyphs[oi].x
                cx = g.x
                tops, bots = [], []
                for mid in layout.glyphs[oi + 1 : gi]:
                    tb = _gtb(mid)
                    if tb:
                        bots.append(tb[0])
                        tops.append(tb[1])
                for r in layout.rects:
                    if ox < r.x < cx:
                        bots.append(r.y)
                        tops.append(r.y + r.height)
                if tops and bots:
                    pad = (max(tops) - min(bots)) * 0.08
                    span = (min(bots) - pad, max(tops) + pad)
                    paren_span[oi] = span
                    paren_span[gi] = span

        for gi, g in enumerate(layout.glyphs):
            if g.char == "√":
                continue  # 上で連結ポリラインとして描画済み
            # 大型括弧 ( ) は skeleton が小さく潰れるので、幾何アーク（左/右に膨らむ
            # 曲線）で描く。縦範囲は中身の上下端(paren_span)に合わせ分数全高を囲む。
            # 通常サイズは手書き(直接ストローク)を使う。
            if g.is_large and g.char in "()":
                pink = glyph_ink_bbox(g.char, g.fontsize)
                if pink is None:
                    continue
                pgx, pgy, pgw, pgh = pink
                pleft = g.x + pgx
                pright = pleft + pgw
                span = paren_span.get(gi)
                if span is not None:
                    pbot, ptop = span
                else:
                    pbot = g.baseline_y + pgy
                    ptop = pbot + pgh
                pgh = ptop - pbot
                pmid = (pbot + ptop) / 2.0
                if g.char == "(":
                    pts_pt = [
                        (pright, ptop),
                        (pleft + 0.35 * pgw, ptop - 0.18 * pgh),
                        (pleft, pmid),
                        (pleft + 0.35 * pgw, pbot + 0.18 * pgh),
                        (pright, pbot),
                    ]
                else:
                    pts_pt = [
                        (pleft, ptop),
                        (pright - 0.35 * pgw, ptop - 0.18 * pgh),
                        (pright, pmid),
                        (pright - 0.35 * pgw, pbot + 0.18 * pgh),
                        (pleft, pbot),
                    ]
                result.append(np.array([to_mm(px, py) for px, py in pts_pt], dtype=np.float64))
                continue
            unit = self._math_glyph_unit_strokes(g.char, g.is_large)
            if not unit:
                continue  # 手書きにできない字は □ を出さずスキップ
            ink = glyph_ink_bbox(g.char, g.fontsize)
            if ink is None:
                continue
            gx, gy, gw, gh = ink  # baseline 原点・上向き正の pt。gy は下端
            # グリフの実インク矩形（pt 絶対）。x は g.x+gx 起点、y は g.baseline_y+gy 起点。
            x_lo_pt = g.x + gx
            y_lo_pt = g.baseline_y + gy
            placed = self._place_unit_in_pt_box(unit, x_lo_pt, y_lo_pt, gw, gh, to_mm)
            # 大型記号は構造線なので素のまま。直接ストローク(既にユーザーの手書き=自然な
            # 揺らぎ持ち)は追加 distortion を乗せると l/i 等の細い字が過剰にうねって歪む
            # ため素のまま。matplotlib skeleton 由来(収集の無い字)だけ手書き揺らぎを乗せる。
            if g.is_large or g.char in self._user_stroke_db:
                result.extend(placed)
            else:
                result.extend(self._apply_distortion(placed, waver_scale=self._WAVER_MATH_IMAGE))

        # ---- 罫線（分数線・根号の横棒）→ 中心線の直線ストローク ----
        for ri, r in enumerate(layout.rects):
            if ri in consumed_rects:
                continue  # √ の屋根は連結ポリラインに含めたので二重描画しない
            cy_pt = r.y + r.height / 2.0
            p0 = to_mm(r.x, cy_pt)
            p1 = to_mm(r.x + r.width, cy_pt)
            result.append(np.array([p0, p1], dtype=np.float64))

        return result

    def _math_glyph_unit_strokes(self, char: str, is_large: bool) -> list[Stroke] | None:
        """数式グリフ 1 字の unit 字形（[0,1]×[0,1] Y-UP）を返す。

        通常グリフはユーザー筆跡（``_direct_stroke``）→ KanjiVG 参照 → matplotlib
        skeleton の順で探す。大型構造記号（√・大括弧・∑・∫）は手書き字形が無いので
        matplotlib skeleton を使う。
        """
        # 数式中の文字も本文と同じ字形置換を適用する。matplotlib は指数のマイナスを
        # U+2212（数学マイナス・未収集）で返すため、収集済みの ASCII "-" 等へ寄せて
        # skeleton の崩れた記号でなく手書きの字形を使う。
        char = self._CHAR_SUBSTITUTIONS.get(char, char)
        # 根号 √ は render_math_handwritten 側で屋根と連結した1本のポリラインとして
        # 描くため、ここには来ない（グリフループで skip 済み）。
        if not is_large:
            # 数式は回転なし(vary=False)の素の字形を使う。instance_variation の回転で
            # 細い字(I, l)が傾く・字形が乱れるのを防ぐ。
            # アスペクト比を保持する（_direct_stroke は _normalize_strokes_to_unit 済み＝
            # 縦横同率で正規化）。軸独立の _normalize_unit_bbox を被せると細い字(1,I,l)が
            # 横に引き伸ばされ、bbox 配置時に斜め・曲がって見えるため使わない。
            direct = self._direct_stroke(char, vary=False)
            if direct is not None:
                return direct
            ref, _ = self._load_reference_strokes(char)
            if ref is not None:
                return self._normalize_strokes_to_unit(ref)
        skel = render_glyph_unit_strokes(char)
        if skel is not None:
            return self._normalize_strokes_to_unit([s.copy() for s in skel])
        return None

    @staticmethod
    def _normalize_unit_bbox(strokes: list[Stroke]) -> list[Stroke]:
        """ストローク列を [0,1]×[0,1] の bbox（アスペクト保持せず各軸独立）へ正規化する。

        ``_place_unit_in_pt_box`` が unit 字形を当該グリフのインク矩形へ各軸独立にスケール
        するため、入力字形を軸ごとに [0,1] へ写す。Y は既に Y-UP 想定のためそのまま。
        """
        all_pts = np.concatenate(strokes, axis=0)
        mins = all_pts.min(axis=0)
        ranges = all_pts.max(axis=0) - mins
        ranges = np.where(ranges < 1e-9, 1.0, ranges)
        return [((s - mins) / ranges) for s in strokes]

    def _place_unit_in_pt_box(
        self,
        unit_strokes: list[Stroke],
        x_lo_pt: float,
        y_lo_pt: float,
        w_pt: float,
        h_pt: float,
        to_mm,
    ) -> list[Stroke]:
        """字形を pt インク矩形へ**アスペクト比を保ったまま**収めて mm へ写す。

        軸独立スケールだと細い字(1, I, l)が横に引き伸ばされ斜め・曲がって見える。
        そこで一様スケール ``s = min(w/uw, h/uh)`` で bbox 内に収め、x は中央寄せ・
        y は下端(ベースライン側)合わせにする。``to_mm`` で mm(Y-UP) に変換する。
        """
        all_pts = np.concatenate(unit_strokes, axis=0)
        umin = all_pts.min(axis=0)
        umax = all_pts.max(axis=0)
        uw = max(umax[0] - umin[0], 1e-6)
        uh = max(umax[1] - umin[1], 1e-6)
        s = min(w_pt / uw, h_pt / uh)
        gw = uw * s
        # x: bbox 中央へ。y: bbox 下端(y_lo)から（インク矩形下端＝グリフ下端を合わせる）。
        x_off = x_lo_pt + (w_pt - gw) / 2.0 - umin[0] * s
        y_off = y_lo_pt - umin[1] * s
        out: list[Stroke] = []
        for st in unit_strokes:
            xs = x_off + st[:, 0] * s
            ys = y_off + st[:, 1] * s
            pts = np.array([to_mm(px, py) for px, py in zip(xs, ys)], dtype=np.float64)
            out.append(pts)
        return out

    # 揺らぎを満額にする画数の上限と、下限まで落とす画数。多画字は画間隔が
    # 狭く、揺らぎで画が隣のレーンへはみ出して「固まる」ため画数で逓減する。
    _WAVER_FULL_STROKES = 10
    _WAVER_MIN_STROKES = 20
    # 多画字と少画字の質感差を「最大3倍(0.3)→2倍(0.5)」に縮め、経路間の質感を揃える。
    _WAVER_FLOOR = 0.5

    # 横棒の右上がり下限保証パラメータ。日本語の手書きは横棒を右上がりに書く
    # 習性があるが、字全体slant・per-stroke回転・per-point offset が全てゼロ平均
    # 対称に乗るため約半数の横棒が右下がりに転ぶ。最終mm座標で横棒だけを下限角
    # まで起こす（既に十分右上がりの画は触らない）。字全体を傾けると斜体化して
    # 不自然なため、ここでは横棒個別を重心まわりに回す。
    _RISE_MIN_ANGLE = np.deg2rad(2.0)
    # 横棒判定: x方向に font_size*係数 以上広がり、かつ y広がりが x広がりの一定
    # 割合未満（概ね水平・直線的）な画のみ対象。縦画・斜め画・曲がった画は除外。
    _RISE_MIN_X_RANGE_RATIO = 0.25
    _RISE_MAX_Y_X_RATIO = 0.35

    @classmethod
    def _waver_scale(cls, n_strokes: int) -> float:
        """画数に応じた揺らぎ倍率 ∈ [_WAVER_FLOOR, 1.0] を返す。

        ``_WAVER_FULL_STROKES`` 画以下は 1.0（満額）、``_WAVER_MIN_STROKES`` 画
        以上は ``_WAVER_FLOOR``、間は線形。多画字ほど tremor/elastic と ML の
        per-point offset を縮らせ、画の重なり（固まり）を防ぐ。
        """
        lo, hi = cls._WAVER_FULL_STROKES, cls._WAVER_MIN_STROKES
        if n_strokes <= lo:
            return 1.0
        if n_strokes >= hi:
            return cls._WAVER_FLOOR
        frac = (n_strokes - lo) / (hi - lo)
        return 1.0 - frac * (1.0 - cls._WAVER_FLOOR)

    def _apply_instance_variation(
        self, strokes: list[Stroke], waver_scale: float = 1.0
    ) -> list[Stroke]:
        """同一字の繰り返しで形が変わるよう、画ごとに微小ランダムaffineを掛ける。

        画中心まわりの微小回転・スケールと、字サイズ比のシフトを画ごとに与える。
        強度は ``self._instance_variation`` ×（多画字で固まり防止のため）``waver_scale``。
        毎回 RNG が進むので「同じ字を書いても毎回違う」を作る。``augmenter`` が無い
        ／無効／強度 0 のときは恒等（クリーンモードでは変動なし）。
        """
        strength = self._instance_variation * waver_scale
        aug = self._augmenter
        if aug is None or not aug._config.enabled or strength <= 0 or not strokes:
            return strokes
        all_pts = np.concatenate(strokes, axis=0)
        span = float((all_pts.max(axis=0) - all_pts.min(axis=0)).max())
        if span < 1e-9:
            return strokes
        rng = aug._rng
        out: list[Stroke] = []
        for s in strokes:
            c = s.mean(axis=0)
            ang = rng.normal(0, strength * 0.04)
            ca, sa = np.cos(ang), np.sin(ang)
            sc = 1.0 + rng.normal(0, strength * 0.03)
            shift = rng.normal(0, strength * 0.04, size=2) * span
            r = (s - c) @ np.array([[ca, -sa], [sa, ca]]) * sc + c + shift
            out.append(r)
        return out

    def _apply_distortion(self, strokes: list[Stroke], waver_scale: float = 1.0) -> list[Stroke]:
        aug = self._augmenter
        if aug is None:
            return strokes
        # 既定振幅(elastic=0.002 bbox比, tremor=0.01mm)を waver_scale で縮める
        strokes = [aug.elastic_distort(s, amplitude=0.002 * waver_scale) for s in strokes]
        strokes = [aug.apply_tremor(s, amplitude=0.01 * waver_scale) for s in strokes]
        return strokes

    def _apply_symbol_distortion(self, strokes: list[Stroke]) -> list[Stroke]:
        """記号・句読点・括弧など幾何経路に微量の手書き揺らぎを乗せる。

        従来この経路は distortion 無し＝定規直線でツルツルだったため、本文の字と
        並ぶと「整いすぎ」で浮く。``_WAVER_SYMBOL`` の控えめな揺らぎで質感だけ
        本文へ寄せる。ただし句点「。」・ピリオドの点・極短ストロークは揺らぎで
        破綻するため ``_SYMBOL_WAVER_MIN_SPAN`` 未満の字形は素のまま返す。

        Args:
            strokes: 配置済みストローク列（mm 座標）。

        Returns:
            微量揺らぎを乗せた（または極短のため素のままの）ストローク列。
        """
        if self._augmenter is None or not strokes:
            return strokes
        all_pts = np.concatenate(strokes, axis=0)
        span = float((all_pts.max(axis=0) - all_pts.min(axis=0)).max())
        if span < 1e-9:
            return strokes
        # 揺らぎは bbox 比なので極端に小さい字形(点)は相対的に大きく崩れる。
        # 相対比・絶対長のどちらかで「短い」と判定したら素のまま返す。
        threshold = max(span * self._SYMBOL_WAVER_MIN_SPAN, self._SYMBOL_WAVER_MIN_LEN_MM)
        out: list[Stroke] = []
        for s in strokes:
            s_span = float((s.max(axis=0) - s.min(axis=0)).max())
            if s_span < threshold:
                out.append(s)
            else:
                out.extend(self._apply_distortion([s], waver_scale=self._WAVER_SYMBOL))
        return out

    def _enforce_horizontal_rise(self, strokes: list[Stroke], font_size: float) -> list[Stroke]:
        """横棒だけを緩い右上がりの下限角まで起こす（日本語手書きの右上がり再現）。

        全変形が乗った最終 mm 座標(Y-UP=上が大)を受け取り、概ね水平・直線的で
        十分に長い画（横棒）のみを対象に、x 昇順の始点→終点角が
        ``_RISE_MIN_ANGLE`` 未満なら重心まわりに反時計回り回転して右端を持ち上げる。
        既に十分右上がりの画・縦画・斜め画・曲がった画・短い画は素通り。

        Args:
            strokes: 配置済みストローク列（mm 座標, Y-UP）。
            font_size: 字の論理高（mm）。横棒の長さ判定の基準に使う。

        Returns:
            横棒のみ下限角まで起こしたストローク列（本数・点数は不変）。
        """
        min_x_range = font_size * self._RISE_MIN_X_RANGE_RATIO
        out: list[Stroke] = []
        for s in strokes:
            if s.shape[0] < 2:
                out.append(s)
                continue
            x = s[:, 0]
            y = s[:, 1]
            x_range = float(x.max() - x.min())
            y_range = float(y.max() - y.min())
            is_horizontal = x_range > min_x_range and y_range < x_range * self._RISE_MAX_Y_X_RATIO
            if not is_horizontal:
                out.append(s)
                continue
            order = np.argsort(x)
            start = s[order[0]]
            end = s[order[-1]]
            theta = float(np.arctan2(end[1] - start[1], end[0] - start[0]))
            if theta >= self._RISE_MIN_ANGLE:
                out.append(s)
                continue
            delta = self._RISE_MIN_ANGLE - theta
            center = s.mean(axis=0)
            ca, sa = np.cos(delta), np.sin(delta)
            rot = np.array([[ca, -sa], [sa, ca]])
            out.append((s - center) @ rot.T + center)
        return out

    def _direct_stroke(self, char: str, vary: bool = True) -> list[Stroke] | None:
        """ユーザーの実筆跡サンプルから字形を返す（文字ごとにサンプル固定）。

        同じ字には常に同じベースサンプルを使う（``_direct_choice_cache``）。初回は
        最も「丁寧に書かれた」＝総点数が多いサンプルを品質スコアとして選ぶ。等確率
        ランダムだと隣接する同一字でベース字形が入れ替わり品質が極端に振れるため。
        毎回の微小な多様性は ``_apply_stroke_variation`` / 温度ノイズが別途与える。

        Args:
            char: 描画対象文字。

        Returns:
            正規化＋微小バリエーション済みのストローク列。サンプルが無ければ None。
        """
        samples = self._user_stroke_db.get(char)
        if not samples:
            return None
        idx = self._direct_choice_cache.get(char)
        if idx is None:
            # 総点数が多い＝丁寧に書かれたサンプルを品質スコアとして best を選ぶ。
            idx = max(
                range(len(samples)),
                key=lambda i: sum(len(s) for s in samples[i]),
            )
            self._direct_choice_cache[char] = idx
        chosen = samples[idx]
        normalized = self._normalize_strokes_to_unit(chosen)
        # vary=False: 数式グリフ用。instance_variation のランダム affine(回転/シアー)を
        # かけない。細い縦字(I, l)が回転で大きく傾いて「斜め」に見える問題を避ける。
        if not vary:
            return normalized
        return self._apply_stroke_variation(normalized)

    @staticmethod
    def _normalize_strokes_to_unit(strokes: list[Stroke]) -> list[Stroke]:
        all_pts = np.concatenate(strokes, axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        span = (maxs - mins).max()
        if span < 1e-6:
            return strokes
        center = (mins + maxs) / 2
        result = []
        for s in strokes:
            normalized = (s - center) / span + 0.5
            normalized[:, 1] = 1.0 - normalized[:, 1]
            result.append(normalized)
        return result

    def _apply_stroke_variation(self, strokes: list[Stroke]) -> list[Stroke]:
        ns = self._NOISE_SCALE
        result = []
        for stroke in strokes:
            center = stroke.mean(axis=0)
            centered = stroke - center
            angle = np.random.normal(0, ns * 0.05)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotated = centered @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            sx = 1.0 + np.random.normal(0, ns * 0.03)
            sy = 1.0 + np.random.normal(0, ns * 0.03)
            scaled = rotated * np.array([sx, sy])
            dx = np.random.normal(0, ns * 0.1)
            dy = np.random.normal(0, ns * 0.1)
            result.append(scaled + center + np.array([dx, dy]))
        return result

    def _simple_punct_strokes(self, char: str) -> list[Stroke] | None:
        if char in ("\u3001", ",", "\uff0c"):
            # \u6b63\u898f\u5316\u5f8c\u306e\u672c\u6587\u8aad\u70b9\u306f\u3059\u3079\u3066\u300c\uff0c\u300d(U+FF0C)\u306b\u5bc4\u305b\u308b\u305f\u3081\u540c\u4e00\u5f62\u306b\u3059\u308b
            return [np.array([[0.58, 0.48], [0.42, 0.26]], dtype=np.float64)]
        elif char in ("\u3002", ".", "\uff0e"):
            # \u53e5\u70b9\u306f\u30ec\u30dd\u30fc\u30c8\u4f53\u88c1\u306b\u5408\u308f\u305b\u3001\u4e38(\u5186)\u3067\u306f\u306a\u304f\u30d4\u30ea\u30aa\u30c9\u98a8\u306e\u77ed\u3044\u70b9(\u63cf\u3051\u308b\u30c9\u30c3\u30c8)
            # \u6b63\u898f\u5316\u5f8c\u306e\u672c\u6587\u53e5\u70b9\u306f\u3059\u3079\u3066\u300c\uff0e\u300d(U+FF0E)\u306b\u5bc4\u305b\u308b\u305f\u3081\u540c\u4e00\u5f62\u306b\u3059\u308b
            return [np.array([[0.475, 0.245], [0.525, 0.205]], dtype=np.float64)]
        elif char == "\u30fb":
            return [self._middle_dot_spiral()]
        elif char == "\u00b7":
            # \u4e2d\u70b9 \u00b7\uff08kg\u00b7m\u00b2 \u7b49\uff09\u3002\u53ce\u96c6\u30b5\u30f3\u30d7\u30eb\u304c\u5857\u308a\u6f70\u3057\u30b9\u30af\u30ea\u30d6\u30eb\u3067\u25a1\u306b\u898b\u3048\u308b/\u77ed\u3044\u7dda\u3060\u3068
            # \u7dda\u306b\u898b\u3048\u308b\u305f\u3081\u3001\u4e2d\u307b\u3069(0.5,0.5)\u306b\u5c0f\u3055\u306a\u5857\u308a\u6f70\u3057\u306e\u70b9(\u30b9\u30d1\u30a4\u30e9\u30eb)\u3092\u63cf\u304f\u3002
            t = np.linspace(0.0, 8.0 * np.pi, 60)
            r = np.linspace(0.07, 0.0, t.size)
            return [
                np.stack([0.5 + r * np.cos(t), 0.5 + r * np.sin(t)], axis=1).astype(np.float64)
            ]
        return None

    @staticmethod
    def _slash_strokes(char: str) -> list[Stroke] | None:
        """``/`` ``\\`` を幾何の斜め線で返す（unit[0,1] Y-UP）。

        これらはディレクトリ名に使えず直接ストローク収集できない構造記号のため、
        m/s² 等の単位で隙間にならないよう斜め1画で描く。
        """
        if char == "/":
            return [np.array([[0.2, 0.0], [0.8, 1.0]], dtype=np.float64)]
        if char == "\\":
            return [np.array([[0.2, 1.0], [0.8, 0.0]], dtype=np.float64)]
        return None

    @staticmethod
    def _middle_dot_spiral() -> Stroke:
        t = np.linspace(0.0, 12.0 * np.pi, 145)
        r = np.linspace(0.15, 0.0, t.size)
        return np.stack([0.5 + r * np.cos(t), 0.5 + r * np.sin(t)], axis=1).astype(np.float64)

    def _simple_paren_strokes(self, char: str, placement: CharPlacement) -> list[Stroke] | None:
        if char in ("(", "\uff08"):
            # \u300c(\u300d\u306f\u5de6\u5bc4\u308a\u3067\u4e2d\u592e\u304c\u5de6\u306b\u51f8\uff08\u958b\u53e3\u306f\u53f3\u5411\u304d\uff09
            points = []
            for i in range(20):
                t = i / 19
                x = 0.40 - 0.25 * np.cos(np.pi * (t - 0.5))
                y = 0.1 + 0.8 * t
                points.append([x, y])
            return [np.array(points)]
        elif char in (")", "\uff09"):
            # \u300c)\u300d\u306f\u53f3\u5bc4\u308a\u3067\u4e2d\u592e\u304c\u53f3\u306b\u51f8\uff08\u958b\u53e3\u306f\u5de6\u5411\u304d\uff09
            points = []
            for i in range(20):
                t = i / 19
                x = 0.60 + 0.25 * np.cos(np.pi * (t - 0.5))
                y = 0.1 + 0.8 * t
                points.append([x, y])
            return [np.array(points)]
        elif char == "\u300c":
            return [
                np.array([[0.8, 0.15], [0.25, 0.15]], dtype=np.float64),
                np.array([[0.25, 0.15], [0.25, 0.45]], dtype=np.float64),
            ]
        elif char == "\u300d":
            return [
                np.array([[0.75, 0.55], [0.75, 0.85]], dtype=np.float64),
                np.array([[0.75, 0.85], [0.2, 0.85]], dtype=np.float64),
            ]
        elif char == "\u300e":
            return [
                np.array([[0.8, 0.15], [0.25, 0.15], [0.25, 0.45]], dtype=np.float64),
                np.array([[0.65, 0.25], [0.35, 0.25], [0.35, 0.45]], dtype=np.float64),
            ]
        elif char == "\u300f":
            return [
                np.array([[0.75, 0.55], [0.75, 0.85], [0.2, 0.85]], dtype=np.float64),
                np.array([[0.65, 0.55], [0.65, 0.75], [0.35, 0.75]], dtype=np.float64),
            ]
        return None

    @staticmethod
    def _strokes_and_types_from_sample(
        sample: StrokeSample,
    ) -> tuple[list[Stroke], list[str]]:
        """``StrokeSample`` を ``len>=2`` 条件で strokes/types を同期フィルタする。

        推論バッチ化（``inference.py``）は ``len>=2`` のストロークだけを同順で
        返すため、筆画タイプ（``stroke_types``）も同じ条件・同じ順序で間引き、
        ``positioned[j]`` と ``types[j]`` の 1 対 1 対応を保つ。

        Args:
            sample: 読み込んだ ``StrokeSample``。

        Returns:
            ``(strokes, types)``。``types`` は対応する ``stroke_types``（無い・
            不足分は ``""`` で補完し、strokes と同数になる）。
        """
        raw_types = sample.stroke_types
        strokes: list[Stroke] = []
        types: list[str] = []
        for i, stroke_points in enumerate(sample.strokes):
            arr = np.array([[p.x, p.y] for p in stroke_points], dtype=np.float64)
            if len(arr) >= 2:
                strokes.append(arr)
                types.append(raw_types[i] if i < len(raw_types) else "")
        return strokes, types

    @staticmethod
    def _is_ml_deformable(char: str) -> bool:
        """ML変形(per-point offset)を適用してよい字か判定する。

        本番モデルはユーザーの CJK（漢字・かな）のみで訓練されており、数字に
        per-point offset を当てると下の横線等が歪んで字形が壊れる。数字は素の
        参照字形（KanjiVG フォント由来）をそのまま使う方が正確なため除外する。
        """
        return char not in "0123456789"

    @staticmethod
    def _is_safe_glyph_key(char: str) -> bool:
        """KanjiVG ディレクトリの glob キーに使って安全な字か判定する。

        Python 3.14 の ``Path.glob`` はパス区切りを含む非相対パターンで
        ``NotImplementedError`` を投げる。``/`` ``\\`` ``.`` ``..`` などを
        含む記号がここに到達すると全描画が落ちるため、事前に弾く。
        """
        return bool(char) and "/" not in char and "\\" not in char and char not in (".", "..")

    def _load_reference_strokes(self, char: str) -> tuple[list[Stroke] | None, list[str]]:
        if self._kanjivg_dir is None or not self._is_safe_glyph_key(char):
            return None, []
        char_dir = self._kanjivg_dir / char
        if not char_dir.is_dir():
            return None, []
        json_files = sorted(char_dir.glob(f"{char}_*.json"))
        if not json_files:
            return None, []
        try:
            sample = StrokeSample.load(json_files[0])
            strokes, types = self._strokes_and_types_from_sample(sample)
            return (strokes if strokes else None), types
        except Exception:
            return None, []

    def _load_kanjivg_json(self, placement: CharPlacement) -> tuple[list[Stroke] | None, list[str]]:
        if self._kanjivg_dir is None or not self._is_safe_glyph_key(placement.char):
            return None, []
        char_dir = self._kanjivg_dir / placement.char
        if not char_dir.is_dir():
            return None, []
        json_files = sorted(char_dir.glob(f"{placement.char}_*.json"))
        if not json_files:
            return None, []
        try:
            sample = StrokeSample.load(json_files[0])
            strokes, types = self._strokes_and_types_from_sample(sample)
            return (strokes if strokes else None), types
        except Exception:
            logger.warning("KanjiVG JSON load failed for '%s'", placement.char, exc_info=True)
            return None, []

    # 半角セル幅 = font_size × この係数。typesetter は字送りを font*0.55 で予約し、
    # 字種係数(半角=0.8)を font_size に焼くため fs=font*0.8。箱を予約と一致させるには
    # fs*(0.55/0.8)=fs*0.6875≈0.7 が必要。これより小さいと数字・英字が横ボックス律速で
    # 縦に潰れ(二重縮小)、漢字比0.6域へ縮む。
    _HALFWIDTH_CELL_FACTOR = 0.7

    def _position_strokes(
        self,
        strokes: list[Stroke],
        placement: CharPlacement,
    ) -> list[Stroke]:
        """\u5b57\u5f62\u3092\u914d\u7f6e\u5ea7\u6a19\u3078\u30b9\u30b1\u30fc\u30eb\u30fb\u79fb\u52d5\u3059\u308b\u3002

        Args:
            strokes: \u5358\u4f4d\u7cfb\u5b57\u5f62\u306e\u30b9\u30c8\u30ed\u30fc\u30af\u5217\u3002
            placement: \u914d\u7f6e\u60c5\u5831\uff08x, y, font_size, char, slant\uff09\u3002

        Returns:
            \u914d\u7f6e\u6e08\u307f\u30b9\u30c8\u30ed\u30fc\u30af\u5217\u3002
        """
        if not strokes:
            return []
        if placement.char in ("\u3001", ",", "\uff0c"):
            return [self._position_comma_mark(strokes, placement)]
        if placement.char in (".", "\u3002", "\uff0e"):
            return [self._position_period_dot(placement)]
        if placement.char == "\u30fb":
            return self._position_middle_dot(strokes, placement)

        all_pts = np.concatenate(strokes, axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        ranges = maxs - mins

        fs = placement.font_size
        # サイズ係数(種別×密度)は typesetter が placement.font_size に1回だけ焼く。
        # renderer は font_size を最終目標高として扱い、再適用しない(二重縮小防止)。
        target_h = fs

        if is_halfwidth(placement.char):
            # 半角の縦長アスペクトはセル幅で表現する(係数の再適用ではない)。
            # 箱は字送り予約(font*0.55)と整合する fs*0.7。狭いと横律速で縦潰れ(二重縮小)。
            cell_width = fs * self._HALFWIDTH_CELL_FACTOR
        else:
            # 全角セルの横はみ出し抑制。effective ではなく固定係数で保つ。
            cell_width = fs * 0.95

        scale_w = cell_width / ranges[0] if ranges[0] > 1e-6 else float("inf")
        scale_h = target_h / ranges[1] if ranges[1] > 1e-6 else float("inf")
        scale = min(scale_w, scale_h)

        scaled = [(stroke - mins) * scale for stroke in strokes]
        rendered_w = ranges[0] * scale
        rendered_h = ranges[1] * scale

        # 半角・全角とも cell_width 基準で中央寄せ(箱拡大と整合させ左ズレを防ぐ)。
        x_offset = placement.x + (cell_width - rendered_w) / 2

        line_spacing = self._page_config.line_spacing
        # 小書き仮名・句読点は字種スケールが小さく、行box中央だと浮くため下寄せにする。
        # font_size に係数が焼かれた後は字種判定でしか小書き/句読点を見分けられないので
        # char_type_scale(係数の再適用ではなく分類用)で旧 effective<0.5 と同じ集合を選ぶ。
        if char_type_scale(placement.char) < 0.6:
            y_offset = placement.y + 0.1 * line_spacing
        else:
            y_offset = placement.y + (line_spacing - rendered_h) / 2

        offset = np.array([x_offset, y_offset])
        positioned = [stroke + offset for stroke in scaled]

        # 文字単位の微小傾き（手書きの揺らぎ）。文字内の全画を同一角で中心回転。
        slant = getattr(placement, "slant", 0.0)
        if slant:
            cx = x_offset + rendered_w / 2
            cy = y_offset + rendered_h / 2
            cos_a, sin_a = np.cos(slant), np.sin(slant)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            center = np.array([cx, cy])
            positioned = [(s - center) @ rot.T + center for s in positioned]

        return positioned

    def _position_comma_mark(self, strokes: list[Stroke], placement: CharPlacement) -> Stroke:
        all_pts = np.concatenate(strokes, axis=0)
        mins = all_pts.min(axis=0)
        ranges = all_pts.max(axis=0) - mins

        fs = placement.font_size
        line_spacing = self._page_config.line_spacing
        cell_width = fs * 0.55 if is_halfwidth(placement.char) else fs * 0.95
        target_h = fs * 0.3675
        target_w = target_h * 0.72
        scale_w = target_w / ranges[0] if ranges[0] > 1e-6 else float("inf")
        scale_h = target_h / ranges[1] if ranges[1] > 1e-6 else float("inf")
        scale = min(scale_w, scale_h)

        rendered_w = ranges[0] * scale
        rendered_h = ranges[1] * scale
        x_offset = placement.x + (cell_width - rendered_w) / 2
        y_offset = placement.y + line_spacing * 0.1
        stroke = (strokes[0] - mins) * scale + np.array([x_offset, y_offset])

        slant = getattr(placement, "slant", 0.0)
        if slant:
            center = np.array([x_offset + rendered_w / 2, y_offset + rendered_h / 2])
            cos_a, sin_a = np.cos(slant), np.sin(slant)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            stroke = (stroke - center) @ rot.T + center

        return stroke

    def _position_middle_dot(self, strokes: list[Stroke], placement: CharPlacement) -> list[Stroke]:
        fs = placement.font_size
        cell_width = fs * 0.95
        dot_diameter = cell_width * 0.5
        scale = dot_diameter / 0.3

        center = np.array(
            [
                placement.x + cell_width / 2,
                placement.y + self._page_config.line_spacing / 2,
            ],
            dtype=np.float64,
        )
        raw_center = np.array([0.5, 0.5], dtype=np.float64)
        positioned = [(stroke - raw_center) * scale + center for stroke in strokes]

        slant = getattr(placement, "slant", 0.0)
        if slant:
            cos_a, sin_a = np.cos(slant), np.sin(slant)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            positioned = [(stroke - center) @ rot.T + center for stroke in positioned]

        return positioned

    def _position_period_dot(self, placement: CharPlacement) -> Stroke:
        fs = placement.font_size
        line_spacing = self._page_config.line_spacing
        cell_width = fs * 0.55
        dot_w = min(0.58, max(0.38, fs * 0.08))
        dot_h = dot_w * 0.75
        center_x = placement.x + cell_width * 0.5
        center_y = placement.y + line_spacing * 0.14
        return np.array(
            [
                [center_x - dot_w / 2, center_y + dot_h / 2],
                [center_x + dot_w / 2, center_y - dot_h / 2],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rect_fallback(p: CharPlacement) -> list[Stroke]:
        half = p.font_size / 2.0
        x0, y0 = p.x, p.y - half
        x1, y1 = p.x + p.font_size, p.y + half
        rect = np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [x0, y0],
            ]
        )
        return [rect]
