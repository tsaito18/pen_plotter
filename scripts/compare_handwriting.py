"""手書き生成結果のA/B目視比較用「定点観測」スクリプト。

固定サンプルテキストを固定 seed で手書き生成し、ページ PNG を保存する。
品質改善の前後で同じ tag を変えて再実行し、出力 PNG を見比べるための道具。

再現性の前提:
- HandwritingAugmenter に seed を渡してレイアウト揺らぎ（行内ドリフト・字間・
  サイズ・傾き）を固定する（PlotterPipeline(seed=...)）。
- StrokeRenderer はストローク選択・instance variation に numpy グローバル乱数を
  使うため、np.random.seed() でグローバル系列も固定する。
両方を固定しないと同一 seed でも絵が変わり比較不能になる。
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui.web_app import PlotterPipeline

# 品質の混在を1ページで網羅するサンプル:
# 横棒の傾き(漢字)・かな・英数・同一字反復(品質ばらつき確認)・%付き文・インライン数式
SAMPLE_TEXT = """一二三 土工言 月日目
あいうえお かきくけこ さしすせそ
本本本本 川川川川 国国国国
Report 2026 test ABCDEF
気温は20度、湿度は50%でした。
電圧と電流の関係は $V=IR$ である。"""

DEFAULT_OUT = Path("data/experiments/handwriting_compare")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-seed handwriting preview PNGs for A/B comparison"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"出力ディレクトリ (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="ファイル名に入れる世代タグ (default: baseline)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数 seed (default: 42)",
    )
    parser.add_argument(
        "--messiness",
        type=float,
        default=1.0,
        help="汚さ倍率 (0=整った字, 1=標準, 2=大きく乱れる) (default: 1.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="字形ゆらぎ温度 (0=決定論, 1=標準) (default: 1.0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/finetuned.pt"),
        help="ML チェックポイント (default: data/models/finetuned.pt)",
    )
    parser.add_argument(
        "--kanjivg-dir",
        type=Path,
        default=Path("data/strokes"),
        help="KanjiVG ストロークディレクトリ (default: data/strokes)",
    )
    parser.add_argument(
        "--user-strokes-dir",
        type=Path,
        default=Path("data/user_strokes"),
        help="ユーザーストロークディレクトリ (default: data/user_strokes)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint if args.checkpoint.exists() else None
    kanjivg_dir = args.kanjivg_dir if args.kanjivg_dir.exists() else None
    user_strokes_dir = args.user_strokes_dir if args.user_strokes_dir.exists() else None

    mode: list[str] = []
    if checkpoint:
        mode.append(f"ML推論({checkpoint})")
    if user_strokes_dir:
        mode.append(f"スタイル({user_strokes_dir})")
    if kanjivg_dir:
        mode.append(f"KanjiVG({kanjivg_dir})")
    if not mode:
        mode.append("矩形フォールバック")
    print(f"描画モード: {', '.join(mode)}")
    print(f"seed={args.seed}  messiness={args.messiness}  tag={args.tag}")

    # グローバル numpy 乱数も固定（StrokeRenderer のサンプル選択・instance variation 用）
    np.random.seed(args.seed)

    pipeline = PlotterPipeline(
        checkpoint_path=checkpoint,
        kanjivg_dir=kanjivg_dir,
        user_strokes_dir=user_strokes_dir,
        messiness=args.messiness,
        temperature=args.temperature,
        seed=args.seed,
    )

    # 既存画像を上書きしないよう tag をファイル名へ。複数ページ時は web_app 側が
    # "<stem>_p<i><suffix>" を付けるため、stem に連番マーカーを含めておく。
    base_path = args.out / f"handwriting_{args.tag}_p.png"
    pages = pipeline.generate_preview(SAMPLE_TEXT, base_path)

    print(f"\n生成ページ数: {len(pages)}")
    for p in pages:
        print(str(Path(p).resolve()))


if __name__ == "__main__":
    main()
