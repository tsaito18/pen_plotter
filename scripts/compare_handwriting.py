"""手書き生成結果のA/B目視比較用「定点観測」スクリプト。

固定サンプルテキストを固定 seed で手書き生成し、ページ PNG を保存する。
品質改善の前後で同じ tag を変えて再実行し、出力 PNG を見比べるための道具。

再現性: ``--seed`` で全ての揺らぎ（配置・字形・ML の温度ノイズ）を固定する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import PlotterPipeline
from src.settings import Settings

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
        help="ユーザー筆跡（プロファイルのルート可）(default: data/user_strokes)",
    )
    parser.add_argument("--profile", default=None, help="人物プロファイル ID（省略時は先頭）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"seed={args.seed}  messiness={args.messiness}  tag={args.tag}")
    pipeline = PlotterPipeline(
        Settings(messiness=args.messiness, temperature=args.temperature),
        checkpoint_path=args.checkpoint,
        kanjivg_dir=args.kanjivg_dir,
        user_strokes_dir=args.user_strokes_dir,
        profile=args.profile,
        seed=args.seed,
    )

    # 既存画像を上書きしないよう tag をファイル名へ（複数ページは "_p<i>" が付く）
    base_path = args.out / f"handwriting_{args.tag}_p.png"
    pages = pipeline.generate_preview(SAMPLE_TEXT, base_path)

    print(f"\n生成ページ数: {len(pages)}")
    for p in pages:
        print(str(Path(p).resolve()))


if __name__ == "__main__":
    main()
