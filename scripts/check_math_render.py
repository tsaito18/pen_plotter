"""数式レンダリング目視確認用の最小スクリプト。

PlotterPipeline をデフォルト設定（KanjiVG のみ、ML 推論なし）で起動し、
複数のサンプル数式を含むテキストを matplotlib プレビューとして PNG に保存する。
レポート用紙背景・罫線・ページ番号も含めて実出力に近い状態で描画する。

A1+A2+B 完了後の確認用:
- ASCII 数式記号（+ - = 等）の新規ストローク（A1）
- LaTeX コマンド → Unicode 変換（\\omega \\pi \\theta 等、A2）
- ブロック数式 $$\\n...\\n$$ の DOTALL 対応（B）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ui.web_app import PlotterPipeline


SAMPLE = r"""# 数式レンダリングテスト

インラインで $E = mc^2$ や $f(x) = x^2 + 2x + 1$ などを書ける。

## ネスト分数

$$
\frac{\frac{1}{2}}{x + 1} \tag{1}
$$

## 累乗内分数と根号

$$
y = x^{\frac{1}{2}} + \sqrt{x^2 + 1} \tag{2}
$$

## 強制改行で連立

$$
y = ax + b \\
y = 2x + 3 \\
y = -x + 1 \tag{3}
$$

## ギリシャ文字と記号

$$
\omega = 2\pi f, \quad \theta \approx 0 \tag{4}
$$

$$
\sigma^2 = \frac{1}{n} \sum (x_i - \mu)^2 \tag{5}
$$
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/math_render_test_v2.png"),
        help="出力 PNG パス（複数ページ時はサフィックス自動付与）",
    )
    parser.add_argument(
        "--kanjivg",
        type=Path,
        default=Path("data/strokes"),
        help="KanjiVG JSON ディレクトリ",
    )
    args = parser.parse_args()

    kanjivg_dir = args.kanjivg if args.kanjivg.is_dir() else None
    pipeline = PlotterPipeline(kanjivg_dir=kanjivg_dir)

    paths = pipeline.generate_preview(SAMPLE, args.out)
    for p in paths:
        size = p.stat().st_size if p.exists() else 0
        print(f"wrote {p} ({size} bytes)")


if __name__ == "__main__":
    main()
