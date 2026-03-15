"""CASIA-OLHWDB .pot ファイルを StrokeSample JSON に変換する CLI。

使い方:
    python scripts/prepare_casia.py --input-dir data/casia_raw --output-dir data/casia_strokes
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector.casia_parser import CASIAParser

logger = logging.getLogger(__name__)


def convert_pot_directory(
    input_dir: Path,
    output_dir: Path,
    target_size: float = 10.0,
    num_points: int = 32,
) -> int:
    """input_dir 内の全 .pot ファイルを StrokeSample JSON に変換する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    parser = CASIAParser()

    pot_files = sorted(input_dir.glob("*.pot"))
    if not pot_files:
        logger.warning("No .pot files found in %s", input_dir)
        return 0

    logger.info("%d .pot files found in %s", len(pot_files), input_dir)

    total_count = 0
    char_counters: dict[str, int] = {}
    for i, pot_file in enumerate(pot_files, 1):
        samples = parser.parse_pot_file(pot_file)
        count = CASIAParser.convert_to_stroke_samples(
            samples, output_dir, target_size=target_size, num_points=num_points,
            char_counters=char_counters,
        )
        total_count += count
        logger.info(
            "[%d/%d] %s: %d samples (%d characters) -> %d converted",
            i,
            len(pot_files),
            pot_file.name,
            len(samples),
            len({s.character for s in samples}),
            count,
        )

    return total_count


def main() -> None:
    argp = argparse.ArgumentParser(
        description="CASIA-OLHWDB .pot ファイルを StrokeSample JSON に変換"
    )
    argp.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=".pot ファイルが含まれるディレクトリ",
    )
    argp.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/casia_strokes"),
        help="出力ディレクトリ (default: data/casia_strokes)",
    )
    argp.add_argument(
        "--target-size",
        type=float,
        default=10.0,
        help="正規化ターゲットサイズ (default: 10.0)",
    )
    argp.add_argument(
        "--num-points",
        type=int,
        default=32,
        help="ストロークあたりのリサンプリング点数 (default: 32)",
    )

    args = argp.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.input_dir.exists():
        print(f"Error: input directory not found: {args.input_dir}")
        sys.exit(1)

    count = convert_pot_directory(
        args.input_dir, args.output_dir, args.target_size, args.num_points
    )
    logger.info("変換完了: %d サンプル", count)

    if count == 0:
        print("Warning: no samples converted. Check that .pot files exist in input directory.")
    else:
        char_dirs = [d for d in args.output_dir.iterdir() if d.is_dir()]
        print(f"Done: {count} samples, {len(char_dirs)} unique characters -> {args.output_dir}")


if __name__ == "__main__":
    main()
