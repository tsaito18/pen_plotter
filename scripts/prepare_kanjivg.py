"""KanjiVG SVGデータをStrokeSample JSON形式に変換するスクリプト。"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector.data_format import StrokePoint, StrokeSample
from src.collector.kanjivg_parser import KanjiVGParser
from src.collector.stroke_recorder import StrokeRecorder

logger = logging.getLogger(__name__)

KANJIVG_URL = (
    "https://github.com/KanjiVG/kanjivg/releases/download/r20240807/kanjivg-20240807.xml.gz"
)


def hex_filename_to_char(stem: str) -> str:
    """SVGファイル名（16進コードポイント）からUnicode文字に変換。"""
    return chr(int(stem, 16))


def download_kanjivg(dest_dir: Path) -> Path:
    """KanjiVG XMLファイルをダウンロードして展開。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    gz_path = dest_dir / "kanjivg.xml.gz"
    xml_path = dest_dir / "kanjivg.xml"

    logger.info("KanjiVGをダウンロード中: %s", KANJIVG_URL)
    urllib.request.urlretrieve(KANJIVG_URL, gz_path)  # noqa: S310

    with gzip.open(gz_path, "rb") as f_in, open(xml_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    gz_path.unlink()
    logger.info("展開完了: %s", xml_path)
    return xml_path


def convert_single_svg(
    svg_path: Path,
    output_dir: Path,
    target_size: float = 10.0,
    num_points: int = 32,
) -> StrokeSample | None:
    """単一のSVGファイルをStrokeSampleに変換して保存。

    Args:
        svg_path: KanjiVG SVGファイルパス
        output_dir: 出力先ディレクトリ
        target_size: 正規化後のサイズ
        num_points: リサンプリング点数

    Returns:
        変換されたStrokeSample。ファイル名が不正な場合はNone。
    """
    stem = svg_path.stem

    try:
        character = hex_filename_to_char(stem)
    except ValueError:
        logger.warning("16進コードポイントとして解析不可: %s", stem)
        return None

    parser = KanjiVGParser()
    raw_strokes = parser.parse_file(svg_path)

    if not raw_strokes:
        logger.warning("ストロークが空: %s", svg_path)
        return None

    recorder = StrokeRecorder(target_size=target_size, output_dir=output_dir)

    stroke_points_list: list[list[StrokePoint]] = []
    for arr in raw_strokes:
        points = [StrokePoint(x=float(x), y=float(y), pressure=1.0, timestamp=0.0) for x, y in arr]
        points = recorder.normalize_points(points)
        points = recorder.resample_points(points, num_points=num_points)
        stroke_points_list.append(points)

    sample = StrokeSample(
        character=character,
        strokes=stroke_points_list,
        metadata={"source": "kanjivg", "svg_file": svg_path.name},
    )

    char_dir = output_dir / character
    char_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{character}_0.json"
    sample.save(char_dir / filename)

    return sample


def convert_kanjivg_to_samples(
    svg_dir: Path,
    output_dir: Path,
    target_size: float = 10.0,
    num_points: int = 32,
) -> int:
    """SVGディレクトリ内の全ファイルをStrokeSampleに一括変換。

    Returns:
        変換に成功した文字数。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for svg_file in sorted(svg_dir.glob("*.svg")):
        result = convert_single_svg(svg_file, output_dir, target_size, num_points)
        if result is not None:
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="KanjiVG SVGデータをStrokeSample JSON形式に変換")
    parser.add_argument(
        "--svg-dir",
        type=Path,
        help="KanjiVG SVGファイルのディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/strokes"),
        help="出力ディレクトリ (default: data/strokes)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="KanjiVGデータをダウンロード",
    )
    parser.add_argument(
        "--target-size",
        type=float,
        default=10.0,
        help="正規化ターゲットサイズ (default: 10.0)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=32,
        help="ストロークあたりのリサンプリング点数 (default: 32)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.download:
        xml_path = download_kanjivg(args.output_dir / "kanjivg_raw")
        logger.info("ダウンロード完了: %s", xml_path)

    if args.svg_dir:
        count = convert_kanjivg_to_samples(
            args.svg_dir, args.output_dir, args.target_size, args.num_points
        )
        logger.info("変換完了: %d 文字", count)


if __name__ == "__main__":
    main()
