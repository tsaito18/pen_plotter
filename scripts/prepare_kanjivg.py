"""KanjiVG XMLデータをStrokeSample JSON形式に変換するスクリプト。"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector.data_format import StrokePoint, StrokeSample
from src.collector.kanjivg_parser import KanjiVGParser, parse_svg_path
from src.collector.stroke_recorder import StrokeRecorder

logger = logging.getLogger(__name__)

KANJIVG_URL = (
    "https://github.com/KanjiVG/kanjivg/releases/download/r20240807/kanjivg-20240807.xml.gz"
)


def hex_filename_to_char(stem: str) -> str:
    """16進コードポイント文字列からUnicode文字に変換。"""
    return chr(int(stem, 16))


def download_kanjivg(dest_dir: Path) -> Path:
    """KanjiVG XMLファイルをダウンロードして展開。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    gz_path = dest_dir / "kanjivg.xml.gz"
    xml_path = dest_dir / "kanjivg.xml"

    if xml_path.exists():
        logger.info("既存のXMLファイルを使用: %s", xml_path)
        return xml_path

    logger.info("KanjiVGをダウンロード中: %s", KANJIVG_URL)
    urllib.request.urlretrieve(KANJIVG_URL, gz_path)  # noqa: S310

    with gzip.open(gz_path, "rb") as f_in, open(xml_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    gz_path.unlink()
    logger.info("展開完了: %s", xml_path)
    return xml_path


def _strokes_to_sample(
    character: str,
    raw_strokes: list,
    recorder: StrokeRecorder,
    num_points: int,
    source_info: str,
    target_size: float = 10.0,
) -> StrokeSample | None:
    """パース済み・正規化済みストローク配列をStrokeSampleに変換。

    入力ストロークはKanjiVGParser.normalize()で全ストローク一括正規化済みを想定。
    SVG座標系（Y下向き）からプロッタ座標系（Y上向き）へのY軸反転も行う。
    """
    if not raw_strokes:
        return None

    stroke_points_list: list[list[StrokePoint]] = []
    for arr in raw_strokes:
        if len(arr) < 2:
            continue
        points = [
            StrokePoint(
                x=float(x), y=float(target_size - y), pressure=1.0, timestamp=0.0
            )
            for x, y in arr
        ]
        points = recorder.resample_points(points, num_points=num_points)
        stroke_points_list.append(points)

    if not stroke_points_list:
        return None

    return StrokeSample(
        character=character,
        strokes=stroke_points_list,
        metadata={"source": "kanjivg", "origin": source_info},
    )


def convert_single_svg(
    svg_path: Path,
    output_dir: Path,
    target_size: float = 10.0,
    num_points: int = 32,
) -> StrokeSample | None:
    """単一のSVGファイルをStrokeSampleに変換して保存。"""
    stem = svg_path.stem

    try:
        character = hex_filename_to_char(stem)
    except ValueError:
        logger.warning("16進コードポイントとして解析不可: %s", stem)
        return None

    parser = KanjiVGParser()
    raw_strokes = parser.parse_file(svg_path)
    normalized = parser.normalize(raw_strokes, target_size=target_size)

    recorder = StrokeRecorder(target_size=target_size, output_dir=output_dir)
    sample = _strokes_to_sample(character, normalized, recorder, num_points, svg_path.name)

    if sample is None:
        return None

    char_dir = output_dir / character
    char_dir.mkdir(parents=True, exist_ok=True)
    sample.save(char_dir / f"{character}_0.json")
    return sample


def convert_kanjivg_to_samples(
    svg_dir: Path,
    output_dir: Path,
    target_size: float = 10.0,
    num_points: int = 32,
) -> int:
    """SVGディレクトリ内の全ファイルをStrokeSampleに一括変換。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for svg_file in sorted(svg_dir.glob("*.svg")):
        result = convert_single_svg(svg_file, output_dir, target_size, num_points)
        if result is not None:
            count += 1
    return count


def convert_xml_to_samples(
    xml_path: Path,
    output_dir: Path,
    target_size: float = 10.0,
    num_points: int = 32,
) -> int:
    """KanjiVG統合XMLを直接パースしてStrokeSample JSONに変換。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    parser = KanjiVGParser()
    recorder = StrokeRecorder(target_size=target_size, output_dir=output_dir)
    count = 0

    for kanji_elem in root.iter("kanji"):
        kanji_id = kanji_elem.get("id", "")
        match = re.search(r"kanji_([0-9a-fA-F]+)", kanji_id)
        if not match:
            continue

        try:
            character = chr(int(match.group(1), 16))
        except (ValueError, OverflowError):
            continue

        raw_strokes = []
        for path_elem in kanji_elem.iter("path"):
            d = path_elem.get("d", "")
            if d:
                points = parse_svg_path(d)
                if len(points) > 0:
                    raw_strokes.append(points)

        normalized = parser.normalize(raw_strokes, target_size=target_size)
        sample = _strokes_to_sample(character, normalized, recorder, num_points, xml_path.name)

        if sample is None:
            continue

        char_dir = output_dir / character
        char_dir.mkdir(parents=True, exist_ok=True)
        sample.save(char_dir / f"{character}_0.json")
        count += 1

        if count % 1000 == 0:
            logger.info("変換中: %d 文字完了...", count)

    return count


def main() -> None:
    argp = argparse.ArgumentParser(description="KanjiVG データをStrokeSample JSON形式に変換")
    argp.add_argument("--xml-path", type=Path, help="KanjiVG XMLファイルパス（直接指定）")
    argp.add_argument("--svg-dir", type=Path, help="KanjiVG SVGファイルのディレクトリ")
    argp.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/strokes"),
        help="出力ディレクトリ (default: data/strokes)",
    )
    argp.add_argument("--download", action="store_true", help="KanjiVGデータをダウンロードして変換")
    argp.add_argument(
        "--target-size", type=float, default=10.0, help="正規化ターゲットサイズ (default: 10.0)"
    )
    argp.add_argument(
        "--num-points", type=int, default=32, help="ストロークあたりのリサンプリング点数 (default: 32)"
    )

    args = argp.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.download:
        xml_path = download_kanjivg(args.output_dir / "kanjivg_raw")
        count = convert_xml_to_samples(xml_path, args.output_dir, args.target_size, args.num_points)
        logger.info("変換完了: %d 文字", count)
    elif args.xml_path:
        count = convert_xml_to_samples(
            args.xml_path, args.output_dir, args.target_size, args.num_points
        )
        logger.info("変換完了: %d 文字", count)
    elif args.svg_dir:
        count = convert_kanjivg_to_samples(
            args.svg_dir, args.output_dir, args.target_size, args.num_points
        )
        logger.info("変換完了: %d 文字", count)
    else:
        argp.print_help()
        print("\n--download, --xml-path, または --svg-dir のいずれかを指定してください。")
        sys.exit(1)


if __name__ == "__main__":
    main()
