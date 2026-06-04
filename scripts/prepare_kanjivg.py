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

_WINDOWS_FORBIDDEN = set(r'\/:*?"<>|')

# ElementTree が xmlns:kvg 宣言を解決した後の kvg:type 属性の完全修飾名。
# KanjiVG 統合 XML はルートに xmlns:kvg を宣言済み。
_KVG_TYPE_ATTR = "{http://kanjivg.tagaini.net}type"


def _is_valid_filename_char(character: str) -> bool:
    """Windowsファイルシステムで使用可能な文字かチェック。"""
    return character not in _WINDOWS_FORBIDDEN


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
    types: list[str] | None = None,
) -> StrokeSample | None:
    """パース済み・正規化済みストローク配列をStrokeSampleに変換。

    入力ストロークはKanjiVGParser.normalize()で全ストローク一括正規化済みを想定。
    SVG座標系（Y下向き）からプロッタ座標系（Y上向き）へのY軸反転も行う。

    Args:
        character: 対象文字。
        raw_strokes: 正規化済みストローク配列のリスト。
        recorder: リサンプリング用 StrokeRecorder。
        num_points: ストロークあたりのリサンプリング点数。
        source_info: メタデータに残す出自情報。
        target_size: 正規化ターゲットサイズ（Y反転に使用）。
        types: raw kvg:type 文字列のリスト。``raw_strokes`` と index 対応し、
            無い場合は全ストローク ``""`` 扱い。``len<2`` でスキップされた
            ストロークの type も同条件で除外され、strokes と stroke_types の
            index 対応が厳密に保たれる。分類はレンダリング時に行うため、ここ
            では raw 値をそのまま保存する。

    Returns:
        変換済み StrokeSample。有効ストロークが無ければ None。
    """
    if not raw_strokes:
        return None

    if types is None:
        types = [""] * len(raw_strokes)

    stroke_points_list: list[list[StrokePoint]] = []
    stroke_types: list[str] = []
    for idx, arr in enumerate(raw_strokes):
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
        stroke_types.append(types[idx] if idx < len(types) else "")

    if not stroke_points_list:
        return None

    return StrokeSample(
        character=character,
        strokes=stroke_points_list,
        metadata={"source": "kanjivg", "origin": source_info},
        stroke_types=stroke_types,
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

    if not _is_valid_filename_char(character):
        logger.debug("ファイル名に使用不可な文字をスキップ: U+%04X", ord(character))
        return None

    parser = KanjiVGParser()
    raw_strokes, raw_types = parser.parse_file_with_types(svg_path)
    normalized = parser.normalize(raw_strokes, target_size=target_size)

    recorder = StrokeRecorder(target_size=target_size, output_dir=output_dir)
    sample = _strokes_to_sample(
        character, normalized, recorder, num_points, svg_path.name, types=raw_types
    )

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

        if not _is_valid_filename_char(character):
            continue

        raw_strokes = []
        raw_types: list[str] = []
        for path_elem in kanji_elem.iter("path"):
            d = path_elem.get("d", "")
            if d:
                points = parse_svg_path(d)
                if len(points) > 0:
                    raw_strokes.append(points)
                    raw_types.append(path_elem.get(_KVG_TYPE_ATTR, ""))

        normalized = parser.normalize(raw_strokes, target_size=target_size)
        sample = _strokes_to_sample(
            character, normalized, recorder, num_points, xml_path.name, types=raw_types
        )

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
