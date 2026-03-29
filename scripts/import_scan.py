"""スキャン画像から手書き文字ストロークを取り込むCLIスクリプト。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector.data_format import StrokePoint, StrokeSample

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="スキャン画像から手書き文字ストロークを取り込む")
    parser.add_argument("--image", type=Path, required=True, help="入力画像パス")
    parser.add_argument("--text", type=str, default=None, help="画像内のテキスト内容（省略時は--autoが必要）")
    parser.add_argument("--auto", action="store_true", help="OCRで自動テキスト認識")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scan_strokes"),
        help="出力ディレクトリ (default: data/scan_strokes)",
    )
    parser.add_argument("--start-line", type=int, default=0, help="処理開始行 (default: 0)")
    parser.add_argument("--end-line", type=int, default=None, help="処理終了行 (default: 最後まで)")
    parser.add_argument("--preview", action="store_true", help="プレビュー画像を生成して保存")
    return parser


def build_char_list(text: str) -> list[str]:
    """テキストから空白を除去して文字リストを返す。"""
    return [ch for ch in text if not ch.isspace()]


def save_stroke_sample(
    character: str,
    strokes: list[np.ndarray],
    output_dir: Path,
    *,
    source: str = "scan",
) -> Path:
    """ストロークをStrokeSample形式でJSONに保存する。

    Returns:
        保存先ファイルパス。
    """
    char_dir = output_dir / character
    char_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(char_dir.glob(f"{character}_scan_*.json"))
    next_num = len(existing)

    stroke_points = []
    for stroke_arr in strokes:
        points = [StrokePoint(x=float(pt[0]), y=float(pt[1])) for pt in stroke_arr]
        stroke_points.append(points)

    sample = StrokeSample(
        character=character,
        strokes=stroke_points,
        metadata={"source": source},
    )

    filename = f"{character}_scan_{next_num:03d}.json"
    save_path = char_dir / filename
    sample.save(save_path)
    return save_path


def process_image(
    importer: object,
    image_path: str | Path,
    text_lines: list[str],
    output_dir: Path,
    *,
    start_line: int = 0,
    end_line: int | None = None,
    preview: bool = False,
) -> int:
    """画像から文字を抽出し、ストロークを復元して保存する。

    Returns:
        保存した文字数。
    """
    all_rows = importer.extract_all_chars(
        Path(image_path) if isinstance(image_path, str) else image_path
    )
    effective_end = end_line if end_line is not None else len(all_rows)
    total_saved = 0

    for line_idx in range(start_line, min(effective_end, len(all_rows))):
        if line_idx >= len(text_lines):
            logger.warning("行 %d: テキスト行が不足、スキップ", line_idx)
            continue

        row_cells = all_rows[line_idx]
        line_chars = build_char_list(text_lines[line_idx])

        if len(row_cells) != len(line_chars):
            # セル数と文字数が合わない場合、短い方に合わせて処理
            n = min(len(row_cells), len(line_chars))
            if n == 0:
                continue
            logger.warning(
                "行 %d: セル数(%d) != 文字数(%d)、%d文字まで処理",
                line_idx,
                len(row_cells),
                len(line_chars),
                n,
            )
            row_cells = row_cells[:n]
            line_chars = line_chars[:n]

        for char_idx, (cell_img, char) in enumerate(zip(row_cells, line_chars)):
            strokes = importer.image_to_strokes(cell_img)
            if not strokes:
                continue
            save_stroke_sample(char, strokes, output_dir)
            total_saved += 1
            print(
                f"\r行{line_idx} {char_idx + 1}/{len(line_chars)} "
                f"'{char}' 保存完了 (計{total_saved}文字)",
                end="",
            )

        print()

    return total_saved


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.image.exists():
        print(f"エラー: 画像ファイルが見つかりません: {args.image}")
        sys.exit(1)

    if not args.text and not args.auto:
        print("エラー: --text または --auto を指定してください")
        sys.exit(1)

    try:
        from src.collector.scan_import import ScanImporter
    except ImportError:
        print("エラー: src/collector/scan_import.py が見つかりません。")
        sys.exit(1)

    importer = ScanImporter()

    if args.auto:
        print(f"画像: {args.image}")
        print("OCR実行中...")
        ocr_lines = importer.ocr_all_lines(args.image)
        text_lines = [line for line in ocr_lines if line.strip()]
        all_chars = []
        for line in text_lines:
            all_chars.extend(build_char_list(line))
        print(f"OCR結果: {len(text_lines)}行, {len(all_chars)}文字")
        for i, line in enumerate(text_lines):
            print(f"  行{i}: {line}")
        print()
    else:
        all_chars = build_char_list(args.text)
        raw_lines = args.text.split("\n")
        text_lines = [line for line in raw_lines if line.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"出力先: {args.output_dir}")
    print(f"テキスト文字数: {len(all_chars)}")
    print()

    saved = process_image(
        importer,
        args.image,
        text_lines,
        args.output_dir,
        start_line=args.start_line,
        end_line=args.end_line,
        preview=args.preview,
    )

    print(f"\n完了: {saved}文字を保存しました")


if __name__ == "__main__":
    main()
