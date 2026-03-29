"""レポートサンプルを一括で取り込むスクリプト。

テキストからMarkdown/LaTeX/図表参照を除去し、各ページ画像とマッチングしてストロークを抽出する。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.import_scan import build_char_list, process_image, save_stroke_sample


def clean_report_text(raw: str) -> list[str]:
    """Markdown/LaTeX/図表参照を除去し、行リストを返す。"""
    # ==...== で囲まれた図表参照を除去
    text = re.sub(r"==.*?==", "", raw)
    # $$...$$ ブロック数式を除去
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    # $...$ インライン数式を除去
    text = re.sub(r"\$[^$]+?\$", "", text)
    # \tag{...} を除去
    text = re.sub(r"\\tag\{[^}]*\}", "", text)
    # Markdown見出し記号を除去（テキストは残す）
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # **太字** を除去
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # \[...\] を除去（単位表記）
    text = re.sub(r"\\\[.*?\\\]", "", text)
    # LaTeXコマンドの残骸
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # 連続空行を1つに
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def estimate_page_lines(lines: list[str], chars_per_line: int = 30) -> list[list[str]]:
    """テキスト行をページごとに分割（推定）。

    1ページ ≈ 25行 × 30文字。長い行は折り返してカウント。
    """
    pages: list[list[str]] = []
    current_page: list[str] = []
    line_count = 0
    lines_per_page = 25

    for line in lines:
        # この行が何行分になるか推定
        n_lines = max(1, (len(line) + chars_per_line - 1) // chars_per_line)
        if line_count + n_lines > lines_per_page and current_page:
            pages.append(current_page)
            current_page = []
            line_count = 0
        current_page.append(line)
        line_count += n_lines

    if current_page:
        pages.append(current_page)
    return pages


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="レポートサンプルを一括取り込み")
    parser.add_argument("--text-file", type=Path, required=True, help="レポートテキスト（Markdown）")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/report_examples/jpg"),
        help="ページ画像ディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scan_strokes"),
        help="出力ディレクトリ",
    )
    parser.add_argument("--start-page", type=int, default=2, help="開始ページ番号（デフォルト: 2、01.jpgは表紙）")
    parser.add_argument("--dry-run", action="store_true", help="実際の抽出は行わず、テキスト分割結果のみ表示")
    args = parser.parse_args()

    raw_text = args.text_file.read_text(encoding="utf-8")
    lines = clean_report_text(raw_text)
    pages = estimate_page_lines(lines)

    print(f"テキスト: {len(lines)}行 → {len(pages)}ページに分割")

    # 画像ファイル一覧
    image_files = sorted(args.image_dir.glob("*.jpg"))
    # start_page (1-indexed) から対応
    available_images = [f for f in image_files if int(f.stem) >= args.start_page]

    print(f"画像: {len(available_images)}ページ（{args.start_page:02d}.jpg〜）")
    print()

    for i, page_lines in enumerate(pages):
        if i >= len(available_images):
            print(f"ページ {i}: 対応する画像なし、スキップ")
            break

        img = available_images[i]
        page_text = "\n".join(page_lines)
        char_count = len(build_char_list(page_text))

        print(f"--- ページ {i} ({img.name}) ---")
        for j, line in enumerate(page_lines[:3]):
            print(f"  {line[:50]}{'...' if len(line) > 50 else ''}")
        if len(page_lines) > 3:
            print(f"  ... ({len(page_lines)}行, {char_count}文字)")
        print()

        if args.dry_run:
            continue

        from src.collector.scan_import import ScanImporter

        importer = ScanImporter()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        saved = process_image(
            importer,
            img,
            page_lines,
            args.output_dir,
        )
        print(f"  → {saved}文字保存\n")

    print("完了")


if __name__ == "__main__":
    main()
