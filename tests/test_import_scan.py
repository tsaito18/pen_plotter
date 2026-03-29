"""tests for scripts/import_scan.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.import_scan import (
    build_char_list,
    process_image,
    save_stroke_sample,
)


class TestBuildCharList:
    """テキストから文字リストを構築するテスト。"""

    def test_simple_text(self):
        result = build_char_list("ABC")
        assert result == ["A", "B", "C"]

    def test_ignores_spaces(self):
        result = build_char_list("A B C")
        assert result == ["A", "B", "C"]

    def test_ignores_newlines(self):
        result = build_char_list("AB\nCD")
        assert result == ["A", "B", "C", "D"]

    def test_ignores_tabs_and_mixed_whitespace(self):
        result = build_char_list("A\t B\n C")
        assert result == ["A", "B", "C"]

    def test_japanese_text(self):
        result = build_char_list("実験目的")
        assert result == ["実", "験", "目", "的"]

    def test_empty_string(self):
        result = build_char_list("")
        assert result == []

    def test_only_whitespace(self):
        result = build_char_list("   \n\t  ")
        assert result == []


class TestSaveStrokeSample:
    """StrokeSample保存のテスト。"""

    def test_saves_json_file(self, tmp_path):
        strokes = [np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])]
        save_stroke_sample("あ", strokes, tmp_path, source="scan")

        char_dir = tmp_path / "あ"
        assert char_dir.exists()
        files = list(char_dir.glob("あ_scan_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["character"] == "あ"
        assert len(data["strokes"]) == 1
        assert len(data["strokes"][0]) == 3
        assert data["metadata"]["source"] == "scan"

    def test_multiple_strokes(self, tmp_path):
        strokes = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
        ]
        save_stroke_sample("木", strokes, tmp_path, source="scan")

        char_dir = tmp_path / "木"
        files = list(char_dir.glob("木_scan_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert len(data["strokes"]) == 2
        assert len(data["strokes"][0]) == 2
        assert len(data["strokes"][1]) == 3

    def test_sequential_numbering(self, tmp_path):
        """同じ文字を複数回保存したときに連番になる。"""
        strokes = [np.array([[0.0, 0.0], [1.0, 1.0]])]
        save_stroke_sample("A", strokes, tmp_path, source="scan")
        save_stroke_sample("A", strokes, tmp_path, source="scan")
        save_stroke_sample("A", strokes, tmp_path, source="scan")

        char_dir = tmp_path / "A"
        files = sorted(char_dir.glob("A_scan_*.json"))
        assert len(files) == 3
        assert files[0].name == "A_scan_000.json"
        assert files[1].name == "A_scan_001.json"
        assert files[2].name == "A_scan_002.json"

    def test_stroke_point_format(self, tmp_path):
        """保存されたポイントがStrokePoint互換形式であることを確認。"""
        strokes = [np.array([[1.5, 2.5]])]
        save_stroke_sample("B", strokes, tmp_path, source="scan")

        files = list((tmp_path / "B").glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))
        point = data["strokes"][0][0]
        assert "x" in point
        assert "y" in point
        assert "pressure" in point
        assert "timestamp" in point
        assert point["x"] == pytest.approx(1.5)
        assert point["y"] == pytest.approx(2.5)


class TestProcessImage:
    """画像処理パイプラインのテスト（ScanImporterをモック）。"""

    def test_char_count_mismatch_warns(self, tmp_path, caplog):
        """テキスト文字数とセル数が合わない行はスキップ。"""
        mock_importer = MagicMock()
        # 2行: 1行目は3セル、2行目は2セル
        mock_importer.extract_all_chars.return_value = [
            [np.zeros((32, 32)) for _ in range(3)],
            [np.zeros((32, 32)) for _ in range(2)],
        ]
        mock_importer.image_to_strokes.return_value = [np.array([[0.0, 0.0], [1.0, 1.0]])]

        # テキスト: 1行目3文字、2行目3文字（2セルなのでミスマッチ）
        text_lines = ["ABC", "DEF"]
        process_image(mock_importer, "dummy.jpg", text_lines, tmp_path)

        # 1行目は処理される
        assert (tmp_path / "A").exists()
        assert (tmp_path / "B").exists()
        assert (tmp_path / "C").exists()
        # 2行目はセル数2 < 文字数3なので、2文字まで処理される
        assert (tmp_path / "D").exists()
        assert (tmp_path / "E").exists()
        assert not (tmp_path / "F").exists()

    def test_start_end_line(self, tmp_path):
        """start_line/end_lineで処理行を制限。"""
        mock_importer = MagicMock()
        mock_importer.extract_all_chars.return_value = [
            [np.zeros((32, 32)) for _ in range(2)],
            [np.zeros((32, 32)) for _ in range(2)],
            [np.zeros((32, 32)) for _ in range(2)],
        ]
        mock_importer.image_to_strokes.return_value = [np.array([[0.0, 0.0], [1.0, 1.0]])]

        text_lines = ["AB", "CD", "EF"]
        process_image(
            mock_importer,
            "dummy.jpg",
            text_lines,
            tmp_path,
            start_line=1,
            end_line=2,
        )

        # 0行目はスキップ
        assert not (tmp_path / "A").exists()
        # 1行目のみ処理
        assert (tmp_path / "C").exists()
        assert (tmp_path / "D").exists()
        # 2行目以降はスキップ
        assert not (tmp_path / "E").exists()

    def test_progress_tracking(self, tmp_path, capsys):
        """進捗表示がされることを確認。"""
        mock_importer = MagicMock()
        mock_importer.extract_all_chars.return_value = [
            [np.zeros((32, 32)) for _ in range(2)],
        ]
        mock_importer.image_to_strokes.return_value = [np.array([[0.0, 0.0], [1.0, 1.0]])]

        text_lines = ["AB"]
        process_image(mock_importer, "dummy.jpg", text_lines, tmp_path)

        captured = capsys.readouterr()
        assert "1/2" in captured.out or "2/2" in captured.out


class TestCLIArgs:
    """コマンドライン引数パースのテスト。"""

    def test_required_args(self):
        from scripts.import_scan import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--image",
                "test.jpg",
                "--text",
                "ABC",
            ]
        )
        assert args.image == Path("test.jpg")
        assert args.text == "ABC"
        assert args.output_dir == Path("data/scan_strokes")
        assert args.start_line == 0
        assert args.end_line is None
        assert args.preview is False

    def test_all_args(self):
        from scripts.import_scan import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--image",
                "scan.png",
                "--text",
                "テスト",
                "--output-dir",
                "/tmp/out",
                "--start-line",
                "2",
                "--end-line",
                "5",
                "--preview",
            ]
        )
        assert args.image == Path("scan.png")
        assert args.text == "テスト"
        assert args.output_dir == Path("/tmp/out")
        assert args.start_line == 2
        assert args.end_line == 5
        assert args.preview is True
