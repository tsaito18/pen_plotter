"""Tests for scan_import module: character extraction from scanned images."""

import numpy as np
import pytest

from src.collector.scan_import import ScanImporter


@pytest.fixture
def importer():
    return ScanImporter(line_spacing_mm=8.0)


class TestLoadImage:
    def test_load_grayscale(self, tmp_path, importer):
        """画像をグレースケールで読み込めること"""
        import cv2

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[30:35, :] = 50
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        gray = importer.load_image(path)
        assert gray.ndim == 2
        assert gray.shape == (100, 200)

    def test_load_nonexistent_raises(self, tmp_path, importer):
        with pytest.raises(FileNotFoundError):
            importer.load_image(tmp_path / "nonexistent.png")


class TestDetectLines:
    def _make_ruled_image(self, height=400, width=600, line_ys=None):
        """罫線付きの人工画像を生成"""
        img = np.ones((height, width), dtype=np.uint8) * 255
        if line_ys is None:
            line_ys = [50, 100, 150, 200, 250]
        for y in line_ys:
            img[y : y + 2, 20 : width - 20] = 0  # 罫線（黒い水平線）
        return img, line_ys

    def test_detect_horizontal_lines(self, importer):
        gray, expected_ys = self._make_ruled_image()
        detected = importer.detect_lines(gray)
        assert len(detected) >= 3
        for det_y in detected:
            assert any(abs(det_y - ey) < 10 for ey in expected_ys)

    def test_no_lines_in_blank_image(self, importer):
        gray = np.ones((200, 300), dtype=np.uint8) * 255
        detected = importer.detect_lines(gray)
        assert len(detected) == 0

    def test_evenly_spaced_lines(self, importer):
        """等間隔の罫線を正しく検出"""
        line_ys = [50, 100, 150, 200, 250, 300]
        gray, _ = self._make_ruled_image(height=400, line_ys=line_ys)
        detected = importer.detect_lines(gray)
        if len(detected) >= 3:
            spacings = np.diff(detected)
            assert np.std(spacings) < 15  # 間隔がほぼ等しい


class TestExtractCharsFromLine:
    def _make_line_with_chars(self, n_chars=5, cell_w=40, height=50, width=250):
        """1行分の人工画像（等間隔に黒い矩形＝文字）"""
        img = np.ones((height, width), dtype=np.uint8) * 255
        for i in range(n_chars):
            x_start = 10 + i * cell_w
            x_end = x_start + 25
            img[10:40, x_start : min(x_end, width)] = 0
        return img, n_chars

    def test_extract_correct_number(self, importer):
        img, n_chars = self._make_line_with_chars(n_chars=5)
        chars = importer.extract_chars_from_line(img, 0, img.shape[0])
        assert len(chars) == n_chars

    def test_each_char_is_numpy_array(self, importer):
        img, _ = self._make_line_with_chars(n_chars=3)
        chars = importer.extract_chars_from_line(img, 0, img.shape[0])
        for c in chars:
            assert isinstance(c, np.ndarray)
            assert c.ndim == 2

    def test_empty_line(self, importer):
        img = np.ones((50, 200), dtype=np.uint8) * 255
        chars = importer.extract_chars_from_line(img, 0, 50)
        assert len(chars) == 0


class TestDeskew:
    def test_no_change_for_straight_image(self, importer):
        """傾きのない画像はそのまま返す"""
        img = np.ones((200, 300), dtype=np.uint8) * 255
        img[100, 20:280] = 0  # 水平線
        result = importer.deskew(img)
        assert result.shape == img.shape

    def test_corrects_tilted_image(self, importer):
        """傾いた画像を補正"""
        import cv2

        img = np.ones((400, 600), dtype=np.uint8) * 255
        for y in [100, 200, 300]:
            img[y : y + 2, 50:550] = 0

        # 2度回転させる
        center = (300, 200)
        matrix = cv2.getRotationMatrix2D(center, 2.0, 1.0)
        tilted = cv2.warpAffine(img, matrix, (600, 400), borderValue=255)

        corrected = importer.deskew(tilted)
        assert corrected.shape == tilted.shape


class TestExtractAllChars:
    def test_extract_from_synthetic_page(self, tmp_path, importer):
        """罫線+文字のある合成画像から文字を抽出"""
        import cv2

        height, width = 400, 300
        img = np.ones((height, width), dtype=np.uint8) * 255

        line_ys = [50, 100, 150, 200, 250]
        for y in line_ys:
            img[y : y + 2, 10 : width - 10] = 0

        for i in range(4):
            y_top = line_ys[i] + 5
            y_bot = line_ys[i + 1] - 5
            for j in range(3):
                x_start = 20 + j * 80
                img[y_top:y_bot, x_start : x_start + 30] = 0

        path = tmp_path / "page.png"
        cv2.imwrite(str(path), img)

        result = importer.extract_all_chars(path)
        assert isinstance(result, list)
        assert len(result) >= 2
        for line_chars in result:
            assert isinstance(line_chars, list)


class TestImageToStrokes:
    def _make_simple_stroke_image(self, size=64):
        """横線1本の文字画像"""
        img = np.ones((size, size), dtype=np.uint8) * 255
        img[size // 2, 10 : size - 10] = 0  # 水平線
        return img

    def _make_two_stroke_image(self, size=64):
        """離れた2本の線の文字画像（2ストローク）"""
        img = np.ones((size, size), dtype=np.uint8) * 255
        img[15, 10 : size - 10] = 0  # 上の水平線
        img[48, 10 : size - 10] = 0  # 下の水平線
        return img

    def test_single_stroke(self, importer):
        img = self._make_simple_stroke_image()
        strokes = importer.image_to_strokes(img)
        assert len(strokes) >= 1
        for s in strokes:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2
            assert s.shape[1] == 2

    def test_two_strokes_from_separated_lines(self, importer):
        img = self._make_two_stroke_image()
        strokes = importer.image_to_strokes(img)
        assert len(strokes) >= 2

    def test_short_strokes_filtered(self, importer):
        """3点未満のストロークは除外"""
        img = np.ones((64, 64), dtype=np.uint8) * 255
        img[30, 30] = 0  # 孤立した1ピクセル
        strokes = importer.image_to_strokes(img)
        assert len(strokes) == 0

    def test_empty_image(self, importer):
        img = np.ones((64, 64), dtype=np.uint8) * 255
        strokes = importer.image_to_strokes(img)
        assert len(strokes) == 0

    def test_stroke_points_ordered(self, importer):
        """ストロークの点列が連続していること"""
        img = self._make_simple_stroke_image(size=64)
        strokes = importer.image_to_strokes(img)
        if len(strokes) > 0:
            s = strokes[0]
            for i in range(1, len(s)):
                dist = np.sqrt(np.sum((s[i] - s[i - 1]) ** 2))
                assert dist < 5.0  # 隣接点間が離れすぎない


class TestTraceSkeletonComponent:
    def test_trace_line(self, importer):
        """直線の骨格をトレースできること"""
        skeleton = np.zeros((20, 50), dtype=bool)
        skeleton[10, 5:45] = True
        points = importer._trace_skeleton_component(skeleton)
        assert isinstance(points, np.ndarray)
        assert points.shape[1] == 2
        assert len(points) >= 30

    def test_trace_single_pixel(self, importer):
        """孤立1ピクセルの場合"""
        skeleton = np.zeros((20, 20), dtype=bool)
        skeleton[10, 10] = True
        points = importer._trace_skeleton_component(skeleton)
        assert len(points) == 1
