import json
import threading
import time
import urllib.request

import pytest

from src.collector.data_format import StrokePoint, StrokeSample
from src.collector.ipad_sync import (
    GUIDED_CHARS,
    StrokeCollectorApp,
    _TIER1_CHARS,
    _TIER2_CHARS,
    select_next_char,
)


class TestStrokeCollectorApp:
    @pytest.fixture
    def app(self, tmp_path):
        """テスト用アプリケーション（一時ディレクトリに保存）"""
        application = StrokeCollectorApp(output_dir=tmp_path, port=0)
        return application

    def test_app_creation(self, app):
        assert app is not None
        assert app.output_dir.exists()

    def test_build_html_page(self, app):
        html = app.build_html()
        assert "canvas" in html.lower()
        assert "stroke" in html.lower()

    def test_parse_stroke_data(self, app):
        data = {
            "character": "あ",
            "strokes": [
                [
                    {"x": 0, "y": 0, "pressure": 1.0, "timestamp": 0},
                    {"x": 10, "y": 10, "pressure": 0.5, "timestamp": 100},
                ]
            ],
        }
        sample = app.parse_stroke_data(data)
        assert sample.character == "あ"
        assert len(sample.strokes) == 1
        assert len(sample.strokes[0]) == 2

    def test_save_stroke_data(self, app, tmp_path):
        data = {
            "character": "い",
            "strokes": [
                [
                    {"x": 0, "y": 0, "pressure": 1.0, "timestamp": 0},
                    {"x": 5, "y": 5, "pressure": 1.0, "timestamp": 50},
                ]
            ],
        }
        sample = app.parse_stroke_data(data)
        path = app.save_stroke(sample)
        assert path.exists()

    def test_get_character_list(self, app, tmp_path):
        for ch in ["あ", "い"]:
            data = {
                "character": ch,
                "strokes": [
                    [
                        {"x": 0, "y": 0, "pressure": 1.0, "timestamp": 0},
                        {"x": 1, "y": 1, "pressure": 1.0, "timestamp": 10},
                    ]
                ],
            }
            sample = app.parse_stroke_data(data)
            app.save_stroke(sample)
        chars = app.list_saved_characters()
        assert "あ" in chars
        assert "い" in chars


class TestGuidedCollection:
    def _make_sample(self, char: str) -> StrokeSample:
        return StrokeSample(
            character=char,
            strokes=[
                [
                    StrokePoint(x=0, y=0, pressure=1.0, timestamp=0),
                    StrokePoint(x=1, y=1, pressure=1.0, timestamp=10),
                ]
            ],
        )

    def test_guided_chars_defined(self):
        assert isinstance(GUIDED_CHARS, list)
        assert len(GUIDED_CHARS) >= 200
        assert "あ" in GUIDED_CHARS
        assert "ア" in GUIDED_CHARS
        assert "一" in GUIDED_CHARS
        assert "0" in GUIDED_CHARS

    def test_get_progress_empty(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        progress = app.get_progress()
        assert progress["total"] == len(GUIDED_CHARS)
        assert progress["completed"] == 0
        # Tier1の0サンプル文字が最優先で選ばれる
        assert progress["current_char"] in _TIER1_CHARS
        assert progress["samples_for_current"] == 0
        assert progress["target_samples"] == 3
        assert "tier" in progress

    def test_get_progress_with_samples(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        first_char = app.get_progress()["current_char"]
        sample = self._make_sample(first_char)
        app.save_stroke(sample)
        progress = app.get_progress()
        # 1サンプルの文字より0サンプルのTier1文字が優先される
        assert progress["current_char"] in GUIDED_CHARS
        assert progress["completed"] == 0

    def test_get_progress_char_completed(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        first_char = app.get_progress()["current_char"]
        for _ in range(3):
            sample = self._make_sample(first_char)
            app.save_stroke(sample)
            time.sleep(0.001)
        progress = app.get_progress()
        assert progress["completed"] == 1
        # 完了した文字ではなく別のTier1文字が選ばれる
        assert progress["current_char"] != first_char
        assert progress["current_char"] in GUIDED_CHARS

    def test_progress_endpoint(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        t = threading.Thread(target=app.serve, daemon=True)
        t.start()
        time.sleep(0.3)

        url = f"http://127.0.0.1:{app.port}/api/progress"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert "total" in data
        assert "completed" in data
        assert "current_char" in data
        assert "current_index" in data
        assert "samples_for_current" in data
        assert "target_samples" in data
        assert data["current_char"] in _TIER1_CHARS


class TestSelectNextChar:
    """select_next_char の優先度ロジックテスト"""

    def test_empty_counts_returns_tier1(self):
        """サンプルなしではTier1文字が選ばれる"""
        result = select_next_char({}, target_samples=3, seed=42)
        assert result in _TIER1_CHARS

    def test_tier1_before_tier2(self):
        """Tier1がTier2より優先される（サンプル数が多くても）"""
        # Tier1文字を全て1サンプルにし、Tier2/3を0サンプルにする
        counts = {c: 1 for c in GUIDED_CHARS if c in _TIER1_CHARS}
        result = select_next_char(counts, target_samples=3, seed=42)
        # Tier1の1サンプル文字が、Tier2/3の0サンプル文字より優先される
        assert result in _TIER1_CHARS

    def test_zero_samples_before_one_sample(self):
        """0サンプルの文字が1サンプルの文字より優先される"""
        tier1_list = [c for c in GUIDED_CHARS if c in _TIER1_CHARS]
        # 半分を1サンプルにする
        counts = {c: 1 for c in tier1_list[:len(tier1_list) // 2]}
        result = select_next_char(counts, target_samples=3, seed=42)
        assert counts.get(result, 0) == 0

    def test_all_completed_returns_none(self):
        """全文字が目標サンプル数に達したらNone"""
        counts = {c: 3 for c in GUIDED_CHARS}
        result = select_next_char(counts, target_samples=3)
        assert result is None

    def test_deterministic_with_seed(self):
        """同じseedで同じ結果"""
        r1 = select_next_char({}, target_samples=3, seed=123)
        r2 = select_next_char({}, target_samples=3, seed=123)
        assert r1 == r2

    def test_varies_without_seed(self):
        """seedなしでは結果がバラけうる（ランダム性確認）"""
        results = set()
        for _ in range(20):
            r = select_next_char({}, target_samples=3)
            results.add(r)
        assert len(results) > 1
