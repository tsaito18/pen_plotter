import json
import threading
import time
import urllib.request

import pytest

from src.collector.ipad_sync import StrokeCollectorApp


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
