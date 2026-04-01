import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from src.collector.data_format import StrokePoint, StrokeSample
from src.collector.ipad_sync import (
    GUIDED_CHARS,
    StrokeCollectorApp,
    _TIER1_CHARS,
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

    def test_build_html_reads_from_template_file(self):
        """build_html() はテンプレートファイルから読み込む"""
        from pathlib import Path
        template_path = Path(__file__).resolve().parent.parent / "src" / "collector" / "templates" / "collector.html"
        assert template_path.exists(), f"Template file not found: {template_path}"
        content = template_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

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


class TestManagementAPI:
    """サンプル管理・統計APIのエンドポイントテスト"""

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

    def _start_server(self, app: StrokeCollectorApp) -> None:
        t = threading.Thread(target=app.serve, daemon=True)
        t.start()
        time.sleep(0.3)

    def _encode_path(self, path: str) -> str:
        """パス中のクエリパラメータをURLエンコードする。"""
        if "?" not in path:
            return path
        base, query = path.split("?", 1)
        params = urllib.parse.parse_qs(query, keep_blank_values=True)
        encoded = urllib.parse.urlencode(
            {k: v[0] for k, v in params.items()}, quote_via=urllib.parse.quote
        )
        return f"{base}?{encoded}"

    def _get(self, port: int, path: str) -> tuple[int, dict | list | str]:
        url = f"http://127.0.0.1:{port}{self._encode_path(path)}"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return resp.status, data
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            try:
                return e.code, json.loads(body)
            except json.JSONDecodeError:
                return e.code, body

    def _delete(self, port: int, path: str) -> tuple[int, dict | str]:
        url = f"http://127.0.0.1:{port}{self._encode_path(path)}"
        req = urllib.request.Request(url, method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return resp.status, data
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            try:
                return e.code, json.loads(body)
            except json.JSONDecodeError:
                return e.code, body

    def _post(self, port: int, path: str, body: dict | None = None) -> tuple[int, dict | str]:
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(body or {}).encode("utf-8") if body else b"{}"
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8")
            try:
                return e.code, json.loads(raw)
            except json.JSONDecodeError:
                return e.code, raw

    def test_get_samples_endpoint(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        app.save_stroke(self._make_sample("あ"))
        self._start_server(app)

        status, data = self._get(app.port, "/api/samples?char=あ")

        assert status == 200
        assert isinstance(data, list)
        assert len(data) == 1
        assert "filename" in data[0]
        assert "stroke_count" in data[0]

    def test_get_samples_unknown_char(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        self._start_server(app)

        status, data = self._get(app.port, "/api/samples?char=ん")

        assert status == 200
        assert data == []

    def test_delete_single_sample_endpoint(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        for _ in range(2):
            app.save_stroke(self._make_sample("あ"))
            time.sleep(0.001)
        path = app.save_stroke(self._make_sample("あ"))
        filename = path.name
        self._start_server(app)

        status, data = self._delete(app.port, f"/api/samples?char=あ&file={filename}")

        assert status == 200
        assert data["status"] == "ok"
        assert data["remaining"] == 2
        assert not path.exists()

    def test_delete_all_samples_endpoint(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        for _ in range(3):
            app.save_stroke(self._make_sample("あ"))
            time.sleep(0.001)
        self._start_server(app)

        status, data = self._delete(app.port, "/api/samples?char=あ")

        assert status == 200
        assert data["deleted_count"] == 3

    def test_delete_missing_sample_returns_404(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        self._start_server(app)

        status, _data = self._delete(app.port, "/api/samples?char=あ&file=nonexistent.json")

        assert status == 404

    def test_stats_endpoint(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        for _ in range(3):
            app.save_stroke(self._make_sample("の"))
            time.sleep(0.001)
        app.save_stroke(self._make_sample("は"))
        self._start_server(app)

        status, data = self._get(app.port, "/api/stats")

        assert status == 200
        assert "tiers" in data
        assert "total_samples" in data
        assert data["total_samples"] == 4

    def test_undo_last_endpoint(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        app.save_stroke(self._make_sample("あ"))
        time.sleep(0.001)
        last_path = app.save_stroke(self._make_sample("あ"))
        self._start_server(app)

        status, data = self._post(app.port, "/api/undo-last")

        assert status == 200
        assert data["status"] == "ok"
        assert data["deleted_file"] == last_path.name
        assert data["character"] == "あ"
        assert not last_path.exists()

    def test_undo_last_empty(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        self._start_server(app)

        status, _data = self._post(app.port, "/api/undo-last")

        assert status == 404

    def test_progress_with_char_param(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        app.save_stroke(self._make_sample("い"))
        self._start_server(app)

        status, data = self._get(app.port, "/api/progress?char=い")

        assert status == 200
        assert data["current_char"] == "い"
        assert data["samples_for_current"] == 1

    def test_html_contains_management_tab(self, tmp_path):
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        self._start_server(app)

        url = f"http://127.0.0.1:{app.port}/"
        with urllib.request.urlopen(url, timeout=5) as resp:
            html = resp.read().decode("utf-8")

        assert "管理" in html

    def test_set_metadata_endpoint(self, tmp_path):
        """POST /api/samples/metadata → 200 {"status": "ok"}"""
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        path = app.save_stroke(self._make_sample("あ"))
        self._start_server(app)

        status, data = self._post(
            app.port,
            "/api/samples/metadata",
            {"char": "あ", "file": path.name, "key": "ignore_anomaly", "value": True},
        )

        assert status == 200
        assert data["status"] == "ok"

    def test_set_metadata_missing_file_returns_404(self, tmp_path):
        """存在しないファイルへの metadata 設定 → 404"""
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        self._start_server(app)

        status, _data = self._post(
            app.port,
            "/api/samples/metadata",
            {"char": "あ", "file": "nonexistent_12345.json", "key": "ignore_anomaly", "value": True},
        )

        assert status == 404

    def test_stroke_mismatches_endpoint(self, tmp_path):
        """GET /api/stroke-mismatches → 画数不一致のグループ一覧"""
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        # "あ" に 3画×2, 5画×1
        for _ in range(2):
            app.save_stroke(
                StrokeSample(
                    character="あ",
                    strokes=[
                        [
                            StrokePoint(x=0, y=0, pressure=1.0, timestamp=0),
                            StrokePoint(x=1, y=1, pressure=1.0, timestamp=10),
                        ]
                        for _ in range(3)
                    ],
                )
            )
            time.sleep(0.001)
        app.save_stroke(
            StrokeSample(
                character="あ",
                strokes=[
                    [
                        StrokePoint(x=0, y=0, pressure=1.0, timestamp=0),
                        StrokePoint(x=1, y=1, pressure=1.0, timestamp=10),
                    ]
                    for _ in range(5)
                ],
            )
        )
        self._start_server(app)

        status, data = self._get(app.port, "/api/stroke-mismatches")

        assert status == 200
        assert isinstance(data, list)
        assert len(data) == 1
        group = data[0]
        assert group["character"] == "あ"
        assert group["mode_count"] == 3
        assert len(group["samples"]) == 3
        outliers = [s for s in group["samples"] if s["is_outlier"]]
        assert len(outliers) == 1
        assert outliers[0]["stroke_count"] == 5

    def test_anomalies_endpoint(self, tmp_path):
        """GET /api/anomalies → 異常サンプルのみ返る"""
        app = StrokeCollectorApp(output_dir=tmp_path, port=0)
        # 正常サンプル（十分なポイント・サイズ・時間）
        normal = StrokeSample(
            character="あ",
            strokes=[
                [
                    StrokePoint(x=100.0, y=100.0, pressure=1.0, timestamp=0.0),
                    StrokePoint(x=200.0, y=100.0, pressure=1.0, timestamp=500.0),
                    StrokePoint(x=300.0, y=200.0, pressure=1.0, timestamp=1000.0),
                ],
                [
                    StrokePoint(x=100.0, y=300.0, pressure=1.0, timestamp=1500.0),
                    StrokePoint(x=200.0, y=300.0, pressure=1.0, timestamp=2000.0),
                    StrokePoint(x=300.0, y=400.0, pressure=1.0, timestamp=2500.0),
                ],
            ],
        )
        app.save_stroke(normal)
        # 異常サンプル（点数少 = 2ポイント）
        anomalous = StrokeSample(
            character="い",
            strokes=[
                [
                    StrokePoint(x=100.0, y=100.0, pressure=1.0, timestamp=0.0),
                    StrokePoint(x=200.0, y=200.0, pressure=1.0, timestamp=2000.0),
                ],
            ],
        )
        app.save_stroke(anomalous)
        self._start_server(app)

        status, data = self._get(app.port, "/api/anomalies")

        assert status == 200
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["character"] == "い"
        assert "点数少" in data[0]["reasons"]


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
        counts = {c: 1 for c in tier1_list[: len(tier1_list) // 2]}
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
