"""手書きサンプル収集: データ形式・KanjiVG・保存/管理・プロファイル・収集サーバ・訓練ジョブ。"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from src.collector.data_format import StrokePoint, StrokeSample
from src.collector.ipad_sync import GUIDED_CHARS, StrokeCollectorApp, select_next_char
from src.collector.kanjivg_parser import KanjiVGParser, parse_svg_path
from src.collector.profiles import (
    list_profiles,
    resolve_character_root,
    resolve_training_dirs,
    validate_profile_id,
)
from src.collector.stroke_recorder import StrokeRecorder
from src.collector.training_jobs import TrainingCancelled, TrainingJobManager

SVG_WITH_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109" viewBox="0 0 109 109">
  <g id="kvg:06728" kvg:element="木">
    <path kvg:type="㇐" d="M 20,40 L 90,40"/>
    <path kvg:type="㇑a" d="M 55,15 L 55,95"/>
    <path d="M 50,45 C 40,60 30,75 18,85"/>
  </g>
</svg>"""


def _sample(char: str = "あ", n_strokes: int = 2, n_points: int = 5, ms: float = 2000.0):
    total = n_strokes * n_points - 1
    strokes = [
        [
            StrokePoint(100 + 200 * k / total, 100 + 200 * k / total, 1.0, ms * k / total)
            for k in range(s * n_points, (s + 1) * n_points)
        ]
        for s in range(n_strokes)
    ]
    return StrokeSample(character=char, strokes=strokes)


# --- データ形式 ---


def test_stroke_sample_roundtrip_and_legacy_json(tmp_path: Path):
    sample = _sample()
    sample.stroke_types = ["㇐", ""]
    path = tmp_path / "s.json"
    sample.save(path)
    assert StrokeSample.load(path) == sample
    legacy = json.dumps({"character": "い", "strokes": [], "metadata": {}})
    assert StrokeSample.from_json(legacy).stroke_types == []


# --- KanjiVG パーサ ---


def test_svg_path_with_smooth_and_repeated_cubic_segments():
    pts = parse_svg_path("M 0,0 C 10,0 20,10 20,20 s 10,10 20,0")
    assert np.allclose(pts[-1], [40, 20]) and np.ptp(pts[:, 0]) > 30
    pts = parse_svg_path("M 0,0 c 10,0 10,10 20,10 10,0 10,10 20,10")
    assert len(pts) == 19 and np.allclose(pts[-1], [40, 20])
    pts = parse_svg_path("M 0,0 C 5,0 10,0 10,10 S 20,20 30,10 40,0 50,10")
    assert len(pts) == 28 and np.allclose(pts[-1], [50, 10])
    real = (  # 実データ（學 の 6 画目）
        "M37.25,46.5c1,0.25,3.75,0.25,5.5,-0.25s18.25,-4,20,-4s2.75,0.75,1,2.25S54.5,53.5,53,54.75"
    )
    assert np.allclose(parse_svg_path(real)[-1], [53, 54.75], atol=0.1)
    assert len(parse_svg_path("")) == 0


def test_parser_extracts_aligned_stroke_types_and_normalizes():
    parser = KanjiVGParser()
    strokes, types = parser.parse_svg_with_types(SVG_WITH_TYPES)
    assert len(strokes) == 3 and types == ["㇐", "㇑a", ""]
    normalized = np.concatenate(parser.normalize(strokes, target_size=10.0))
    assert normalized.min() >= 0 and normalized.max() <= 10


# --- 保存・管理 ---


def test_recorder_save_list_delete(tmp_path: Path):
    rec = StrokeRecorder(output_dir=tmp_path)
    paths = [rec.save_sample(_sample("い")) for _ in range(3)]
    assert rec.list_characters() == ["い"]
    assert len(rec.get_sample_info("い")) == 3
    assert rec.delete_sample("い", paths[0].name) and not paths[0].exists()
    assert not rec.delete_sample("い", "い_1.json")
    with pytest.raises(ValueError):
        rec.delete_sample("い", "../evil.json")
    assert rec.set_metadata("い", paths[1].name, "ignored", True)
    assert rec.delete_all_samples("い") == 2


def test_recorder_flags_anomalies_and_stroke_count_outliers(tmp_path: Path):
    rec = StrokeRecorder(output_dir=tmp_path)
    rec.save_sample(_sample("あ", ms=2000))
    rec.save_sample(_sample("い", ms=100))  # 速すぎる
    (anomaly,) = rec.find_anomalies()
    assert anomaly["character"] == "い" and "描画時間短" in anomaly["reasons"]

    for n in (3, 3, 5):
        rec.save_sample(_sample("う", n_strokes=n))
    (group,) = rec.find_stroke_mismatches()
    outliers = [s for s in group["samples"] if s["is_outlier"]]
    assert len(outliers) == 1 and outliers[0]["stroke_count"] == 5


def test_resample_preserves_endpoints():
    points = [StrokePoint(float(i), 0.0) for i in range(5)]
    out = StrokeRecorder().resample_points(points, num_points=32)
    assert len(out) == 32 and (out[0].x, out[-1].x) == (0.0, 4.0)


# --- プロファイル ---


def test_profiles(user_strokes_root: Path, tmp_path: Path):
    (profile,) = list_profiles(user_strokes_root)
    assert (profile.id, profile.character_count, profile.sample_count) == ("taro", 2, 3)
    assert resolve_character_root(user_strokes_root) == user_strokes_root / "taro"
    assert resolve_character_root(user_strokes_root, "nobody") == user_strokes_root / "taro"
    char_root = user_strokes_root / "taro"
    assert resolve_character_root(char_root) == char_root  # 文字ディレクトリの親はそのまま
    assert resolve_character_root(tmp_path / "none") is None
    assert resolve_training_dirs(user_strokes_root, {"mode": "all"}) == [char_root]
    with pytest.raises(ValueError):
        resolve_training_dirs(user_strokes_root, {"mode": "profiles", "profiles": ["jiro"]})
    for bad in ["", "default", "a b", "../x"]:
        with pytest.raises(ValueError):
            validate_profile_id(bad)


# --- 収集サーバ ---


def test_next_char_prioritizes_unwritten_chars():
    first = select_next_char({}, target_samples=3, seed=0)
    assert first in GUIDED_CHARS
    counts = {c: 3 for c in GUIDED_CHARS}
    assert select_next_char(counts, target_samples=3) is None
    counts[GUIDED_CHARS[-1]] = 0
    assert select_next_char(counts, target_samples=3) == GUIDED_CHARS[-1]


class _Server:
    def __init__(self, root: Path) -> None:
        self.app = StrokeCollectorApp(output_dir=root, port=0, person_id="taro")
        threading.Thread(target=self.app.serve, daemon=True).start()
        for _ in range(50):
            if self.app.port:
                break
            time.sleep(0.02)

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        url = f"http://127.0.0.1:{self.app.port}{urllib.parse.quote(path, safe='/?=&')}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            return err.code, {}


def test_collector_http_workflow(tmp_path: Path):
    server = _Server(tmp_path)
    stroke = {"character": "あ", "strokes": [[{"x": 0, "y": 0, "pressure": 1, "timestamp": 0}]]}
    assert server.request("POST", "/api/stroke", stroke)[0] == 200
    assert server.request("POST", "/api/stroke", stroke)[0] == 200

    status, samples = server.request("GET", "/api/samples?char=あ")
    assert status == 200 and len(samples) == 2
    filename = samples[0]["filename"]
    meta = {"char": "あ", "file": filename, "key": "ignored", "value": True}
    assert server.request("POST", "/api/samples/metadata", meta)[0] == 200

    status, undone = server.request("POST", "/api/undo-last")
    assert status == 200 and undone["character"] == "あ"
    assert server.request("DELETE", "/api/samples?char=あ")[0] == 200
    assert server.request("POST", "/api/undo-last")[0] == 404

    status, progress = server.request("GET", "/api/progress")
    assert status == 200 and progress["current_char"] in GUIDED_CHARS
    assert server.request("GET", "/api/stats")[0] == 200
    assert "canvas" in server.app.build_html().lower()


# --- 訓練ジョブ ---


def _wait(manager: TrainingJobManager) -> dict:
    for _ in range(200):
        status = manager.status()
        if status["state"] != "running":
            return status
        time.sleep(0.01)
    raise AssertionError("training job did not finish")


def test_training_job_success_failure_and_cancel(tmp_path: Path, monkeypatch):
    manager = TrainingJobManager(root_dir=tmp_path, ref_dir=None)

    monkeypatch.setattr(manager, "_run_training", lambda config, profile: tmp_path / "ok.pt")
    manager.start({"kind": "finetune"}, "taro")
    ok = _wait(manager)
    assert ok["state"] == "succeeded" and ok["checkpoint_path"].endswith("ok.pt")
    assert ok["total_epochs"] == 20

    def fail(config, profile):
        raise RuntimeError("boom")

    monkeypatch.setattr(manager, "_run_training", fail)
    manager.start({}, "taro")
    assert _wait(manager)["error"] == "boom"

    def cancellable(config, profile):
        while True:
            manager._check_cancel(0)
            time.sleep(0.005)

    monkeypatch.setattr(manager, "_run_training", cancellable)
    manager.start({}, "taro")
    with pytest.raises(RuntimeError, match="already running"):
        manager.start({}, "taro")
    manager.cancel()
    assert _wait(manager)["state"] == "cancelled"
    assert issubclass(TrainingCancelled, RuntimeError)
