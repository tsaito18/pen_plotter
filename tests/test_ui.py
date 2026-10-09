"""Web UI のサーバー API（入力 → 組版ドラフト → 清書ストローク＋同一の G-code）。"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from src.gcode.config import PlotterConfig
from src.settings import Settings

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.ui.server import create_app

TEXT = "# 十人\n人十一 $x^2$\n\n| a | b |\n|---|---|\n| 1 | 2 |"


@pytest.fixture(scope="module")
def client(kanjivg_dir: Path, user_strokes_root: Path) -> TestClient:
    return TestClient(create_app(kanjivg_dir=kanjivg_dir, user_strokes_dir=user_strokes_root))


def _render(client: TestClient, **body: object) -> list[dict]:
    response = client.post("/api/render", json=body)
    assert response.status_code == 200
    return [json.loads(line) for line in response.iter_lines() if line]


def test_index_serves_the_app_shell_and_assets(client: TestClient):
    html = client.get("/").text
    assert '<script type="module"' in html and "Pen Plotter" in html
    for asset in ["app.js", "plotter.js", "paper.js", "editor.js", "styles.css"]:
        assert client.get(f"/static/{asset}").status_code == 200, asset


def test_bootstrap_describes_every_setting_profiles_and_examples(client: TestClient):
    data = client.get("/api/bootstrap").json()
    controls = {c["field"] for section in data["sections"] for c in section["controls"]}
    assert controls | {"paper_width", "paper_height"} == {f.name for f in fields(Settings)}
    assert data["settings"] == Settings().to_dict()
    assert [p["id"] for p in data["profiles"]] == ["taro"]
    assert data["examples"] and all(e["label"] and e["text"] for e in data["examples"])
    assert data["paper"] == {
        "width": 210.0,
        "height": 297.0,
        "background": data["paper"]["background"],
        "pen_width_mm": PlotterConfig().pen_width_mm,  # ビューアの線幅も同じペン幅
    }
    assert data["syntax"] and data["sources"]["kanjivg"] is True


def test_layout_is_a_fast_draft_with_validation(client: TestClient):
    data = client.post("/api/layout", json={"text": TEXT, "settings": {}}).json()
    assert data["errors"] == [] and len(data["pages"]) == 1
    page = data["pages"][0]
    chars = [c[0] for c in page["chars"]]
    assert {"十", "人", "一", "a", "1"} <= set(chars)
    assert page["math"] and page["rules"]  # インライン数式の枠・表の罫線
    assert data["ruled"]  # 用紙の罫線（背景画像が無いとき描く）
    bad = client.post("/api/layout", json={"text": "十", "settings": {"font_size": 0}}).json()
    assert bad["errors"]


def test_render_streams_progress_then_strokes_with_the_matching_gcode(client: TestClient):
    events = _render(client, text=TEXT, seed=3)
    assert events[0]["type"] == "progress" and events[-1]["type"] == "result"
    result = events[-1]
    assert result["seed"] == 3 and len(result["pages"]) == 1
    page = result["pages"][0]
    assert len(page["strokes"]) == len(page["spans"]) > 0
    gcode = page["gcode"]
    assert gcode[0] == "$H" and not any(line.startswith(";") for line in gcode)
    for stroke, (_start, end) in zip(page["strokes"], page["spans"]):
        x, y = stroke["points"][-2:]
        assert gcode[end - 1].startswith(f"G1 X{x:.2f} Y{y:.2f}")
        n_segments = len(stroke["points"]) // 2 - 1
        assert isinstance(stroke["contact"], float) or len(stroke["contact"]) == n_segments
    coverage = result["coverage"]
    assert coverage["kanjivg"]["count"] > 0 and "十" in coverage["user_strokes"]["chars"]


def test_render_is_reproducible_by_seed(client: TestClient):
    first = _render(client, text="十人十人", seed=7)[-1]
    again = _render(client, text="十人十人", seed=7)[-1]
    other = _render(client, text="十人十人", seed=8)[-1]
    assert first["pages"] == again["pages"]
    assert first["pages"][0]["strokes"] != other["pages"][0]["strokes"]


def test_render_options_profile_and_japanese_only(client: TestClient):
    events = _render(client, text="十a", seed=1, profile="taro", japanese_only=True)
    coverage = events[-1]["coverage"]
    assert "a" in coverage["skipped"]["chars"]


@pytest.mark.parametrize("body", [{"text": "  "}, {"text": "十", "settings": {"font_size": 0}}])
def test_render_reports_errors_instead_of_failing(client: TestClient, body: dict):
    events = _render(client, seed=1, **body)
    assert events[-1]["type"] == "error" and events[-1]["message"]


def test_paper_background_is_served_when_available(client: TestClient):
    response = client.get("/api/paper")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert response.headers["content-type"] == "image/jpeg"
