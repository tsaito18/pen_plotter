"""Web UI のサーバー API（入力 → 組版ドラフト → 清書ストローク＋同一の G-code）。"""

from __future__ import annotations

import json
import re
import shutil
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
    for asset in ["app.js", "plotter.js", "paper.js", "editor.js", "base.css", "studio.css"]:
        assert client.get(f"/static/{asset}").status_code == 200, asset


def test_every_module_import_resolves_without_a_build_step(client: TestClient):
    """画面は import map とブラウザの ES Modules だけで動く（ビルドしない）ので、

    両画面の import map の行き先と、static 配下の JS の import 先がすべて配信されること。
    """
    static = Path(__file__).resolve().parents[1] / "src" / "ui" / "static"
    bare: set[str] = set()
    for page in ["/", "/collect"]:
        html = client.get(page).text
        found = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
        imports = json.loads(found.group(1))
        for target in imports["imports"].values():
            assert client.get(target).status_code == 200, target
        bare |= set(imports["imports"])
    for path in static.rglob("*.js"):
        if "vendor" in path.parts:
            continue
        for spec in re.findall(r'^import [^;]*? from "([^"]+)";', path.read_text(), re.M):
            if spec.startswith("."):
                url = "/static/" + (path.parent / spec).resolve().relative_to(static).as_posix()
                assert client.get(url).status_code == 200, f"{path.name}: {spec}"
            else:
                assert spec in bare, f"{path.name}: {spec} が import map に無い"


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


# --- 筆跡（収集・見直し・学習）---

POINTS = [
    [
        {"x": 100 + 20 * k, "y": 100 + 15 * k, "pressure": 0.5, "timestamp": 300.0 * k}
        for k in range(8)
    ]
]


@pytest.fixture
def studio(tmp_path: Path, kanjivg_dir: Path, user_strokes_root: Path) -> TestClient:
    root = tmp_path / "strokes"
    shutil.copytree(user_strokes_root, root)
    models = tmp_path / "models"
    (models / "runs").mkdir(parents=True)
    (models / "finetuned.pt").write_bytes(b"x")
    (models / "runs" / "user_train.pt").write_bytes(b"x")
    app = create_app(kanjivg_dir=kanjivg_dir, user_strokes_dir=root, models_dir=models)
    return TestClient(app)


def test_collect_page_and_shared_assets(client: TestClient):
    html = client.get("/collect").text
    assert '<script type="module"' in html and "collect.js" in html
    for asset in ["collect.js", "pad.js", "common.js", "base.css", "collect.css", "icons.svg"]:
        assert client.get(f"/static/{asset}").status_code == 200, asset


def test_collect_roundtrip_with_trash_and_undo(studio: TestClient):
    taro = {"profile": "taro"}
    char = studio.get("/api/collect/next", params=taro).json()["char"]
    saved = studio.post("/api/collect/samples", json={**taro, "character": char, "strokes": POINTS})
    assert saved.status_code == 200 and saved.json()["count"] == 1
    (sample,) = studio.get("/api/collect/samples", params={**taro, "char": char}).json()
    target = {**taro, "char": char, "file": sample["filename"]}
    assert studio.delete("/api/collect/samples", params=target).json()["remaining"] == 0
    restore = {**taro, "char": char, "files": [sample["filename"]]}
    assert studio.post("/api/collect/samples/restore", json=restore).json()["restored"] == 1
    meta = {
        **taro,
        "char": char,
        "file": sample["filename"],
        "key": "ignore_anomaly",
        "value": True,
    }
    assert studio.post("/api/collect/samples/metadata", json=meta).status_code == 200
    assert studio.post("/api/collect/undo", json=taro).json()["character"] == char
    stats = studio.get("/api/collect/stats", params=taro).json()
    assert stats["char_counts"]["十"] == 2
    assert set(studio.get("/api/collect/issues", params=taro).json()) == {"anomalies", "mismatches"}
    assert studio.get("/api/glyph", params={"char": "十"}).json()["strokes"]


@pytest.mark.parametrize(
    "body", [{"profile": "taro", "character": "../"}, {"profile": "../x", "character": "あ"}]
)
def test_collect_rejects_unsafe_paths(studio: TestClient, body: dict):
    assert studio.post("/api/collect/samples", json={**body, "strokes": POINTS}).status_code == 400


def test_profiles_are_shared_with_the_studio(studio: TestClient):
    assert studio.post("/api/profiles", json={"id": "hana"}).status_code == 200
    assert studio.post("/api/profiles", json={"id": "a b"}).status_code == 400
    ids = [p["id"] for p in studio.get("/api/bootstrap").json()["profiles"]]
    assert ids == ["hana", "taro"]


def test_studio_requests_reach_the_collector_queue(studio: TestClient):
    studio.post("/api/collect/queue", json={"profile": "taro", "chars": "一人一"})
    assert studio.get("/api/collect/queue", params={"profile": "taro"}).json()["chars"] == [
        "一",
        "人",
    ]
    body = {"profile": "taro", "character": "人", "strokes": POINTS}
    studio.post("/api/collect/samples", json=body)
    assert studio.get("/api/collect/queue", params={"profile": "taro"}).json()["chars"] == ["一"]


def test_char_preview_uses_the_studio_handwriting(studio: TestClient):
    params = {"profile": "taro", "char": "十", "n": 3}
    data = studio.get("/api/collect/preview", params=params).json()
    assert data["source"] == "user_strokes" and len(data["variants"]) == 3
    assert all(v and all(len(s) >= 4 for s in v) for v in data["variants"])


def test_models_are_listed_and_switched_inside_the_models_dir(studio: TestClient):
    data = studio.get("/api/models").json()
    assert {m["name"] for m in data["models"]} == {"finetuned.pt", "runs/user_train.pt"}
    assert data["active"] is None
    assert studio.post("/api/models/use", json={"name": "runs/user_train.pt"}).status_code == 200
    assert studio.get("/api/models").json()["active"] == "runs/user_train.pt"
    assert studio.get("/api/bootstrap").json()["sources"]["ml"] is True
    for bad in ["../x.pt", "/etc/passwd", "missing.pt"]:
        assert studio.post("/api/models/use", json={"name": bad}).status_code == 400
    assert studio.post("/api/models/use", json={"name": None}).json()["active"] is None


@pytest.mark.parametrize("output", ["../evil", "/tmp/evil", "a/../../b"])
def test_training_writes_only_inside_the_models_dir(studio: TestClient, output: str):
    body = {"profile": "taro", "output": output}
    assert studio.post("/api/training/start", json=body).status_code == 400
    assert studio.get("/api/training").json()["state"] == "idle"
