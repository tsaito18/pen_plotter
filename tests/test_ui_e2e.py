"""Web UI の画面テスト（実ブラウザ）。スタジオ（書く→清書→描く）と筆跡（集める・見直す・学習）。

プロッタはブラウザ内の偽シリアルポート（受けた行に即 ok を返す）で置き換える。
Playwright とブラウザが無い環境では丸ごとスキップする（``uv sync --extra e2e`` と
``playwright install chromium`` で有効になる）。
"""

from __future__ import annotations

import json
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
playwright_api = pytest.importorskip("playwright.sync_api")

from src.pipeline import PlotterPipeline  # noqa: E402
from src.ui.server import create_app  # noqa: E402

pytestmark = pytest.mark.e2e

# 大きな字・広い行間にして、短い文で 2 ページになるようにする（送信テストを速くする）
BIG = {"font_size": 10.0, "line_spacing": 15.0}
TWO_PAGES = "十人口一" * 75

# 受けた行をそのまま記録し、すぐ ok を返す偽の WebSerial
FAKE_SERIAL = """
(() => {
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  window.__sent = [];
  class FakePort extends EventTarget {
    async open() {
      let ctrl;
      this.readable = new ReadableStream({ start(c) { ctrl = c; } });
      setTimeout(() => ctrl.enqueue(enc.encode("\\r\\nGrbl 1.1h ['$' for help]\\r\\n")), 50);
      let buf = "";
      this.writable = new WritableStream({
        write(chunk) {
          buf += dec.decode(chunk);
          let i;
          while ((i = buf.indexOf("\\n")) >= 0) {
            window.__sent.push(buf.slice(0, i));
            buf = buf.slice(i + 1);
            setTimeout(() => ctrl.enqueue(enc.encode("ok\\r\\n")), 2);
          }
        },
      });
    }
    async close() {}
  }
  const port = new FakePort();
  const serial = new EventTarget();
  serial.requestPort = async () => port;
  serial.getPorts = async () => [];
  Object.defineProperty(navigator, "serial", { value: serial, configurable: true });
})();
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """テスト用データで Web UI を別スレッドに立てる。"""

    def __init__(self, root: Path, kanjivg_dir: Path, models: Path) -> None:
        import uvicorn

        self.root = root
        self.port = _free_port()
        app = create_app(kanjivg_dir=kanjivg_dir, user_strokes_dir=root, models_dir=models)
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                break
            time.sleep(0.02)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # ブラウザ未インストール
            pytest.skip(f"Chromium を起動できません: {exc}")
        yield b
        b.close()


@pytest.fixture
def server(tmp_path: Path, kanjivg_dir: Path, user_strokes_root: Path) -> Iterator[_Server]:
    root = tmp_path / "strokes"
    shutil.copytree(user_strokes_root, root)
    models = tmp_path / "models"
    models.mkdir()
    for name in ("a.pt", "b.pt"):
        (models / name).write_bytes(b"not a real checkpoint")
    s = _Server(root, kanjivg_dir, models)
    yield s
    s.stop()


def _page(browser, server: _Server, path: str, *, width=1440, height=900, settings=None):
    context = browser.new_context(viewport={"width": width, "height": height}, has_touch=True)
    context.add_init_script(FAKE_SERIAL)
    script = "localStorage.setItem('pp.profile', JSON.stringify('taro'));"
    if settings:
        script += f"localStorage.setItem('pp.settings', {json.dumps(json.dumps(settings))});"
    context.add_init_script(script)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server.url + path)
    page.errors = errors  # type: ignore[attr-defined]
    return page


def _type(page, text: str) -> None:
    page.fill("#textInput", text)


def _render(page) -> None:
    page.keyboard.press("Control+Enter")
    page.wait_for_selector("#plotBtn:not([hidden])", timeout=60_000)


def _sample_files(root: Path, profile: str, char: str) -> list[Path]:
    return sorted((root / profile / char).glob("*.json"))


# --------------------------------------------------------------------- スタジオ


def test_studio_draft_render_and_two_page_plot(browser, server: _Server):
    page = _page(browser, server, "/", settings=BIG)
    _type(page, TWO_PAGES)
    page.wait_for_function("document.getElementById('statPages').textContent === '2'")
    assert page.text_content("#modeText") == "下書き"
    _render(page)
    assert "清書" in page.text_content("#modeText")
    assert page.is_visible("#coverageCard")

    page.click("#plotBtn")  # 未接続なら接続まで
    page.wait_for_function("document.getElementById('machineState').textContent === '待機中'")
    page.click("#startBtn")
    page.wait_for_function("window.__sent.length > 60")

    # 一時停止はペンを上げた直後で止まる
    page.click("#pauseBtn")
    page.wait_for_function("document.getElementById('machineState').textContent === '一時停止中'")
    sent = page.evaluate("window.__sent.length")
    page.wait_for_timeout(300)
    assert page.evaluate("window.__sent.length") == sent
    assert page.evaluate("window.__sent.at(-1)") == "G1G90 Z0.5 F5000"

    page.click("#resumeBtn")
    page.wait_for_function("document.getElementById('paperDialog').open", timeout=60_000)
    page.keyboard.press("Enter")  # 用紙交換して続ける
    page.wait_for_function("document.getElementById('modeText').textContent.includes('2ページ目')")

    # 停止はペンを上げて終わる
    page.click("#stopBtn")
    page.wait_for_function("document.getElementById('machineState').textContent === '待機中'")
    assert page.evaluate("window.__sent.at(-1)") == "G1G90 Z0.5 F5000"
    assert not page.errors


def test_studio_marks_render_stale_and_rerolls(browser, server: _Server):
    page = _page(browser, server, "/")
    _type(page, "十人一口")
    _render(page)
    seed = page.input_value("#seedInput")
    page.eval_on_selector(
        "#ctl-font_size",
        "el => { el.value = 6; el.dispatchEvent(new Event('input', { bubbles: true })); }",
    )
    page.wait_for_function("document.getElementById('modeText').textContent.includes('古い')")
    assert page.is_visible("#renderBtn")
    _render(page)
    page.click("#rerollBtn")
    page.wait_for_function(f"document.getElementById('seedInput').value !== '{seed}'")
    assert not page.errors


def test_studio_opens_uploaded_gcode(browser, server: _Server, tmp_path: Path, kanjivg_dir):
    (gcode,) = PlotterPipeline(kanjivg_dir=kanjivg_dir, seed=0).generate_gcode(
        "十人", tmp_path / "up.gcode"
    )
    page = _page(browser, server, "/")
    page.click("#tabPlot")
    page.set_input_files("#fileInput", str(gcode))
    page.wait_for_function("document.getElementById('modeText').textContent.includes('up.gcode')")
    assert page.is_visible("#useRenderBtn") is False  # 清書が無ければ戻り先も無い
    assert not page.errors


def test_studio_sends_unlearned_chars_to_the_collector(browser, server: _Server):
    page = _page(browser, server, "/")
    _type(page, "十人一口")  # 十 だけがユーザー筆跡
    _render(page)
    page.wait_for_selector("#teach:not([hidden])")
    page.click("#teachBtn")
    page.wait_for_selector(".toast")
    collect = _page(browser, server, "/collect#write", width=1180, height=820)
    collect.wait_for_selector("#request:not([hidden])")
    assert set(collect.text_content("#requestChars")) == {"人", "一", "口"}
    collect.click("#requestStart")
    collect.wait_for_function(
        "document.getElementById('subjectTier').textContent.startsWith('依頼 1')"
    )
    assert not page.errors and not collect.errors


# --------------------------------------------------------------------- 筆跡


def _draw(page, strokes: list[list[tuple[float, float]]]) -> None:
    box = page.locator("#pad").bounding_box()
    for stroke in strokes:
        x, y = stroke[0]
        page.mouse.move(box["x"] + box["width"] * x, box["y"] + box["height"] * y)
        page.mouse.down()
        for x, y in stroke[1:]:
            page.mouse.move(box["x"] + box["width"] * x, box["y"] + box["height"] * y, steps=8)
        page.mouse.up()


def test_collect_first_run_write_save_and_undo(browser, server: _Server, tmp_path: Path):
    shutil.rmtree(server.root / "taro")
    page = _page(browser, server, "/collect", width=1180, height=820)
    page.wait_for_selector("#welcome:not([hidden])")
    page.fill("#welcomeInput", "hana")
    page.keyboard.press("Enter")
    page.wait_for_selector("#welcome", state="hidden")
    char = page.text_content("#subjectChar")

    _draw(page, [[(0.2, 0.3), (0.8, 0.3)], [(0.5, 0.15), (0.5, 0.85)]])
    assert page.text_content("#strokeText").startswith("2 画")
    page.click("#saveBtn")
    page.wait_for_function(f"document.getElementById('subjectChar').textContent !== '{char}'")
    assert len(_sample_files(server.root, "hana", char)) == 1
    assert page.text_content("#tallySession") == "1"

    page.click(".toast-action")  # 保存の取り消し
    stats = "fetch('/api/collect/stats?profile=hana').then(r => r.json())"
    page.wait_for_function(f"{stats}.then(s => s.total_samples === 0)")
    assert _sample_files(server.root, "hana", char) == []
    assert not page.errors


def test_collect_pen_rejects_palm_and_two_finger_tap_undoes(browser, server: _Server):
    page = _page(browser, server, "/collect", width=1180, height=820)
    page.wait_for_selector("#subjectChar")
    counts = page.evaluate(
        """async () => {
          const live = document.querySelector('.pad-live');
          const r = live.getBoundingClientRect();
          const fire = (type, id, kind, x, y) => live.dispatchEvent(new PointerEvent(type, {
            pointerId: id, pointerType: kind, pressure: 0.6, bubbles: true, cancelable: true,
            clientX: r.left + r.width * x, clientY: r.top + r.height * y }));
          const text = () => document.getElementById('strokeText').textContent;
          const out = [];
          for (const y of [0.3, 0.6]) {
            fire('pointerdown', 2, 'pen', 0.2, y);
            for (let i = 1; i <= 10; i++) fire('pointermove', 2, 'pen', 0.2 + i * 0.06, y);
            fire('pointerup', 2, 'pen', 0.8, y);
          }
          out.push(text());
          fire('pointerdown', 5, 'touch', 0.3, 0.8); fire('pointermove', 5, 'touch', 0.6, 0.8);
          fire('pointerup', 5, 'touch', 0.6, 0.8);
          out.push(text());
          fire('pointerdown', 7, 'touch', 0.4, 0.5); fire('pointerdown', 8, 'touch', 0.6, 0.5);
          await new Promise((res) => setTimeout(res, 60));
          fire('pointerup', 7, 'touch', 0.4, 0.5); fire('pointerup', 8, 'touch', 0.6, 0.5);
          out.push(text());
          fire('pointerdown', 1, 'mouse', 0.3, 0.3); fire('pointerup', 1, 'mouse', 0.3, 0.3);
          out.push(text());
          return out.map((t) => t.split(' ')[0]);
        }"""
    )
    assert counts == ["2", "2", "1", "1"]  # 手のひら・ペン検出後のマウスは無視、二本指で 1 画戻る
    assert not page.errors


def test_collect_review_delete_and_restore(browser, server: _Server):
    page = _page(browser, server, "/collect#review", width=1180, height=820)
    page.click(".cell[title^='十']")
    page.wait_for_selector("#drawerSamples .thumb")
    assert page.locator("#drawerSamples .thumb").count() == 2
    page.wait_for_selector("#drawerPreview .thumb")  # 清書での試し書き
    page.locator("#drawerSamples .thumb-del").first.click()
    page.wait_for_function("document.querySelectorAll('#drawerSamples .thumb').length === 1")
    assert len(_sample_files(server.root, "taro", "十")) == 1
    page.click(".toast-action")
    page.wait_for_function("document.querySelectorAll('#drawerSamples .thumb').length === 2")
    assert len(_sample_files(server.root, "taro", "十")) == 2
    assert not page.errors


def test_model_switch_reaches_the_studio(browser, server: _Server):
    page = _page(browser, server, "/collect#train", width=1180, height=820)
    page.wait_for_selector("#models .model")
    page.locator("#models .model", has_text="b.pt").locator("button").click()
    page.wait_for_function("document.getElementById('modelsMeta').textContent.includes('b.pt')")
    studio = _page(browser, server, "/")
    studio.wait_for_function("document.getElementById('modelSelect').value === 'b.pt'")
    assert not page.errors and not studio.errors
