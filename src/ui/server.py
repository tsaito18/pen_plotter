"""Web UI サーバー（FastAPI）。テキスト → 組版ドラフト → 清書（手書きストローク＋G-code）。

画面は ``static/`` の単一ページアプリ。サーバーは状態を持たず、リクエストごとに
:class:`Settings` の snapshot からパイプラインを組み立てる。

- ``POST /api/layout``: 組版だけの高速ドラフト（入力中のライブ表示・設定検証）。
- ``POST /api/render``: 清書。進捗を NDJSON で流し、最後にページごとのストロークと、
  **同じストロークから作った G-code** を返す。プレビューした字形がそのまま描かれる。
- プロッタへの送信はブラウザの WebSerial が行う（``static/plotter.js``）。
"""

from __future__ import annotations

import io
import json
import logging
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.collector.service import CollectorService
from src.collector.training_jobs import TrainingJobManager
from src.layout.placement import CharPlacement
from src.pipeline import PageStrokes, PlotterPipeline
from src.render.char_renderer import CharCoverageReport
from src.render.preview import stroke_contact
from src.resources import report_paper_path
from src.settings import Settings
from src.ui import content
from src.ui.collect_api import ModelRegistry, build_collect_router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).with_name("static")
# 進捗イベントの最短間隔（秒）。文字ごとに呼ばれるので間引く
_PROGRESS_INTERVAL = 0.05


@dataclass(frozen=True)
class Control:
    """設定パネルの 1 項目（``Settings`` のフィールドと 1 対 1）。"""

    field: str
    label: str
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.05
    unit: str = ""
    info: str = ""
    kind: str = "range"  # "range" | "toggle"
    hardware_note: bool = False  # 実機では 0 推奨の項目


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    controls: tuple[Control, ...]
    collapsed: bool = False


SECTIONS: tuple[Section, ...] = (
    Section(
        "hand",
        "筆跡",
        (
            Control(
                "temperature",
                "字形の揺らぎ",
                0.0,
                2.0,
                0.1,
                info="ML 変形の温度。上げると同じ字でも形が揺れる",
            ),
            Control(
                "messiness",
                "乱雑さ",
                0.0,
                2.0,
                0.1,
                info="行の上下動・字間・大きさ・傾きのばらつき。0=整った字",
            ),
            Control(
                "instance_variation",
                "字ごとの変化",
                0.0,
                1.0,
                0.05,
                info="同じ字を書くたびに形を変える強さ",
            ),
        ),
    ),
    Section(
        "layout",
        "レイアウト",
        (
            Control("font_size", "文字の大きさ", 3.0, 10.0, 0.1, "mm"),
            Control("line_spacing", "行間（罫線間隔）", 5.0, 15.0, 0.01, "mm"),
            Control("margin_top", "上余白", 5, 60, 1, "mm"),
            Control("margin_bottom", "下余白", 5, 50, 1, "mm"),
            Control("margin_left", "左余白", 1, 50, 1, "mm"),
            Control("margin_right", "右余白", 1, 50, 1, "mm"),
            Control(
                "plot_page_numbers",
                "ページ番号を書く",
                kind="toggle",
                info="用紙の「P.」欄に手書きで入れる",
            ),
        ),
    ),
    Section(
        "brush",
        "筆遣い",
        (
            Control(
                "connection_strength",
                "連綿（続け字）",
                0.0,
                1.0,
                0.05,
                info="近い画を細い線で続ける。Z 一定なので実機でも点線にならない",
            ),
            Control(
                "pressure_variation",
                "筆圧の濃淡",
                0.0,
                1.0,
                0.05,
                info="画の中の濃淡。描画中に Z が振れるため実機は 0 推奨",
                hardware_note=True,
            ),
            Control(
                "entry_taper",
                "入筆",
                0.0,
                1.0,
                0.05,
                info="始筆を軽く入れる。実機では始筆がかすれ得るため 0 推奨",
                hardware_note=True,
            ),
        ),
        collapsed=True,
    ),
)
assert {c.field for s in SECTIONS for c in s.controls} | {"paper_width", "paper_height"} == {
    f.name for f in fields(Settings)
}


class LayoutRequest(BaseModel):
    text: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)


class RenderRequest(LayoutRequest):
    profile: str | None = None
    japanese_only: bool = False
    seed: int = 0


def create_app(
    checkpoint_path: Path | str | None = None,
    kanjivg_dir: Path | str | None = None,
    user_strokes_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
) -> FastAPI:
    """Web UI の ASGI アプリを作る（スタジオ ``/`` と筆跡 ``/collect``）。

    Args:
        checkpoint_path: 起動時に使う ML チェックポイント（無ければ ML 変形なし）。
        kanjivg_dir: KanjiVG 参照字形ディレクトリ。
        user_strokes_dir: ユーザー筆跡（プロファイルのルート）。筆跡の収集もここへ保存する。
        models_dir: モデルの置き場（選択できるモデルと学習の保存先）。省略時は
            ``checkpoint_path`` のあるディレクトリ。
    """
    user_root = Path(user_strokes_dir) if user_strokes_dir is not None else None
    if models_dir is None and checkpoint_path is not None:
        models_dir = Path(checkpoint_path).parent
    models = ModelRegistry(models_dir, checkpoint_path)
    collector = CollectorService(user_root, kanjivg_dir=kanjivg_dir) if user_root else None
    training = TrainingJobManager(
        root_dir=user_root or Path("data/user_strokes"),
        ref_dir=Path(kanjivg_dir) if kanjivg_dir else None,
    )
    # パイプライン（特に ML 推論）は並行実行に対応しないので清書は 1 件ずつ
    render_lock = threading.Lock()

    def profile_list() -> list:
        if collector is not None:
            return collector.profiles()
        return []

    def build(
        settings: Settings,
        req: RenderRequest | None = None,
        *,
        profile: str | None = None,
    ) -> PlotterPipeline:
        ids = {p.id for p in profile_list()}
        wanted = profile or (req.profile if req else None)
        return PlotterPipeline(
            settings,
            checkpoint_path=models.active,
            kanjivg_dir=kanjivg_dir,
            user_strokes_dir=user_strokes_dir,
            profile=wanted if wanted in ids else None,
            japanese_only=bool(req and req.japanese_only),
            seed=req.seed if req else 0,
            style_revision=collector.revision if collector else 0,
        )

    app = FastAPI(title="Pen Plotter", docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/collect", include_in_schema=False)
    def collect_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "collect.html", headers={"Cache-Control": "no-cache"})

    if collector is not None:
        app.include_router(
            build_collect_router(
                collector, training, models, lambda pid: build(Settings(), profile=pid)
            )
        )

    @app.get("/api/bootstrap")
    def bootstrap() -> dict[str, Any]:
        defaults = Settings()
        return {
            "settings": defaults.to_dict(),
            "sections": [
                {
                    "id": s.id,
                    "title": s.title,
                    "collapsed": s.collapsed,
                    "controls": [asdict(c) for c in s.controls],
                }
                for s in SECTIONS
            ],
            "profiles": [
                {"id": p.id, "characters": p.character_count, "samples": p.sample_count}
                for p in profile_list()
            ],
            "examples": [{"label": k, "text": v} for k, v in content.EXAMPLES.items()],
            "syntax": [asdict(row) for row in content.SYNTAX],
            "paper": {
                "width": defaults.paper_width,
                "height": defaults.paper_height,
                "background": report_paper_path() is not None,
            },
            "model": models.name_of(models.active),
            "collect": collector is not None,
            "sources": {
                "ml": models.active is not None,
                "user": bool(profile_list()),
                "kanjivg": kanjivg_dir is not None and Path(kanjivg_dir).is_dir(),
            },
        }

    @app.post("/api/layout")
    def layout(req: LayoutRequest) -> dict[str, Any]:
        settings = Settings.from_dict(req.settings)
        errors = settings.validate()
        if errors:
            return {"errors": errors, "pages": [], "ruled": []}
        pipeline = build(settings)
        pages = pipeline.typeset(req.text) if req.text.strip() else [[]]
        ruled = [_r(s.ravel()) for s in pipeline.typesetter.layout.ruled_line_strokes()]
        return {"errors": [], "pages": [_draft_page(p) for p in pages], "ruled": ruled}

    @app.post("/api/render")
    def render(req: RenderRequest) -> StreamingResponse:
        return StreamingResponse(
            _ndjson(_render_events(req)),
            media_type="application/x-ndjson",
            # gzip は全体をバッファしてしまい進捗が届かないので、この応答だけ無圧縮にする
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Content-Encoding": "identity",
            },
        )

    def _render_events(req: RenderRequest) -> Iterator[dict[str, Any]]:
        yield {"type": "progress", "fraction": 0.0, "message": "準備中"}
        settings = Settings.from_dict(req.settings)
        if errors := settings.validate():
            yield {"type": "error", "message": " / ".join(errors)}
            return
        if not req.text.strip():
            yield {"type": "error", "message": "テキストを入力してください"}
            return
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        last = [0.0]

        def progress(fraction: float, message: str) -> None:
            now = time.monotonic()
            if now - last[0] >= _PROGRESS_INTERVAL:
                last[0] = now
                f = round(min(max(fraction, 0.0), 1.0), 3)
                events.put({"type": "progress", "fraction": f, "message": message})

        def work() -> None:
            try:
                with render_lock:
                    start = time.monotonic()
                    pipeline = build(settings, req)
                    pages = pipeline.render_document(req.text, progress)
                    events.put(
                        {
                            "type": "result",
                            "seed": req.seed,
                            "elapsed": round(time.monotonic() - start, 2),
                            "pages": [_render_page(pipeline, p) for p in pages],
                            "coverage": _coverage(pipeline.coverage),
                        }
                    )
            except Exception as exc:
                logger.exception("render failed")
                events.put({"type": "error", "message": f"生成に失敗しました: {exc}"})
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (event := events.get()) is not None:
            yield event

    @app.get("/api/paper")
    def paper() -> Response:
        data = _paper_jpeg()
        if data is None:
            return Response(status_code=404)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    return app


def _ndjson(events: Iterator[dict[str, Any]]) -> Iterator[bytes]:
    for event in events:
        yield (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _r(values: Any, digits: int = 2) -> list[float]:
    return [round(float(v), digits) for v in values]


def _draft_page(placements: list[CharPlacement]) -> dict[str, list]:
    """組版結果を軽量な配列へ（文字 / 罫線 / 数式の枠）。"""
    chars, rules, math = [], [], []
    for p in placements:
        if p.line_segment is not None:
            rules.append(_r(p.line_segment))
        elif p.math is not None:
            x, y, w, h = p.math.bbox
            math.append([*_r((x, y, w, h)), p.math.align, p.math.source])
        elif p.char.strip():
            chars.append([p.char, *_r((p.x, p.y, p.font_size)), round(p.slant, 4)])
    return {"chars": chars, "rules": rules, "math": math}


def _render_page(pipeline: PlotterPipeline, page: PageStrokes) -> dict[str, Any]:
    """1 ページ分のストローク（点列・接触率・筆法）と、それと同一の G-code。"""
    config = pipeline.plotter_config
    lines, spans = pipeline.generator.generate_indexed(
        page.strokes, finishes=page.finishes, vary_speed=True
    )
    # 送信側はコメント行を送らないので、ここで落として行番号を揃える
    keep = [i for i, line in enumerate(lines) if line.strip() and not line.startswith(";")]
    new_index = {old: new for new, old in enumerate(keep)}
    new_index[len(lines)] = len(keep)

    def remap(i: int) -> int:
        while i not in new_index:
            i += 1
        return new_index[i]

    strokes = []
    for stroke, finish in zip(page.strokes, page.finishes):
        contact = stroke_contact(stroke, finish, config)
        uniform = contact.size == 0 or bool((abs(contact - contact[0]) < 1e-3).all())
        strokes.append(
            {
                "points": _r(stroke.ravel()),
                "contact": round(float(contact[0]) if contact.size else 1.0, 3)
                if uniform
                else _r(contact, 3),
                "finish": finish,
            }
        )
    return {
        "strokes": strokes,
        "spans": [[remap(s), remap(e)] for s, e in spans],
        "gcode": [lines[i] for i in keep],
    }


def _coverage(report: CharCoverageReport) -> dict[str, dict[str, Any]]:
    return {
        name: {"count": len(chars), "chars": "".join(sorted(set(chars)))}
        for name, chars in asdict(report).items()
    }


@lru_cache(maxsize=1)
def _paper_jpeg() -> bytes | None:
    """レポート用紙のスキャン画像を表示用に縮小した JPEG。無ければ None。"""
    path = report_paper_path()
    if path is None:
        return None
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((1600, 1600))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85, optimize=True)
    except OSError:
        logger.exception("report paper load failed")
        return None
    return buf.getvalue()
