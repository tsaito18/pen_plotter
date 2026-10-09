"""「筆跡」画面の API（集める・見直す・学習）と、ML モデルの選択。

収集の実体は :class:`src.collector.service.CollectorService`、学習は
:class:`src.collector.training_jobs.TrainingJobManager`。ここは HTTP への薄い変換と、
LAN に公開しても安全なように書き込み先（プロファイル・文字・モデル保存先）を絞る役。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.collector.profiles import validate_profile_id
from src.collector.service import CollectorService, validate_char
from src.collector.training_jobs import TrainingJobManager
from src.layout.placement import CharPlacement
from src.pipeline import PlotterPipeline


class ModelRegistry:
    """``models_dir`` 配下のチェックポイント（*.pt）と、清書に使う 1 つ。

    Args:
        models_dir: モデルの置き場（学習の保存先もここに限る）。
        active: 起動時に使うチェックポイント。
    """

    def __init__(self, models_dir: Path | str | None, active: Path | str | None = None) -> None:
        self.dir = Path(models_dir).resolve() if models_dir else None
        self.active: Path | None = Path(active).resolve() if active else None
        if self.active is not None and not self.active.exists():
            self.active = None

    def name_of(self, path: Path | str | None) -> str | None:
        if path is None:
            return None
        path = Path(path).resolve()
        if self.dir is not None and path.is_relative_to(self.dir):
            return path.relative_to(self.dir).as_posix()
        return path.name

    def list(self) -> list[dict[str, Any]]:
        if self.dir is None or not self.dir.is_dir():
            return []
        files = sorted(self.dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [
            {
                "name": self.name_of(p),
                "size": p.stat().st_size,
                "modified": p.stat().st_mtime,
            }
            for p in files
        ]

    def inside(self, name: str) -> Path:
        """``models_dir`` 配下の相対パスを解決する（外へ出るものは ValueError）。"""
        if self.dir is None:
            raise ValueError("モデルの保存先が設定されていません")
        if not name or Path(name).is_absolute() or "\\" in name:
            raise ValueError(f"不正なパス: {name}")
        path = (self.dir / name).resolve()
        if not path.is_relative_to(self.dir):
            raise ValueError(f"モデルの保存先の外は指定できません: {name}")
        return path

    def use(self, name: str | None) -> None:
        if name is None:
            self.active = None
            return
        path = self.inside(name)
        if path.suffix != ".pt" or not path.is_file():
            raise ValueError(f"モデルが見つかりません: {name}")
        self.active = path


class SampleBody(BaseModel):
    profile: str
    character: str
    strokes: list[list[dict[str, float]]]


class ProfileBody(BaseModel):
    profile: str


class RestoreBody(ProfileBody):
    char: str
    files: list[str]


class MetadataBody(ProfileBody):
    char: str
    file: str
    key: Literal["ignore_anomaly", "ignore_stroke_mismatch"]
    value: bool = True


class QueueBody(ProfileBody):
    chars: str


class ModelBody(BaseModel):
    name: str | None = None


class TrainingBody(ProfileBody):
    kind: Literal["user_train", "finetune"] = "user_train"
    dataset: Literal["current", "all"] = "current"
    epochs: int | None = Field(default=None, ge=1, le=10000)
    batch_size: int | None = Field(default=None, ge=1, le=4096)
    learning_rate: float | None = Field(default=None, gt=0, lt=1)
    deformer_type: Literal["twostage", "transformer", "offset"] = "twostage"
    device: str | None = None
    output: str | None = None  # models_dir 配下の保存先
    checkpoint: str | None = None  # finetune の元（models_dir 配下）


def _bad(error: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(error))


def build_collect_router(
    service: CollectorService,
    training: TrainingJobManager,
    models: ModelRegistry,
    make_pipeline: Callable[[str], PlotterPipeline],
) -> APIRouter:
    """``/api/collect/*``・``/api/glyph``・``/api/training*``・``/api/models*`` のルーター。

    Args:
        make_pipeline: プロファイル ID から清書パイプラインを作る（試し書き用）。
    """
    router = APIRouter()

    def guard(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise _bad(e) from e

    # ------------------------------------------------------------------ プロファイル

    @router.post("/api/profiles")
    def create_profile(body: dict[str, Any]) -> dict[str, Any]:
        profile = guard(lambda: service.create_profile(str(body.get("id", ""))))
        return {"id": profile.id}

    # ------------------------------------------------------------------ 集める

    @router.get("/api/collect/next")
    def next_char(profile: str, prefer: str | None = None, char: str | None = None) -> dict:
        if char:
            return guard(lambda: service.char_info(profile, char))
        return guard(lambda: service.next_char(profile, prefer=prefer))

    @router.post("/api/collect/samples")
    def save(body: SampleBody) -> dict:
        return guard(lambda: service.save(body.profile, body.character, body.strokes))

    @router.get("/api/glyph")
    def glyph(char: str) -> dict:
        return {"char": char, "strokes": guard(lambda: service.glyph(char))}

    @router.get("/api/collect/queue")
    def queue(profile: str) -> dict:
        return {"chars": guard(lambda: service.queue(profile))}

    @router.post("/api/collect/queue")
    def set_queue(body: QueueBody) -> dict:
        return {"chars": guard(lambda: service.set_queue(body.profile, body.chars))}

    @router.post("/api/collect/undo")
    def undo(body: ProfileBody) -> dict:
        result = guard(lambda: service.undo_last(body.profile))
        if result is None:
            raise HTTPException(status_code=404, detail="取り消すサンプルがありません")
        return result

    # ------------------------------------------------------------------ 見直す

    @router.get("/api/collect/stats")
    def stats(profile: str) -> dict:
        return guard(lambda: service.stats(profile))

    @router.get("/api/collect/issues")
    def issues(profile: str) -> dict:
        return guard(lambda: service.issues(profile))

    @router.get("/api/collect/samples")
    def samples(profile: str, char: str) -> list[dict]:
        return guard(lambda: service.samples(profile, char))

    @router.delete("/api/collect/samples")
    def delete(profile: str, char: str, file: str | None = None) -> dict:
        if file:
            return {"remaining": guard(lambda: service.delete(profile, char, file))}
        return {"files": guard(lambda: service.delete_all(profile, char))}

    @router.post("/api/collect/samples/restore")
    def restore(body: RestoreBody) -> dict:
        return {"restored": guard(lambda: service.restore(body.profile, body.char, body.files))}

    @router.post("/api/collect/samples/metadata")
    def metadata(body: MetadataBody) -> dict:
        ok = guard(
            lambda: service.set_metadata(body.profile, body.char, body.file, body.key, body.value)
        )
        if not ok:
            raise HTTPException(status_code=404, detail="サンプルが見つかりません")
        return {"status": "ok"}

    @router.get("/api/collect/preview")
    def preview(profile: str, char: str, n: int = 3) -> dict:
        """その字が清書でどう書かれるか（スタジオと同じ経路で ``n`` 通り）。"""
        guard(lambda: validate_char(char))
        pipeline = make_pipeline(profile)
        variants = []
        for i in range(max(1, min(n, 8))):
            rendered = pipeline.renderer.render(CharPlacement(char, i * 12.0, 0.0, 10.0))
            variants.append([[round(float(v), 2) for v in s.ravel()] for s in rendered.strokes])
        coverage = pipeline.coverage
        source = next(
            (
                name
                for name in ("user_strokes", "ml_inference", "kanjivg", "geometric")
                if char in getattr(coverage, name)
            ),
            "missing_glyphs",
        )
        return {"char": char, "source": source, "variants": variants}

    # ------------------------------------------------------------------ 学習・モデル

    def training_status() -> dict:
        status = training.status()
        status["checkpoint_name"] = models.name_of(status.get("checkpoint_path"))
        return status

    @router.get("/api/training")
    def get_training() -> dict:
        return training_status()

    @router.post("/api/training/start")
    def start_training(body: TrainingBody) -> dict:
        def config() -> dict[str, Any]:
            validate_profile_id(body.profile)
            output = body.output or f"ui_{body.kind}_{time.strftime('%Y%m%d_%H%M%S')}"
            cfg: dict[str, Any] = {
                "kind": body.kind,
                "dataset": {"mode": body.dataset},
                "epochs": body.epochs,
                "batch_size": body.batch_size,
                "learning_rate": body.learning_rate,
                "deformer_type": body.deformer_type,
                "device": body.device or None,
                "output_dir": str(models.inside(output)),
                "use_aligner": True,
            }
            if body.kind == "finetune":
                source = models.inside(body.checkpoint or "pretrain_checkpoint.pt")
                if not source.is_file():
                    raise ValueError(f"元のモデルが見つかりません: {body.checkpoint}")
                cfg["checkpoint"] = str(source)
            return cfg

        cfg = guard(config)
        try:
            training.start(cfg, body.profile)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return training_status()

    @router.post("/api/training/cancel")
    def cancel_training() -> dict:
        training.cancel()
        return training_status()

    @router.get("/api/models")
    def list_models() -> dict:
        return {"models": models.list(), "active": models.name_of(models.active)}

    @router.post("/api/models/use")
    def use_model(body: ModelBody) -> dict:
        guard(lambda: models.use(body.name))
        return {"active": models.name_of(models.active)}

    return router
