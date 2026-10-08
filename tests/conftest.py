"""共通フィクスチャ。実データ（data/ は git 管理外）に依存しない小さな字形ソースを作る。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.collector.data_format import StrokePoint, StrokeSample


def write_sample(
    root: Path,
    char: str,
    strokes: list[list[tuple[float, float]]],
    types: list[str] | None = None,
    index: int = 0,
) -> Path:
    """``<root>/<char>/<char>_<index>.json`` に StrokeSample を書く。"""
    sample = StrokeSample(
        character=char,
        strokes=[[StrokePoint(x, y) for x, y in s] for s in strokes],
        stroke_types=types or [],
    )
    path = root / char / f"{char}_{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.save(path)
    return path


def line(x0: float, y0: float, x1: float, y1: float, n: int = 8) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n))]


# KanjiVG 形式（Y-UP・0..10）の最小字形セット
KANJIVG_CHARS: dict[str, tuple[list[list[tuple[float, float]]], list[str]]] = {
    "一": ([line(1, 5, 9, 5)], ["㇐"]),
    "十": ([line(1, 6, 9, 6), line(5, 9, 5, 1)], ["㇐", "㇑"]),
    "人": ([line(5, 9, 1, 1), line(5, 6, 9, 1)], ["㇒", "㇏"]),
    "口": ([line(2, 8, 2, 2), line(2, 8, 8, 8) + line(8, 8, 8, 2)[1:], line(7.8, 2, 2, 2)], []),
    "あ": ([line(2, 7, 8, 7), line(5, 9, 5, 1), line(3, 5, 7, 2)], []),
    "り": ([line(4, 8, 4, 4), line(6, 9, 6, 1)], []),  # 細い字（字間の検証用）
    "1": ([line(5, 9, 5, 1)], []),
    "2": ([line(2, 8, 8, 8) + line(8, 8, 2, 1)[1:], line(2, 1, 8, 1)], []),
}


@pytest.fixture(scope="session")
def kanjivg_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("kanjivg")
    for char, (strokes, types) in KANJIVG_CHARS.items():
        write_sample(root, char, strokes, types)
    return root


@pytest.fixture(scope="session")
def user_strokes_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """プロファイルのルート（``<root>/taro/<文字>/``）。筆跡は Y-DOWN。"""
    root = tmp_path_factory.mktemp("user_strokes")
    profile = root / "taro"
    write_sample(profile, "十", [line(10, 50, 90, 45), line(50, 10, 52, 90)])
    write_sample(profile, "十", [line(10, 50, 90, 50, n=30), line(50, 10, 50, 90, n=30)], index=1)
    write_sample(profile, "a", [line(20, 20, 20, 80), line(20, 50, 60, 50)])
    return root


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """ランダム重みの小さな twostage チェックポイント（推論経路の動作確認用）。"""
    import torch

    from src.model.deformers import build_deformer
    from src.model.style_encoder import StyleEncoder

    torch.manual_seed(0)
    config = {
        "deformer_type": "twostage",
        "style_dim": 16,
        "hidden_dim": 16,
        "num_points": 32,
        "dropout": 0.0,
        "d_model": 16,
        "nhead": 2,
        "num_self_attn_layers": 1,
        "ff_dim": 16,
    }
    path = tmp_path_factory.mktemp("models") / "pretrain_checkpoint.pt"
    torch.save(
        {
            "deformer_state_dict": build_deformer(config).state_dict(),
            "style_encoder_state_dict": StyleEncoder(style_dim=16).state_dict(),
            "config": config,
            "norm_stats": None,
            "version": 3,
        },
        path,
    )
    return path
