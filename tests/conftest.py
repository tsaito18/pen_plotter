import numpy as np
import numpy.typing as npt
import pytest

from src.gcode.config import PlotterConfig
from src.gcode.generator import GCodeGenerator, Stroke


@pytest.fixture
def plotter_config() -> PlotterConfig:
    """デフォルトの PlotterConfig"""
    return PlotterConfig()


@pytest.fixture
def gcode_generator(plotter_config: PlotterConfig) -> GCodeGenerator:
    """デフォルトの GCodeGenerator"""
    return GCodeGenerator(plotter_config)


@pytest.fixture
def square_stroke() -> Stroke:
    """四角形ストローク"""
    return np.array([
        [10.0, 10.0],
        [50.0, 10.0],
        [50.0, 50.0],
        [10.0, 50.0],
        [10.0, 10.0],
    ])


@pytest.fixture
def triangle_stroke() -> Stroke:
    """三角形ストローク"""
    return np.array([
        [30.0, 10.0],
        [50.0, 50.0],
        [10.0, 50.0],
        [30.0, 10.0],
    ])


@pytest.fixture
def line_stroke() -> Stroke:
    """単純直線ストローク"""
    return np.array([
        [0.0, 0.0],
        [10.0, 10.0],
    ])


@pytest.fixture
def scattered_strokes() -> list[Stroke]:
    """最適化テスト用の散らばったストローク群（seed固定）"""
    rng = np.random.default_rng(42)
    strokes = []
    for _ in range(10):
        start = rng.uniform(0, 300, size=2)
        end = start + rng.uniform(-20, 20, size=2)
        strokes.append(np.array([start, end]))
    return strokes


@pytest.fixture
def sample_gcode(gcode_generator: GCodeGenerator, square_stroke: Stroke) -> list[str]:
    """テスト用G-code行リスト"""
    return gcode_generator.generate([square_stroke])
