"""PyTorch の実行デバイス選択。"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cuda_is_usable() -> bool:
    """CUDA カーネルが実際に動くか（``is_available()`` だけでは不十分な環境がある）。"""
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.ones(1, device="cuda")
        probe.add_(1)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 — どんな失敗でも CPU へ退避する
        logger.warning("CUDA unavailable; falling back to CPU: %s", exc)
        return False
    return True


def detect_device(device: str | torch.device | None = None) -> torch.device:
    """明示指定があればそれを、無ければ CUDA → XPU(Intel Arc) → CPU の順で選ぶ。"""
    if device is not None:
        return torch.device(device)
    if _cuda_is_usable():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
