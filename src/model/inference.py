"""訓練済み V3 変形モデルによる推論（KanjiVG 参照字形 → ユーザーの書き癖へ変形）。"""

from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from src.geometry import resample_stroke, rotation_matrix
from src.model.data import limit_style_points, load_style_sample, normalize_deltas
from src.model.deformers import OFFSET_CLAMP, build_deformer, smooth_offsets
from src.model.device import detect_device
from src.model.style_encoder import StyleEncoder

logger = logging.getLogger(__name__)

# temperature=1 での点ごとのオフセット揺らぎ振幅（オフセット座標系, clamp=0.4 の約 1/3）
TEMP_NOISE_AMP = 0.12
# 変形後の画ごとの微小な回転・拡縮・移動の強さ
STROKE_NOISE_SCALE = 0.02


def temperature_noise(
    num_strokes: int,
    num_points: int,
    amp: float,
    rng: np.random.Generator,
    num_ctrl: int = 6,
) -> NDArray:
    """画ごと・x/y 独立の低周波ノイズ ``(num_strokes, num_points, 2)``（float32）。

    少数の制御点に置いたガウスノイズを点数へ線形補間するので、点間で相関した
    滑らかな揺らぎになる（高周波のガタつきにならない）。``amp=0`` でゼロ。
    """
    if amp <= 0.0:
        return np.zeros((num_strokes, num_points, 2), dtype=np.float32)
    ctrl = rng.normal(0.0, amp, size=(num_strokes, num_ctrl, 2))
    t_ctrl = np.linspace(0.0, 1.0, num_ctrl)
    t_pts = np.linspace(0.0, 1.0, num_points)
    out = np.empty((num_strokes, num_points, 2), dtype=np.float32)
    for b in range(num_strokes):
        for d in range(2):
            out[b, :, d] = np.interp(t_pts, t_ctrl, ctrl[b, :, d])
    return out


def upsample_stroke(
    points: NDArray, pts_per_unit: float = 8.0, corner_thresh: float = 0.85
) -> NDArray[np.float32]:
    """角で区切った区間ごとに 3 次スプラインで補間し、曲線を滑らかに増点する。

    入力点は動かさない（補間のみ）ので、角は鋭いまま保たれる。
    """
    if len(points) < 3:
        return points
    corners = [0]
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-8 or len2 < 1e-8:
            continue
        if np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0) < corner_thresh:
            corners.append(i)
    corners.append(len(points) - 1)

    parts: list[NDArray] = []
    for start, end in pairwise(corners):
        seg = points[start : end + 1]
        seg_diffs = np.diff(seg, axis=0)
        seg_lens = np.sqrt((seg_diffs**2).sum(axis=1))
        n_out = max(int(seg_lens.sum() * pts_per_unit), 2)
        if len(seg) < 3:
            t_orig = np.linspace(0, 1, len(seg))
            t_new = np.linspace(0, 1, n_out)
            part = np.stack(
                [np.interp(t_new, t_orig, seg[:, 0]), np.interp(t_new, t_orig, seg[:, 1])], axis=1
            )
        else:
            cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
            if cum[-1] < 1e-12:
                part = seg
            else:
                part = CubicSpline(cum, seg, bc_type="clamped")(np.linspace(0.0, cum[-1], n_out))
        parts.append(part[1:] if parts else part)
    return np.concatenate(parts, axis=0).astype(np.float32)


class StrokeInference:
    """V3 チェックポイントを読み込み、参照字形をユーザーのスタイルで変形する。

    Args:
        checkpoint_path: ``deformer_state_dict`` を持つ V3 チェックポイント。
        style_sample: StyleEncoder へ渡す筆跡の差分系列 ``(1, N, 3)``。
        device: 実行デバイス（省略時は自動選択）。
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        style_sample: torch.Tensor | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = detect_device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "deformer_state_dict" not in checkpoint:
            raise ValueError(f"not a V3 deformation checkpoint: {checkpoint_path}")
        config = checkpoint.get("config", {})
        self.norm_stats: dict | None = checkpoint.get("norm_stats")
        self.deformer_type: str = config.get("deformer_type", "offset")
        self.num_points: int = config.get("num_points", 32)

        self.deformer = build_deformer(config)
        self.deformer.load_state_dict(checkpoint["deformer_state_dict"])
        self.deformer.to(self.device).eval()
        self.style_encoder = StyleEncoder(style_dim=config.get("style_dim", 128))
        self.style_encoder.load_state_dict(checkpoint["style_encoder_state_dict"])
        self.style_encoder.to(self.device).eval()

        self._style: torch.Tensor | None = None
        if style_sample is not None:
            self.set_style(style_sample)

    @classmethod
    def from_user_strokes(
        cls, checkpoint_path: Path | str, user_strokes_dir: Path | str | None
    ) -> StrokeInference:
        """ユーザー筆跡ディレクトリの全サンプルをスタイルとして使う推論器を作る。"""
        return cls(checkpoint_path, style_sample=load_style_sample(user_strokes_dir))

    @torch.no_grad()
    def set_style(self, style_sample: torch.Tensor) -> None:
        """スタイル（筆跡の差分系列）を設定する。スタイルベクトルは 1 回だけ計算する。"""
        if self.norm_stats is not None:
            style_sample = normalize_deltas(style_sample, self.norm_stats)
        self._style = self.style_encoder(limit_style_points(style_sample).to(self.device))

    @torch.no_grad()
    def generate(
        self,
        reference_strokes: list[NDArray],
        temperature: float = 0.0,
        deform_scale: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> list[NDArray[np.float32]]:
        """参照ストロークを変形する（2 点未満の画は除く）。

        Args:
            reference_strokes: KanjiVG 参照字形（Y-UP）。
            temperature: 点ごとの低周波揺らぎの強さ（0 で決定的）。
            deform_scale: 変形量の倍率（<1 で参照字形へ近づける。多画字の固まり防止）。
            rng: 揺らぎの乱数源（省略時は毎回新しい非決定的な乱数）。
        """
        rng = rng if rng is not None else np.random.default_rng()
        if self._style is None:
            raise RuntimeError("style is not set; call set_style() first")
        refs = [
            (i, resample_stroke(np.asarray(s, dtype=np.float32), self.num_points))
            for i, s in enumerate(reference_strokes)
            if len(s) >= 2
        ]
        if not refs:
            raise ValueError("at least one reference stroke with >= 2 points is required")

        ref_batch = torch.tensor(np.stack([r for _, r in refs]), device=self.device)
        idx_batch = torch.tensor([i for i, _ in refs], device=self.device)
        style_batch = self._style.expand(len(refs), -1)

        if self.deformer_type == "affine":
            transformed, _params = self.deformer(ref_batch, style_batch, idx_batch)
            if deform_scale != 1.0:
                transformed = ref_batch + (transformed - ref_batch) * deform_scale
            deformed = transformed.cpu().numpy()
        else:
            offsets = smooth_offsets(self.deformer(ref_batch, style_batch, idx_batch))
            if temperature > 0:
                # クランプ前に足すので、合算後も ±OFFSET_CLAMP に収まる
                noise = temperature_noise(
                    offsets.shape[0], offsets.shape[1], temperature * TEMP_NOISE_AMP, rng
                )
                offsets = offsets + torch.from_numpy(noise).to(offsets.device)
            offsets = offsets.clamp(-OFFSET_CLAMP, OFFSET_CLAMP) * deform_scale
            deformed = (ref_batch + offsets).cpu().numpy()

        strokes: list[NDArray[np.float32]] = []
        ns = STROKE_NOISE_SCALE
        for stroke in deformed:
            center = stroke.mean(axis=0)
            rotated = (stroke - center) @ rotation_matrix(rng.normal(0, ns * 0.05))
            sx = 1.0 + rng.normal(0, ns * 0.03)
            sy = 1.0 + rng.normal(0, ns * 0.03)
            shift = rng.normal(0, ns * 0.1, size=2)
            varied = rotated * np.array([sx, sy]) + center + shift
            strokes.append(upsample_stroke(varied.astype(np.float32)))
        return strokes
