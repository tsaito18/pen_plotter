"""訓練データの読み込みと前処理（ユーザー筆跡 × KanjiVG 参照のストローク対）。

ユーザー筆跡（iPad, Y-DOWN）は Y 軸を反転して KanjiVG（Y-UP）の座標系・スケールへ
揃えてから、参照ストロークとの対応（:class:`StrokeAligner`）を取る。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.geometry import resample_stroke

logger = logging.getLogger(__name__)

# 推論時に StyleEncoder へ渡す点数の上限（長すぎる系列は LSTM が遅い）
MAX_STYLE_POINTS = 4096

PointDict = dict[str, float]


def strokes_to_deltas(strokes: list[list[PointDict]]) -> torch.Tensor:
    """ストローク列を ``(N, 3)`` の ``[dx, dy, pen_up]`` 系列へ変換する（StyleEncoder 入力）。

    ``pen_up=1`` は各ストロークの最終点（その後ペンを上げる）。先頭点の差分は 0。
    """
    points = [(pt["x"], pt["y"]) for stroke in strokes for pt in stroke]
    pen = [1.0 if j == len(stroke) - 1 else 0.0 for stroke in strokes for j in range(len(stroke))]
    result = torch.zeros(len(points), 3, dtype=torch.float32)
    if points:
        xy = torch.tensor(points, dtype=torch.float64)
        result[1:, :2] = (xy[1:] - xy[:-1]).float()
        result[:, 2] = torch.tensor(pen, dtype=torch.float32)
    return result


def compute_normalization_stats(tensors: list[torch.Tensor]) -> dict[str, float]:
    """``(N, 3)`` 差分系列群の dx/dy の平均・標準偏差。"""
    all_dx = torch.cat([t[:, 0] for t in tensors])
    all_dy = torch.cat([t[:, 1] for t in tensors])
    std_x = all_dx.std().item() if len(all_dx) > 1 else 0.0
    std_y = all_dy.std().item() if len(all_dy) > 1 else 0.0
    return {
        "mean_x": all_dx.mean().item(),
        "mean_y": all_dy.mean().item(),
        "std_x": max(std_x, 1e-6),
        "std_y": max(std_y, 1e-6),
    }


def normalize_deltas(tensor: torch.Tensor, stats: dict[str, float]) -> torch.Tensor:
    """dx/dy 列を正規化する（pen 列はそのまま）。``(N, 3)`` / ``(B, N, 3)`` どちらも可。"""
    result = tensor.clone()
    result[..., 0] = (tensor[..., 0] - stats["mean_x"]) / stats["std_x"]
    result[..., 1] = (tensor[..., 1] - stats["mean_y"]) / stats["std_y"]
    return result


def limit_style_points(style: torch.Tensor, max_points: int = MAX_STYLE_POINTS) -> torch.Tensor:
    """``(1, N, 3)`` の系列を等間隔に間引いて ``max_points`` 点以内にする。"""
    if style.ndim != 3 or style.shape[1] <= max_points:
        return style.contiguous()
    idx = torch.linspace(0, style.shape[1] - 1, max_points, device=style.device).round().long()
    return style.index_select(1, idx).contiguous()


def load_style_sample(char_dir_root: Path | str | None) -> torch.Tensor:
    """ユーザー筆跡の全サンプルを 1 本の差分系列 ``(1, N, 3)`` にまとめる（推論のスタイル）。

    サンプルが無ければ ``(1, 10, 3)`` のゼロ系列。
    """
    root = Path(char_dir_root) if char_dir_root is not None else None
    all_strokes: list[list[PointDict]] = []
    if root is not None and root.is_dir():
        for char_dir in sorted(root.iterdir()):
            if not char_dir.is_dir():
                continue
            for jf in sorted(char_dir.glob("*.json")):
                try:
                    all_strokes.extend(json.loads(jf.read_text(encoding="utf-8"))["strokes"])
                except (json.JSONDecodeError, KeyError):
                    continue
    if not all_strokes:
        return torch.zeros(1, 10, 3)
    deltas = strokes_to_deltas(all_strokes)
    logger.info("Loaded style sample (%d points)", deltas.shape[0])
    return deltas.unsqueeze(0)


def scan_char_pairs(user_dirs: list[Path], ref_dir: Path) -> list[tuple[str, Path, Path]]:
    """ユーザー筆跡と KanjiVG の両方にある文字の ``(文字, ユーザーJSON, 参照JSON)``。"""
    ref_chars: dict[str, Path] = {}
    if Path(ref_dir).is_dir():
        for char_dir in Path(ref_dir).iterdir():
            if char_dir.is_dir() and (ref_files := list(char_dir.glob("*.json"))):
                ref_chars[char_dir.name] = ref_files[0]

    pairs: list[tuple[str, Path, Path]] = []
    for user_dir in user_dirs:
        if not Path(user_dir).is_dir():
            continue
        for char_dir in sorted(Path(user_dir).iterdir()):
            if char_dir.is_dir() and char_dir.name in ref_chars:
                for f in sorted(char_dir.glob("*.json")):
                    pairs.append((char_dir.name, f, ref_chars[char_dir.name]))
    return pairs


def _to_arrays(strokes: list[list[PointDict]]) -> list[np.ndarray]:
    return [np.array([[pt["x"], pt["y"]] for pt in s], dtype=np.float32) for s in strokes]


def user_to_reference_frame(
    user_strokes: list[np.ndarray], ref_strokes: list[np.ndarray]
) -> list[np.ndarray]:
    """ユーザー筆跡を Y 反転し、KanjiVG 参照と同じ原点・スケール（長辺基準）へ写す。"""
    all_u = np.concatenate(user_strokes, axis=0).astype(np.float64)
    y_min, y_max = all_u[:, 1].min(), all_u[:, 1].max()
    flipped = []
    for s in user_strokes:
        f = s.copy()
        f[:, 1] = y_min + y_max - f[:, 1]
        flipped.append(f)
    all_u[:, 1] = y_min + y_max - all_u[:, 1]
    all_r = np.concatenate(ref_strokes, axis=0)
    u_min = all_u.min(axis=0)
    u_range = (all_u.max(axis=0) - u_min).max()
    r_min = all_r.min(axis=0)
    r_range = (all_r.max(axis=0) - r_min).max()
    if u_range <= 0:
        return flipped
    scale = r_range / u_range
    return [(s - u_min) * scale + r_min for s in flipped]


class DeformationDataset(Dataset):
    """ユーザー筆跡 1 画と、それに対応する KanjiVG 参照 1 画の対のデータセット。

    Args:
        user_dirs: ユーザー筆跡の文字ディレクトリ群の親（複数プロファイル可）。
        ref_dir: KanjiVG 参照字形ディレクトリ。
        num_points: 各画の再サンプリング点数。
        augment: 文字単位で回転・拡縮・平行移動の水増しを掛けるか。
        use_aligner: 画の順序・向き・数の不一致を :class:`StrokeAligner` で対応付けるか。
            False なら同じ番号の画どうしを対にする。
    """

    def __init__(
        self,
        user_dirs: list[Path],
        ref_dir: Path,
        num_points: int = 32,
        augment: bool = False,
        use_aligner: bool = True,
    ) -> None:
        self.num_points = num_points
        self.augment = augment
        self._chars: list[tuple[str, list[np.ndarray], list[np.ndarray]]] = []
        # (char_idx, user_stroke_idx, ref_stroke_idx, reversed)
        self.samples: list[tuple[int, int, int, bool]] = []

        aligner = None
        if use_aligner:
            from src.model.aligner import StrokeAligner

            aligner = StrokeAligner(num_points=num_points)

        for char, user_path, ref_path in scan_char_pairs(user_dirs, ref_dir):
            user_raw = json.loads(user_path.read_text(encoding="utf-8"))["strokes"]
            ref_raw = json.loads(ref_path.read_text(encoding="utf-8"))["strokes"]
            char_idx = len(self._chars)
            ref = _to_arrays(ref_raw)
            user = user_to_reference_frame(_to_arrays(user_raw), ref)
            self._chars.append((char, user, ref))
            if aligner is None:
                for i in range(min(len(user), len(ref))):
                    if len(user[i]) >= 2 and len(ref[i]) >= 2:
                        self.samples.append((char_idx, i, i, False))
                continue
            u_valid = [i for i, s in enumerate(user) if len(s) >= 2]
            r_valid = [i for i, s in enumerate(ref) if len(s) >= 2]
            if not u_valid or not r_valid:
                continue
            result = aligner.align([user[i] for i in u_valid], [ref[i] for i in r_valid])
            for u, r, rev in zip(
                result.user_indices, result.ref_indices, result.reversed_flags, strict=True
            ):
                self.samples.append((char_idx, u_valid[u], r_valid[r], rev))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        char_idx, u_idx, r_idx, reversed_ = self.samples[idx]
        char, user, ref = self._chars[char_idx]
        user_stroke = user[u_idx][::-1] if reversed_ else user[u_idx]
        user_all = list(user)

        if self.augment:
            # 同じ字の全画に同じ変換を掛ける（字形のバランスを保つ）
            all_ref = np.concatenate(ref, axis=0)
            angle = np.random.uniform(-5, 5) * np.pi / 180
            scale = np.random.uniform(0.9, 1.1)
            center = all_ref.mean(axis=0)
            bbox_size = (all_ref.max(axis=0) - all_ref.min(axis=0)).max()
            shift = np.array(
                [
                    np.random.uniform(-0.05, 0.05) * bbox_size,
                    np.random.uniform(-0.05, 0.05) * bbox_size,
                ]
            )
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            user_stroke = (user_stroke - center) * scale @ rot + center + shift
            user_all = [(u - center) * scale @ rot + center + shift for u in user_all]

        style = strokes_to_deltas(
            [[{"x": float(p[0]), "y": float(p[1])} for p in pts] for pts in user_all]
        )
        return {
            "reference_points": torch.tensor(resample_stroke(ref[r_idx], self.num_points)),
            "target_points": torch.tensor(resample_stroke(user_stroke, self.num_points)),
            "stroke_index": r_idx,
            "style_strokes": style,
            "character": char,
        }


def augment_style_strokes(style: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """対照学習用の別ビュー: 差分系列 ``(N, 3)`` に微小ジッターと回転を掛ける。"""
    result = style.clone()
    n = result.shape[0]
    if n == 0:
        return result
    bbox = result[:, :2].abs().max().item() + 1e-6
    result[:, :2] += torch.from_numpy(rng.normal(0, 0.02 * bbox, size=(n, 2)).astype(np.float32))
    angle = rng.normal(0, np.radians(5))
    c, s = np.cos(angle), np.sin(angle)
    dx = result[:, 0].clone()
    dy = result[:, 1].clone()
    result[:, 0] = dx * c - dy * s
    result[:, 1] = dx * s + dy * c
    return result


def collate_deformation(batch: list[dict]) -> dict:
    """バッチ化。対照学習用にスタイルの水増しビューを加えてバッチを 2 倍にする。

    元ビューと水増しビューは同じ文字ラベルを持つ（SupCon の正例ペア）。
    """
    rng = np.random.default_rng()
    styles = [item["style_strokes"] for item in batch]
    all_styles = styles + [augment_style_strokes(s, rng) for s in styles]
    chars = [item["character"] for item in batch]
    char_to_idx = {c: i for i, c in enumerate(sorted(set(chars)))}
    return {
        "reference_points": torch.stack([item["reference_points"] for item in batch]).repeat(
            2, 1, 1
        ),
        "target_points": torch.stack([item["target_points"] for item in batch]).repeat(2, 1, 1),
        "stroke_indices": torch.tensor([item["stroke_index"] for item in batch]).repeat(2),
        "style_strokes": pad_sequence(all_styles, batch_first=True, padding_value=0.0),
        "style_lengths": torch.tensor([s.shape[0] for s in all_styles]),
        "characters": chars + chars,
        "character_labels": torch.tensor([char_to_idx[c] for c in chars + chars], dtype=torch.long),
    }
