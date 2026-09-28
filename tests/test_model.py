"""ML: 変形器・StyleEncoder・データ前処理・ストローク対応付け・訓練・推論。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.model.aligner import StrokeAligner
from src.model.data import (
    DeformationDataset,
    collate_deformation,
    limit_style_points,
    load_style_sample,
    strokes_to_deltas,
    user_to_reference_frame,
)
from src.model.deformers import (
    DEFORMER_TYPES,
    OFFSET_CLAMP,
    build_deformer,
    compute_local_curvature,
    postprocess_offsets,
    smooth_offsets,
)
from src.model.inference import StrokeInference, temperature_noise, upsample_stroke
from src.model.style_encoder import StyleEncoder, supervised_contrastive_loss
from src.model.training import (
    DeformationFinetuner,
    FinetuneConfig,
    UserDeformationTrainer,
    UserTrainConfig,
)

SMALL = {"style_dim": 16, "hidden_dim": 16, "d_model": 16, "nhead": 2, "ff_dim": 16}


# --- 変形器 ---


@pytest.mark.parametrize("kind", DEFORMER_TYPES)
def test_all_deformers_share_the_same_interface(kind: str):
    model = build_deformer({"deformer_type": kind, **SMALL})
    ref = torch.rand(3, 32, 2)
    out = model(ref, torch.randn(3, 16), torch.tensor([0, 1, 99]))  # 画番号は上限でクランプ
    if kind == "affine":
        out = out[0]
    assert out.shape == (3, 32, 2)
    out.sum().backward()


@pytest.mark.parametrize("kind", ["affine", "transformer", "twostage"])
def test_zero_initialized_deformers_start_as_identity(kind: str):
    model = build_deformer({"deformer_type": kind, **SMALL})
    ref = torch.rand(2, 32, 2)
    out = model(ref, torch.randn(2, 16))
    deformed = out[0] if kind == "affine" else ref + out
    assert torch.allclose(deformed, ref, atol=1e-6)


def test_unknown_deformer_type_is_rejected():
    with pytest.raises(ValueError, match="unknown deformer_type"):
        build_deformer({"deformer_type": "lstm"})


def test_curvature_is_high_at_corners():
    line = torch.tensor([[[0.0, 0], [1, 0], [2, 0]]])
    corner = torch.tensor([[[0.0, 0], [1, 0], [1, 1]]])
    assert compute_local_curvature(line)[0, 1, 0] < compute_local_curvature(corner)[0, 1, 0]


def test_offset_postprocessing_smooths_and_clamps():
    noisy = torch.randn(1, 32, 2) * 5
    out = postprocess_offsets(noisy)
    assert out.abs().max() <= OFFSET_CLAMP
    assert smooth_offsets(noisy).diff(dim=1).abs().mean() < noisy.diff(dim=1).abs().mean()


# --- StyleEncoder / 対照学習 ---


def test_style_encoder_respects_sequence_lengths():
    enc = StyleEncoder(style_dim=16)
    x = torch.randn(2, 20, 3)
    padded = x.clone()
    padded[1, 10:] = 0
    style = enc(padded, lengths=torch.tensor([20, 10]))
    assert style.shape == (2, 16)
    assert torch.allclose(style[1], enc(x[1:2, :10])[0], atol=1e-5)


def test_projection_head_and_supcon_loss():
    enc = StyleEncoder(style_dim=16)
    enc.enable_projection_head(output_dim=8)
    _style, z = enc(torch.randn(4, 10, 3), return_projection=True)
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)
    labels = torch.tensor([0, 0, 1, 1])
    aligned = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1, 0], [0, 1], [0, 1]]), dim=1)
    mixed = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [0, 1], [1, 0], [0, 1]]), dim=1)
    assert supervised_contrastive_loss(aligned, labels) < supervised_contrastive_loss(mixed, labels)
    assert supervised_contrastive_loss(aligned, torch.arange(4)) == 0  # 正例なし


# --- 前処理 ---


def test_strokes_to_deltas_marks_pen_up_at_stroke_ends():
    deltas = strokes_to_deltas(
        [[{"x": 0, "y": 0}, {"x": 1, "y": 2}], [{"x": 5, "y": 5}, {"x": 5, "y": 6}]]
    )
    assert deltas.tolist() == [[0, 0, 0], [1, 2, 1], [4, 3, 0], [0, 1, 1]]


def test_user_strokes_are_flipped_and_scaled_to_reference_frame():
    user = [np.array([[0, 0], [0, 100]], dtype=np.float32)]  # Y-DOWN: 上から下へ
    ref = [np.array([[1, 9], [1, 1]], dtype=np.float32)]  # Y-UP: 上から下へ
    (out,) = user_to_reference_frame(user, ref)
    assert np.allclose(out, [[1, 9], [1, 1]])


def test_style_sample_and_point_limit(user_strokes_root: Path):
    style = load_style_sample(user_strokes_root / "taro")
    assert style.ndim == 3 and style.shape[0] == 1 and style.shape[1] > 10
    assert load_style_sample(None).shape == (1, 10, 3)
    assert limit_style_points(torch.zeros(1, 9000, 3), 4096).shape == (1, 4096, 3)


def test_dataset_pairs_user_and_reference_strokes(user_strokes_root: Path, kanjivg_dir: Path):
    ds = DeformationDataset([user_strokes_root / "taro"], kanjivg_dir, use_aligner=True)
    assert len(ds) == 4  # 「十」2 サンプル × 2 画（"a" は参照に無い）
    batch = collate_deformation([ds[0], ds[1]])
    assert batch["reference_points"].shape == (4, 32, 2)  # 対照学習用に 2 倍
    assert batch["character_labels"].tolist() == [0, 0, 0, 0]


# --- 対応付け ---


def _h() -> np.ndarray:
    return np.array([[i, 5.0] for i in range(11)], dtype=np.float32)


def _v() -> np.ndarray:
    return np.array([[5.0, i] for i in range(11)], dtype=np.float32)


def test_aligner_handles_order_direction_and_count_mismatch():
    aligner = StrokeAligner()
    reordered = aligner.align([_v(), _h()[::-1].copy()], [_h(), _v()])
    pairs = dict(zip(reordered.user_indices, reordered.ref_indices))
    assert pairs == {0: 1, 1: 0}
    assert reordered.reversed_flags[reordered.user_indices.index(1)]

    h = _h()
    v = np.array([[10.0, 5.0 + i] for i in range(11)], dtype=np.float32)
    merged = aligner.align([np.concatenate([h, v[1:]])], [h, v])  # 2 画を続けて書いた
    assert sorted(merged.ref_indices) == [0, 1]
    split = aligner.align([h[:6].copy(), h[5:].copy(), _v()], [h, _v()])  # 1 画を分けて書いた
    assert sorted(split.ref_indices) == [0, 1]

    far = aligner.align([_h() + 100], [_h()])
    assert far.ref_indices == [] and far.rejected_indices == [0]


# --- 訓練 ---


@pytest.mark.slow
def test_pretrain_then_finetune_twostage(user_strokes_root: Path, kanjivg_dir: Path, tmp_path):
    """本番構成（twostage）で pretrain → finetune → 推論まで通る。"""
    config = UserTrainConfig(epochs=2, batch_size=4, deformer_type="twostage", **SMALL)
    trainer = UserDeformationTrainer(config, [user_strokes_root / "taro"], kanjivg_dir, tmp_path)
    epochs: list[int] = []
    trainer.on_epoch_end = lambda epoch, loss: epochs.append(epoch)
    losses = trainer.train()
    assert len(losses) == 2 and epochs == [0, 1] and all(np.isfinite(losses))

    finetuner = DeformationFinetuner(
        FinetuneConfig(epochs=1, batch_size=4),
        trainer.checkpoint_path,
        [user_strokes_root / "taro"],
        kanjivg_dir,
        tmp_path,
    )
    finetuner.train()
    frozen = {k: v for k, v in finetuner.deformer.state_dict().items()}
    saved = torch.load(finetuner.checkpoint_path, weights_only=False)
    assert all(torch.equal(frozen[k], saved["deformer_state_dict"][k]) for k in frozen)
    assert StrokeInference(finetuner.checkpoint_path, torch.zeros(1, 10, 3)).deformer_type == (
        "twostage"
    )


def test_affine_cannot_be_pretrained(tmp_path):
    with pytest.raises(ValueError, match="affine"):
        UserDeformationTrainer(UserTrainConfig(deformer_type="affine"), [], tmp_path, tmp_path)


# --- 推論 ---


def test_inference_deforms_reference_strokes(tiny_checkpoint: Path):
    engine = StrokeInference(tiny_checkpoint, torch.zeros(1, 10, 3))
    ref = [np.array([[0, 5], [10, 5]], dtype=float), np.array([[1, 1]]), np.array([[5, 0], [5, 9]])]
    np.random.seed(0)
    out = engine.generate(ref, temperature=0.0)
    assert len(out) == 2  # 2 点未満の画は除外
    assert all(s.dtype == np.float32 and s.shape[1] == 2 for s in out)
    assert np.allclose(out[0][[0, -1]], [[0, 5], [10, 5]], atol=0.5)
    with pytest.raises(ValueError):
        engine.generate([np.array([[0.0, 0.0]])])


def test_inference_requires_a_style(tiny_checkpoint: Path):
    with pytest.raises(RuntimeError, match="style"):
        StrokeInference(tiny_checkpoint).generate([np.array([[0.0, 0], [1, 1]])])


def test_temperature_noise_is_smooth_and_scaled():
    np.random.seed(0)
    noise = temperature_noise(4, 32, amp=0.1)
    assert noise.shape == (4, 32, 2)
    assert np.abs(np.diff(noise, axis=1)).max() < 0.1
    assert not temperature_noise(2, 8, amp=0.0).any()


def test_upsample_keeps_corners_and_endpoints():
    corner = np.array([[0, 0], [5, 0], [5, 5]], dtype=np.float32)
    out = upsample_stroke(corner)
    assert len(out) > 3
    assert np.allclose(out[[0, -1]], corner[[0, -1]])
    assert np.min(np.linalg.norm(out - corner[1], axis=1)) < 1e-5
