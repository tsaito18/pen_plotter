"""Stroke alignment via Modified Hausdorff Distance + Hungarian assignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.model.data_utils import resample_stroke


@dataclass
class AlignmentResult:
    user_indices: list[int]
    ref_indices: list[int]
    cost_matrix: np.ndarray
    per_stroke_cost: list[float]
    total_cost: float
    reversed_flags: list[bool]
    rejected_indices: list[int] = field(default_factory=list)


class StrokeAligner:
    def __init__(self, num_points: int = 32, quality_threshold: float = 2.0):
        self.num_points = num_points
        self.quality_threshold = quality_threshold

    def _compute_mhd(self, a: np.ndarray, b: np.ndarray) -> float:
        """Modified Hausdorff Distance (ordered point-to-point)."""
        dists = np.sqrt(((a - b) ** 2).sum(axis=1))
        return float(dists.mean())

    def _min_mhd(self, a: np.ndarray, b: np.ndarray) -> float:
        """MHD considering both forward and reversed directions."""
        return min(self._compute_mhd(a, b), self._compute_mhd(a[::-1], b))

    def _build_cost_matrix(
        self,
        user_strokes: list[np.ndarray],
        ref_strokes: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build MHD cost matrix and reversed-flags matrix."""
        n_user = len(user_strokes)
        n_ref = len(ref_strokes)
        cost = np.zeros((n_user, n_ref))
        reversed_flags = np.zeros((n_user, n_ref), dtype=bool)

        for i, u in enumerate(user_strokes):
            for j, r in enumerate(ref_strokes):
                fwd = self._compute_mhd(u, r)
                rev = self._compute_mhd(u[::-1], r)
                if rev < fwd:
                    cost[i, j] = rev
                    reversed_flags[i, j] = True
                else:
                    cost[i, j] = fwd

        return cost, reversed_flags

    def _hungarian_assign(self, cost_matrix: np.ndarray) -> tuple[list[int], list[int], float]:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = float(cost_matrix[row_ind, col_ind].sum())
        return row_ind.tolist(), col_ind.tolist(), total

    def _find_split_points(
        self,
        stroke: np.ndarray,
        pressure: np.ndarray | None,
        timestamps: np.ndarray | None,
    ) -> list[int]:
        """Find candidate split points from speed dips and curvature peaks."""
        candidates: list[int] = []
        n = len(stroke)

        if timestamps is not None and len(timestamps) >= 3:
            dists = np.sqrt(np.sum(np.diff(stroke, axis=0) ** 2, axis=1))
            dt = np.maximum(np.diff(timestamps), 1e-10)
            speeds = dists / dt
            if len(speeds) > 0:
                median_speed = np.median(speeds)
                if median_speed > 0:
                    for i in range(len(speeds)):
                        if speeds[i] < median_speed * 0.3:
                            candidates.append(i + 1)

        if n >= 3:
            d1 = stroke[1:-1] - stroke[:-2]
            d2 = stroke[2:] - stroke[1:-1]
            cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
            len1 = np.sqrt((d1**2).sum(axis=1))
            len2 = np.sqrt((d2**2).sum(axis=1))
            denom = np.maximum(len1 * len2, 1e-10)
            curvature = cross / denom
            for i in range(len(curvature)):
                if curvature[i] > 0.5:
                    candidates.append(i + 1)

        min_idx = max(2, n // 5)
        max_idx = min(n - 3, n - n // 5)
        return sorted(set(c for c in candidates if min_idx <= c <= max_idx))

    def _detect_merges(
        self,
        resampled_ref: list[np.ndarray],
        raw_user: list[np.ndarray],
        pressure: list[np.ndarray] | None,
        timestamps: list[np.ndarray] | None,
        initial: AlignmentResult,
    ) -> AlignmentResult:
        """Phase 2a: split user strokes to match unmatched ref strokes."""
        if len(raw_user) >= len(resampled_ref):
            return initial

        matched_refs = set(initial.ref_indices)
        unmatched_refs = [i for i in range(len(resampled_ref)) if i not in matched_refs]
        if not unmatched_refs:
            return initial

        candidates = []
        for pair_idx, (u_idx, r_idx) in enumerate(zip(initial.user_indices, initial.ref_indices)):
            raw = raw_user[u_idx]
            p = pressure[u_idx] if pressure is not None else None
            t = timestamps[u_idx] if timestamps is not None else None

            split_pts = self._find_split_points(raw, p, t)
            if not split_pts:
                split_pts = [len(raw) // 2]

            for sp in split_pts:
                p1_raw = raw[: sp + 1]
                p2_raw = raw[sp:]
                if len(p1_raw) < 2 or len(p2_raw) < 2:
                    continue

                p1_r = resample_stroke(p1_raw, self.num_points)
                p2_r = resample_stroke(p2_raw, self.num_points)

                for ur_idx in unmatched_refs:
                    c1a = self._min_mhd(p1_r, resampled_ref[r_idx])
                    c2a = self._min_mhd(p2_r, resampled_ref[ur_idx])
                    c1b = self._min_mhd(p1_r, resampled_ref[ur_idx])
                    c2b = self._min_mhd(p2_r, resampled_ref[r_idx])

                    if c1a + c2a <= c1b + c2b:
                        cost1, cost2, ridx1, ridx2 = c1a, c2a, r_idx, ur_idx
                    else:
                        cost1, cost2, ridx1, ridx2 = c1b, c2b, ur_idx, r_idx

                    if cost1 <= self.quality_threshold and cost2 <= self.quality_threshold:
                        candidates.append(
                            {
                                "pair_idx": pair_idx,
                                "u_idx": u_idx,
                                "ridx1": ridx1,
                                "ridx2": ridx2,
                                "cost1": cost1,
                                "cost2": cost2,
                                "total": cost1 + cost2,
                            }
                        )

        if not candidates:
            return initial

        best = min(candidates, key=lambda c: c["total"])
        idx = best["pair_idx"]

        new_user = list(initial.user_indices)
        new_ref = list(initial.ref_indices)
        new_costs = list(initial.per_stroke_cost)
        new_reversed = list(initial.reversed_flags)

        new_ref[idx] = best["ridx1"]
        new_costs[idx] = best["cost1"]

        new_user.append(best["u_idx"])
        new_ref.append(best["ridx2"])
        new_costs.append(best["cost2"])
        new_reversed.append(False)

        return AlignmentResult(
            user_indices=new_user,
            ref_indices=new_ref,
            cost_matrix=initial.cost_matrix,
            per_stroke_cost=new_costs,
            total_cost=sum(new_costs),
            reversed_flags=new_reversed,
            rejected_indices=[
                r for r in initial.rejected_indices if r not in (best["ridx1"], best["ridx2"])
            ],
        )

    def _detect_splits(
        self,
        resampled_ref: list[np.ndarray],
        raw_user: list[np.ndarray],
        initial: AlignmentResult,
    ) -> AlignmentResult:
        """Phase 2b: join user strokes to match ref strokes better."""
        if len(raw_user) <= len(resampled_ref):
            return initial

        matched_users = set(initial.user_indices)
        unmatched_users = [i for i in range(len(raw_user)) if i not in matched_users]
        if not unmatched_users:
            return initial

        best_join: dict | None = None
        best_cost = float("inf")

        for um_idx in unmatched_users:
            for pair_idx, (u_idx, r_idx) in enumerate(
                zip(initial.user_indices, initial.ref_indices)
            ):
                for first, second in [
                    (raw_user[u_idx], raw_user[um_idx]),
                    (raw_user[um_idx], raw_user[u_idx]),
                ]:
                    joined = np.concatenate([first, second])
                    joined_r = resample_stroke(joined, self.num_points)
                    cost = self._min_mhd(joined_r, resampled_ref[r_idx])

                    if (
                        cost < initial.per_stroke_cost[pair_idx]
                        and cost <= self.quality_threshold
                        and cost < best_cost
                    ):
                        best_cost = cost
                        best_join = {"pair_idx": pair_idx, "cost": cost}

        if best_join is None:
            return initial

        idx = best_join["pair_idx"]
        new_costs = list(initial.per_stroke_cost)
        new_costs[idx] = best_join["cost"]

        return AlignmentResult(
            user_indices=list(initial.user_indices),
            ref_indices=list(initial.ref_indices),
            cost_matrix=initial.cost_matrix,
            per_stroke_cost=new_costs,
            total_cost=sum(new_costs),
            reversed_flags=list(initial.reversed_flags),
            rejected_indices=initial.rejected_indices,
        )

    def align(
        self,
        user_strokes: list[np.ndarray],
        ref_strokes: list[np.ndarray],
        pressure: list[np.ndarray] | None = None,
        timestamps: list[np.ndarray] | None = None,
    ) -> AlignmentResult:
        resampled_user = [resample_stroke(s, self.num_points) for s in user_strokes]
        resampled_ref = [resample_stroke(s, self.num_points) for s in ref_strokes]

        # Phase 1: Hungarian assignment
        cost_matrix, rev_matrix = self._build_cost_matrix(resampled_user, resampled_ref)
        row_ids, col_ids, _ = self._hungarian_assign(cost_matrix)

        initial = AlignmentResult(
            user_indices=list(row_ids),
            ref_indices=list(col_ids),
            cost_matrix=cost_matrix,
            per_stroke_cost=[float(cost_matrix[r, c]) for r, c in zip(row_ids, col_ids)],
            total_cost=sum(float(cost_matrix[r, c]) for r, c in zip(row_ids, col_ids)),
            reversed_flags=[bool(rev_matrix[r, c]) for r, c in zip(row_ids, col_ids)],
            rejected_indices=[],
        )

        # Phase 2: Merge / Split detection
        result = self._detect_merges(resampled_ref, user_strokes, pressure, timestamps, initial)
        result = self._detect_splits(resampled_ref, user_strokes, result)

        # Phase 3: Quality filtering
        final_user: list[int] = []
        final_ref: list[int] = []
        final_costs: list[float] = []
        final_reversed: list[bool] = []
        rejected: list[int] = []

        for u, r, c, rev in zip(
            result.user_indices,
            result.ref_indices,
            result.per_stroke_cost,
            result.reversed_flags,
        ):
            if c > self.quality_threshold:
                rejected.append(r)
            else:
                final_user.append(u)
                final_ref.append(r)
                final_costs.append(c)
                final_reversed.append(rev)

        return AlignmentResult(
            user_indices=final_user,
            ref_indices=final_ref,
            cost_matrix=cost_matrix,
            per_stroke_cost=final_costs,
            total_cost=sum(final_costs),
            reversed_flags=final_reversed,
            rejected_indices=rejected,
        )
