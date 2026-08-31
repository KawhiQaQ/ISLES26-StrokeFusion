#!/usr/bin/env python3
"""Build the frozen V1 center-held-out, constrained-balanced five folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
import pandas as pd


IMAGE_SUFFIX = "_space-orig_desc-brain_T1w.nii.gz"
MASK_SUFFIX = "_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"
META_SUFFIX = "_metadata.csv"
N_FOLDS = 5
V1_SEED = 20260811


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--local-steps", type=int, default=5000)
    return parser.parse_args()


def read_metadata(path: Path, case_id: str) -> tuple[dict[str, str], bool]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) == 0:
        center = path.parents[3].name
        return (
            {
                "SESSION_ID": case_id,
                "ATLAS2_DATASET": "ATLAS3" if center == "SOOP" else "<missing>",
                "DAYS_POST_STROKE": "",
                "CHRONICITY": "",
                "SITE": center,
            },
            False,
        )
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one metadata row")
    return ({key: (value or "").strip() for key, value in rows[0].items()}, True)


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def scan_cases(raw_root: Path, manifest_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    images = sorted(raw_root.rglob(f"*{IMAGE_SUFFIX}"))
    if len(images) != 1453:
        raise RuntimeError(f"expected 1453 images, found {len(images)}")

    for index, image_path in enumerate(images, start=1):
        case_id = image_path.name[: -len(IMAGE_SUFFIX)]
        mask_path = image_path.with_name(f"{case_id}{MASK_SUFFIX}")
        metadata_path = image_path.with_name(f"{case_id}{META_SUFFIX}")
        metadata, metadata_complete = read_metadata(metadata_path, case_id)

        image = nib.load(image_path)
        mask_image = nib.load(mask_path)
        mask = np.asarray(mask_image.dataobj) > 0.5
        lesion_voxels = int(mask.sum())
        voxel_volume_ml = float(np.prod(image.header.get_zooms()[:3]) / 1000.0)
        if lesion_voxels:
            _, lesion_count = cc3d.connected_components(
                np.ascontiguousarray(mask), connectivity=26, return_N=True
            )
        else:
            lesion_count = 0
        spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
        anisotropy_ratio = max(spacing) / min(spacing)

        records.append(
            {
                "case_id": case_id,
                "center": metadata.get("SITE") or "<missing>",
                "source": metadata.get("ATLAS2_DATASET") or "<missing>",
                "days_post_stroke": as_float(metadata.get("DAYS_POST_STROKE", "")),
                "chronicity": metadata.get("CHRONICITY") or "<missing>",
                "metadata_complete": metadata_complete,
                "image_path": os.path.relpath(image_path.resolve(), manifest_dir),
                "mask_path": os.path.relpath(mask_path.resolve(), manifest_dir),
                "lesion_voxels": lesion_voxels,
                "lesion_volume_ml": lesion_voxels * voxel_volume_ml,
                "lesion_count": int(lesion_count),
                "orientation": "".join(nib.aff2axcodes(image.affine)),
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "anisotropy": (
                    "isotropic"
                    if anisotropy_ratio < 1.25
                    else "mild"
                    if anisotropy_ratio < 2.0
                    else "high"
                ),
                "shape": "x".join(str(x) for x in image.shape),
            }
        )
        if index % 200 == 0:
            print(f"scanned {index}/{len(images)} cases", flush=True)
    return pd.DataFrame.from_records(records)


def add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["lesion_status"] = np.where(frame.lesion_voxels > 0, "lesion", "empty")
    frame["volume_bin"] = "empty"
    nonempty = frame.lesion_voxels > 0
    frame.loc[nonempty, "volume_bin"] = pd.qcut(
        np.log1p(frame.loc[nonempty, "lesion_volume_ml"]),
        q=5,
        labels=["q1_tiny", "q2_small", "q3", "q4", "q5_large"],
        duplicates="drop",
    ).astype(str)
    frame["count_bin"] = pd.cut(
        frame.lesion_count,
        bins=[-1, 0, 1, 3, np.inf],
        labels=["0", "1", "2-3", "4+"],
    ).astype(str)
    frame["days_bin"] = pd.cut(
        frame.days_post_stroke,
        bins=[-np.inf, 7, 30, 179, np.inf],
        labels=["acute_le7", "early_8_30", "late_31_179", "chronic_ge180"],
    ).astype(object)
    frame.loc[frame.days_post_stroke.isna(), "days_bin"] = "missing"
    frame["days_bin"] = frame.days_bin.astype(str)
    return frame


@dataclass
class AssignmentState:
    assignment: np.ndarray
    fold_features: np.ndarray
    fold_sizes: np.ndarray
    fold_group_counts: np.ndarray


def objective(
    state: AssignmentState,
    target_features: np.ndarray,
    feature_weights: np.ndarray,
    target_size: float,
    target_groups: float,
) -> float:
    scale = np.maximum(np.sqrt(target_features), 1.0)
    feature_error = ((state.fold_features - target_features) / scale) ** 2
    feature_term = float(np.mean(feature_error * feature_weights[None, :]))
    size_term = float(np.mean(((state.fold_sizes - target_size) / target_size) ** 2))
    group_term = float(
        np.mean(((state.fold_group_counts - target_groups) / target_groups) ** 2)
    )
    return feature_term + 30.0 * size_term + 0.5 * group_term


def make_state(assignment: np.ndarray, group_features: np.ndarray) -> AssignmentState:
    fold_features = np.zeros((N_FOLDS, group_features.shape[1]), dtype=float)
    fold_group_counts = np.zeros(N_FOLDS, dtype=int)
    for group_index, fold in enumerate(assignment):
        fold_features[fold] += group_features[group_index]
        fold_group_counts[fold] += 1
    return AssignmentState(
        assignment=assignment.copy(),
        fold_features=fold_features,
        fold_sizes=fold_features[:, 0].copy(),
        fold_group_counts=fold_group_counts,
    )


def optimize_groups(
    group_features: np.ndarray,
    feature_weights: np.ndarray,
    restarts: int,
    local_steps: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(V1_SEED)
    target_features = group_features.sum(axis=0) / N_FOLDS
    target_size = float(group_features[:, 0].sum() / N_FOLDS)
    target_groups = float(len(group_features) / N_FOLDS)
    order_base = np.argsort(-group_features[:, 0])
    best_assignment: np.ndarray | None = None
    best_score = math.inf

    for _ in range(restarts):
        assignment = np.full(len(group_features), -1, dtype=int)
        fold_features = np.zeros((N_FOLDS, group_features.shape[1]), dtype=float)
        fold_group_counts = np.zeros(N_FOLDS, dtype=int)
        order = order_base.copy()
        if len(order) > N_FOLDS:
            tail = order[N_FOLDS:].copy()
            rng.shuffle(tail)
            tail = sorted(tail, key=lambda idx: -group_features[idx, 0])
            order = np.concatenate([order[:N_FOLDS], np.asarray(tail)])
        first_folds = rng.permutation(N_FOLDS)

        for position, group_index in enumerate(order):
            candidates = [int(first_folds[position])] if position < N_FOLDS else list(range(N_FOLDS))
            candidate_scores: list[tuple[float, int]] = []
            for fold in candidates:
                fold_features[fold] += group_features[group_index]
                fold_group_counts[fold] += 1
                state = AssignmentState(
                    assignment,
                    fold_features,
                    fold_features[:, 0],
                    fold_group_counts,
                )
                score = objective(
                    state,
                    target_features,
                    feature_weights,
                    target_size,
                    target_groups,
                )
                candidate_scores.append((score + float(rng.random()) * 1e-9, fold))
                fold_features[fold] -= group_features[group_index]
                fold_group_counts[fold] -= 1
            chosen = min(candidate_scores)[1]
            assignment[group_index] = chosen
            fold_features[chosen] += group_features[group_index]
            fold_group_counts[chosen] += 1

        state = make_state(assignment, group_features)
        score = objective(
            state, target_features, feature_weights, target_size, target_groups
        )
        for _ in range(local_steps):
            if rng.random() < 0.55:
                group_index = int(rng.integers(len(group_features)))
                old_fold = int(state.assignment[group_index])
                new_fold = int(rng.integers(N_FOLDS - 1))
                if new_fold >= old_fold:
                    new_fold += 1
                if state.fold_group_counts[old_fold] <= 1:
                    continue
                state.assignment[group_index] = new_fold
                state.fold_features[old_fold] -= group_features[group_index]
                state.fold_features[new_fold] += group_features[group_index]
                state.fold_sizes = state.fold_features[:, 0]
                state.fold_group_counts[old_fold] -= 1
                state.fold_group_counts[new_fold] += 1
                proposed = objective(
                    state,
                    target_features,
                    feature_weights,
                    target_size,
                    target_groups,
                )
                if proposed < score:
                    score = proposed
                else:
                    state.assignment[group_index] = old_fold
                    state.fold_features[old_fold] += group_features[group_index]
                    state.fold_features[new_fold] -= group_features[group_index]
                    state.fold_sizes = state.fold_features[:, 0]
                    state.fold_group_counts[old_fold] += 1
                    state.fold_group_counts[new_fold] -= 1
            else:
                left, right = rng.choice(len(group_features), size=2, replace=False)
                left_fold = int(state.assignment[left])
                right_fold = int(state.assignment[right])
                if left_fold == right_fold:
                    continue
                state.assignment[left], state.assignment[right] = right_fold, left_fold
                state.fold_features[left_fold] += group_features[right] - group_features[left]
                state.fold_features[right_fold] += group_features[left] - group_features[right]
                state.fold_sizes = state.fold_features[:, 0]
                proposed = objective(
                    state,
                    target_features,
                    feature_weights,
                    target_size,
                    target_groups,
                )
                if proposed < score:
                    score = proposed
                else:
                    state.assignment[left], state.assignment[right] = left_fold, right_fold
                    state.fold_features[left_fold] += group_features[left] - group_features[right]
                    state.fold_features[right_fold] += group_features[right] - group_features[left]
                    state.fold_sizes = state.fold_features[:, 0]

        if score < best_score:
            best_score = score
            best_assignment = state.assignment.copy()

    if best_assignment is None:
        raise RuntimeError("fold optimization failed")
    return best_assignment, best_score


def build_folds(frame: pd.DataFrame, restarts: int, local_steps: int) -> tuple[pd.DataFrame, float]:
    categorical = [
        "source",
        "lesion_status",
        "volume_bin",
        "count_bin",
        "days_bin",
        "chronicity",
        "orientation",
        "anisotropy",
    ]
    encoded = pd.get_dummies(frame[categorical], prefix=categorical, dtype=float)
    encoded.insert(0, "case_count", 1.0)
    grouped = encoded.groupby(frame.center, sort=True).sum()
    centers = grouped.index.to_numpy()
    feature_names = grouped.columns.tolist()
    feature_weights = np.ones(len(feature_names), dtype=float)
    for index, name in enumerate(feature_names):
        if name == "case_count":
            feature_weights[index] = 4.0
        elif name.startswith(("source_", "volume_bin_", "lesion_status_")):
            feature_weights[index] = 2.0
        elif name.startswith("count_bin_"):
            feature_weights[index] = 1.5
        elif name.startswith(("orientation_", "anisotropy_")):
            feature_weights[index] = 0.4
        elif name.startswith("chronicity_"):
            feature_weights[index] = 0.7

    assignment, score = optimize_groups(
        grouped.to_numpy(dtype=float), feature_weights, restarts, local_steps
    )
    center_to_fold = dict(zip(centers, assignment, strict=True))
    frame = frame.copy()
    frame["fold"] = frame.center.map(center_to_fold).astype(int)
    return frame.sort_values("case_id").reset_index(drop=True), score


def validate_and_export(frame: pd.DataFrame, output_dir: Path, objective_score: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(frame) != 1453 or frame.case_id.nunique() != 1453:
        raise AssertionError("case IDs are not one-to-one")
    if frame.groupby("center").fold.nunique().max() != 1:
        raise AssertionError("center leakage detected")
    if sorted(frame.fold.unique().tolist()) != list(range(N_FOLDS)):
        raise AssertionError("missing fold")

    all_cases = set(frame.case_id)
    splits: list[dict[str, list[str]]] = []
    for fold in range(N_FOLDS):
        validation = set(frame.loc[frame.fold == fold, "case_id"])
        training = all_cases - validation
        if training & validation or training | validation != all_cases:
            raise AssertionError(f"fold {fold}: train/validation leakage")
        splits.append({"train": sorted(training), "val": sorted(validation)})

    frame.to_csv(output_dir / "manifest_v1.csv", index=False)
    (output_dir / "splits_final.json").write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8"
    )

    summary: dict[str, object] = {
        "version": "V1",
        "seed": V1_SEED,
        "group": "CENTER/SITE",
        "objective_score": objective_score,
        "center_leakage": False,
        "folds": {},
    }
    for fold, fold_frame in frame.groupby("fold", sort=True):
        summary["folds"][str(fold)] = {
            "cases": int(len(fold_frame)),
            "centers": int(fold_frame.center.nunique()),
            "center_names": sorted(fold_frame.center.unique().tolist()),
            "empty_masks": int((fold_frame.lesion_voxels == 0).sum()),
            "lesion_volume_ml_median": float(fold_frame.lesion_volume_ml.median()),
            "lesion_volume_ml_mean": float(fold_frame.lesion_volume_ml.mean()),
            "sources": {
                str(key): int(value)
                for key, value in fold_frame.source.value_counts().items()
            },
            "days_bins": {
                str(key): int(value)
                for key, value in fold_frame.days_bin.value_counts().items()
            },
        }
    (output_dir / "fold_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    frame = add_strata(scan_cases(args.raw_root.resolve(), output_dir))
    frame, score = build_folds(frame, args.restarts, args.local_steps)
    validate_and_export(frame, output_dir, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
