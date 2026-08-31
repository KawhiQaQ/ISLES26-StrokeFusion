#!/usr/bin/env python3
"""Evaluate nnU-Net validation outputs with the published ISLES'26 metrics."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from panoptica import (
    ConnectedComponentsInstanceApproximator,
    InputType,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
)
from sklearn.metrics import auc, precision_recall_curve


METRICS = [
    "pr_auc",
    "dice",
    "abs_volume_diff_ml",
    "abs_lesion_count_diff",
    "lesion_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--label-dir",
        type=Path,
        help="Optional nnU-Net labelsTr directory containing grid-corrected labels.",
    )
    parser.add_argument(
        "--foreground-aggregation",
        choices=("single", "sum"),
        default="single",
        help=(
            "Use class-1 probability for a binary model, or sum all foreground "
            "probabilities for a multi-class labeling model such as MSL."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Evaluate cases in parallel without changing metric definitions.",
    )
    return parser.parse_args()


def compute_instance_metrics(
    ground_truth: np.ndarray, prediction: np.ndarray, empty_value: float = 1.0
) -> tuple[float, int, float]:
    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(matching_threshold=0.25),
    )
    result = evaluator.evaluate(prediction.astype(int), ground_truth.astype(int))[
        "ungrouped"
    ]
    count_difference = abs(result.n_ref_instances - result.n_pred_instances)
    if result.n_ref_instances == 0 and result.n_pred_instances == 0:
        return empty_value, count_difference, empty_value
    return float(result.rq), int(count_difference), float(result.global_bin_dsc)


def compute_pr_auc(
    ground_truth: np.ndarray, prediction_map: np.ndarray, empty_value: float = 1.0
) -> float:
    gt_flat = np.asarray(ground_truth).astype(np.bool_).ravel()
    pred_flat = np.asarray(prediction_map).astype(np.float32).ravel()
    if gt_flat.shape != pred_flat.shape:
        raise ValueError(f"soft-map shape mismatch: {gt_flat.shape} vs {pred_flat.shape}")
    if int(gt_flat.sum()) == 0:
        return empty_value if np.all(pred_flat == pred_flat[0]) else 0.0
    precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
    return float(auc(recall, precision))


def load_probabilities(path: Path) -> tuple[np.ndarray, str]:
    with np.load(path) as payload:
        if "foreground_probability" in payload:
            foreground = payload["foreground_probability"]
            if foreground.ndim != 3:
                raise ValueError(
                    f"{path}: unexpected compact foreground shape {foreground.shape}"
                )
            return foreground[None], "compact_foreground"
        if "probabilities" in payload:
            probabilities = payload["probabilities"]
            if probabilities.ndim != 4 or probabilities.shape[0] < 2:
                raise ValueError(
                    f"{path}: unexpected probabilities shape {probabilities.shape}"
                )
            return probabilities, "full_multiclass"
    raise KeyError(f"{path}: missing probability payload")


def restore_probability_orientation(
    probabilities: np.ndarray, properties_path: Path
) -> np.ndarray:
    """Restore nnU-Net probabilities from internal z-y-x/RAS to native NIfTI axes."""
    if not properties_path.is_file():
        raise FileNotFoundError(f"missing nnU-Net properties: {properties_path}")
    with properties_path.open("rb") as handle:
        properties = pickle.load(handle)
    nibabel_stuff = properties.get("nibabel_stuff")
    if not isinstance(nibabel_stuff, dict):
        raise KeyError(f"{properties_path}: missing nibabel_stuff")

    # nnU-Net stores exported probabilities as C, z, y, x on its reoriented grid.
    probabilities = probabilities.transpose(0, 3, 2, 1)
    source = io_orientation(np.asarray(nibabel_stuff["reoriented_affine"]))
    target = io_orientation(np.asarray(nibabel_stuff["original_affine"]))
    transform = ornt_transform(source, target)
    return np.stack(
        [apply_orientation(channel, transform) for channel in probabilities], axis=0
    )


def evaluate_case(
    row: pd.Series,
    prediction_dir: Path,
    label_dir: Path | None,
    foreground_aggregation: str,
) -> dict[str, object]:
    case_id = str(row.case_id)
    binary_path = prediction_dir / f"{case_id}.nii.gz"
    soft_path = prediction_dir / f"{case_id}.npz"
    properties_path = prediction_dir / f"{case_id}.pkl"
    if (
        not binary_path.is_file()
        or not soft_path.is_file()
        or not properties_path.is_file()
    ):
        raise FileNotFoundError(f"missing prediction for {case_id}")

    ground_truth_path = (
        label_dir / f"{case_id}.nii.gz" if label_dir is not None else Path(row.mask_path)
    )
    ground_truth_image = nib.load(ground_truth_path)
    prediction_image = nib.load(binary_path)
    ground_truth = np.asarray(ground_truth_image.dataobj) > 0.5
    exported_foreground = np.asarray(prediction_image.dataobj) > 0.5
    probability_payload, probability_format = load_probabilities(soft_path)
    probabilities = restore_probability_orientation(
        probability_payload, properties_path
    )
    if probability_format == "compact_foreground":
        soft_map = probabilities[0]
    elif foreground_aggregation == "single":
        if probabilities.shape[0] != 2:
            raise ValueError(
                f"{case_id}: binary aggregation requires 2 probability channels, "
                f"found {probabilities.shape[0]}"
            )
        soft_map = probabilities[1]
    else:
        soft_map = probabilities[1:].sum(axis=0)
    if (
        exported_foreground.shape != ground_truth.shape
        or soft_map.shape != ground_truth.shape
    ):
        raise ValueError(
            f"{case_id}: shapes gt={ground_truth.shape}, "
            f"exported={exported_foreground.shape}, "
            f"soft={soft_map.shape}"
        )
    if probability_format == "full_multiclass":
        argmax_foreground = np.argmax(probabilities, axis=0) > 0
        if not np.array_equal(argmax_foreground, exported_foreground):
            disagreements = int(
                np.count_nonzero(argmax_foreground != exported_foreground)
            )
            raise ValueError(
                f"{case_id}: restored probabilities disagree with binary prediction "
                f"at {disagreements} voxels"
            )
    prediction = (
        exported_foreground
        if foreground_aggregation == "single"
        else soft_map > 0.5
    )
    if not np.allclose(prediction_image.affine, ground_truth_image.affine, atol=1e-3):
        raise ValueError(f"{case_id}: prediction affine does not match ground truth")

    lesion_f1, count_difference, dice = compute_instance_metrics(
        ground_truth, prediction
    )
    voxel_size_ml = float(np.prod(ground_truth_image.header.get_zooms()[:3]) / 1000.0)
    volume_difference = abs(int(ground_truth.sum()) - int(prediction.sum())) * voxel_size_ml
    return {
        "case_id": case_id,
        "center": row.center,
        "fold": int(row.fold),
        "pr_auc": compute_pr_auc(ground_truth, soft_map),
        "dice": dice,
        "abs_volume_diff_ml": volume_difference,
        "abs_lesion_count_diff": count_difference,
        "lesion_f1": lesion_f1,
        "gt_volume_ml": float(ground_truth.sum()) * voxel_size_ml,
        "pred_volume_ml": float(prediction.sum()) * voxel_size_ml,
        "probability_format": probability_format,
    }


def evaluate_case_worker(
    payload: tuple[dict[str, object], str, str | None, str]
) -> dict[str, object]:
    """Pickle-friendly wrapper used by ProcessPoolExecutor."""
    row, prediction_dir, label_dir, foreground_aggregation = payload
    return evaluate_case(
        pd.Series(row),
        Path(prediction_dir),
        Path(label_dir) if label_dir is not None else None,
        foreground_aggregation,
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    frame = pd.read_csv(args.manifest)
    manifest_dir = args.manifest.resolve().parent
    for column in ("image_path", "mask_path"):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: str(
                    (manifest_dir / value).resolve()
                    if not Path(value).is_absolute()
                    else Path(value)
                )
            )
    if args.fold is not None:
        frame = frame.loc[frame.fold == args.fold].copy()
    results = []
    prediction_dir = str(args.prediction_dir.resolve())
    label_dir = str(args.label_dir.resolve()) if args.label_dir is not None else None
    payloads = [
        (row.to_dict(), prediction_dir, label_dir, args.foreground_aggregation)
        for _, row in frame.iterrows()
    ]
    if args.workers == 1:
        iterator = map(evaluate_case_worker, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(evaluate_case_worker, payloads, chunksize=1)
    try:
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index % 25 == 0:
                print(f"evaluated {index}/{len(frame)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    result_frame = pd.DataFrame(results)
    case_mean = result_frame[METRICS].mean().to_dict()
    center_table = result_frame.groupby("center", as_index=False)[METRICS].mean()
    center_macro = center_table[METRICS].mean().to_dict()
    fold_table = result_frame.groupby("fold", as_index=False)[METRICS].mean()
    summary = {
        "cases": int(len(result_frame)),
        "centers": int(result_frame.center.nunique()),
        "foreground_aggregation": args.foreground_aggregation,
        "case_mean": {key: float(value) for key, value in case_mean.items()},
        "center_macro_mean": {
            key: float(value) for key, value in center_macro.items()
        },
        "fold_means": fold_table.to_dict(orient="records"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(args.output_dir / "per_case_metrics.csv", index=False)
    center_table.to_csv(args.output_dir / "per_center_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
