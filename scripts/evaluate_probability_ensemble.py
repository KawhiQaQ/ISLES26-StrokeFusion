#!/usr/bin/env python3
"""Evaluate a fixed uniform average of binary nnU-Net probability maps.

This is intentionally an analysis utility, not a weight-search tool. Every
member receives exactly the same weight so validation cases cannot be used to
tune ensemble coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from evaluate_isles26 import (
    METRICS,
    compute_instance_metrics,
    compute_pr_auc,
    load_probabilities,
    restore_probability_orientation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("prediction_dirs", nargs="+", type=Path)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    return parser.parse_args()


def load_foreground(case_id: str, prediction_dir: Path) -> np.ndarray:
    payload, probability_format = load_probabilities(
        prediction_dir / f"{case_id}.npz"
    )
    probabilities = restore_probability_orientation(
        payload, prediction_dir / f"{case_id}.pkl"
    )
    if probability_format == "compact_foreground":
        foreground = probabilities[0]
    else:
        if probabilities.shape[0] != 2:
            raise ValueError(
                f"{prediction_dir}: uniform ensemble expects a binary member, "
                f"found {probabilities.shape[0]} probability channels"
            )
        foreground = probabilities[1]

    exported = np.asarray(
        nib.load(prediction_dir / f"{case_id}.nii.gz").dataobj
    ) > 0.5
    if not np.array_equal(foreground > 0.5, exported):
        raise ValueError(f"{case_id}: member probability/mask disagreement")
    return np.asarray(foreground, dtype=np.float32)


def main() -> int:
    args = parse_args()
    if len(args.prediction_dirs) < 2:
        raise ValueError("an ensemble requires at least two prediction directories")
    frame = pd.read_csv(args.manifest)
    frame = frame.loc[frame.fold == args.fold].copy()
    results: list[dict[str, object]] = []

    for index, (_, row) in enumerate(frame.iterrows(), start=1):
        case_id = str(row.case_id)
        label_image = nib.load(args.label_dir / f"{case_id}.nii.gz")
        ground_truth = np.asarray(label_image.dataobj) > 0.5
        members = [
            load_foreground(case_id, directory.resolve())
            for directory in args.prediction_dirs
        ]
        if any(member.shape != ground_truth.shape for member in members):
            raise ValueError(f"{case_id}: member/label shape disagreement")
        soft_map = np.mean(np.stack(members, axis=0), axis=0, dtype=np.float32)
        prediction = soft_map > 0.5
        lesion_f1, count_difference, dice = compute_instance_metrics(
            ground_truth, prediction
        )
        voxel_size_ml = float(np.prod(label_image.header.get_zooms()[:3]) / 1000.0)
        results.append(
            {
                "case_id": case_id,
                "center": row.center,
                "fold": int(row.fold),
                "pr_auc": compute_pr_auc(ground_truth, soft_map),
                "dice": dice,
                "abs_volume_diff_ml": abs(
                    int(ground_truth.sum()) - int(prediction.sum())
                )
                * voxel_size_ml,
                "abs_lesion_count_diff": count_difference,
                "lesion_f1": lesion_f1,
                "gt_volume_ml": float(ground_truth.sum()) * voxel_size_ml,
                "pred_volume_ml": float(prediction.sum()) * voxel_size_ml,
            }
        )
        if index % 25 == 0:
            print(f"evaluated {index}/{len(frame)}", flush=True)

    result_frame = pd.DataFrame(results)
    center_table = result_frame.groupby("center", as_index=False)[METRICS].mean()
    summary = {
        "cases": int(len(result_frame)),
        "members": [str(path.resolve()) for path in args.prediction_dirs],
        "weights": [1.0 / len(args.prediction_dirs)] * len(args.prediction_dirs),
        "weight_selection": "fixed_uniform_no_validation_search",
        "case_mean": {
            key: float(value)
            for key, value in result_frame[METRICS].mean().to_dict().items()
        },
        "center_macro_mean": {
            key: float(value)
            for key, value in center_table[METRICS].mean().to_dict().items()
        },
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
