#!/usr/bin/env python3
"""Evaluate the frozen full-data submission with the predeclared safe cleanup."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


CONNECTIVITY = ndimage.generate_binary_structure(3, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def remove_low_confidence_tiny_components(
    prediction: np.ndarray,
    probability: np.ndarray,
    voxel_volume_ml: float,
    minimum_volume_ml: float,
    maximum_probability: float,
) -> np.ndarray:
    """Reference implementation of confidence-aware component cleanup."""

    labels, count = ndimage.label(prediction, structure=CONNECTIVITY)
    if count == 0:
        return prediction.copy()
    component_voxels = np.bincount(labels.ravel())
    minimum_voxels = max(1, int(np.ceil(minimum_volume_ml / voxel_volume_ml)))
    keep = component_voxels >= minimum_voxels
    keep[0] = False
    for component in range(1, count + 1):
        if (
            not keep[component]
            and float(probability[labels == component].max()) >= maximum_probability
        ):
            keep[component] = True
    return keep[labels]


def metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    voxel_volume_ml: float,
    compute_instance_metrics,
    compute_pr_auc,
) -> dict[str, float | int]:
    lesion_f1, count_difference, dice = compute_instance_metrics(
        ground_truth, prediction
    )
    return {
        "pr_auc": compute_pr_auc(ground_truth, probability),
        "dice": dice,
        "abs_volume_diff_ml": abs(
            int(ground_truth.sum()) - int(prediction.sum())
        )
        * voxel_volume_ml,
        "abs_lesion_count_diff": count_difference,
        "lesion_f1": lesion_f1,
        "positive_voxels": int(prediction.sum()),
    }


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    run_root = args.run_root.resolve()
    input_root = run_root / "input"
    output_root = run_root / "output"
    scratch_root = run_root / "scratch"
    results_root = run_root / "results"
    for path in (input_root, output_root, scratch_root, results_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    os.environ.update(
        {
            "ISLES26_INPUT_PATH": str(input_root),
            "ISLES26_OUTPUT_PATH": str(output_root),
            "ISLES26_MODEL_PATH": os.environ.get(
                "ISLES26_TEST_MODEL_PATH",
                str(workspace / "docker" / "model"),
            ),
            "ISLES26_SCRATCH_PATH": str(scratch_root),
            "nnUNet_compile": "false",
        }
    )
    submission = workspace / "docker"
    sys.path.insert(0, str(submission))
    sys.path.insert(0, str(workspace / "scripts"))
    import inference  # noqa: PLC0415
    from evaluate_isles26 import compute_instance_metrics, compute_pr_auc  # noqa: PLC0415

    manifest = pd.read_csv(workspace / "outputs" / "local_sanity_check" / "manifest.csv")
    bundle = inference.init_model()
    all_results: dict[str, dict[str, dict[str, float | int]]] = {}
    for row in manifest.itertuples(index=False):
        case_id = str(row.case_id)
        shutil.rmtree(input_root / "images", ignore_errors=True)
        shutil.rmtree(output_root / "images", ignore_errors=True)
        image_dir = input_root / "images" / "t1-brain-mri"
        image_dir.mkdir(parents=True)
        (image_dir / "input.nii.gz").symlink_to(Path(row.image_path))
        (input_root / "inputs.json").write_text(
            json.dumps(
                [
                    {"socket": {"slug": "t1-brain-mri"}},
                    {"socket": {"slug": "stroke-metadata"}},
                ]
            ),
            encoding="utf-8",
        )
        (input_root / "stroke-metadata.json").write_text(
            json.dumps({"CENTER": str(row.center)}), encoding="utf-8"
        )
        inference.run(bundle)

        probability_image = sitk.ReadImage(
            str(output_root / "images" / "lesion-probability-map" / "output.mha")
        )
        segmentation_image = sitk.ReadImage(
            str(output_root / "images" / "stroke-lesion-segmentation" / "output.mha")
        )
        ground_truth_image = sitk.ReadImage(str(row.mask_path))
        if probability_image.GetSize() != ground_truth_image.GetSize():
            raise ValueError(f"{case_id}: probability/label size mismatch")
        if segmentation_image.GetSize() != ground_truth_image.GetSize():
            raise ValueError(f"{case_id}: segmentation/label size mismatch")
        probability = sitk.GetArrayFromImage(probability_image).astype(np.float32)
        submitted = sitk.GetArrayFromImage(segmentation_image) > 0
        baseline = probability > np.float32(inference.SEGMENTATION_THRESHOLD)
        ground_truth = sitk.GetArrayFromImage(ground_truth_image) > 0
        voxel_volume_ml = float(np.prod(ground_truth_image.GetSpacing()) / 1000.0)
        cleaned = remove_low_confidence_tiny_components(
            baseline,
            probability,
            voxel_volume_ml,
            inference.MINIMUM_COMPONENT_VOLUME_ML,
            inference.PRESERVE_COMPONENT_MAX_PROBABILITY,
        )
        if not np.array_equal(submitted, cleaned):
            raise ValueError(
                f"{case_id}: submitted mask and frozen cleanup implementation disagree"
            )
        all_results[case_id] = {
            "thresholded_before_cleanup": metrics(
                ground_truth,
                baseline,
                probability,
                voxel_volume_ml,
                compute_instance_metrics,
                compute_pr_auc,
            ),
            "submitted_after_cleanup": metrics(
                ground_truth,
                cleaned,
                probability,
                voxel_volume_ml,
                compute_instance_metrics,
                compute_pr_auc,
            ),
        }
        case_dir = results_root / case_id
        case_dir.mkdir(parents=True)
        shutil.copy2(
            output_root / "images" / "lesion-probability-map" / "output.mha",
            case_dir / "probability.mha",
        )
        cleaned_image = sitk.GetImageFromArray(cleaned.astype(np.uint8))
        cleaned_image.CopyInformation(ground_truth_image)
        sitk.WriteImage(cleaned_image, str(case_dir / "cleaned.mha"), True)

    metric_names = (
        "pr_auc",
        "dice",
        "abs_volume_diff_ml",
        "abs_lesion_count_diff",
        "lesion_f1",
    )
    summary: dict[str, dict[str, float]] = {}
    for candidate in ("thresholded_before_cleanup", "submitted_after_cleanup"):
        summary[candidate] = {
            name: float(
                np.mean([case[candidate][name] for case in all_results.values()])
            )
            for name in metric_names
        }
    payload = {
        "configuration": {
            "threshold": inference.SEGMENTATION_THRESHOLD,
            "minimum_component_volume_ml": inference.MINIMUM_COMPONENT_VOLUME_ML,
            "preserve_component_max_probability": (
                inference.PRESERVE_COMPONENT_MAX_PROBABILITY
            ),
        },
        "cases": all_results,
        "summary": summary,
    }
    (run_root / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
