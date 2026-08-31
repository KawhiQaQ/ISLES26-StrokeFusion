#!/usr/bin/env python3
"""Expose the licensed RAW release to nnU-Net without duplicating images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


IMAGE_SUFFIX = "_space-orig_desc-brain_T1w.nii.gz"
MASK_SUFFIX = "_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("nnunet_raw", type=Path)
    parser.add_argument("--dataset-id", type=int, default=26)
    parser.add_argument("--dataset-name", default="ISLES26")
    parser.add_argument("--affine-tolerance", type=float, default=1e-2)
    return parser.parse_args()


def replace_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.symlink_to(source.resolve())


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    dataset_dir = args.nnunet_raw.resolve() / (
        f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    )
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(raw_root.rglob(f"*{IMAGE_SUFFIX}"))
    if len(images) != 1453:
        raise RuntimeError(f"expected 1453 images, found {len(images)}")

    corrected: list[dict[str, object]] = []
    cases: list[str] = []
    for image_path in images:
        case_id = image_path.name[: -len(IMAGE_SUFFIX)]
        mask_path = image_path.with_name(f"{case_id}{MASK_SUFFIX}")
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        output_image = images_dir / f"{case_id}_0000.nii.gz"
        output_label = labels_dir / f"{case_id}.nii.gz"
        replace_symlink(image_path, output_image)

        image = nib.load(image_path)
        mask = nib.load(mask_path)
        affine_delta = float(np.max(np.abs(image.affine - mask.affine)))
        same_grid = image.shape == mask.shape and affine_delta <= args.affine_tolerance
        if same_grid:
            replace_symlink(mask_path, output_label)
        else:
            if output_label.is_symlink():
                output_label.unlink()
            if output_label.exists():
                previous = nib.load(output_label)
                if previous.shape != image.shape or not np.allclose(
                    previous.affine, image.affine, atol=1e-5
                ):
                    raise FileExistsError(f"stale corrected label: {output_label}")
            else:
                aligned = resample_from_to(
                    mask,
                    (image.shape, image.affine),
                    order=0,
                    mode="constant",
                    cval=0,
                )
                data = (np.asarray(aligned.dataobj) > 0.5).astype(np.uint8)
                header = image.header.copy()
                header.set_data_dtype(np.uint8)
                nib.save(nib.Nifti1Image(data, image.affine, header), output_label)
            corrected.append(
                {
                    "case_id": case_id,
                    "image_shape": list(image.shape),
                    "mask_shape": list(mask.shape),
                    "max_affine_delta": affine_delta,
                    "image_orientation": "".join(nib.aff2axcodes(image.affine)),
                    "mask_orientation": "".join(nib.aff2axcodes(mask.affine)),
                }
            )
        cases.append(case_id)

    dataset_json = {
        "channel_names": {"0": "T1"},
        "labels": {"background": 0, "stroke_lesion": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "dataset_dir": os.fspath(dataset_dir),
        "cases": len(cases),
        "corrected_label_grids": corrected,
    }
    (dataset_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
