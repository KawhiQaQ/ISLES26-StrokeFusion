#!/usr/bin/env python3
"""Audit ATLAS R3.0 RAW image/mask/metadata pairing and NIfTI geometry."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import struct
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


IMAGE_SUFFIX = "_space-orig_desc-brain_T1w.nii.gz"
MASK_SUFFIX = "_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"
META_SUFFIX = "_metadata.csv"
META_COLUMNS = {
    "SESSION_ID",
    "ATLAS2_DATASET",
    "DAYS_POST_STROKE",
    "CHRONICITY",
    "SITE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="ATLAS3_Training_Raw directory")
    parser.add_argument("--expected-cases", type=int, default=1453)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Read every mask payload and verify binary labels (slower).",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("missing CSV header")
        missing = META_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"missing metadata columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"expected one metadata row, found {len(rows)}")
    return rows[0]


def raw_pixdim(path: Path) -> tuple[float, ...]:
    """Read pixdim without nibabel's automatic absolute-value correction."""
    with gzip.open(path, "rb") as handle:
        header = handle.read(108)
    if len(header) != 108:
        raise ValueError("short NIfTI-1 header")
    little_size = struct.unpack("<i", header[:4])[0]
    big_size = struct.unpack(">i", header[:4])[0]
    if little_size == 348:
        endian = "<"
    elif big_size == 348:
        endian = ">"
    else:
        raise ValueError("invalid NIfTI-1 sizeof_hdr")
    return struct.unpack(f"{endian}8f", header[76:108])


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    images = sorted(root.rglob(f"*{IMAGE_SUFFIX}"))
    errors: list[str] = []
    audit_warnings: list[str] = []
    sites: Counter[str] = Counter()
    atlas2_split: Counter[str] = Counter()
    chronicity: Counter[str] = Counter()
    metadata_missing: Counter[str] = Counter()
    days_post_stroke: list[float] = []
    orientations: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    zoom_min = np.array([np.inf, np.inf, np.inf], dtype=float)
    zoom_max = np.array([0.0, 0.0, 0.0], dtype=float)
    empty_mask_cases: list[str] = []

    if len(images) != args.expected_cases:
        errors.append(
            f"image count is {len(images)}, expected {args.expected_cases}"
        )

    for image_path in images:
        case_id = image_path.name[: -len(IMAGE_SUFFIX)]
        mask_path = image_path.with_name(f"{case_id}{MASK_SUFFIX}")
        meta_path = image_path.with_name(f"{case_id}{META_SUFFIX}")

        if not mask_path.is_file():
            errors.append(f"{case_id}: missing mask")
            continue
        if not meta_path.is_file():
            errors.append(f"{case_id}: missing metadata")
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = nib.load(image_path)
                mask = nib.load(mask_path)
            if image.shape != mask.shape:
                errors.append(
                    f"{case_id}: shape mismatch {image.shape} vs {mask.shape}"
                )
            affine_delta = float(np.max(np.abs(image.affine - mask.affine)))
            image_axcodes = nib.aff2axcodes(image.affine)
            mask_axcodes = nib.aff2axcodes(mask.affine)
            if image_axcodes != mask_axcodes or affine_delta > 1e-2:
                errors.append(
                    f"{case_id}: image/mask affine mismatch "
                    f"(max_abs_delta={affine_delta:g}, "
                    f"orientation={image_axcodes}/{mask_axcodes})"
                )
            elif affine_delta > 1e-4:
                audit_warnings.append(
                    f"{case_id}: minor affine rounding delta={affine_delta:g}"
                )

            for role, path in (("image", image_path), ("mask", mask_path)):
                pixdim = raw_pixdim(path)
                if any(value <= 0 for value in pixdim[1:4]):
                    audit_warnings.append(
                        f"{case_id}: {role} has non-positive raw pixdim "
                        f"{pixdim[1:4]}"
                    )

            image_zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
            zoom_min = np.minimum(zoom_min, image_zooms)
            zoom_max = np.maximum(zoom_max, image_zooms)
            orientations["".join(image_axcodes)] += 1
            shapes["x".join(str(value) for value in image.shape)] += 1

            if args.full:
                proxy = mask.dataobj
                mask_data = np.asanyarray(
                    proxy.get_unscaled() if hasattr(proxy, "get_unscaled") else proxy
                )
                if not np.isfinite(mask_data).all():
                    errors.append(f"{case_id}: mask contains NaN/Inf")
                raw_min = mask_data.min()
                raw_max = mask_data.max()
                slope = float(getattr(proxy, "slope", None) or 1.0)
                intercept = float(getattr(proxy, "inter", None) or 0.0)
                scaled_endpoints = sorted(
                    (
                        float(raw_min) * slope + intercept,
                        float(raw_max) * slope + intercept,
                    )
                )
                if raw_min == raw_max:
                    is_binary = bool(
                        np.isclose(scaled_endpoints[0], 0, rtol=0, atol=1e-6)
                        or np.isclose(scaled_endpoints[0], 1, rtol=0, atol=1e-6)
                    )
                else:
                    scaled_binary = bool(
                        np.isclose(scaled_endpoints[0], 0, rtol=0, atol=1e-6)
                        and np.isclose(scaled_endpoints[1], 1, rtol=0, atol=1e-6)
                    )
                    consecutive_integer_endpoints = bool(
                        np.issubdtype(mask_data.dtype, np.integer)
                        and int(raw_max) - int(raw_min) == 1
                    )
                    only_endpoints = consecutive_integer_endpoints or bool(
                        np.logical_or(
                            mask_data == raw_min, mask_data == raw_max
                        ).all()
                    )
                    is_binary = scaled_binary and only_endpoints
                if not is_binary:
                    labels = np.unique(mask_data)
                    errors.append(
                        f"{case_id}: non-binary raw labels {labels[:10].tolist()} "
                        f"with scaled endpoints {scaled_endpoints}"
                    )
                if scaled_endpoints[-1] <= 0.5:
                    empty_mask_cases.append(case_id)
                    audit_warnings.append(f"{case_id}: empty lesion mask")

            metadata = load_metadata(meta_path)
            sites[metadata["SITE"] or "<missing>"] += 1
            atlas2_split[metadata["ATLAS2_DATASET"] or "<missing>"] += 1
            chronicity[metadata["CHRONICITY"] or "<missing>"] += 1
            for column in META_COLUMNS:
                if not metadata[column].strip():
                    metadata_missing[column] += 1
            if metadata["DAYS_POST_STROKE"].strip():
                try:
                    days_post_stroke.append(float(metadata["DAYS_POST_STROKE"]))
                except ValueError:
                    errors.append(
                        f"{case_id}: invalid DAYS_POST_STROKE="
                        f"{metadata['DAYS_POST_STROKE']!r}"
                    )
            if metadata["SESSION_ID"] != case_id:
                errors.append(
                    f"{case_id}: metadata SESSION_ID={metadata['SESSION_ID']!r}"
                )
        except Exception as exc:  # Keep auditing remaining cases.
            errors.append(f"{case_id}: {type(exc).__name__}: {exc}")

    mask_count = sum(1 for _ in root.rglob(f"*{MASK_SUFFIX}"))
    metadata_count = sum(1 for _ in root.rglob(f"*{META_SUFFIX}"))
    if mask_count != len(images):
        errors.append(f"mask count is {mask_count}, image count is {len(images)}")
    if metadata_count != len(images):
        errors.append(
            f"metadata count is {metadata_count}, image count is {len(images)}"
        )

    summary: dict[str, Any] = {
        "root": str(root),
        "full_payload_check": args.full,
        "counts": {
            "images": len(images),
            "masks": mask_count,
            "metadata": metadata_count,
            "sites": len(sites),
            "unique_shapes": len(shapes),
            "empty_masks": len(empty_mask_cases) if args.full else None,
        },
        "voxel_spacing_mm": {
            "min": zoom_min.tolist() if images else None,
            "max": zoom_max.tolist() if images else None,
        },
        "orientations": dict(orientations.most_common()),
        "site_counts": dict(sites.most_common()),
        "atlas2_dataset_counts": dict(atlas2_split.most_common()),
        "chronicity_counts": dict(chronicity.most_common()),
        "metadata_missing_counts": dict(metadata_missing.most_common()),
        "days_post_stroke": {
            "available": len(days_post_stroke),
            "min": min(days_post_stroke) if days_post_stroke else None,
            "max": max(days_post_stroke) if days_post_stroke else None,
        },
        "most_common_shapes": dict(shapes.most_common(20)),
        "empty_mask_cases": empty_mask_cases if args.full else None,
        "error_count": len(errors),
        "errors": errors[:100],
        "warning_count": len(audit_warnings),
        "warnings": audit_warnings[:200],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
