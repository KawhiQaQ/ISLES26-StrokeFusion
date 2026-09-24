#!/usr/bin/env python3
"""Run the frozen StrokeFusion ensemble on one native-space T1 MRI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

from verify_inference_release import _normalized_members, verify_archive


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Predict a binary stroke-lesion mask and foreground probability map "
            "from one native-space T1 MRI. An NVIDIA CUDA GPU is required."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=".nii.gz, .nii, or .mha T1 MRI",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-archive",
        type=Path,
        default=workspace / "submission_artifacts" / "model-postprocess.tar.gz",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="optional JSON metadata; accepted for interface parity but not used",
    )
    parser.add_argument(
        "--output-format",
        choices=("nii.gz", "mha"),
        default="nii.gz",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-model-verification",
        action="store_true",
        help=(
            "skip the archive checksum pass "
            "(checkpoint hashes are still checked at load time)"
        ),
    )
    return parser.parse_args()


def _install_inference_trainers(workspace: Path) -> None:
    import nnunetv2

    source = workspace / "docker" / "deploy_trainers"
    destination = (
        Path(nnunetv2.__file__).resolve().parent / "training" / "nnUNetTrainer"
    )
    for name in ("nnUNetTrainerV12RASS.py", "nnUNetTrainerV15NativeRefinement.py"):
        shutil.copy2(source / name, destination / name)


def _load_inference_module(workspace: Path):
    location = workspace / "docker" / "inference.py"
    specification = importlib.util.spec_from_file_location(
        "strokefusion_frozen_inference", location
    )
    if specification is None or specification.loader is None:
        raise ImportError(location)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _extract_model_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        _normalized_members(bundle)
        bundle.extractall(destination)


def _output_paths(output_dir: Path, extension: str) -> tuple[Path, Path]:
    return (
        output_dir / f"stroke-lesion-segmentation.{extension}",
        output_dir / f"lesion-probability-map.{extension}",
    )


def main() -> int:
    args = parse_args()
    workspace = Path(__file__).resolve().parents[1]
    input_path = args.input.expanduser().resolve()
    archive = args.model_archive.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not input_path.name.endswith((".nii.gz", ".nii", ".mha")):
        raise ValueError("--input must end in .nii.gz, .nii, or .mha")
    if args.metadata is not None and not args.metadata.expanduser().is_file():
        raise FileNotFoundError(args.metadata)

    segmentation_path, probability_path = _output_paths(
        output_dir, args.output_format
    )
    existing = [
        path for path in (segmentation_path, probability_path) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing outputs: {existing}; pass --overwrite"
        )

    if not args.skip_model_verification:
        verify_archive(archive)

    with tempfile.TemporaryDirectory(prefix="strokefusion-predict-") as temporary:
        root = Path(temporary)
        model_dir = root / "model"
        input_dir = root / "input"
        gc_output_dir = root / "output"
        scratch_dir = root / "scratch"
        image_dir = input_dir / "images" / "t1-brain-mri"
        for directory in (model_dir, image_dir, gc_output_dir, scratch_dir):
            directory.mkdir(parents=True, exist_ok=True)

        _extract_model_archive(archive, model_dir)
        shutil.copy2(input_path, image_dir / input_path.name)
        (input_dir / "inputs.json").write_text(
            json.dumps(
                [
                    {"socket": {"slug": "t1-brain-mri"}},
                    {"socket": {"slug": "stroke-metadata"}},
                ]
            ),
            encoding="utf-8",
        )
        if args.metadata is None:
            metadata = {}
        else:
            metadata = json.loads(
                args.metadata.expanduser().read_text(encoding="utf-8")
            )
        (input_dir / "stroke-metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

        os.environ.update(
            {
                "ISLES26_INPUT_PATH": os.fspath(input_dir),
                "ISLES26_OUTPUT_PATH": os.fspath(gc_output_dir),
                "ISLES26_MODEL_PATH": os.fspath(model_dir),
                "ISLES26_SCRATCH_PATH": os.fspath(scratch_dir),
                "nnUNet_compile": "false",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            }
        )
        _install_inference_trainers(workspace)
        inference = _load_inference_module(workspace)
        inference.run(inference.init_model())

        source_segmentation = (
            gc_output_dir / "images" / "stroke-lesion-segmentation" / "output.mha"
        )
        source_probability = (
            gc_output_dir / "images" / "lesion-probability-map" / "output.mha"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for source, destination in (
            (source_segmentation, segmentation_path),
            (source_probability, probability_path),
        ):
            image = inference.sitk.ReadImage(os.fspath(source))
            inference.sitk.WriteImage(image, os.fspath(destination), useCompression=True)

    print(f"Segmentation: {segmentation_path}")
    print(f"Probability map: {probability_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
