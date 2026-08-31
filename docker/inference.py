"""StrokeFusion native-space inference for ISLES'26."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy import ndimage


INPUT_PATH = Path(os.environ.get("ISLES26_INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.environ.get("ISLES26_OUTPUT_PATH", "/output"))
MODEL_PATH = Path(os.environ.get("ISLES26_MODEL_PATH", "/opt/ml/model"))
SCRATCH_ROOT = Path(os.environ.get("ISLES26_SCRATCH_PATH", "/tmp"))

# Defaults reproduce the final full-data submission. Environment variables are
# exposed for controlled ablations; the submitted probability map is unchanged.
POSTPROCESS_CONNECTIVITY = ndimage.generate_binary_structure(3, 1)
SEGMENTATION_THRESHOLD = float(
    os.environ.get("ISLES26_SEGMENTATION_THRESHOLD", "0.5")
)
MINIMUM_COMPONENT_VOLUME_ML = float(
    os.environ.get("ISLES26_MINIMUM_COMPONENT_VOLUME_ML", "0.002")
)
PRESERVE_COMPONENT_MAX_PROBABILITY = float(
    os.environ.get("ISLES26_PRESERVE_COMPONENT_MAX_PROBABILITY", "0.65")
)
if not 0.0 < SEGMENTATION_THRESHOLD < 1.0:
    raise ValueError(f"invalid segmentation threshold: {SEGMENTATION_THRESHOLD}")
if MINIMUM_COMPONENT_VOLUME_ML < 0.0:
    raise ValueError(
        f"invalid minimum component volume: {MINIMUM_COMPONENT_VOLUME_ML}"
    )
if not SEGMENTATION_THRESHOLD <= PRESERVE_COMPONENT_MAX_PROBABILITY <= 1.0:
    raise ValueError(
        "component-preservation probability must be between the segmentation "
        "threshold and 1"
    )


@dataclass
class ModelBundle:
    e1: nnUNetPredictor
    v15: nnUNetPredictor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_manifest() -> None:
    manifest_path = MODEL_PATH / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_path, expected_hash in manifest["sha256"].items():
        path = MODEL_PATH / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"model hash mismatch for {relative_path}: "
                f"{actual_hash} != {expected_hash}"
            )


def _make_predictor(model_dir: Path) -> nnUNetPredictor:
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        # Preserve nnU-Net predictions while retaining headroom on the 16 GiB T4.
        perform_everything_on_device=False,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=("all",),
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.network.eval()
    return predictor


def init_model() -> ModelBundle:
    """Validate resources and initialize both frozen full-data models."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "StrokeFusion requires the challenge NVIDIA T4 GPU; "
            "configure this algorithm image with a GPU instance"
        )
    print(
        f"CUDA device: {torch.cuda.get_device_name(0)}; "
        f"torch={torch.__version__}, cuda={torch.version.cuda}",
        flush=True,
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    _verify_model_manifest()

    started = time.monotonic()
    bundle = ModelBundle(
        e1=_make_predictor(MODEL_PATH / "e1"),
        v15=_make_predictor(MODEL_PATH / "v15"),
    )
    print(f"Models initialized in {time.monotonic() - started:.2f}s", flush=True)
    return bundle


def get_interface_key() -> tuple[str, ...]:
    with (INPUT_PATH / "inputs.json").open(encoding="utf-8") as handle:
        inputs = json.load(handle)
    return tuple(sorted(item["socket"]["slug"] for item in inputs))


def _find_input_image() -> Path:
    location = INPUT_PATH / "images" / "t1-brain-mri"
    candidates = sorted(
        glob.glob(str(location / "*.mha"))
        + glob.glob(str(location / "*.nii.gz"))
        + glob.glob(str(location / "*.nii"))
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one T1 image in {location}, found {len(candidates)}"
        )
    return Path(candidates[0])


def _predict_member(
    predictor: nnUNetPredictor,
    normalized_input: Path,
    output_stem: Path,
    label: str,
) -> None:
    started = time.monotonic()
    # Grand Challenge may keep the API container alive for more than one invoke.
    # Members are parked on CPU after use, so always restore the active member.
    predictor.network.to(torch.device("cuda", 0))
    predictor.predict_from_files(
        [[str(normalized_input)]],
        [str(output_stem)],
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    predictor.network.to(torch.device("cpu"))
    torch.cuda.empty_cache()
    print(f"{label} inference completed in {time.monotonic() - started:.2f}s", flush=True)


def _load_native_foreground(output_stem: Path) -> np.ndarray:
    probability_path = output_stem.with_suffix(".npz")
    properties_path = output_stem.with_suffix(".pkl")
    with np.load(probability_path) as payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != 2:
        raise ValueError(f"unexpected binary probability shape: {probabilities.shape}")

    with properties_path.open("rb") as handle:
        properties = pickle.load(handle)
    nibabel_stuff = properties.get("nibabel_stuff")
    if not isinstance(nibabel_stuff, dict):
        raise KeyError("nnU-Net prediction properties are missing nibabel_stuff")

    # nnU-Net stores exported maps as C,z,y,x on its reoriented grid.
    probabilities_xyz = probabilities.transpose(0, 3, 2, 1)
    source = io_orientation(np.asarray(nibabel_stuff["reoriented_affine"]))
    target = io_orientation(np.asarray(nibabel_stuff["original_affine"]))
    transform = ornt_transform(source, target)
    foreground_xyz = apply_orientation(probabilities_xyz[1], transform)
    # SimpleITK arrays are z,y,x.
    return np.asarray(foreground_xyz.transpose(2, 1, 0), dtype=np.float32)


def _write_output(location: Path, array: np.ndarray, reference: sitk.Image) -> None:
    location.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(location / "output.mha"), useCompression=True)


def _remove_low_confidence_tiny_components(
    segmentation: np.ndarray,
    probability: np.ndarray,
    voxel_volume_ml: float,
) -> tuple[np.ndarray, int, int]:
    """Apply the frozen confidence-aware 6-connected-component cleanup."""

    labels, component_count = ndimage.label(
        segmentation,
        structure=POSTPROCESS_CONNECTIVITY,
    )
    if component_count == 0:
        return segmentation.copy(), 0, 0

    component_voxels = np.bincount(labels.ravel())
    minimum_voxels = max(
        1,
        int(np.ceil(MINIMUM_COMPONENT_VOLUME_ML / voxel_volume_ml)),
    )
    keep = component_voxels >= minimum_voxels
    keep[0] = False
    component_maxima = np.asarray(
        ndimage.maximum(
            probability,
            labels=labels,
            index=np.arange(1, component_count + 1),
        )
    )
    keep[1:] |= component_maxima >= PRESERVE_COMPONENT_MAX_PROBABILITY
    removed = ~keep[1:]
    removed_components = int(removed.sum())
    removed_voxels = int(component_voxels[1:][removed].sum())
    return keep[labels].astype(np.uint8), removed_components, removed_voxels


def run(model: ModelBundle) -> int:
    interface_key = get_interface_key()
    expected_key = ("stroke-metadata", "t1-brain-mri")
    if interface_key != expected_key:
        raise KeyError(f"unsupported interface {interface_key}; expected {expected_key}")

    invocation_started = time.monotonic()
    input_image_path = _find_input_image()
    input_image = sitk.ReadImage(str(input_image_path))
    input_array = sitk.GetArrayViewFromImage(input_image)
    with (INPUT_PATH / "stroke-metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    print(f"Metadata received (not used by the frozen models): {metadata}", flush=True)

    scratch = SCRATCH_ROOT / f"isles26-{uuid.uuid4().hex}"
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        normalized_input = scratch / "case_0000.nii.gz"
        sitk.WriteImage(input_image, str(normalized_input), useCompression=True)

        e1_stem = scratch / "e1" / "case"
        v15_stem = scratch / "v15" / "case"
        e1_stem.parent.mkdir(parents=True)
        v15_stem.parent.mkdir(parents=True)

        _predict_member(
            model.e1,
            normalized_input,
            e1_stem,
            "ResEnc-L RASS SWA",
        )
        e1_probability = _load_native_foreground(e1_stem)
        _predict_member(
            model.v15,
            normalized_input,
            v15_stem,
            "Primus-M Local-Refinement SWA",
        )
        v15_probability = _load_native_foreground(v15_stem)

        if e1_probability.shape != input_array.shape:
            raise ValueError(
                "ResEnc-L/native shape mismatch: "
                f"{e1_probability.shape} != {input_array.shape}"
            )
        if v15_probability.shape != input_array.shape:
            raise ValueError(
                "Primus-M/native shape mismatch: "
                f"{v15_probability.shape} != {input_array.shape}"
            )

        probability = np.clip(
            (e1_probability + v15_probability) * np.float32(0.5),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        if not np.isfinite(probability).all():
            raise FloatingPointError("non-finite values in lesion probability map")
        segmentation = (
            probability > np.float32(SEGMENTATION_THRESHOLD)
        ).astype(np.uint8)
        voxel_volume_ml = float(np.prod(input_image.GetSpacing()) / 1000.0)
        segmentation, removed_components, removed_voxels = (
            _remove_low_confidence_tiny_components(
                segmentation,
                probability,
                voxel_volume_ml,
            )
        )

        _write_output(
            OUTPUT_PATH / "images" / "stroke-lesion-segmentation",
            segmentation,
            input_image,
        )
        _write_output(
            OUTPUT_PATH / "images" / "lesion-probability-map",
            probability,
            input_image,
        )
        print(
            f"Invoke completed in {time.monotonic() - invocation_started:.2f}s; "
            f"shape={tuple(input_array.shape)}, positive_voxels={int(segmentation.sum())}, "
            f"threshold={SEGMENTATION_THRESHOLD}, "
            f"minimum_component_ml={MINIMUM_COMPONENT_VOLUME_ML}, "
            f"preserve_component_max_probability={PRESERVE_COMPONENT_MAX_PROBABILITY}, "
            f"removed_components={removed_components}, removed_voxels={removed_voxels}",
            flush=True,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(init_model()))
