#!/usr/bin/env python3
"""Bounded ResEnc-L RASS launch check with optimizer state resident."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerV12RASS import (
    RandomAmplitudeSpectrumSynthesisTransform,
)


EXPECTED_SPLIT_SHA256 = (
    "da10108f65fdff2954c7f68f9e69456a5d9bb8f78b28512e3714656ba2bd9885"
)
EXPECTED_PRETRAINED_SHA256 = (
    "7a847af785635335c00e711d16ff4d225d86ecd5992b14c059df2b520e3ee933"
)
WORKSPACE = Path(
    os.environ.get("ISLES26_WORKSPACE", Path(__file__).resolve().parents[1])
).resolve()
PRETRAINED_CHECKPOINT = (
    WORKSPACE
    / "external_models"
    / "ResEncL-OpenMind-MAE"
    / "checkpoint_final.pth"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finish_loader(loader) -> None:
    finish = getattr(loader, "_finish", None)
    if callable(finish):
        finish()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path(
            os.environ.get(
                "nnUNet_preprocessed", WORKSPACE / ".cache/nnUNet_preprocessed"
            )
        )
        / "Dataset026_ISLES26/splits_final.json",
    )
    args = parser.parse_args()

    split_hash = sha256(args.split_file)
    pretrained_hash = sha256(PRETRAINED_CHECKPOINT)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"frozen split hash changed: {split_hash}")
    if pretrained_hash != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError(f"pretrained checkpoint hash changed: {pretrained_hash}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    direct_transform = RandomAmplitudeSpectrumSynthesisTransform(3.0, 0.25, 2.0)
    synthetic = torch.zeros((1, 24, 28, 20), dtype=torch.float32)
    synthetic[:, 3:-3, 4:-4, 2:-2] = torch.randn((1, 18, 20, 16))
    transformed = direct_transform(image=synthetic.clone())["image"]
    if not torch.isfinite(transformed).all():
        raise RuntimeError("RASS produced non-finite values")
    if torch.any(transformed[synthetic == 0] != 0):
        raise RuntimeError("RASS changed the skull-stripped background support")
    if torch.equal(transformed, synthetic):
        raise RuntimeError("RASS did not alter the image")

    trainer = get_trainer_from_args(
        "26",
        "3d_fullres",
        args.fold,
        trainer_name="nnUNetTrainerV12RASS",
        plans_identifier="nnUNetResEncUNetLPlans",
        device=torch.device("cuda"),
    )
    trainer.initialize()
    network = trainer._network_module()
    if trainer.enable_deep_supervision:
        raise RuntimeError("ResEnc-L RASS must disable deep supervision")
    if trainer.configuration_manager.batch_size != 3:
        raise RuntimeError("ResEnc-L RASS check requires batch size 3")

    rotation, dummy_2d, _, mirror_axes = (
        trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    )
    transforms = trainer.get_training_transforms(
        trainer.configuration_manager.patch_size,
        rotation,
        trainer._get_deep_supervision_scales(),
        mirror_axes,
        dummy_2d,
        use_mask_for_norm=trainer.configuration_manager.use_mask_for_norm,
        is_cascaded=trainer.is_cascaded,
        foreground_labels=trainer.label_manager.foreground_labels,
        regions=None,
        ignore_label=trainer.label_manager.ignore_label,
    )
    rass_wrappers = [
        transform
        for transform in transforms.transforms
        if isinstance(transform, RandomTransform)
        and isinstance(
            transform.transform, RandomAmplitudeSpectrumSynthesisTransform
        )
    ]
    if len(rass_wrappers) != 1:
        raise RuntimeError(f"expected one RASS transform, found {len(rass_wrappers)}")
    if not math.isclose(rass_wrappers[0].apply_probability, 0.30):
        raise RuntimeError("unexpected ResEnc-L RASS probability")

    # Force the most memory-intensive scheduled stage before any long run.
    trainer.optimizer, trainer.lr_scheduler = trainer.configure_optimizers("warmup_all")
    trainer.network.train()
    if not all(parameter.requires_grad for parameter in network.parameters()):
        raise RuntimeError("the full network is not unfrozen in the memory test")
    audited_parameter = next(network.encoder.parameters())
    audited_before = audited_parameter.detach().clone()

    train_keys, validation_keys = trainer.do_split()
    train_loader = validation_loader = None
    losses: list[float] = []
    try:
        train_loader, validation_loader = trainer.get_dataloaders()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch_shape = None
        for _ in range(2):
            batch = next(train_loader)
            batch_shape = list(batch["data"].shape)
            output = trainer.train_step(batch)
            losses.append(float(output["loss"]))
        torch.cuda.synchronize()
        if not all(math.isfinite(loss) for loss in losses):
            raise RuntimeError(f"non-finite ResEnc-L RASS losses: {losses}")
        if not trainer.optimizer.state:
            raise RuntimeError("optimizer momentum state was not materialized")
        if torch.equal(audited_before, audited_parameter.detach()):
            raise RuntimeError("the unfrozen encoder did not update")
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    finally:
        if train_loader is not None:
            finish_loader(train_loader)
        if validation_loader is not None:
            finish_loader(validation_loader)

    print(
        json.dumps(
            {
                "model": "ResEnc-L RASS",
                "fold": args.fold,
                "train_cases": len(train_keys),
                "validation_cases": len(validation_keys),
                "split_sha256": split_hash,
                "pretrained_sha256": pretrained_hash,
                "raw_native_input": True,
                "rass": {
                    "probability": trainer.rass_probability,
                    "alpha": trainer.rass_alpha,
                    "beta": trainer.rass_beta,
                    "gamma": trainer.rass_gamma,
                    "training_only": True,
                },
                "full_network_unfrozen": True,
                "optimizer_state_entries": len(trainer.optimizer.state),
                "optimizer_steps": 2,
                "losses": losses,
                "batch_shape": batch_shape,
                "peak_cuda_allocated_gib": peak_allocated,
                "peak_cuda_reserved_gib": peak_reserved,
                "cuda_free_after_test_gib": free_bytes / 1024**3,
                "cuda_total_gib": total_bytes / 1024**3,
                "cuda_device": torch.cuda.get_device_name(0),
                "status": "ok",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
