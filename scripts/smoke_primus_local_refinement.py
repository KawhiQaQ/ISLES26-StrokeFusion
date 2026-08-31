#!/usr/bin/env python3
"""Minimal Primus-M Local-Refinement start/backward and split check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from nnunetv2.run.run_training import get_trainer_from_args


EXPECTED_SPLIT_SHA256 = (
    "da10108f65fdff2954c7f68f9e69456a5d9bb8f78b28512e3714656ba2bd9885"
)
WORKSPACE = Path(
    os.environ.get("ISLES26_WORKSPACE", Path(__file__).resolve().parents[1])
).resolve()


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
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"frozen split changed: {split_hash}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    trainer = get_trainer_from_args(
        "26",
        "3d_fullres",
        args.fold,
        trainer_name="nnUNetTrainerV15NativeRefinement",
        plans_identifier="nnUNetResEncUNetLPlansV15",
        device=torch.device("cuda"),
    )
    trainer.initialize()
    network = trainer._network_module()
    train_keys, validation_keys = trainer.do_split()
    overlap = set(train_keys) & set(validation_keys)
    if overlap:
        raise RuntimeError(f"train/validation leakage: {sorted(overlap)[:5]}")
    if trainer.num_epochs != 400 or trainer.warmup_duration_whole_net != 15:
        raise RuntimeError("Primus-M refinement must use its 400-epoch schedule")
    if not math.isclose(trainer.initial_lr, 1e-4):
        raise RuntimeError(f"unexpected initial LR: {trainer.initial_lr}")
    if list(trainer.configuration_manager.patch_size) != [160, 160, 160]:
        raise RuntimeError(f"unexpected patch: {trainer.configuration_manager.patch_size}")
    if trainer.pretrained_encoder_tensors != 371 or trainer.pretrained_stem_tensors != 2:
        raise RuntimeError("incomplete public Primus-M initialization")

    adapter_output = network.local_stem_refinement[4].weight
    if torch.count_nonzero(adapter_output).item() != 0:
        raise RuntimeError("Primus-M local refinement is not zero initialized")
    presence_output = network.token_presence_head.weight
    if torch.count_nonzero(presence_output).item() != 0:
        raise RuntimeError("Primus-M token-presence head is not zero initialized")
    if not math.isclose(network.local_gate_amplitude, 0.25):
        raise RuntimeError("unexpected Primus-M local gate amplitude")
    if not math.isclose(trainer.token_presence_loss_weight, 0.05):
        raise RuntimeError("unexpected Primus-M token-presence loss weight")
    encoder_parameter = next(network.eva.parameters())
    encoder_before = encoder_parameter.detach().clone()
    adapter_before = adapter_output.detach().clone()
    presence_before = presence_output.detach().clone()

    train_loader = validation_loader = None
    try:
        train_loader, validation_loader = trainer.get_dataloaders()
        trainer.network.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = next(train_loader)
        result = trainer.train_step(batch)
        loss = float(result["loss"])
        torch.cuda.synchronize()
        if not math.isfinite(loss):
            raise RuntimeError(f"non-finite Primus-M refinement loss: {loss}")
        if torch.equal(encoder_before, encoder_parameter.detach()):
            raise RuntimeError("pretrained encoder did not update")
        if torch.equal(adapter_before, adapter_output.detach()):
            raise RuntimeError("local refinement did not update")
        if torch.equal(presence_before, presence_output.detach()):
            raise RuntimeError("token-presence head did not update")
        segmentation_loss = float(result["segmentation_loss"])
        token_presence_loss = float(result["token_presence_loss"])
        if not math.isfinite(segmentation_loss) or not math.isfinite(
            token_presence_loss
        ):
            raise RuntimeError("non-finite Primus-M component loss")
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        trainer.network.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=True):
            inference_output = trainer.network(batch["data"].to("cuda"))
        if not isinstance(inference_output, torch.Tensor):
            raise RuntimeError("Primus-M inference must return one segmentation tensor")
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        batch_shape = list(batch["data"].shape)
    finally:
        if train_loader is not None:
            finish_loader(train_loader)
        if validation_loader is not None:
            finish_loader(validation_loader)

    print(
        json.dumps(
            {
                "model": "Primus-M Local-Refinement",
                "fold": args.fold,
                "train_cases": len(train_keys),
                "validation_cases": len(validation_keys),
                "split_sha256": split_hash,
                "train_validation_overlap": 0,
                "raw_native_input": True,
                "single_model": True,
                "epochs_from_zero": trainer.num_epochs,
                "loss": loss,
                "segmentation_loss": segmentation_loss,
                "token_presence_loss": token_presence_loss,
                "token_presence_loss_weight": trainer.token_presence_loss_weight,
                "batch_shape": batch_shape,
                "peak_cuda_allocated_gib": peak_allocated,
                "peak_cuda_reserved_gib": peak_reserved,
                "status": "ok",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
