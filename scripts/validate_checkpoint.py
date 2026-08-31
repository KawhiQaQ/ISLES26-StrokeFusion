#!/usr/bin/env python3
"""Run native-space whole-case validation for an explicit nnU-Net checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from nnunetv2.run.run_training import get_trainer_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--dataset", default="26")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument(
        "--inference-only-weights",
        action="store_true",
        help="Load network weights without requiring optimizer resume state.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_folder = args.output_folder.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output_folder == Path("/") or output_folder == Path.home():
        raise ValueError(f"unsafe output folder: {output_folder}")

    trainer = get_trainer_from_args(
        args.dataset,
        args.configuration,
        args.fold,
        args.trainer,
        args.plans,
        False,
        device=torch.device("cuda"),
    )
    if args.inference_only_weights:
        trainer.initialize()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        network = (
            trainer._network_module()
            if hasattr(trainer, "_network_module")
            else trainer.network
        )
        state = {}
        expected = network.state_dict()
        for key, value in payload["network_weights"].items():
            normalized = key
            if normalized not in expected and normalized.startswith("module."):
                normalized = normalized[7:]
            state[normalized] = value
        network.load_state_dict(state, strict=True)
        trainer.current_epoch = int(payload.get("current_epoch", 0))
        if "inference_allowed_mirroring_axes" in payload:
            trainer.inference_allowed_mirroring_axes = payload[
                "inference_allowed_mirroring_axes"
            ]
    else:
        trainer.load_checkpoint(str(checkpoint))
    # perform_actual_validation always writes to <output_folder>/validation.
    # Redirect only its disposable predictions; model inputs and frozen split
    # remain those encoded in the checkpoint and installed nnU-Net plans.
    trainer.output_folder = str(output_folder)
    trainer.perform_actual_validation(args.save_probabilities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
