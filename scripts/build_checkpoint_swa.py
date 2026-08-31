#!/usr/bin/env python3
"""Build one inference checkpoint by uniform streaming weight averaging."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise ValueError("SWA requires at least two checkpoints")
    output = args.output.resolve()
    sources = [path.resolve() for path in args.checkpoints]
    if output in sources:
        raise ValueError("output must differ from every source checkpoint")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    payload = torch.load(sources[0], map_location="cpu", weights_only=False)
    average = payload["network_weights"]
    expected_keys = tuple(average)
    source_hashes = [sha256(sources[0])]

    for count, source in enumerate(sources[1:], start=1):
        candidate_payload = torch.load(
            source, map_location="cpu", weights_only=False
        )
        candidate = candidate_payload["network_weights"]
        if tuple(candidate) != expected_keys:
            raise RuntimeError(f"network key mismatch: {source}")
        for key in expected_keys:
            left = average[key]
            right = candidate[key]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError(f"tensor mismatch for {key}: {source}")
            if left.is_floating_point() or left.is_complex():
                left.mul_(count / (count + 1.0)).add_(
                    right, alpha=1.0 / (count + 1.0)
                )
            elif not torch.equal(left, right):
                raise RuntimeError(f"non-floating tensor changed for {key}: {source}")
        payload["current_epoch"] = candidate_payload.get(
            "current_epoch", payload.get("current_epoch")
        )
        source_hashes.append(sha256(source))
        del candidate, candidate_payload

    payload["optimizer_state"] = {}
    payload["grad_scaler_state"] = None
    payload["checkpoint_swa"] = {
        "method": "uniform_weight_average",
        "sources": [str(path) for path in sources],
        "source_sha256": source_hashes,
        "num_checkpoints": len(sources),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"output={output} checkpoints={len(sources)} "
        f"tensors={len(expected_keys)} sha256={sha256(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
