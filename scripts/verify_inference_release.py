#!/usr/bin/env python3
"""Verify the frozen StrokeFusion inference Model resource."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


ARCHIVE_SHA256 = "06c159dce3059f319f916d264c78b2b5e76e29456f91f907077c50520446ffd6"
EXPECTED_CHECKPOINTS = {
    "e1/fold_all/checkpoint_final.pth": (
        "b974c29405011d4ddcb1850544c6d6fc054cc39544d3ada99b4b2808db88647b"
    ),
    "v15/fold_all/checkpoint_final.pth": (
        "98429330ea0e800d0ce9d38b446494343fdaeba04c7e83caa6a50ff032dc05d2"
    ),
}
REQUIRED_MEMBERS = {
    "e1/dataset.json",
    "e1/plans.json",
    "v15/dataset.json",
    "v15/plans.json",
    "model_manifest.json",
    *EXPECTED_CHECKPOINTS,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_members(bundle: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in bundle.getmembers():
        normalized = member.name.removeprefix("./")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe archive member: {member.name}")
        if normalized:
            members[normalized] = member
    return members


def verify_archive(path: Path, *, verify_container_hash: bool = True) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify_container_hash:
        actual_archive_hash = sha256_file(path)
        if actual_archive_hash != ARCHIVE_SHA256:
            raise RuntimeError(
                f"model archive SHA-256 mismatch: {actual_archive_hash} != "
                f"{ARCHIVE_SHA256}"
            )

    with tarfile.open(path, "r:gz") as bundle:
        members = _normalized_members(bundle)
        missing = sorted(REQUIRED_MEMBERS - set(members))
        if missing:
            raise RuntimeError(f"model archive is missing: {missing}")

        manifest_file = bundle.extractfile(members["model_manifest.json"])
        if manifest_file is None:
            raise RuntimeError("model_manifest.json is not a regular file")
        manifest = json.load(manifest_file)
        if manifest.get("threshold") != 0.5:
            raise RuntimeError(f"unexpected segmentation threshold: {manifest}")
        postprocessing = str(manifest.get("postprocessing", "")).lower()
        if "0.002 ml" not in postprocessing or "0.65" not in postprocessing:
            raise RuntimeError(f"unexpected postprocessing contract: {manifest}")

        manifest_hashes = manifest.get("sha256", {})
        for relative, expected in EXPECTED_CHECKPOINTS.items():
            if manifest_hashes.get(relative) != expected:
                raise RuntimeError(f"manifest hash mismatch for {relative}")
            source = bundle.extractfile(members[relative])
            if source is None:
                raise RuntimeError(f"checkpoint is not a regular file: {relative}")
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
            actual = digest.hexdigest()
            if actual != expected:
                raise RuntimeError(f"{relative}: {actual} != {expected}")


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Verify the released StrokeFusion inference model bundle."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=workspace / "submission_artifacts" / "model-postprocess.tar.gz",
        help="path to model-postprocess.tar.gz",
    )
    parser.add_argument(
        "--allow-repacked-archive",
        action="store_true",
        help="verify embedded artifacts but do not require the published tarball hash",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_archive(
        args.archive,
        verify_container_hash=not args.allow_repacked_archive,
    )
    print(f"StrokeFusion inference model verified: {args.archive.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
