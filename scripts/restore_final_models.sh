#!/usr/bin/env bash

# Restore the immutable final SWA checkpoints from the Grand Challenge Model archive.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace=${ISLES26_WORKSPACE:-$(cd -- "$script_dir/.." && pwd)}
archive=${ISLES26_MODEL_ARCHIVE:-$workspace/submission_artifacts/model-postprocess.tar.gz}
output_dir=${ISLES26_FINAL_MODELS:-$workspace/outputs/final_full_models}

python3 - "$archive" "$output_dir" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

archive = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
expected = {
    "e1/fold_all/checkpoint_final.pth": (
        "checkpoint_E1_swa.pth",
        "b974c29405011d4ddcb1850544c6d6fc054cc39544d3ada99b4b2808db88647b",
    ),
    "v15/fold_all/checkpoint_final.pth": (
        "checkpoint_V15_swa_e200_e250_e300.pth",
        "98429330ea0e800d0ce9d38b446494343fdaeba04c7e83caa6a50ff032dc05d2",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if not archive.is_file():
    raise FileNotFoundError(archive)
output_dir.mkdir(parents=True, exist_ok=True)

with tarfile.open(archive, "r:gz") as bundle, tempfile.TemporaryDirectory(
    prefix="isles26-restore-"
) as temporary:
    names = {name.removeprefix("./"): name for name in bundle.getnames()}
    for member, (filename, wanted) in expected.items():
        destination = output_dir / filename
        if destination.exists():
            actual = sha256(destination)
            if actual != wanted:
                raise RuntimeError(f"refusing to overwrite mismatched {destination}")
            print(f"{destination}: already verified")
            continue
        source = bundle.extractfile(names[member])
        if source is None:
            raise RuntimeError(f"archive member is not a file: {member}")
        staged = Path(temporary) / filename
        digest = hashlib.sha256()
        with staged.open("wb") as target:
            while block := source.read(8 * 1024 * 1024):
                digest.update(block)
                target.write(block)
        actual = digest.hexdigest()
        if actual != wanted:
            raise RuntimeError(f"{member}: {actual} != {wanted}")
        shutil.move(staged, destination)
        print(f"restored {destination}")
PY
