#!/usr/bin/env bash
set -euo pipefail

workspace="${ISLES26_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify() {
  local expected="$1"
  local relative="$2"
  local path="$workspace/$relative"
  if [[ ! -f "$path" ]]; then
    echo "missing required reproducibility artifact: $relative" >&2
    exit 1
  fi
  local actual
  actual="$(hash_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "$relative hash mismatch: $actual != $expected" >&2
    exit 1
  fi
  echo "$relative: OK"
}

verify 62d55b2fe862961ec598ebe28ec8c9ae17711359d7e60475ef88e3785bcdeacc atlas3_training_raw.tar.gz
verify da10108f65fdff2954c7f68f9e69456a5d9bb8f78b28512e3714656ba2bd9885 versions/V1/splits_final.json
verify 7a847af785635335c00e711d16ff4d225d86ecd5992b14c059df2b520e3ee933 external_models/ResEncL-OpenMind-MAE/checkpoint_final.pth
verify b866ac5f61d7e90d3a6cbb00a759ffc9d73beb5e63baa6b3cd654671ebc9a552 external_models/PrimusM-OpenMind-MAE/checkpoint_final.pth
verify 3a87ff03f71e4d05eb4e57e7bb4bf43716cf599612f9b0599c465ab07fff6484 submission_artifacts/isles26-e1-v15-swa-postprocess-fixed.tar.gz
verify 06c159dce3059f319f916d264c78b2b5e76e29456f91f907077c50520446ffd6 submission_artifacts/model-postprocess.tar.gz

python3 - "$workspace/submission_artifacts/model-postprocess.tar.gz" <<'PY'
import hashlib
import json
import sys
import tarfile

expected = {
    "e1/fold_all/checkpoint_final.pth": "b974c29405011d4ddcb1850544c6d6fc054cc39544d3ada99b4b2808db88647b",
    "v15/fold_all/checkpoint_final.pth": "98429330ea0e800d0ce9d38b446494343fdaeba04c7e83caa6a50ff032dc05d2",
}
with tarfile.open(sys.argv[1], "r:gz") as archive:
    names = set(archive.getnames())
    manifest_name = next(
        name for name in names if name.removeprefix("./") == "model_manifest.json"
    )
    manifest = json.load(archive.extractfile(manifest_name))
    if manifest["threshold"] != 0.5 or "0.002 ml" not in manifest["postprocessing"]:
        raise RuntimeError(f"unexpected final model manifest: {manifest}")
    for relative, wanted in expected.items():
        name = next(name for name in (relative, f"./{relative}") if name in names)
        digest = hashlib.sha256()
        payload = archive.extractfile(name)
        for block in iter(lambda: payload.read(8 * 1024 * 1024), b""):
            digest.update(block)
        actual = digest.hexdigest()
        if actual != wanted:
            raise RuntimeError(f"{relative}: {actual} != {wanted}")
        print(f"embedded {relative}: OK")
PY

echo "Final release reproducibility verification passed."
