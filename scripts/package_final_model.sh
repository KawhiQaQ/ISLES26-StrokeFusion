#!/usr/bin/env bash

# Rebuild the Grand Challenge Model resource for the frozen full-data ensemble.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace=${ISLES26_WORKSPACE:-$(cd -- "$script_dir/.." && pwd)}
final_models=${ISLES26_FINAL_MODELS:-$workspace/outputs/final_full_models}
output=${1:-$workspace/submission_artifacts/model-rebuilt.tar.gz}
template_dir=${ISLES26_DOCKER_TEMPLATE:-$workspace/docker}

e1_source=$final_models/checkpoint_E1_swa.pth
v15_source=$final_models/checkpoint_V15_swa_e200_e250_e300.pth
e1_expected=b974c29405011d4ddcb1850544c6d6fc054cc39544d3ada99b4b2808db88647b
v15_expected=98429330ea0e800d0ce9d38b446494343fdaeba04c7e83caa6a50ff032dc05d2

if [[ -e "$output" ]]; then
    printf 'Refusing to overwrite existing output: %s\n' "$output" >&2
    exit 2
fi

for required in \
    "$e1_source" \
    "$v15_source" \
    "$workspace/reproducibility/model_metadata/e1/plans.json" \
    "$workspace/reproducibility/model_metadata/e1/dataset.json" \
    "$workspace/reproducibility/model_metadata/v15/plans.json" \
    "$workspace/reproducibility/model_metadata/v15/dataset.json" \
    "$template_dir/model/model_manifest.json"
do
    if [[ ! -f "$required" ]]; then
        printf 'Required release input is missing: %s\n' "$required" >&2
        exit 2
    fi
done

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

verify_hash() {
    local expected=$1
    local path=$2
    local actual
    actual=$(hash_file "$path")
    if [[ "$actual" != "$expected" ]]; then
        printf 'SHA-256 mismatch for %s: %s != %s\n' "$path" "$actual" "$expected" >&2
        exit 3
    fi
}

verify_hash "$e1_expected" "$e1_source"
verify_hash "$v15_expected" "$v15_source"

stage=$(mktemp -d "${TMPDIR:-/tmp}/isles26-final-model.XXXXXX")
trap 'find "$stage" -depth -delete 2>/dev/null || true' EXIT
mkdir -p "$stage/e1/fold_all" "$stage/v15/fold_all" "$stage/a_tarball_subdirectory"

install -m 0644 "$workspace/reproducibility/model_metadata/e1/plans.json" "$stage/e1/plans.json"
install -m 0644 "$workspace/reproducibility/model_metadata/e1/dataset.json" "$stage/e1/dataset.json"
install -m 0644 "$workspace/reproducibility/model_metadata/v15/plans.json" "$stage/v15/plans.json"
install -m 0644 "$workspace/reproducibility/model_metadata/v15/dataset.json" "$stage/v15/dataset.json"
install -m 0644 "$e1_source" "$stage/e1/fold_all/checkpoint_final.pth"
install -m 0644 "$v15_source" "$stage/v15/fold_all/checkpoint_final.pth"
install -m 0644 "$template_dir/model/model_manifest.json" "$stage/model_manifest.json"

mkdir -p "$(dirname -- "$output")"
tar -czf "$output" -C "$stage" .
printf 'Created %s\n' "$output"
printf 'SHA-256 %s\n' "$(hash_file "$output")"
