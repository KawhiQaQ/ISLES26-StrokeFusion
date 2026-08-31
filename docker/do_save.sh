#!/usr/bin/env bash

# Build and export the two Grand Challenge upload resources.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd -- "$script_dir/.." && pwd)
image_tag=isles26-e1-v15-swa-final
image_output=${ISLES26_IMAGE_OUTPUT:-$workspace/submission_artifacts/isles26-e1-v15-swa-rebuilt.tar.gz}
model_output=${ISLES26_MODEL_OUTPUT:-$workspace/submission_artifacts/model-rebuilt.tar.gz}

for output in "$image_output" "$model_output"; do
    if [[ -e "$output" ]]; then
        printf 'Refusing to overwrite existing output: %s\n' "$output" >&2
        exit 2
    fi
done

bash "$script_dir/do_build.sh"
mkdir -p "$(dirname -- "$image_output")"
docker save "$image_tag" | gzip -c > "$image_output"
bash "$workspace/scripts/package_final_model.sh" "$model_output"

printf 'Created algorithm image: %s\n' "$image_output"
printf 'Created model resource: %s\n' "$model_output"
