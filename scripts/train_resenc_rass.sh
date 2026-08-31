#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="${ISLES26_WORKSPACE:-$(cd -- "$script_dir/.." && pwd)}"
split_dir="$workspace/data/derived/center_grouped_folds"
resenc_plan="$workspace/plans/resenc_l.json"

conda_exe="${ISLES_CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" && -x /opt/conda/bin/conda ]]; then conda_exe=/opt/conda/bin/conda; fi
if [[ -z "$conda_exe" ]]; then echo "conda executable not found" >&2; exit 1; fi
source "$("$conda_exe" info --base)/etc/profile.d/conda.sh"
conda activate "${ISLES_CONDA_ENV:-isles26}"

export ISLES26_WORKSPACE="$workspace"
export nnUNet_raw="${nnUNet_raw:-$workspace/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$workspace/.cache/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$workspace/outputs/nnUNet_results}"
export TORCH_HOME="${TORCH_HOME:-$workspace/.cache/torch}"
export MPLCONFIGDIR="$workspace/.cache/matplotlib"
export OMP_NUM_THREADS=8
export nnUNet_def_n_proc="${nnUNet_def_n_proc:-2}"
# Pin the actual augmentation-pool setting. V11 demonstrated that
# nnUNet_def_n_proc does not control these workers.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export nnUNet_compile=false

dataset_name=Dataset026_ISLES26
preprocessed_dataset="$nnUNet_preprocessed/$dataset_name"
expected_split_sha256=da10108f65fdff2954c7f68f9e69456a5d9bb8f78b28512e3714656ba2bd9885
expected_pretrained_sha256=7a847af785635335c00e711d16ff4d225d86ecd5992b14c059df2b520e3ee933
pretrained_checkpoint="$workspace/external_models/ResEncL-OpenMind-MAE/checkpoint_final.pth"
plans_name=nnUNetResEncUNetLPlans
trainer_name=nnUNetTrainerV12RASS
extension_dir="$workspace/nnunet_extensions"
trainer_destination_dir="$(python - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).resolve().parent / "training" / "nnUNetTrainer")
PY
)"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

verify_hash() {
  local file="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label hash changed: $actual" >&2
    exit 1
  fi
}

install_inputs() {
  verify_hash "$split_dir/splits_final.json" "$expected_split_sha256" "frozen split"
  cp "$split_dir/splits_final.json" "$preprocessed_dataset/splits_final.json"
  verify_hash "$preprocessed_dataset/splits_final.json" "$expected_split_sha256" "installed split"
  verify_hash "$pretrained_checkpoint" "$expected_pretrained_sha256" "OpenMind checkpoint"
  cmp --silent "$resenc_plan" "$preprocessed_dataset/${plans_name}.json" || {
    echo "ResEnc-L plans differ from the frozen repository copy" >&2
    exit 1
  }
  python -m py_compile \
    "$extension_dir/nnUNetTrainerV3UFL.py" \
    "$extension_dir/nnUNetTrainerV9OpenMindMAE.py" \
    "$extension_dir/${trainer_name}.py"
  for source in nnUNetTrainerV3UFL.py nnUNetTrainerV9OpenMindMAE.py "${trainer_name}.py"; do
    install -m 0644 "$extension_dir/$source" "$trainer_destination_dir/$source"
  done
}

action="${1:-}"
fold="${2:-0}"

case "$action" in
  smoke)
    install_inputs
    python "$workspace/scripts/smoke_resenc_rass.py" --fold "$fold"
    ;;
  train)
    install_inputs
    nnUNetv2_train 26 3d_fullres "$fold" -tr "$trainer_name" -p "$plans_name" --npz
    ;;
  resume)
    install_inputs
    nnUNetv2_train 26 3d_fullres "$fold" -tr "$trainer_name" -p "$plans_name" --npz --c
    ;;
  validate)
    install_inputs
    nnUNetv2_train 26 3d_fullres "$fold" -tr "$trainer_name" -p "$plans_name" --val --npz
    ;;
  *)
    echo "usage: $0 {smoke|train|resume|validate} [fold]" >&2
    exit 2
    ;;
esac
