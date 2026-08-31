#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="${ISLES26_WORKSPACE:-$(cd -- "$script_dir/.." && pwd)}"
transformer_plan="$workspace/configs/primus_m_local_refinement_plans.json"
split_dir="$workspace/data/derived/center_grouped_folds"

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
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export nnUNet_n_proc_val="${nnUNet_n_proc_val:-2}"
export nnUNet_compile=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

dataset_name=Dataset026_ISLES26
preprocessed_dataset="$nnUNet_preprocessed/$dataset_name"
expected_split_sha256=da10108f65fdff2954c7f68f9e69456a5d9bb8f78b28512e3714656ba2bd9885
expected_plans_sha256=59dd27a82d75639614d225bb34c3acf1024b4f64b73f44388edf9f1c8fd74519
expected_pretrained_sha256=b866ac5f61d7e90d3a6cbb00a759ffc9d73beb5e63baa6b3cd654671ebc9a552
pretrained_checkpoint="$workspace/external_models/PrimusM-OpenMind-MAE/checkpoint_final.pth"
plans_name=nnUNetResEncUNetLPlansV15
trainer_name=nnUNetTrainerV15NativeRefinement
extension_dir="$workspace/strokefusion/trainers"
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
  install -m 0644 "$split_dir/splits_final.json" "$preprocessed_dataset/splits_final.json"
  verify_hash "$preprocessed_dataset/splits_final.json" "$expected_split_sha256" "installed split"
  verify_hash "$transformer_plan" "$expected_plans_sha256" "Primus-M plans"
  install -m 0644 "$transformer_plan" "$preprocessed_dataset/${plans_name}.json"
  verify_hash "$preprocessed_dataset/${plans_name}.json" "$expected_plans_sha256" "installed Primus-M plans"
  verify_hash "$pretrained_checkpoint" "$expected_pretrained_sha256" "Primus-M checkpoint"

  python -m py_compile \
    "$extension_dir/nnUNetTrainerV3UFL.py" \
    "$extension_dir/nnUNetTrainerV9OpenMindMAE.py" \
    "$extension_dir/nnUNetTrainerV12RASS.py" \
    "$extension_dir/nnUNetTrainerV14PrimusMAE.py" \
    "$extension_dir/${trainer_name}.py"
  for source in \
    nnUNetTrainerV3UFL.py \
    nnUNetTrainerV9OpenMindMAE.py \
    nnUNetTrainerV12RASS.py \
    nnUNetTrainerV14PrimusMAE.py \
    "${trainer_name}.py"; do
    install -m 0644 "$extension_dir/$source" "$trainer_destination_dir/$source"
  done
}

action="${1:-}"
fold="${2:-0}"

case "$action" in
  smoke)
    install_inputs
    python "$workspace/scripts/smoke_primus_local_refinement.py" --fold "$fold"
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
