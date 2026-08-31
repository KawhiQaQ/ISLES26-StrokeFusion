#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="${ISLES26_WORKSPACE:-$(cd -- "$script_dir/.." && pwd)}"
full_results="${ISLES26_FULL_RESULTS:-$workspace/outputs/full_training}"
durable_models="$workspace/outputs/final_full_models"
log_dir="$workspace/logs/final_full"

conda_exe="${ISLES_CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" && -x /opt/conda/bin/conda ]]; then conda_exe=/opt/conda/bin/conda; fi
if [[ -z "$conda_exe" ]]; then echo "conda executable not found" >&2; exit 1; fi
source "$("$conda_exe" info --base)/etc/profile.d/conda.sh"
conda activate "${ISLES_CONDA_ENV:-isles26}"

export nnUNet_results="$full_results"
export nnUNet_n_proc_DA=4
export nnUNet_def_n_proc=2

mkdir -p "$full_results" "$durable_models" "$log_dir"

run_or_resume() {
  local runner="$1"
  local result_dir="$2"
  local stage="$3"
  if [[ -f "$result_dir/checkpoint_final.pth" ]]; then
    echo "$stage training already complete"
  elif [[ -f "$result_dir/checkpoint_latest.pth" ]]; then
    echo "$stage resuming from checkpoint_latest.pth"
    bash "$runner" resume all
  else
    echo "$stage starting from epoch 0"
    bash "$runner" train all
  fi
}

v12_dir="$full_results/Dataset026_ISLES26/nnUNetTrainerV12RASS__nnUNetResEncUNetLPlans__3d_fullres/fold_all"
run_or_resume "$workspace/scripts/run_v12.sh" "$v12_dir" V12
python "$workspace/scripts/build_checkpoint_swa.py" \
  "$v12_dir/checkpoint_E1_swa.pth" \
  "$v12_dir/checkpoint_epoch300.pth" \
  "$v12_dir/checkpoint_epoch350.pth" \
  "$v12_dir/checkpoint_epoch400.pth"
install -m 0644 "$v12_dir/checkpoint_E1_swa.pth" "$durable_models/checkpoint_E1_swa.pth"
sha256sum "$durable_models/checkpoint_E1_swa.pth" > "$durable_models/checkpoint_E1_swa.sha256"
echo "V12/E1 full-data artifact persisted"

v15_dir="$full_results/Dataset026_ISLES26/nnUNetTrainerV15NativeRefinement__nnUNetResEncUNetLPlansV15__3d_fullres/fold_all"
run_or_resume "$workspace/scripts/run_v15.sh" "$v15_dir" V15
python "$workspace/scripts/build_checkpoint_swa.py" \
  "$v15_dir/checkpoint_V15_swa_e200_e250_e300.pth" \
  "$v15_dir/checkpoint_epoch200.pth" \
  "$v15_dir/checkpoint_epoch250.pth" \
  "$v15_dir/checkpoint_epoch300.pth"
install -m 0644 \
  "$v15_dir/checkpoint_V15_swa_e200_e250_e300.pth" \
  "$durable_models/checkpoint_V15_swa_e200_e250_e300.pth"
sha256sum \
  "$durable_models/checkpoint_V15_swa_e200_e250_e300.pth" \
  > "$durable_models/checkpoint_V15_swa_e200_e250_e300.sha256"
echo "V15-SWA full-data artifact persisted"

date -Is > "$durable_models/TRAINING_COMPLETE"
echo "FULL FINAL TRAINING COMPLETE"
