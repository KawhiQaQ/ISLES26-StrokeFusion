#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="${ISLES26_WORKSPACE:-$(cd -- "$script_dir/.." && pwd)}"
raw_root="$workspace/data/raw/ATLAS3_Training_Raw"
version_dir="$workspace/versions/V1"
resenc_plan="$workspace/versions/V2/nnUNetResEncUNetLPlans.json"
v15_plan="$workspace/versions/V15/nnUNetResEncUNetLPlansV15.json"

conda_exe="${ISLES_CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" && -x /opt/conda/bin/conda ]]; then conda_exe=/opt/conda/bin/conda; fi
if [[ -z "$conda_exe" ]]; then echo "conda executable not found" >&2; exit 1; fi
source "$("$conda_exe" info --base)/etc/profile.d/conda.sh"
conda activate "${ISLES_CONDA_ENV:-isles26}"

export nnUNet_raw="${nnUNet_raw:-$workspace/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$workspace/.cache/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$workspace/outputs/nnUNet_results}"
export TORCH_HOME="${TORCH_HOME:-$workspace/.cache/torch}"
export MPLCONFIGDIR="$workspace/.cache/matplotlib"
export OMP_NUM_THREADS=8
# Keep native-space NIfTI/probability export conservative. Eight exporters
# exhausted the multiprocessing resource tracker midway through fold0.
export nnUNet_def_n_proc="${nnUNet_def_n_proc:-2}"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$version_dir"

action="${1:-}"
fold="${2:-0}"

case "$action" in
  setup)
    python "$workspace/scripts/prepare_nnunet_dataset.py" \
      "$raw_root" "$nnUNet_raw" --dataset-id 26 --dataset-name ISLES26
    if [[ ! -f "$version_dir/splits_final.json" ]]; then
      python "$workspace/scripts/build_v1_folds.py" "$raw_root" "$version_dir"
    fi
    ;;
  plan)
    nnUNetv2_plan_and_preprocess -d 26 -npfp 8 --no_pp --no_pbar
    install -m 0644 "$resenc_plan" \
      "$nnUNet_preprocessed/Dataset026_ISLES26/nnUNetResEncUNetLPlans.json"
    install -m 0644 "$v15_plan" \
      "$nnUNet_preprocessed/Dataset026_ISLES26/nnUNetResEncUNetLPlansV15.json"
    cp "$version_dir/splits_final.json" \
      "$nnUNet_preprocessed/Dataset026_ISLES26/splits_final.json"
    ;;
  preprocess)
    nnUNetv2_preprocess -d 26 -c 3d_fullres -np 6 --no_pbar
    install -m 0644 "$resenc_plan" \
      "$nnUNet_preprocessed/Dataset026_ISLES26/nnUNetResEncUNetLPlans.json"
    install -m 0644 "$v15_plan" \
      "$nnUNet_preprocessed/Dataset026_ISLES26/nnUNetResEncUNetLPlansV15.json"
    cp "$version_dir/splits_final.json" \
      "$nnUNet_preprocessed/Dataset026_ISLES26/splits_final.json"
    ;;
  smoke)
    python "$workspace/scripts/smoke_v1.py" --fold "$fold"
    ;;
  train)
    nnUNetv2_train 26 3d_fullres "$fold" \
      -tr nnUNetTrainer_250epochs --npz
    ;;
  validate)
    nnUNetv2_train 26 3d_fullres "$fold" \
      -tr nnUNetTrainer_250epochs --val --npz
    ;;
  evaluate)
    prediction_dir="$nnUNet_results/Dataset026_ISLES26/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_${fold}/validation"
    python "$workspace/scripts/evaluate_isles26.py" \
      "$version_dir/manifest_v1.csv" "$prediction_dir" \
      "$workspace/outputs/V1/fold${fold}/metrics" --fold "$fold" \
      --label-dir "$nnUNet_raw/Dataset026_ISLES26/labelsTr"
    ;;
  *)
    echo "usage: $0 {setup|plan|preprocess|smoke|train|validate|evaluate} [fold]" >&2
    exit 2
    ;;
esac
