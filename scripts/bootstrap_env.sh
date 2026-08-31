#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-isles26}"
CONDA_EXE_PATH="${ISLES_CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_EXE_PATH" && -x /opt/conda/bin/conda ]]; then
  CONDA_EXE_PATH=/opt/conda/bin/conda
fi

if [[ -z "$CONDA_EXE_PATH" || ! -x "$CONDA_EXE_PATH" ]]; then
  echo "Conda executable not found: $CONDA_EXE_PATH" >&2
  exit 1
fi

if "$CONDA_EXE_PATH" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  "$CONDA_EXE_PATH" env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
else
  "$CONDA_EXE_PATH" env create -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml"
fi

PIP_NO_CACHE_DIR=1 "$CONDA_EXE_PATH" run -n "$ENV_NAME" \
  python -m pip install -r "$PROJECT_DIR/requirements-core.txt"

echo "Environment ready. Activate with: conda activate $ENV_NAME"
