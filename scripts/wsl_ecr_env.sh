#!/usr/bin/env bash
# Activate the WSL conda env "ecr" and set CUDA library paths for PyG/torch_sparse.
# Usage: source scripts/wsl_ecr_env.sh
#        (from /mnt/d/ECR-official-main)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ecr

SITE="$CONDA_PREFIX/lib/python3.8/site-packages"
EXTRA_LIBS=(
  "$SITE/nvidia/cusparse/lib"
  "$SITE/torch/lib"
  "/usr/lib/wsl/lib"
)
for d in "${EXTRA_LIBS[@]}"; do
  if [[ -d "$d" ]]; then
    export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$REPO" || return 1
echo "ecr env ready at $REPO (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
