#!/usr/bin/env bash
# Create the pinned conda environment for the official ECR code (Linux + NVIDIA GPU, driver >= 455).
#   bash setup_env.sh                  # environment named "ecr"
#   ENV_NAME=ecr2 bash setup_env.sh    # another name
set -euo pipefail
cd "$(dirname "$0")"
ENV_NAME="${ENV_NAME:-ecr}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniforge first: https://github.com/conda-forge/miniforge" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warning: the CUDA 11.1 wheels below exist only for Linux; this will fail on $(uname -s)." >&2
fi

conda create -y -n "$ENV_NAME" -c conda-forge python=3.8.13 pip
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
echo "using $PY"

"$PY" -m pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
"$PY" -m pip install torch-scatter==2.0.8 torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.0+cu111.html
"$PY" -m pip install -r requirements.txt

# DialoGPT-small and RoBERTa-base backbones into src_emo/save/ (see fetch_backbones.sh)
bash fetch_backbones.sh

echo
echo "Environment '$ENV_NAME' is ready. Next:"
echo "  conda activate $ENV_NAME"
echo "  bash download_data.sh"
echo "  python check_env.py"
