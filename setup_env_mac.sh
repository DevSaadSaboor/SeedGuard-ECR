#!/usr/bin/env bash
# UNOFFICIAL CPU-only environment for macOS (Apple Silicon), for evaluating checkpoints and debugging on a laptop.
# It is NOT the paper's environment: PyTorch 2.4 (CPU) instead of 1.8.1 + CUDA 11.1, and tokenizers 0.11.6 instead of
# 0.10.3. Train and produce reported numbers with setup_env.sh on a Linux GPU machine.
#   bash setup_env_mac.sh      # creates .venv/ in this folder; needs Python 3.9 (macOS ships it as /usr/bin/python3)
set -euo pipefail
cd "$(dirname "$0")"
PY_BIN="${PY_BIN:-/usr/bin/python3}"

if [[ "$("$PY_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != "3.9" ]]; then
  echo "need Python 3.9 (the pinned macOS builds are for 3.9); set PY_BIN=/path/to/python3.9" >&2
  exit 1
fi

"$PY_BIN" -m venv .venv
P=.venv/bin/python
"$P" -m pip install -q --upgrade pip
"$P" -m pip install torch==2.4.0 "numpy<2"
"$P" -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
"$P" -m pip install --no-deps transformers==4.15.0
"$P" -m pip install tokenizers==0.11.6 "huggingface-hub>=0.1,<0.17" filelock packaging pyyaml regex requests sacremoses \
  torch-geometric==2.0.1 accelerate==0.8.0 loguru==0.6.0 einops==0.4.1 nltk==3.7 scikit-learn tqdm wandb==0.13.10 \
  scipy pandas jinja2 pyparsing gdown==5.2.0

SITE="$("$P" -c 'import site; print(site.getsitepackages()[0])')"
# transformers 4.15 accepts tokenizers <0.11, but 0.11.6 is the oldest version with an Apple Silicon build.
# For DialoGPT and RoBERTa it produces the same tokens.
sed -i '' 's/"tokenizers": "tokenizers>=0.10.1,<0.11"/"tokenizers": "tokenizers>=0.10.1,<0.12"/' \
  "$SITE/transformers/dependency_versions_table.py"

# Two shims so the authors' code (written for PyTorch 1.8.1 on CUDA) runs here without editing it.
cat > "$SITE/sitecustomize.py" <<'EOF'
# Mac CPU environment only: lets the ECR code (written for PyTorch 1.8.1) run on PyTorch 2.x without editing it.
# torch.set_deterministic() was renamed torch.use_deterministic_algorithms() in PyTorch 1.10; config.py calls the old name.
try:
    import torch

    if not hasattr(torch, "set_deterministic"):
        torch.set_deterministic = torch.use_deterministic_algorithms
    # dataset_dbpedia.py moves the KG edges with a hard-coded .cuda(); without a GPU, keep them on the CPU
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *args, **kwargs: self
except ImportError:
    pass
EOF

bash fetch_backbones.sh

echo
echo "Mac CPU environment ready in .venv/ (unofficial). Next:"
echo "  source .venv/bin/activate"
echo "  bash download_data.sh"
