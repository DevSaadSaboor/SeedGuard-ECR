#!/usr/bin/env bash
# Option B: point this SeedGuard repo at a full ECR install inside WSL.
# Does not copy multi‑GB assets; creates symlinks. Safe for GitHub (targets are gitignored).
#
# Usage (from WSL):
#   bash scripts/link_wsl_ecr_assets.sh
#   bash scripts/link_wsl_ecr_assets.sh /home/saad/ECR
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ECR_SRC="${1:-$HOME/ECR}"

if [[ ! -d "$ECR_SRC/src_emo" ]]; then
  echo "ECR install not found at: $ECR_SRC" >&2
  echo "Pass the path: bash scripts/link_wsl_ecr_assets.sh /path/to/ECR" >&2
  exit 1
fi

link_dir() {
  local dest="$1" src="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -L "$dest" ]]; then
    rm -f "$dest"
  elif [[ -d "$dest" ]]; then
    # Keep a tiny backup of placeholder dirs (note.md only, etc.)
    local bak="${dest}.prelink_backup_$(date +%Y%m%d%H%M%S)"
    mv "$dest" "$bak"
    echo "moved $dest -> $bak"
  elif [[ -e "$dest" ]]; then
    echo "refusing to replace non-directory: $dest" >&2
    exit 1
  fi
  ln -s "$src" "$dest"
  echo "linked $dest -> $src"
}

link_file() {
  local dest="$1" src="$2"
  if [[ -e "$dest" || -L "$dest" ]]; then
    rm -f "$dest"
  fi
  ln -s "$src" "$dest"
  echo "linked $dest -> $src"
}

echo "REPO=$REPO"
echo "ECR_SRC=$ECR_SRC"

# Whole-directory assets (downloaded)
link_dir "$REPO/src_emo/data/emo_data" "$ECR_SRC/src_emo/data/emo_data"
link_dir "$REPO/src_emo/data/saved" "$ECR_SRC/src_emo/data/saved"

mkdir -p "$REPO/src_emo/save"
link_dir "$REPO/src_emo/save/dialogpt" "$ECR_SRC/src_emo/save/dialogpt"
link_dir "$REPO/src_emo/save/roberta" "$ECR_SRC/src_emo/save/roberta"

# evaluate_rec.py writes under save/redial_rec/
if [[ -d "$ECR_SRC/src_emo/save/redial_rec" ]]; then
  link_dir "$REPO/src_emo/save/redial_rec" "$ECR_SRC/src_emo/save/redial_rec"
else
  mkdir -p "$REPO/src_emo/save/redial_rec"
fi

# Keep tracked process/merge scripts; link all other data files from ECR.
mkdir -p "$REPO/src_emo/data/redial" "$REPO/src_emo/data/redial_gen"
for src in "$ECR_SRC/src_emo/data/redial"/*; do
  [[ -e "$src" ]] || continue
  base="$(basename "$src")"
  case "$base" in
    process.py|process_mask.py|process_empthetic.py) continue ;;
  esac
  link_file "$REPO/src_emo/data/redial/$base" "$src"
done

for src in "$ECR_SRC/src_emo/data/redial_gen"/*; do
  [[ -e "$src" ]] || continue
  base="$(basename "$src")"
  case "$base" in
    merge.py|merge_rec.py) continue ;;
  esac
  link_file "$REPO/src_emo/data/redial_gen/$base" "$src"
done

echo
echo "Checking required paths..."
need=(
  "$REPO/src_emo/save/dialogpt/config.json"
  "$REPO/src_emo/save/roberta/config.json"
  "$REPO/src_emo/data/emo_data/entity2id.json"
  "$REPO/src_emo/data/saved/rec"
  "$REPO/src_emo/data/redial_gen/test_data_processed.jsonl"
  "$REPO/src_emo/data/redial_gen/train_data_processed.jsonl"
  "$REPO/src_emo/data/redial_gen/valid_data_processed.jsonl"
)
ok=1
for p in "${need[@]}"; do
  if [[ -e "$p" ]]; then
    echo "  OK  $p"
  else
    echo "  MISSING  $p"
    ok=0
  fi
done

if [[ "$ok" -ne 1 ]]; then
  echo "Some assets are missing in $ECR_SRC" >&2
  exit 1
fi

echo
echo "Done. Next (in WSL, from repo root):"
echo "  source ~/miniconda3/etc/profile.d/conda.sh && conda activate ecr"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo "  python make_quick_eval_set.py --seed 0"
echo "  python run_seedguard.py --smoke 1x1 --skip-dataset --per-device-train-batch-size 4 --per-device-eval-batch-size 8"
