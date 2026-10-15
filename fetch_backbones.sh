#!/usr/bin/env bash
# Download the backbones the training scripts load from src_emo/save/dialogpt/ and src_emo/save/roberta/.
# The authors don't ship or document these folders. ECR builds on UniCRS, which uses microsoft/DialoGPT-small and
# roberta-base, and the authors' generator checkpoint confirms DialoGPT-small (12 layers, 768 dims).
# transformers 4.15 can no longer follow the Hugging Face Hub's download redirects, so files are fetched directly.
# Safe to re-run: finished files are skipped and interrupted downloads resume.
set -euo pipefail
cd "$(dirname "$0")"

fetch_hf() {  # fetch_hf <hub repo> <folder> <files...>
  local repo="$1" dir="$2" f try ok
  shift 2
  mkdir -p "$dir"
  for f in "$@"; do
    [[ -s "$dir/$f" ]] && continue
    ok=0
    for try in 1 2 3 4 5; do
      if curl -fsSL -C - -o "$dir/$f.part" "https://huggingface.co/$repo/resolve/main/$f"; then ok=1; break; fi
      echo "  retrying $f ($try/5)" >&2
      sleep 3
    done
    [[ $ok == 1 ]] || { echo "failed to download $repo/$f" >&2; return 1; }
    mv "$dir/$f.part" "$dir/$f"  # only complete files get their real name
  done
  echo "$repo -> $dir"
}

fetch_hf microsoft/DialoGPT-small src_emo/save/dialogpt config.json pytorch_model.bin vocab.json merges.txt tokenizer_config.json
fetch_hf FacebookAI/roberta-base src_emo/save/roberta config.json pytorch_model.bin vocab.json merges.txt tokenizer_config.json
