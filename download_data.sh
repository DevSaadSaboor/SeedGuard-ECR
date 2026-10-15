#!/usr/bin/env bash
# Download the authors' data and checkpoints from Google Drive (links in README.md) and unpack them
# where the README says: emo_data.zip -> src_emo/data/emo_data/, ckpt.zip -> src_emo/data/saved/.
# Run inside the conda environment (needs gdown and unzip).
#   bash download_data.sh          # both
#   bash download_data.sh data     # only emo_data.zip
#   bash download_data.sh ckpt     # only ckpt.zip
set -euo pipefail
cd "$(dirname "$0")/src_emo"
WHAT="${1:-all}"

fetch() {  # fetch <drive file id> <zip name> <target folder>
  local id="$1" zip="$2" target="$3" tmp src
  tmp="$(mktemp -d)"
  gdown "https://drive.google.com/uc?id=$id" -O "$tmp/$zip"
  unzip -q "$tmp/$zip" -d "$tmp/x"
  rm -rf "$tmp/x/__MACOSX"
  src="$tmp/x"
  # the README says to move all files into the folder, so flatten a single top-level folder
  if [[ "$(ls -A "$src" | wc -l)" -eq 1 && -d "$src/$(ls -A "$src")" ]]; then
    src="$src/$(ls -A "$src")"
  fi
  mkdir -p "$target"
  cp -R "$src"/. "$target"/
  rm -rf "$tmp"
  echo "$zip -> src_emo/$target"
}

if [[ "$WHAT" == "data" || "$WHAT" == "all" ]]; then
  fetch 1fb9kDo8uSRLlwc5c4nUw8DZHR5XOY_l_ emo_data.zip data/emo_data
fi
if [[ "$WHAT" == "ckpt" || "$WHAT" == "all" ]]; then
  fetch 1uBtcqbQByVrrJ1hEwk2dvsAOxuvEgE19 ckpt.zip data/saved
fi
