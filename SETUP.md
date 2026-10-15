# Setup

This repo is the official code of *Towards Empathetic Conversational Recommender Systems* (Zhang et al.,
RecSys 2024) from [zxd-octopus/ECR](https://github.com/zxd-octopus/ECR) at commit `26e2342`, plus setup
files and the SeedGuard evaluation layer. ECR recommendation training/loss/architecture are preserved;
SeedGuard adds orchestration (`run_seedguard.py`, `seedguard/`), result collection, Precision@10, and
small provenance hooks in `train_rec.py` / `evaluate_rec.py`. Full experiment instructions:
[SEEDGUARD.md](SEEDGUARD.md).

## What you need

- Linux with an NVIDIA GPU and driver 455 or newer. The authors used a 24 GB GPU.
- conda (e.g. [Miniforge](https://github.com/conda-forge/miniforge)).
- About 1–2 GB of disk for the environment, plus the data downloads.

GPU compatibility: the authors pinned PyTorch 1.8.1 with CUDA 11.1, which targets GPUs up to compute capability 8.6
(e.g. RTX 30xx, A100, A40, A6000). Newer GPUs (RTX 40xx, L4/L40, H100) may not run it. `check_env.py` runs a
small GPU test that tells you. Upgrading PyTorch is possible, but results may then differ from the paper.

## Steps

```bash
git clone https://github.com/rHidden/ECR-official.git
cd ECR-official
bash setup_env.sh          # conda env "ecr": Python 3.8.13, PyTorch 1.8.1+cu111, pinned packages, backbones
conda activate ecr
bash download_data.sh      # authors' data (emo_data.zip) and checkpoints (ckpt.zip) from Google Drive
python check_env.py        # versions, GPU test, folders
```

Then follow the Quick-Start in [README.md](README.md), starting with `cd src_emo`.

## Unofficial: macOS (Apple Silicon), CPU only

For evaluating the released checkpoints, preprocessing data and debugging on a laptop. Not for training or for
numbers you report: it uses PyTorch 2.4 on the CPU instead of 1.8.1 on CUDA, so results can differ slightly.

```bash
bash setup_env_mac.sh       # .venv/ with Python 3.9 (macOS ships it as /usr/bin/python3)
source .venv/bin/activate
bash download_data.sh
```

What it changes, without editing the authors' code:

- **tokenizers 0.11.6** instead of 0.10.3, the oldest version with an Apple Silicon build. transformers 4.15's version
  check is relaxed to accept it. The tokens for DialoGPT and RoBERTa are identical.
- **A `sitecustomize.py` shim** in the environment:
  - It maps `torch.set_deterministic` (removed after PyTorch 1.8) to `torch.use_deterministic_algorithms`.
  - Without a GPU, it makes `.cuda()` a no-op, because `dataset_dbpedia.py` hard-codes `.cuda()` for the knowledge-graph edges.
- **accelerate 0.8** has no Apple-GPU (MPS) support, so everything runs on the CPU.

**It uses every CPU core and makes the laptop hard to use.** Evaluating the released recommender (`--test`) takes
about 15 minutes at full load on an M4 Pro. To keep the machine usable, run it throttled, which is slower:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 python train_rec.py ...
```

## Notes and assumptions

- **Backbones.** The scripts load DialoGPT and RoBERTa from `src_emo/save/dialogpt/` and `src_emo/save/roberta/`,
  which the authors don't ship or document. `fetch_backbones.sh` (run by both setup scripts) fills them with
  `microsoft/DialoGPT-small` and `roberta-base`, the models used by UniCRS, which ECR builds on. `ckpt.zip` holds
  no backbones, but its generator config (12 layers, 768 dims, 50,257 + 2 tokens) confirms DialoGPT-small.
  The files are downloaded directly, because transformers 4.15 can no longer follow the Hub's redirects.
- **Google Drive downloads** need gdown 5.x. Older versions fail on large files with "Access denied".
- **wandb** is off unless you pass `--use_wandb`.
- **Where trained models go.** `train_rec.py` appends a timestamp to `--output_dir` with no separator
  (`save/run1` becomes `save/run12026-10-07-05-27-49/`), with `best/` and `final/` inside. End the path with a
  slash (`--output_dir save/run1/`) to get one folder per run. The default, `data/saved/rec`, gets a timestamp too,
  so it doesn't overwrite the authors' released checkpoint, but it does clutter that folder.
- **Evaluate without training:** add `--test --prompt_encoder data/saved/rec/` to the README's full `train_rec.py`
  command. It scores the saved prompt encoder on the validation and test sets and trains nothing. Keep
  `--num_warmup_steps` in the command: the learning-rate scheduler is built even in test mode and crashes without it.
- **Downloads:** `emo_data.zip` is 111 MB and `ckpt.zip` 679 MB. Following the README's copy steps, the data
  takes about 2.5 GB on disk, and the two backbones another 0.75 GB.
- **Package versions.** The authors pinned Python, PyTorch, CUDA, transformers, accelerate and PyG. All other
  versions in `requirements.txt` are our choice of releases from the same period.

## SeedGuard workflow (recommendation robustness)

See **[SEEDGUARD.md](SEEDGUARD.md)** for the full guide (commands, metrics, aggregation,
determinism note, outputs). Summary:

- SeedGuard wraps `train_rec.py`; it does **not** add sampling to recommendation inference.
- **ECR recommendation inference is deterministic** (`torch.topk` on fixed `kg['item_ids']`).
  Inference-seed variation is **n/a**; the meaningful result is training-seed STD across
  per-model means (`np.std(..., ddof=1)`).
- Quick start after data/setup:

```bash
python make_quick_eval_set.py --seed 0
python run_seedguard.py --smoke 1x1 --skip-dataset
python run_seedguard.py --training-only --skip-dataset   # 10 models × 1 eval
python -m unittest test_seedguard_metrics test_seedguard_collect -v
```

## Seeds: what the code actually does

| Script | README seed | How it seeds |
|---|---|---|
| `train_pre.py` (pre-training) | 42 | Only with `--use_new_seed` does it call `seed_torch()` (Python, NumPy, torch, cuDNN disabled). Without it (the README command) it calls accelerate's `set_seed()` and sets `cudnn.benchmark = False`, but not `cudnn.deterministic`, so GPU runs may not repeat exactly. |
| `train_rec.py` (recommendation) | 8 | Always `seed_torch()`: Python, NumPy and torch seeded, cuDNN disabled, and `torch.set_deterministic(True)` (PyTorch errors on any non-deterministic operation). |
| `train_emp.py`, `infer_emp.py` (generation) | default 42 | accelerate's `set_seed()` only. |

For seed experiments:

- Pass `--use_new_seed` to `train_pre.py` for deterministic pre-training.
- Run each seed twice to measure how much results move with the same seed.
- Use one GPU model and a single GPU. Batch-wise AUC is computed per evaluation batch, so changing
  `--per_device_eval_batch_size` or the number of GPUs changes it.

## Upstream

```bash
git remote add upstream https://github.com/zxd-octopus/ECR.git   # if not already set
git fetch upstream
```

The upstream repository has no license file. All credit for the ECR code goes to its authors.
