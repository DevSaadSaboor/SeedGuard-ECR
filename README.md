# SeedGuard + ECR

**SeedGuard** evaluates whether an LLM-based conversational recommender stays reliable when **training random seeds** change.

This repository contains:

1. The official **ECR** code (*Towards Empathetic Conversational Recommender Systems*, Zhang et al., RecSys 2024) from [zxd-octopus/ECR](https://github.com/zxd-octopus/ECR)
2. The **SeedGuard** experiment layer on top of ECR’s recommendation component (`train_rec.py`)

ECR’s training loss, architecture, and original recommendation metrics are preserved. SeedGuard adds multi-seed orchestration, Precision@10, result collection, and aggregation (mean / sample STD / min / max / range across training seeds).

| Guide | Content |
|---|---|
| **[SEEDGUARD.md](SEEDGUARD.md)** | SeedGuard experiment commands, metrics, determinism notes |
| **[SETUP.md](SETUP.md)** | Environment, data download, GPU notes |

## What SeedGuard measures

- **Dataset:** ReDial conversational movie dialogues + a DBpedia knowledge-graph subgraph (ECR’s prepared `redial_gen` / `redial_gen_quick` files — not a SQL database)
- **Model under test:** ECR emotion-aware item recommendation (`src_emo/train_rec.py`)
- **Main question:** How much do Recall / Recall_True / AUC / Precision@10 change across training seeds?
- **Inference note:** ECR recommendation ranking is **deterministic** (`torch.topk` on a fixed item set). Meaningful variation is across **training seeds**, not inference seeds.

## Quick start (SeedGuard)

### Environment

```bash
# Linux / WSL with conda (recommended)
bash setup_env.sh
conda activate ecr
bash download_data.sh
python check_env.py
```

If ECR is already installed elsewhere in WSL (e.g. `~/ECR`), use Option B:

```bash
bash scripts/link_wsl_ecr_assets.sh
source scripts/wsl_ecr_env.sh
python check_env.py
```

### Run a smoke experiment

```bash
python make_quick_eval_set.py --seed 0          # once; fixed subset
python run_seedguard.py --smoke 1x1 --skip-dataset
# 6 GB GPU example:
python run_seedguard.py --smoke 1x1 --skip-dataset \
  --per-device-train-batch-size 4 --per-device-eval-batch-size 8 \
  --gradient-accumulation-steps 16
```

### 10 training-seed study (recommended scientific protocol)

```bash
python run_seedguard.py --training-only --skip-dataset
```

Outputs appear under `src_emo/save/` as `*_seedguard_*.csv` and `*_seedguard_report.md`.

### Unit tests (no GPU)

```bash
python -m unittest test_seedguard_metrics test_seedguard_collect -v
```

## Repository layout (SeedGuard-related)

```
run_seedguard.py          # multi-seed experiment runner
collect_results.py        # log → CSV / summary / report
make_quick_eval_set.py    # fixed ReDial subset
seedguard/                # metrics, STD formulas, ECR adapter
scripts/                  # WSL link + env helpers
src_emo/train_rec.py      # ECR recommendation (+ SeedGuard hooks)
src_emo/evaluate_rec.py   # ECR metrics + Precision@K
SEEDGUARD.md              # full experiment guide
SETUP.md                  # install / data / seeds
```

Large data, checkpoints, and model backbones are **gitignored**. Download them with `download_data.sh` / `fetch_backbones.sh`, or link an existing ECR install.

---

# Upstream ECR (original README)

The sections below are from the ECR authors’ README and describe their full pipeline (pre-training, recommendation, response generation).

## Requirements

- python == 3.8.13
- pytorch == 1.8.1
- cudatoolkit == 11.1.1
- transformers == 4.15.0
- pyg == 2.0.1
- accelerate == 0.8.0

## Data

The data we used has been uploaded [here](https://drive.google.com/file/d/1fb9kDo8uSRLlwc5c4nUw8DZHR5XOY_l_/view?usp=sharing).
The downloaded ckpt files should be moved into `src_emo/data/emo_data/`.

## Saved Models

We have saved the parameters of our model, all of which have been uploaded [here](https://drive.google.com/file/d/1uBtcqbQByVrrJ1hEwk2dvsAOxuvEgE19/view?usp=sharing).
The downloaded ckpt files should be moved into `src_emo/data/saved/`.

## Quick-Start

We run all experiments and tune hyperparameters on a GPU with 24GB memory, you can adjust `per_device_train_batch_size` and `per_device_eval_batch_size` according to your GPU, and then the optimization hyperparameters (e.g., `learning_rate`) may also need to be tuned.

### Emotional Semantic Fusion Subtask

```bash
cd src_emo 
cp -r data/emo_data/* data/redial/
python data/redial/process.py 
accelerate launch train_pre.py \
--dataset redial \
--num_train_epochs 10 \
--gradient_accumulation_steps 4  \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 64  \
--num_warmup_steps 1389 \
--max_length 200 \
--prompt_max_length 200  \
--entity_max_length 32  \
--learning_rate 5e-4 \
--seed 42 \
--nei_mer
```

### Emotion-aware Item Recommendation Training

```bash
# merge infer results from conversation model of UniCRS
cd src_emo
cp -r data/emo_data/* data/redial/
python data/redial/process_mask.py
cp -r data/redial/* data/redial_gen/
python data/redial_gen/merge.py
accelerate launch train_rec.py   \
--dataset redial_gen   \
--n_prefix_rec 10    \
--num_train_epochs 5   \
--per_device_train_batch_size 16   \
--per_device_eval_batch_size 32   \
--gradient_accumulation_steps 8   \
--num_warmup_steps 530   \
--context_max_length 200   \
--prompt_max_length 200   \
--entity_max_length 32   \
--learning_rate 1e-4   \
--seed 8   \
--like_score 2.0   \
--dislike_score 1.0   \
--notsay_score 0.5    \
--weighted_loss   \
--nei_mer  \
--use_sentiment 
```

### Emotion-aligned Response Generation Training and Inference

#### Backbone: DialoGPT

```bash
# merge infer results from Recommend Response Generation
# train
cd src_emo
python data/redial_gen/merge_rec.py
python imdb_review_entity_filter.py
python data/redial/process_empthetic.py
accelerate launch train_emp.py   \
 --dataset redial   \
--num_train_epochs 15    \
--gradient_accumulation_steps 1    \
--ignore_pad_token_for_loss    \
--per_device_train_batch_size 20    \
--per_device_eval_batch_size 64    \
--num_warmup_steps 9965    \
--context_max_length 150    \
--resp_max_length 150    \
--learning_rate 1e-04  

# infer
accelerate launch infer_emp.py   \
--dataset redial_gen \
--split test \
--per_device_eval_batch_size 256 \
--context_max_length 150 \
--resp_max_length 150
```

#### Backbone: Llama 2-Chat

We use [LLaMA Board](https://github.com/hiyouga/LLaMA-Efficient-Tuning) to fine-tune Llama 2-Chat.

```
Training Data Path: src_emo/data/emo_data/llama_train.json
Testing Data Path: src_emo/data/emo_data/llama_test.json
```

## Acknowledgement

ECR is developed based on [UniCRS](https://github.com/RUCAIBox/UniCRS).
Any scientific publications that use the ECR codes or dataset should cite the ECR paper as the reference.
The upstream ECR repository has no license file; rights remain with the original authors.
SeedGuard is an evaluation layer for studying training-seed robustness on top of that code.
