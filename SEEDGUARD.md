# SeedGuard experiment guide

SeedGuard evaluates whether ECR recommendation metrics stay reliable when
**training seeds** change. It wraps the existing `src_emo/train_rec.py`
pipeline; it does not replace ECR with a toy recommender.

## Environment

Follow [SETUP.md](SETUP.md) first (`setup_env.sh` or `setup_env_mac.sh`,
`download_data.sh`, `check_env.py`, README preprocessing for `redial_gen`).

### Option B (recommended): SeedGuard on Windows + full ECR in WSL

If ECR is already installed in WSL (e.g. `~/ECR` with conda env `ecr`), keep this
GitHub repo at `D:\ECR-official-main` and **symlink** assets instead of copying them:

```bash
# in WSL
cd /mnt/d/ECR-official-main
bash scripts/link_wsl_ecr_assets.sh          # once; or pass /path/to/ECR
source scripts/wsl_ecr_env.sh                # conda ecr + CUDA libs + CUDA_VISIBLE_DEVICES=0
python make_quick_eval_set.py --seed 0       # once
python check_env.py                          # should pass
# 6 GB GPU smoke:
python run_seedguard.py --smoke 1x1 --skip-dataset \
  --per-device-train-batch-size 4 --per-device-eval-batch-size 8 \
  --gradient-accumulation-steps 16
```

Linked data/backbones are gitignored — safe to push SeedGuard code to GitHub.
Do **not** commit `*.prelink_backup_*` folders or `src_emo/save/train_seed_*` checkpoints.

Then generate the quick-eval subset **once** and reuse it for every seed:

```bash
python make_quick_eval_set.py --seed 0
```

## Files that matter

| File | Role |
|---|---|
| `src_emo/train_rec.py` | ECR recommendation training/eval; `--seed`, `--training_seed`, `--inference_seed`, `--no_timestamp` |
| `src_emo/evaluate_rec.py` | Original ECR metrics + Precision@K from ranked lists |
| `make_quick_eval_set.py` | Fixed subset (`redial_gen_quick`), generated once |
| `run_seedguard.py` | Multi-seed orchestrator (uses `seedguard.adapters.ecr`) |
| `collect_results.py` | Log parser + SeedGuard CSV / report aggregation |
| `seedguard/` | Shared metrics, STD formulas, recommender adapters |

## Inference randomness (important)

Recommendation evaluation is **deterministic**:

- `prompt_encoder.eval()` and `torch.no_grad()`
- logits restricted to the fixed item set `kg['item_ids']`
- ranking via `torch.topk` (no sampling, no random negatives at eval)

Changing `inference_seed` does **not** change recommendation metrics. The
field is kept for protocol compatibility and provenance. Inference variation
is marked **n/a** for the ECR recommendation adapter.

Do **not** confuse this with:

- **response generation** (`infer_emp.py`) — different randomness
- **LLM-rephrased dialogues** — input/prompt variation, not inference seeds

## Commands

### Environment check

```bash
python check_env.py
```

### One training seed + one eval (smoke)

```bash
python run_seedguard.py --smoke 1x1 --skip-dataset
```

### Two training seeds × two inference labels (smoke)

```bash
python run_seedguard.py --smoke 2x2 --skip-dataset
```

### Training-only (one eval per model; recommended once determinism is accepted)

```bash
python run_seedguard.py --training-only --skip-dataset
```

### Train seeds without evaluating (no invented zero metrics)

```bash
python run_seedguard.py --train-only-no-eval --skip-dataset
# later:
python run_seedguard.py --skip-train --skip-dataset --inference-seeds 1
```

### Full 10×10 protocol (completeness; inference cells should match within a row)

```bash
python run_seedguard.py --skip-dataset
```

### Collect only

```bash
python collect_results.py --seedguard --out src_emo/save/seedguard \
  --manifest src_emo/save/seedguard_manifest.json
```

### Unit tests (no GPU)

```bash
python -m unittest test_seedguard_metrics test_seedguard_collect -v
```

## Outputs

Under `src_emo/save/` (basename from `--out`, default `seedguard`):

| Path | Meaning |
|---|---|
| `{out}_manifest.json` | Per-run status, seeds, checkpoint paths |
| `{out}_seedguard_raw.csv` | One row per (training_seed, inference_seed) cell |
| `{out}_seedguard_matrix_*.csv` | Matrices (labelled repeated deterministic eval for ECR rec) |
| `{out}_seedguard_training_seed_means.csv` | **Main analysis**: one mean per training seed |
| `{out}_seedguard_summary.csv` | Per-model mean / inference STD / min / max / range |
| `{out}_seedguard_overall.csv` | Mean + training-seed STD / min / max / range across means |
| `{out}_seedguard_report.md` | Human-readable variation explanation |

Failed or missing cells stay **blank** in the CSV (never filled with 0).

## Metric definitions

Original ECR recommendation metrics (preserved):

- Recall@1 / @10 / @50
- Recall_True@1 / @10 / @50 (`RT@10` = Recall_True@10, not response time)
- AUC
- NDCG@10, MRR@10, loss (when logged)

Added SeedGuard metric:

- **Precision@10** = (number of relevant items in top-10) / 10  
  Computed from the ranked list and labels per example, then averaged over
  examples like ECR Recall@K. **Not** derived from Recall@10.

## Aggregation formulas

For each training seed $t$ with valid inference scores $x_{t,1},\ldots,x_{t,m}$:

1. $m_t = \mathrm{mean}(x_{t,\cdot})$  
   (If inference is deterministic and scores are identical, $m_t$ equals that score.)
2. Overall mean = $\mathrm{mean}(m_1,\ldots,m_N)$
3. Training-seed sample STD = `np.std([m_1,...,m_N], ddof=1)`
4. Range = $\max m_t - \min m_t$

Do **not** compute training-seed STD over all matrix cells.

Inference STD is `np.std` within one training seed only when that variation
is scientifically meaningful. For ECR recommendation it is **n/a** (repeats
are identical → numeric 0 if computed for transparency).

## Checkpoints

Each training seed writes to `src_emo/save/train_seed_{N}/` with `best/` and
`final/`. `--no_timestamp` prevents overwriting across runs. ECR selects the
best validation-loss checkpoint; it does not terminate epochs early
(`early_stop=best_checkpoint_only` in the train log).

## Limitations

- Requires the ECR conda/venv, data, and backbones from SETUP.md.
- Hard-coded `CUDA_VISIBLE_DEVICES` default remains `3` if unset; set the
  environment variable to choose another GPU.
- Batch-wise AUC depends on eval batch size / GPU count.
- Full 10×10 is costly; prefer `--training-only` for the scientific claim
  about training-seed robustness once determinism is verified.

## Reproducibility checklist

1. Fixed dataset: `make_quick_eval_set.py --seed 0` once.
2. Same README training hyperparameters for every seed.
3. Unique checkpoint directories per training seed.
4. Manifest + raw CSV record seeds, paths, and run status.
5. Report training-seed STD from means, with failed runs omitted.
