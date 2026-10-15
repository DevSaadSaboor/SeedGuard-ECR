#!/usr/bin/env python3
"""Run SeedGuard robustness experiments on ECR recommendation (train_rec.py).

ECR recommendation inference is deterministic (eval + torch.topk over a fixed
item set). This script does not add sampling. The default protocol can still
run 10 training seeds × 10 inference seeds = 100 evaluations for completeness.
Inference-seed scores for the same trained model are expected to be identical
(inference STD = 0; marked n/a as a source of variation). The robustness
analysis uses the N training-seed means, then np.std(..., ddof=1) across those
means.

Examples:
  python run_seedguard.py --smoke 1x1
  python run_seedguard.py --smoke 2x2
  python run_seedguard.py                              # full 10×10 (default)
  python run_seedguard.py --training-seeds 10 --inference-seeds 10
  python run_seedguard.py --training-only              # 10 trains × 1 eval
  python run_seedguard.py --inference-seeds 1          # same as --training-only
  python run_seedguard.py --train-only-no-eval         # train seeds, skip eval loop
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

from seedguard.adapters import get_adapter

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_checked(cmd, cwd, env=None):
    print("\n>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="ecr_rec",
                    help="recommender adapter (default: ecr_rec)")
    ap.add_argument("--smoke", choices=["1x1", "2x2"],
                    help="run 1×1 or 2×2 smoke test (overrides seed counts)")
    ap.add_argument("--training-seeds", type=int, default=10,
                    help="number of training seeds 1..N (default: 10)")
    ap.add_argument("--inference-seeds", type=int, default=None,
                    help="number of inference seeds 1..M (default: 10; use 1 to skip redundant evals)")
    ap.add_argument("--training-only", action="store_true",
                    help="evaluate each trained model once (inference_seed=1). "
                         "Equivalent to --inference-seeds 1.")
    ap.add_argument("--train-only-no-eval", action="store_true",
                    help="train the requested seeds and write the manifest, but do not "
                         "run the separate inference loop or invent zero metrics")
    ap.add_argument("--dataset-seed", type=int, default=0,
                    help="fixed seed for make_quick_eval_set.py (run once)")
    ap.add_argument("--num-train-epochs", type=int, default=None,
                    help="training epochs (default: 1 for smoke, 5 for full)")
    ap.add_argument("--num-warmup-steps", type=int, default=None,
                    help="LR warmup steps (default: 1 for smoke, 530 for full)")
    ap.add_argument("--skip-dataset", action="store_true",
                    help="reuse existing redial_gen_quick (do not regenerate)")
    ap.add_argument("--skip-train", action="store_true",
                    help="skip training; only run eval + collect (checkpoints must exist)")
    ap.add_argument("--out", default="seedguard",
                    help="basename for collect_results --out (under src_emo/save/)")
    ap.add_argument("--experiment-id", default="",
                    help="id written into the manifest and SeedGuard CSV")
    ap.add_argument("--per-device-train-batch-size", type=int, default=None,
                    help="override train batch size (use 4 on 6 GB GPUs)")
    ap.add_argument("--per-device-eval-batch-size", type=int, default=None,
                    help="override eval batch size (use 8 on 6 GB GPUs)")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=None,
                    help="override gradient accumulation steps")
    args = ap.parse_args()

    adapter_kwargs = {"root": ROOT}
    if args.adapter == "ecr_rec":
        adapter_kwargs.update({
            "train_batch_size": args.per_device_train_batch_size,
            "eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        })
    adapter = get_adapter(args.adapter, **adapter_kwargs)
    if args.smoke == "1x1":
        n_train, n_infer = 1, 1
        epochs = args.num_train_epochs or 1
        warmup = args.num_warmup_steps or 1
        out = args.out if args.out != "seedguard" else "seedguard_smoke_1x1"
    elif args.smoke == "2x2":
        n_train, n_infer = 2, 2
        epochs = args.num_train_epochs or 1
        warmup = args.num_warmup_steps or 1
        out = args.out if args.out != "seedguard" else "seedguard_smoke_2x2"
    else:
        n_train = args.training_seeds
        if args.train_only_no_eval:
            n_infer = 0
        elif args.training_only:
            n_infer = 1
            if args.inference_seeds not in (None, 1):
                print("note: --training-only forces --inference-seeds 1 "
                      f"(ignoring {args.inference_seeds})", flush=True)
        else:
            n_infer = 10 if args.inference_seeds is None else args.inference_seeds
        epochs = args.num_train_epochs or 5
        warmup = args.num_warmup_steps or 530
        out = args.out

    experiment_id = args.experiment_id or f"seedguard_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    save_dir = os.path.join(ROOT, "src_emo", "save")
    os.makedirs(save_dir, exist_ok=True)
    manifest_path = os.path.join(save_dir, f"{out}_manifest.json")
    runs = []

    env = os.environ.copy()
    # Respect an already-set GPU; train_rec.py also prefers an existing value.

    if not args.skip_dataset:
        ds_cmd = adapter.dataset_command(args.dataset_seed)
        if ds_cmd:
            rc = run_checked(ds_cmd, ROOT, env=env)
            if rc.returncode != 0:
                raise SystemExit(f"dataset preparation failed with code {rc.returncode}")

    for t in range(1, n_train + 1):
        ckpt = adapter.checkpoint_dir(t)
        train_status = "skipped"
        if not args.skip_train:
            train_entry = {
                "experiment_id": experiment_id,
                "phase": "train",
                "training_seed": t,
                "inference_seed": "",
                "dataset": getattr(adapter, "dataset_id", ""),
                "model_config": getattr(adapter, "config_id", adapter.name),
                "checkpoint_path": ckpt,
                "started_at": _utc_now(),
            }
            rc = run_checked(
                adapter.train_command(t, ckpt, epochs, warmup),
                adapter.working_directory,
                env=env,
            )
            train_status = "completed" if rc.returncode == 0 else "failed"
            train_entry["run_status"] = train_status
            train_entry["returncode"] = rc.returncode
            train_entry["finished_at"] = _utc_now()
            runs.append(train_entry)
            if train_status == "failed":
                print(f"training seed {t} failed; continuing with remaining seeds", flush=True)
                continue
        else:
            train_status = "completed" if os.path.isdir(os.path.join(ckpt, "best")) else "missing"

        if n_infer <= 0:
            # Training-only mode: do not invent inference metric rows as zeros.
            runs.append({
                "experiment_id": experiment_id,
                "phase": "train_only",
                "training_seed": t,
                "inference_seed": "",
                "dataset": getattr(adapter, "dataset_id", ""),
                "model_config": getattr(adapter, "config_id", adapter.name),
                "checkpoint_path": ckpt,
                "run_status": train_status if train_status != "skipped" else (
                    "completed" if os.path.isdir(os.path.join(ckpt, "best")) else "missing"
                ),
            })
            continue

        for i in range(1, n_infer + 1):
            eval_entry = {
                "experiment_id": experiment_id,
                "phase": "eval",
                "training_seed": t,
                "inference_seed": i,
                "dataset": getattr(adapter, "dataset_id", ""),
                "model_config": getattr(adapter, "config_id", adapter.name),
                "checkpoint_path": os.path.join(ckpt, "best"),
                "started_at": _utc_now(),
            }
            if train_status == "failed":
                eval_entry["run_status"] = "skipped_inference"
                eval_entry["returncode"] = None
            else:
                rc = run_checked(
                    adapter.evaluate_command(t, i, ckpt, warmup),
                    adapter.working_directory,
                    env=env,
                )
                eval_entry["run_status"] = "completed" if rc.returncode == 0 else "failed"
                eval_entry["returncode"] = rc.returncode
            eval_entry["finished_at"] = _utc_now()
            runs.append(eval_entry)

    manifest = {
        "experiment_id": experiment_id,
        "adapter": adapter.name,
        "inference_variation_meaningful": bool(adapter.inference_variation_meaningful),
        "n_training_seeds": n_train,
        "n_inference_seeds": n_infer,
        "dataset_seed": args.dataset_seed,
        "num_train_epochs": epochs,
        "num_warmup_steps": warmup,
        "created_at": _utc_now(),
        "runs": runs,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}", flush=True)

    if n_infer <= 0:
        print("\nTrain-only-no-eval finished: checkpoints and manifest written. "
              "No inference metrics were invented as zeros.")
        print(f"Re-run with --skip-train --inference-seeds N to evaluate later.")
        return

    out_prefix = os.path.join("src_emo", "save", out)
    collect_cmd = [
        sys.executable, os.path.join(ROOT, "collect_results.py"),
        "--seedguard",
        "--out", out_prefix,
        "--experiment-id", experiment_id,
        "--model-config", getattr(adapter, "config_id", adapter.name),
        "--manifest", manifest_path,
        "--inference-variation", "yes" if adapter.inference_variation_meaningful else "no",
    ]
    rc = run_checked(collect_cmd, ROOT, env=env)
    if rc.returncode != 0:
        raise SystemExit(f"collect_results failed with code {rc.returncode}")

    expected = n_train * n_infer
    print(f"\nSeedGuard finished: {n_train} training seed(s) × {n_infer} inference seed(s) "
          f"= {expected} scheduled eval run(s). Results under src_emo/save/{out}_seedguard_*.csv")
    if not adapter.inference_variation_meaningful:
        print("Recommendation inference is deterministic: inference variation is n/a; "
              "repeated cells are expected to match (STD 0).")
    print("Main analysis: training-seed STD across the N training-seed means "
          "(np.std(means, ddof=1)), not across all cells.")


if __name__ == "__main__":
    main()
