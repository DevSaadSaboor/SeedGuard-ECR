"""Built so that we can test on a subset of the real data that takes a fraction of the time to run
    created dataset is saved in gitignored files redial_gen_quick under src_emo/data

    python3 make_quick_eval_set.py --percent 10                     # random 10% of the validation and test sets
    python3 make_quick_eval_set.py --percent 10 --train-percent 5   # ...plus 5% of the training set, to train on
    python3 make_quick_eval_set.py --percent 5                      # random 5% of the validation and test sets
    python3 make_quick_eval_set.py --percent 5 --train-percent 1    # ...plus 1% of the training set, to train on
    python3 make_quick_eval_set.py                                  # ~1000 test and ~160 validation recommendations
    python3 make_quick_eval_set.py --test-targets 500               # smaller and faster

    After you are finished creating this smaller data chunk and training just certain percentage of train data, navigate to src_emo (cd src_emo)
    Then you will be able to run something like: (feel free to adjust epochs and/or other parameters)
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 19 ../.venv/bin/python train_rec.py --dataset redial_gen_quick --n_prefix_rec 10 --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --gradient_accumulation_steps 8 --num_warmup_steps 1 --context_max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --seed 1 --like_score 2.0 --dislike_score 1.0 --notsay_score 0.5 --weighted_loss --nei_mer --use_sentiment --output_dir save/train_seed1 2>&1 | tee save/train_seed1.log

    Go back to the original folder (ECR-official) (cd ..)
    Run python3 collect_results.py - collects results from the logs (custom built)

    logs can be found in save/train_seed1.log
    clear(er) results can be found in results.csv / md
"""

import argparse
import json
import os
import random

SRC = os.path.join("src_emo", "data", "redial_gen")
DST = os.path.join("src_emo", "data", "redial_gen_quick")
SPLITS = ["train", "valid", "test"]


def sample_split(split, targets, percent, rng):
    """Random dialogue turns from one split until they hold `targets` recommendations (or `percent` of the split's
    recommendations, if given), kept in file order."""
    with open(os.path.join(SRC, f"{split}_data_processed.jsonl"), encoding="utf-8") as f:
        lines = [line for line in f if json.loads(line).get("rec")]
    if percent is not None:
        targets = round(sum(len(json.loads(line)["rec"]) for line in lines) * percent / 100)
    order = list(range(len(lines)))
    rng.shuffle(order)
    chosen, n = [], 0
    for i in order:
        if n >= targets:
            break
        chosen.append(i)
        n += len(json.loads(lines[i])["rec"])
    return [lines[i] for i in sorted(chosen)], n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-targets", type=int, default=1000, help="test recommendations to keep (full: 4810)")
    ap.add_argument("--valid-targets", type=int, default=160, help="validation recommendations to keep (full: 3733)")
    ap.add_argument("--percent", type=float, help="keep this %% of the validation and test sets (overrides targets)")
    ap.add_argument("--train-percent", type=float,
                    help="keep this %% of the training set, for training runs (default: 32 rows, enough for --test)")
    ap.add_argument("--seed", type=int, default=0, help="which random sample")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isfile(os.path.join(SRC, "test_data_processed.jsonl")):
        raise SystemExit(f"{SRC} has no processed data yet: run the README's preprocessing steps first")
    os.makedirs(DST, exist_ok=True)

    processed = {f"{s}_data_processed.jsonl" for s in SPLITS}
    for name in os.listdir(SRC):
        link = os.path.join(DST, name)
        if name in processed or os.path.lexists(link):
            continue
        os.symlink(os.path.join("..", "redial_gen", name), link)

    rng = random.Random(args.seed)
    budget = {"train": 32, "valid": args.valid_targets, "test": args.test_targets}
    for split in SPLITS:
        percent = args.train_percent if split == "train" else args.percent
        lines, n = sample_split(split, budget[split], percent, rng)
        with open(os.path.join(DST, f"{split}_data_processed.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"{split:<5} {len(lines):>5} dialogue turns, {n:>5} recommendations")
    print(f"-> {DST}; run train_rec.py with --dataset redial_gen_quick")


if __name__ == "__main__":
    main()
