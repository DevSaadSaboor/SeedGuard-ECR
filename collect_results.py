"""Collects the results of train_rec.py runs into one table.

SeedGuard mode (--seedguard) aggregates test-split eval rows that include
training_seed and inference_seed into raw CSV, per-training-seed summaries,
overall statistics, matrices, a run manifest merge, and a short markdown report.

Aggregation (training-seed STD):
  For each training seed t, m_t = mean of valid inference evaluations for that model.
  training_seed_std = np.std([m_1, ..., m_N], ddof=1)
  Never compute this STD over all matrix cells.

Missing / failed runs stay blank — they are never treated as zero.
"""

import argparse
import csv
import glob
import json
import os
import re
from datetime import datetime, timezone

import numpy as np

from seedguard.stats import seed_stats

METRICS = [("R@1", "recall@1"), ("R@10", "recall@10"), ("R@50", "recall@50"),
           ("RT@1", "recall_true@1"), ("RT@10", "recall_true@10"), ("RT@50", "recall_true@50"),
           ("AUC", "auc"), ("Precision@10", "precision@10"),
           ("NDCG@10", "ndcg@10"), ("MRR@10", "mrr@10"), ("loss", "loss")]
COLUMNS = ["started", "run", "mode", "seed", "training_seed", "inference_seed",
           "dataset", "train_examples", "epoch", "split"] + [m for m, _ in METRICS] + ["best"]

SEEDGUARD_RAW_COLUMNS = [
    "experiment_id", "training_seed", "inference_seed", "dataset", "model_config",
    "checkpoint_path", "run_status", "best_epoch", "epoch",
    "R@1", "R@10", "R@50", "RT@1", "RT@10", "RT@50",
    "AUC", "Precision@10", "NDCG@10", "MRR@10", "loss",
]

SEEDGUARD_STAT_METRICS = ["RT@10", "Precision@10", "R@10", "AUC"]
SEEDGUARD_MATRICES = [("RT@10", "RT10"), ("Precision@10", "Precision10"),
                      ("R@10", "R10"), ("AUC", "AUC")]

# ECR recommendation ranking is deterministic; inference_seed is a run label.
DEFAULT_INFERENCE_VARIATION_MEANINGFUL = False


def setting(text, key):
    m = re.search(rf"'{key}': '?([^,'}}]*)'?", text)
    if not m:
        return ""
    value = m.group(1).strip()
    # argparse None is logged as the literal None (unquoted).
    if value in ("None", "none", "null"):
        return ""
    return value


def parse_log(path):
    """One dict per (epoch, split) with the run's settings and metrics, in log order."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    mode = "eval" if setting(text, "test") == "True" else "train"
    examples = re.search(r"Num examples = (\d+)", text)
    seed = setting(text, "seed")
    training_seed = setting(text, "training_seed") or seed
    inference_seed = setting(text, "inference_seed")
    output_dir = setting(text, "output_dir")
    prompt_encoder = setting(text, "prompt_encoder")
    run = {
        "started": os.path.basename(path)[:-len(".log")] if re.match(r"\d{4}-", os.path.basename(path)) else "",
        "run": (output_dir or prompt_encoder or "").rstrip("/").rstrip("\\").split("/")[-1].split("\\")[-1],
        "mode": mode,
        "seed": seed,
        "training_seed": training_seed,
        "inference_seed": inference_seed,
        "dataset": setting(text, "dataset"),
        "train_examples": examples.group(1) if mode == "train" and examples else "",
        "checkpoint_path": prompt_encoder or output_dir,
        "log_path": path,
    }
    rows, best_epoch, last_valid_epoch = [], None, None
    failed = "Traceback" in text or "Error:" in text and "Random seed" not in text
    completed = ("save final model" in text) or (mode == "eval" and "test/recall@10" in text)
    for line in text.splitlines():
        if "new best model" in line:
            best_epoch = last_valid_epoch
        report = re.search(r"\{'(valid|test)/.*\}", line)
        if not report:
            continue
        split = report.group(1)
        values = dict(re.findall(rf"'{split}/([a-z_@0-9]+)': ([-0-9.e+]+)", report.group(0)))
        epoch = int(setting(report.group(0), "epoch") or 0)
        row_training_seed = setting(report.group(0), "training_seed") or training_seed
        row_inference_seed = setting(report.group(0), "inference_seed") or inference_seed
        if split == "valid":
            last_valid_epoch = epoch
        row = dict(run, epoch=epoch + 1 if mode == "train" else "", split=split, _epoch=epoch,
                   training_seed=row_training_seed, inference_seed=row_inference_seed,
                   best_epoch="" if best_epoch is None else best_epoch + 1)
        row.update({name: f"{float(values[key]):.4f}" if key in values else "" for name, key in METRICS})
        rows.append(row)
    for row in rows:
        row["best"] = "yes" if mode == "train" and row.pop("_epoch") == best_epoch else ""
        row.pop("_epoch", None)
        if failed:
            row["run_status"] = "failed"
        elif completed:
            row["run_status"] = "completed"
        else:
            row["run_status"] = "incomplete"
    return rows


def load_manifest(path):
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("runs", data if isinstance(data, list) else [])


def seedguard_rows(rows, experiment_id="", model_config="ecr_rec",
                   expected_cells=None, manifest_runs=None):
    """Keep one test row per (training_seed, inference_seed); merge manifest status."""
    def _seed_ok(value):
        return value not in ("", None, "None", "none", "null")

    eval_rows = [r for r in rows if r["split"] == "test" and _seed_ok(r.get("inference_seed"))]
    if not eval_rows:
        # Fallback: dedicated --test eval runs without an inference_seed label.
        eval_rows = [r for r in rows if r["split"] == "test" and r["mode"] == "eval"]
    seen = {}
    for row in eval_rows:
        inf = row.get("inference_seed", "")
        if not _seed_ok(inf):
            if row.get("mode") == "eval":
                inf = "1"
            else:
                continue
        key = (str(row["training_seed"]), str(inf))
        row = dict(row)
        row["inference_seed"] = str(inf)
        seen[key] = row

    manifest_by_key = {}
    for m in manifest_runs or []:
        # Training-only manifest rows (no inference_seed) are provenance, not metric cells.
        if not _seed_ok(m.get("inference_seed")):
            continue
        key = (str(m.get("training_seed", "")), str(m.get("inference_seed", "")))
        manifest_by_key[key] = m

    def enrich(row, key):
        m = manifest_by_key.get(key, {})
        status = m.get("run_status") or row.get("run_status", "completed")
        out = {
            "experiment_id": experiment_id or m.get("experiment_id", ""),
            "training_seed": row.get("training_seed", m.get("training_seed", "")),
            "inference_seed": row.get("inference_seed", m.get("inference_seed", "")),
            "dataset": row.get("dataset") or m.get("dataset", ""),
            "model_config": model_config or m.get("model_config", "ecr_rec"),
            "checkpoint_path": row.get("checkpoint_path") or m.get("checkpoint_path", ""),
            "run_status": status,
            "best_epoch": row.get("best_epoch", m.get("best_epoch", "")),
            "epoch": row.get("epoch", ""),
        }
        for name, _ in METRICS:
            # Never invent zeros for failed / missing / skipped cells.
            if status in ("failed", "missing", "skipped_inference", "incomplete"):
                out[name] = ""
            else:
                out[name] = row.get(name, "")
        return out

    results = []
    keys = set(seen) | set(manifest_by_key)
    if expected_cells:
        keys |= {(str(t), str(i)) for t, i in expected_cells}
    for key in sorted(keys, key=lambda k: (int(k[0]) if str(k[0]).lstrip("-").isdigit() else 0,
                                           int(k[1]) if str(k[1]).lstrip("-").isdigit() else 0)):
        if key in seen:
            results.append(enrich(seen[key], key))
        else:
            m = manifest_by_key.get(key, {})
            results.append({
                "experiment_id": experiment_id or m.get("experiment_id", ""),
                "training_seed": key[0],
                "inference_seed": key[1],
                "dataset": m.get("dataset", ""),
                "model_config": model_config or m.get("model_config", "ecr_rec"),
                "checkpoint_path": m.get("checkpoint_path", ""),
                "run_status": m.get("run_status", "missing"),
                "best_epoch": m.get("best_epoch", ""),
                "epoch": "",
                **{name: "" for name, _ in METRICS},
            })
    return results


def write_seedguard_outputs(rows, out_prefix, experiment_id="", model_config="ecr_rec",
                            inference_variation_meaningful=DEFAULT_INFERENCE_VARIATION_MEANINGFUL,
                            expected_cells=None, manifest_runs=None):
    sg_rows = seedguard_rows(
        rows, experiment_id=experiment_id, model_config=model_config,
        expected_cells=expected_cells, manifest_runs=manifest_runs,
    )
    if not sg_rows:
        raise SystemExit("SeedGuard: no test eval rows found in logs or manifest")

    raw_path = out_prefix + "_seedguard_raw.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEEDGUARD_RAW_COLUMNS)
        writer.writeheader()
        for row in sg_rows:
            writer.writerow({col: row.get(col, "") for col in SEEDGUARD_RAW_COLUMNS})

    completed = [r for r in sg_rows if r.get("run_status") == "completed"
                 and r.get("RT@10", "") != ""]
    failed = [r for r in sg_rows if r.get("run_status") in ("failed", "missing", "incomplete")]
    training_seeds = sorted({int(r["training_seed"]) for r in sg_rows
                             if str(r["training_seed"]).lstrip("-").isdigit()})
    inference_seeds = sorted({int(r["inference_seed"]) for r in sg_rows
                              if str(r.get("inference_seed", "")).lstrip("-").isdigit()})
    lookup = {}
    for r in completed:
        lookup[(int(r["training_seed"]), int(r["inference_seed"]))] = r

    per_train = []
    train_means = {metric: [] for metric in SEEDGUARD_STAT_METRICS}
    for ts in training_seeds:
        entry = {"training_seed": ts}
        for metric in SEEDGUARD_STAT_METRICS:
            vals = [float(lookup[(ts, inf)][metric]) for inf in inference_seeds
                    if (ts, inf) in lookup and lookup[(ts, inf)].get(metric, "") != ""]
            stats = seed_stats(vals)
            entry[f"mean_{metric}"] = stats["mean"]
            if inference_variation_meaningful:
                entry[f"inference_std_{metric}"] = stats["std"]
            else:
                # Transparent numeric STD of repeated deterministic cells, plus label.
                entry[f"inference_std_{metric}"] = stats["std"] if stats["n"] else ""
                entry[f"inference_variation_{metric}"] = "n/a (deterministic ranking)"
            entry[f"min_{metric}"] = stats["min"]
            entry[f"max_{metric}"] = stats["max"]
            entry[f"range_{metric}"] = stats["range"]
            entry[f"n_inference_{metric}"] = stats["n"]
            if stats["mean"] != "":
                train_means[metric].append(stats["mean"])
        per_train.append(entry)

    summary_path = out_prefix + "_seedguard_summary.csv"
    summary_cols = ["training_seed"]
    for metric in SEEDGUARD_STAT_METRICS:
        summary_cols += [f"mean_{metric}", f"inference_std_{metric}",
                         f"inference_variation_{metric}",
                         f"min_{metric}", f"max_{metric}", f"range_{metric}",
                         f"n_inference_{metric}"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols, extrasaction="ignore")
        writer.writeheader()
        for entry in per_train:
            writer.writerow({
                col: (f"{entry[col]:.6f}" if isinstance(entry.get(col), float) else entry.get(col, ""))
                for col in summary_cols
            })

    overall_path = out_prefix + "_seedguard_overall.csv"
    overall_cols = [
        "metric", "n_training_seeds", "overall_mean", "training_seed_std",
        "minimum", "maximum", "range",
        "inference_variation_meaningful", "n_completed_cells", "n_failed_or_missing_cells",
    ]
    overall_rows = []
    for metric in SEEDGUARD_STAT_METRICS:
        stats = seed_stats(train_means[metric])
        overall_rows.append({
            "metric": metric,
            "n_training_seeds": stats["n"],
            "overall_mean": stats["mean"],
            "training_seed_std": stats["std"],
            "minimum": stats["min"],
            "maximum": stats["max"],
            "range": stats["range"],
            "inference_variation_meaningful": str(bool(inference_variation_meaningful)).lower(),
            "n_completed_cells": len(completed),
            "n_failed_or_missing_cells": len(failed),
        })
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overall_cols)
        writer.writeheader()
        for row in overall_rows:
            writer.writerow({
                col: (f"{row[col]:.6f}" if isinstance(row.get(col), float) else row.get(col, ""))
                for col in overall_cols
            })

    means_path = out_prefix + "_seedguard_training_seed_means.csv"
    means_cols = ["training_seed"] + [f"mean_{m}" for m in SEEDGUARD_STAT_METRICS]
    with open(means_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=means_cols)
        writer.writeheader()
        for entry in per_train:
            writer.writerow({
                "training_seed": entry["training_seed"],
                **{f"mean_{m}": (f"{entry[f'mean_{m}']:.6f}" if isinstance(entry.get(f"mean_{m}"), float)
                                 else entry.get(f"mean_{m}", ""))
                   for m in SEEDGUARD_STAT_METRICS},
            })

    matrix_paths = []
    matrix_note = ("repeated_deterministic_eval" if not inference_variation_meaningful
                   else "inference_seed_variation")
    for metric, suffix in SEEDGUARD_MATRICES:
        matrix_path = out_prefix + f"_seedguard_matrix_{suffix}.csv"
        with open(matrix_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"training_seed \\ inference_seed ({matrix_note})"] + inference_seeds)
            for ts in training_seeds:
                writer.writerow(
                    [ts] + [lookup.get((ts, inf), {}).get(metric, "") for inf in inference_seeds]
                )
        matrix_paths.append(matrix_path)

    report_path = out_prefix + "_seedguard_report.md"
    _write_report(report_path, overall_rows, per_train, completed, failed,
                  inference_variation_meaningful, experiment_id)

    print(f"SeedGuard: {len(completed)} completed / {len(failed)} failed-or-missing "
          f"of {len(sg_rows)} cell(s) -> {raw_path}")
    print(f"  training-seed means (main analysis): {means_path}")
    print(f"  summary (per-training-seed mean / inference STD / min / max / range): {summary_path}")
    print(f"  overall (mean + training-seed STD across column means, not all cells): {overall_path}")
    print(f"  report: {report_path}")
    for p in matrix_paths:
        print(f"  matrix: {p}")
    if not inference_variation_meaningful:
        print("Note: ECR recommendation inference is deterministic. "
              "Inference STD of identical repeats is 0; variation is marked n/a.")
    print("Training-seed STD = np.std([mean of each training seed's inference results], ddof=1).")
    return sg_rows


def _write_report(path, overall_rows, per_train, completed, failed,
                  inference_variation_meaningful, experiment_id):
    lines = [
        "# SeedGuard variation report",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Experiment: {experiment_id or '(from logs)'}",
        f"- Completed cells: {len(completed)}",
        f"- Failed / missing / incomplete cells: {len(failed)}",
        "",
        "## Scientific note on inference seeds",
        "",
    ]
    if not inference_variation_meaningful:
        lines += [
            "ECR recommendation evaluation ranks items with `torch.topk` over the fixed",
            "item vocabulary under `torch.no_grad()`. There is no sampling and no random",
            "candidate construction at eval time. **Changing `inference_seed` does not",
            "change recommendation metrics.** The training×inference matrix, if produced,",
            "is labelled as repeated deterministic evaluation, not evidence of inference",
            "randomness.",
            "",
            "Response generation (`infer_emp.py`) is a different component and may use",
            "different randomness; do not pool it with recommendation metrics.",
            "",
            "LLM-rephrased conversations would be **input/prompt variation**, not",
            "inference-seed variation, and must be analysed separately.",
            "",
        ]
    else:
        lines += [
            "Inference-seed variation is treated as meaningful for this adapter.",
            "",
        ]
    lines += [
        "## Aggregation formulas",
        "",
        "For each training seed $t$ with valid inference scores $x_{t,1},\\ldots,x_{t,m}$:",
        "",
        "- $m_t = \\mathrm{mean}(x_{t,\\cdot})$",
        "- Training-seed sample STD $= \\mathrm{np.std}([m_1,\\ldots,m_N],\\,\\mathrm{ddof}=1)$",
        "- Range $= \\max m_t - \\min m_t$",
        "- Missing/failed runs are omitted (never filled with 0).",
        "",
        "Precision@10 (per example) $= |\\{\\text{relevant}\\} \\cap \\mathrm{top}\\text{-}10| / 10$,",
        "then averaged over examples like ECR Recall@K. It is **not** derived from Recall@10.",
        "",
        "## Overall statistics (across training-seed means)",
        "",
        "| metric | n | mean | training-seed STD | min | max | range |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in overall_rows:
        def fmt(v):
            return f"{v:.6f}" if isinstance(v, float) else str(v)
        lines.append(
            f"| {row['metric']} | {row['n_training_seeds']} | {fmt(row['overall_mean'])} | "
            f"{fmt(row['training_seed_std'])} | {fmt(row['minimum'])} | {fmt(row['maximum'])} | "
            f"{fmt(row['range'])} |"
        )
    lines += ["", "## Per training seed (means)", ""]
    if per_train:
        lines.append("| training_seed | " + " | ".join(f"mean_{m}" for m in SEEDGUARD_STAT_METRICS) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in SEEDGUARD_STAT_METRICS) + " |")
        for entry in per_train:
            cells = []
            for m in SEEDGUARD_STAT_METRICS:
                v = entry.get(f"mean_{m}", "")
                cells.append(f"{v:.6f}" if isinstance(v, float) else "")
            lines.append(f"| {entry['training_seed']} | " + " | ".join(cells) + " |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="log files (default: src_emo/log/*.log)")
    ap.add_argument("--out", default=os.path.join("src_emo", "save", "results"),
                    help="output path without extension")
    ap.add_argument("--seedguard", action="store_true",
                    help="also write SeedGuard raw/summary/overall/matrix/report outputs")
    ap.add_argument("--experiment-id", default="", help="optional experiment id for SeedGuard CSV")
    ap.add_argument("--model-config", default="ecr_rec", help="model/configuration identifier")
    ap.add_argument("--manifest", default="",
                    help="optional SeedGuard run manifest JSON (status / expected cells)")
    ap.add_argument("--inference-variation", choices=["auto", "yes", "no"], default="auto",
                    help="whether inference_seed variation is scientifically meaningful "
                         "(auto: no for ecr_rec)")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    paths = args.logs or sorted(glob.glob(os.path.join("src_emo", "log", "*.log")))
    rows = [row for path in paths for row in parse_log(path)]
    manifest_runs = load_manifest(args.manifest)
    if not rows and not manifest_runs:
        raise SystemExit("no results found in: " + ", ".join(paths or ["src_emo/log/ (empty)"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if rows:
        with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        table = [COLUMNS] + [[str(row.get(c, "")) for c in COLUMNS] for row in rows]
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(table[0]) + " |\n|" + "|".join("---" for _ in COLUMNS) + "|\n")
            f.writelines("| " + " | ".join(r) + " |\n" for r in table[1:])

        widths = [max(len(r[i]) for r in table) for i in range(len(COLUMNS))]
        for r in table:
            print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))
        runs = len({(row["started"], row["run"]) for row in rows})
        print(f"\n{runs} run(s), {len(rows)} rows -> {args.out}.csv and {args.out}.md")

    if args.seedguard:
        if args.inference_variation == "auto":
            meaningful = DEFAULT_INFERENCE_VARIATION_MEANINGFUL
        else:
            meaningful = args.inference_variation == "yes"
        def _manifest_seed_ok(value):
            return value not in (None, "", "None", "none", "null")

        expected = [(m.get("training_seed"), m.get("inference_seed")) for m in manifest_runs
                    if _manifest_seed_ok(m.get("inference_seed"))]
        write_seedguard_outputs(
            rows, args.out,
            experiment_id=args.experiment_id,
            model_config=args.model_config,
            inference_variation_meaningful=meaningful,
            expected_cells=expected or None,
            manifest_runs=manifest_runs,
        )


if __name__ == "__main__":
    main()
