"""Unit test for SeedGuard result collection (no GPU/data required)."""

import csv
import os
import tempfile
import unittest

import numpy as np

import collect_results


MOCK_HEADER = ("{'seed': 1, 'training_seed': 1, 'inference_seed': 2, "
               "'dataset': 'redial_gen_quick', 'test': True, "
               "'output_dir': 'save/train_seed_1/', "
               "'prompt_encoder': 'save/train_seed_1/best/'}\n"
               "Num examples = 32\n")
MOCK_REPORT = ("{'test/recall@1': 0.1, 'test/recall@10': 0.2, 'test/recall@50': 0.3, "
               "'test/recall_true@1': 0.05, 'test/recall_true@10': 0.15, "
               "'test/recall_true@50': 0.25, "
               "'test/auc': 0.6, 'test/precision@10': 0.02, 'test/ndcg@10': 0.11, "
               "'test/mrr@10': 0.09, "
               "'test/loss': 1.2, 'test/training_seed': 1, 'test/inference_seed': 2, "
               "'epoch': 0}\n")


def _write_log(tmp, t, i, r10, rt10, auc, p10):
    header = MOCK_HEADER.replace("'training_seed': 1", f"'training_seed': {t}")
    header = header.replace("'inference_seed': 2", f"'inference_seed': {i}")
    report = MOCK_REPORT.replace("'test/recall@10': 0.2", f"'test/recall@10': {r10}")
    report = report.replace("'test/recall_true@10': 0.15", f"'test/recall_true@10': {rt10}")
    report = report.replace("'test/auc': 0.6", f"'test/auc': {auc}")
    report = report.replace("'test/precision@10': 0.02", f"'test/precision@10': {p10}")
    report = report.replace("'test/training_seed': 1", f"'test/training_seed': {t}")
    report = report.replace("'test/inference_seed': 2", f"'test/inference_seed': {i}")
    path = os.path.join(tmp, f"log_{t}_{i}.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report)
    return path


class SeedGuardCollectTest(unittest.TestCase):
    def test_seedguard_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            metrics = {
                (1, 1): ("0.20", "0.15", "0.60", "0.02"),
                (1, 2): ("0.21", "0.16", "0.61", "0.03"),
                (2, 1): ("0.18", "0.14", "0.58", "0.01"),
                (2, 2): ("0.19", "0.13", "0.59", "0.02"),
            }
            for (t, i), vals in metrics.items():
                rows.extend(collect_results.parse_log(_write_log(tmp, t, i, *vals)))

            sg = collect_results.seedguard_rows(rows, experiment_id="unit")
            self.assertEqual(len(sg), 4)

            out = os.path.join(tmp, "results")
            collect_results.write_seedguard_outputs(
                rows, out, experiment_id="unit",
                inference_variation_meaningful=False,
            )

            with open(out + "_seedguard_raw.csv", encoding="utf-8") as f:
                raw = f.read().strip().splitlines()
            self.assertEqual(len(raw) - 1, 4)
            self.assertIn("experiment_id", raw[0])
            self.assertIn("run_status", raw[0])

            with open(out + "_seedguard_summary.csv", encoding="utf-8") as f:
                summary = f.read()
            self.assertIn("inference_std_RT@10", summary)
            self.assertIn("n/a (deterministic ranking)", summary)

            with open(out + "_seedguard_overall.csv", encoding="utf-8", newline="") as f:
                overall = list(csv.DictReader(f))
            self.assertIn("overall_mean", overall[0])
            self.assertTrue(os.path.isfile(out + "_seedguard_report.md"))

    def test_training_seed_std_uses_column_means_not_all_cells(self):
        """Deterministic 10×10: identical inference cells; STD over 10 means."""
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for t in range(1, 11):
                rt10 = f"{0.10 + 0.01 * (t - 1):.4f}"
                p10 = f"{0.010 + 0.001 * (t - 1):.4f}"
                r10 = f"{0.20 + 0.01 * (t - 1):.4f}"
                auc = f"{0.50 + 0.01 * (t - 1):.4f}"
                for i in range(1, 11):
                    rows.extend(collect_results.parse_log(
                        _write_log(tmp, t, i, r10, rt10, auc, p10)))

            out = os.path.join(tmp, "results")
            collect_results.write_seedguard_outputs(
                rows, out, inference_variation_meaningful=False)
            self.assertEqual(len(collect_results.seedguard_rows(rows)), 100)

            with open(out + "_seedguard_summary.csv", encoding="utf-8", newline="") as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 10)
            for row in summary:
                self.assertAlmostEqual(float(row["inference_std_RT@10"]), 0.0, places=6)
                self.assertAlmostEqual(float(row["range_RT@10"]), 0.0, places=6)

            with open(out + "_seedguard_training_seed_means.csv", encoding="utf-8", newline="") as f:
                means = list(csv.DictReader(f))
            rt_means = [float(r["mean_RT@10"]) for r in means]
            expected_std = float(np.std(rt_means, ddof=1))
            with open(out + "_seedguard_overall.csv", encoding="utf-8", newline="") as f:
                overall = {r["metric"]: r for r in csv.DictReader(f)}
            self.assertAlmostEqual(float(overall["RT@10"]["training_seed_std"]),
                                   expected_std, places=6)
            self.assertAlmostEqual(float(overall["RT@10"]["overall_mean"]),
                                   float(np.mean(rt_means)), places=6)
            self.assertNotAlmostEqual(
                expected_std,
                float(np.std([0.10 + 0.01 * t for t in range(10)] * 10, ddof=1)),
            )

    def test_failed_runs_not_filled_with_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = collect_results.parse_log(
                _write_log(tmp, 1, 1, "0.20", "0.15", "0.60", "0.02"))
            manifest = [
                {"training_seed": 1, "inference_seed": 1, "run_status": "completed",
                 "checkpoint_path": "save/train_seed_1/best"},
                {"training_seed": 2, "inference_seed": 1, "run_status": "failed",
                 "checkpoint_path": "save/train_seed_2/best"},
            ]
            out = os.path.join(tmp, "results")
            sg = collect_results.write_seedguard_outputs(
                rows, out, expected_cells=[(1, 1), (2, 1)], manifest_runs=manifest)
            by_key = {(int(r["training_seed"]), int(r["inference_seed"])): r for r in sg}
            self.assertEqual(by_key[(1, 1)]["RT@10"], "0.1500")
            self.assertEqual(by_key[(2, 1)]["run_status"], "failed")
            self.assertEqual(by_key[(2, 1)]["RT@10"], "")
            with open(out + "_seedguard_overall.csv", encoding="utf-8", newline="") as f:
                overall = {r["metric"]: r for r in csv.DictReader(f)}
            # Only training seed 1 contributes a mean; failed seed omitted, not zero.
            self.assertEqual(int(overall["RT@10"]["n_training_seeds"]), 1)
            self.assertAlmostEqual(float(overall["RT@10"]["overall_mean"]), 0.15, places=4)


if __name__ == "__main__":
    unittest.main()
