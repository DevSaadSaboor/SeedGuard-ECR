"""Unit tests for SeedGuard Precision@K and standard-deviation formulas."""

import unittest

import numpy as np

from seedguard.metrics import compute_precision_at_k, compute_recall_at_k
from seedguard.stats import inference_std_for_model, seed_stats, training_seed_means


class PrecisionRecallTest(unittest.TestCase):
    def test_precision_at_10_hit(self):
        ranked = list(range(10))  # label 3 is in top-10
        self.assertAlmostEqual(compute_precision_at_k(ranked, 3, 10), 0.1)
        self.assertEqual(compute_recall_at_k(ranked, 3, 10), 1)

    def test_precision_at_10_miss(self):
        ranked = list(range(10))
        self.assertAlmostEqual(compute_precision_at_k(ranked, 99, 10), 0.0)
        self.assertEqual(compute_recall_at_k(ranked, 99, 10), 0)

    def test_precision_not_equal_to_recall(self):
        ranked = [5] + list(range(10, 19))  # single hit at rank 1
        recall = compute_recall_at_k(ranked, 5, 10)
        precision = compute_precision_at_k(ranked, 5, 10)
        self.assertEqual(recall, 1)
        self.assertAlmostEqual(precision, 0.1)
        self.assertNotAlmostEqual(precision, recall)

    def test_multi_relevant(self):
        ranked = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertAlmostEqual(compute_precision_at_k(ranked, {1, 3, 99}, 10), 0.2)


class StdAggregationTest(unittest.TestCase):
    def test_missing_not_zero(self):
        stats = seed_stats([])
        self.assertEqual(stats["mean"], "")
        self.assertEqual(stats["std"], "")
        self.assertEqual(stats["n"], 0)
        stats2 = seed_stats([None, "", "nan"])
        # "nan" parses as float nan and is dropped
        self.assertEqual(stats2["n"], 0)

    def test_sample_std_ddof_1(self):
        vals = [0.10, 0.12, 0.14]
        stats = seed_stats(vals)
        self.assertAlmostEqual(stats["std"], float(np.std(vals, ddof=1)))
        self.assertAlmostEqual(stats["range"], 0.04)

    def test_training_seed_std_uses_means_not_cells(self):
        # Each model has 10 identical inference scores; means are 0.10..0.19
        per_model = {t: [0.10 + 0.01 * (t - 1)] * 10 for t in range(1, 11)}
        means = [m for _, m in training_seed_means(per_model)]
        expected = float(np.std(means, ddof=1))
        wrong = float(np.std([v for vals in per_model.values() for v in vals], ddof=1))
        self.assertAlmostEqual(expected, float(np.std(
            [0.10 + 0.01 * (t - 1) for t in range(1, 11)], ddof=1)))
        self.assertNotAlmostEqual(expected, wrong)

    def test_inference_std_n_a_when_not_meaningful(self):
        self.assertEqual(inference_std_for_model([0.2, 0.2, 0.2], False), "n/a")
        self.assertAlmostEqual(inference_std_for_model([0.2, 0.2, 0.2], True), 0.0)

    def test_omit_failed_model_from_means(self):
        per_model = {1: [0.2, 0.2], 2: [], 3: [0.4]}
        means = training_seed_means(per_model)
        self.assertEqual([ts for ts, _ in means], [1, 3])
        self.assertEqual(len(means), 2)


if __name__ == "__main__":
    unittest.main()
