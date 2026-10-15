"""SeedGuard: multi-seed robustness evaluation around ECR recommendation.

This package is intentionally thin. ECR training/evaluation remain in
``src_emo/``. SeedGuard only orchestrates seeds, collects results, and
aggregates statistics so another compatible recommender can be plugged in
later via ``seedguard.adapters``.
"""

from .metrics import compute_precision_at_k, compute_recall_at_k
from .stats import seed_stats, training_seed_means, inference_std_for_model

__all__ = [
    "compute_precision_at_k",
    "compute_recall_at_k",
    "seed_stats",
    "training_seed_means",
    "inference_std_for_model",
]
