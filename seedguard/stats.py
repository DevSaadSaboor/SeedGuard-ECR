"""Aggregation formulas for training-seed and inference variation.

Training-seed STD is always computed across per-model means, never across
all matrix cells. Missing values are never treated as zero.
"""

import math

import numpy as np


def seed_stats(values):
    """Mean / sample STD (ddof=1) / min / max / range over finite values.

    Empty input returns blank fields (not zeros). A single value yields STD 0.
    """
    cleaned = []
    for v in values:
        if v is None or v == "":
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        cleaned.append(fv)

    if len(cleaned) == 0:
        return {"mean": "", "std": "", "min": "", "max": "", "range": "", "n": 0}
    arr = np.asarray(cleaned, dtype=float)
    if len(arr) == 1:
        return {
            "mean": float(arr[0]),
            "std": 0.0,
            "min": float(arr[0]),
            "max": float(arr[0]),
            "range": 0.0,
            "n": 1,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "n": int(len(arr)),
    }


def training_seed_means(per_model_values):
    """One mean per training seed from that model's valid inference values.

    ``per_model_values`` maps training_seed -> iterable of metric values.
    Models with no valid values are omitted (not zero-filled).
    """
    means = []
    for ts in sorted(per_model_values.keys(), key=lambda x: int(x) if str(x).lstrip("-").isdigit() else str(x)):
        stats = seed_stats(per_model_values[ts])
        if stats["n"] > 0:
            means.append((ts, stats["mean"]))
    return means


def inference_std_for_model(values, inference_variation_meaningful=True):
    """Sample STD across inference evaluations for one fixed trained model.

    If inference variation is not scientifically meaningful (e.g. deterministic
    ranking), returns ``"n/a"`` instead of implying stochastic robustness.
    Identical repeated deterministic scores still yield numeric 0.0 when
    ``inference_variation_meaningful`` is True (for transparency of the matrix).
    """
    if not inference_variation_meaningful:
        return "n/a"
    return seed_stats(values)["std"]
