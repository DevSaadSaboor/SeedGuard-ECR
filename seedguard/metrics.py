"""Ranking metrics used by SeedGuard.

Precision@K is computed from the ranked recommendation list and labels.
It is not derived from Recall@K.
"""


def compute_recall_at_k(ranked_items, relevant_items, k):
    """Return 1 if any relevant item appears in the top-k, else 0.

    Matches ECR's single-positive ``compute_recall`` when ``relevant_items``
    is a single label (or a one-element collection).
    """
    if isinstance(relevant_items, (int, str)):
        relevant = {relevant_items}
    else:
        relevant = set(relevant_items)
    top = ranked_items[:k]
    return int(any(item in relevant for item in top))


def compute_precision_at_k(ranked_items, relevant_items, k):
    """Precision@K = (# relevant items among top K) / K.

    For ECR's usual single-positive protocol this is 1/K if the label is in
    the top K, otherwise 0. Aggregation across examples follows ECR:
    mean over examples (same denominator as Recall@K).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if isinstance(relevant_items, (int, str)):
        relevant = {relevant_items}
    else:
        relevant = set(relevant_items)
    hits = sum(1 for item in ranked_items[:k] if item in relevant)
    return hits / k
