"""Recommender adapters for SeedGuard experiment runners."""

from .base import RecommenderAdapter
from .ecr import ECRRecAdapter

ADAPTERS = {
    "ecr_rec": ECRRecAdapter,
}


def get_adapter(name="ecr_rec", **kwargs):
    if name not in ADAPTERS:
        raise KeyError(f"Unknown adapter {name!r}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[name](**kwargs)


__all__ = ["RecommenderAdapter", "ECRRecAdapter", "ADAPTERS", "get_adapter"]
