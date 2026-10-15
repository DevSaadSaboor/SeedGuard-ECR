"""Abstract adapter so SeedGuard can drive another compatible recommender later."""

from abc import ABC, abstractmethod


class RecommenderAdapter(ABC):
    """Minimal interface for a SeedGuard-compatible recommender."""

    name = "base"
    # Whether recommendation evaluation scores change with inference_seed.
    inference_variation_meaningful = False

    @abstractmethod
    def train_command(self, training_seed, output_dir, num_epochs, num_warmup):
        """Return argv list to train one model for ``training_seed``."""

    @abstractmethod
    def evaluate_command(self, training_seed, inference_seed, checkpoint_dir, num_warmup):
        """Return argv list to evaluate one (training_seed, inference_seed) cell."""

    @abstractmethod
    def checkpoint_dir(self, training_seed):
        """Filesystem path for the checkpoint belonging to ``training_seed``."""

    def dataset_command(self, dataset_seed):
        """Optional argv to build a fixed eval subset once. Default: None."""
        return None
