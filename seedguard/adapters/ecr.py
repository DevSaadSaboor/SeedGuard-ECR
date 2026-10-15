"""ECR recommendation adapter (UniCRS-style ``train_rec.py``)."""

import os
import sys

from .base import RecommenderAdapter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src_emo")

# Matches README Emotion-aware Item Recommendation settings, on the quick-eval set.
TRAIN_ARGS = [
    "--dataset", "redial_gen_quick",
    "--n_prefix_rec", "10",
    "--per_device_train_batch_size", "16",
    "--per_device_eval_batch_size", "32",
    "--gradient_accumulation_steps", "8",
    "--context_max_length", "200",
    "--prompt_max_length", "200",
    "--entity_max_length", "32",
    "--learning_rate", "1e-4",
    "--like_score", "2.0",
    "--dislike_score", "1.0",
    "--notsay_score", "0.5",
    "--weighted_loss",
    "--nei_mer",
    "--use_sentiment",
    "--no_timestamp",
]


class ECRRecAdapter(RecommenderAdapter):
    """Drive ECR's existing ``train_rec.py`` without replacing the model."""

    name = "ecr_rec"
    # Ranking is torch.topk over fixed kg['item_ids'] under torch.no_grad().
    inference_variation_meaningful = False
    working_directory = SRC
    dataset_id = "redial_gen_quick"
    config_id = "ecr_rec_redial_gen_quick_readme"

    def __init__(self, root=None, train_batch_size=None, eval_batch_size=None,
                 gradient_accumulation_steps=None):
        self.root = root or ROOT
        self.src = os.path.join(self.root, "src_emo")
        self.working_directory = self.src
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def _python(self):
        return sys.executable

    def _train_args(self):
        args = list(TRAIN_ARGS)
        replacements = {
            "--per_device_train_batch_size": self.train_batch_size,
            "--per_device_eval_batch_size": self.eval_batch_size,
            "--gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        for flag, value in replacements.items():
            if value is None:
                continue
            if flag in args:
                args[args.index(flag) + 1] = str(value)
        return args

    def checkpoint_dir(self, training_seed):
        return os.path.join(self.src, "save", f"train_seed_{training_seed}")

    def dataset_command(self, dataset_seed):
        return [self._python(), os.path.join(self.root, "make_quick_eval_set.py"),
                "--seed", str(dataset_seed)]

    def train_command(self, training_seed, output_dir, num_epochs, num_warmup):
        return [
            self._python(), "train_rec.py",
            *self._train_args(),
            "--num_train_epochs", str(num_epochs),
            "--num_warmup_steps", str(num_warmup),
            "--training_seed", str(training_seed),
            "--seed", str(training_seed),
            "--output_dir", output_dir if output_dir.endswith(os.sep) else output_dir + os.sep,
        ]

    def evaluate_command(self, training_seed, inference_seed, checkpoint_dir, num_warmup):
        best = os.path.join(checkpoint_dir, "best") + os.sep
        return [
            self._python(), "train_rec.py",
            *self._train_args(),
            "--test",
            "--num_warmup_steps", str(num_warmup),
            "--training_seed", str(training_seed),
            "--inference_seed", str(inference_seed),
            "--seed", str(training_seed),
            "--prompt_encoder", best,
        ]
