"""Use the Nerqova MLX scorer with Kev's frozen-suite evaluator."""

from kev.predictors import LocalPredictor
from kev.suite import CONTEXT

from .checkpoint import load_model


class MLXPredictor(LocalPredictor):
    """Score Kev records through the same MLX path used by Nerqova's server."""

    def __init__(self, run, *, context=CONTEXT,
                 unmasked_branches=False, packed_delta=False,
                 exit_head=None, exit_gap=None, exit_gap_wide=None,
                 verify_head=None, verify_threshold=None):
        checkpoint, self.tok, self.model = load_model(
            run, unmasked_branches=unmasked_branches, packed_delta=packed_delta,
            exit_head=exit_head, exit_gap=exit_gap, exit_gap_wide=exit_gap_wide,
            verify_head=verify_head, verify_threshold=verify_threshold,
        )
        self.checkpoint = checkpoint
        self.run = checkpoint.path
        self.device = "mps"
        self.context = context
        self.temperature = self.model.head.temperature

    def __call__(self, record):
        result = super().__call__(record)
        if hasattr(self.model, "last_exit_stats"):
            result["exit_stats"] = self.model.last_exit_stats
        return result
