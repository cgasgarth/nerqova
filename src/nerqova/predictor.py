"""Use the Nerqova MLX scorer with Kev's frozen-suite evaluator."""

from kev.predictors import LocalPredictor
from kev.suite import CONTEXT

from .checkpoint import load_model


class MLXPredictor(LocalPredictor):
    """Score Kev records through the same MLX path used by Nerqova's server."""

    def __init__(self, run, *, context=CONTEXT):
        checkpoint, self.tok, self.model = load_model(run)
        self.checkpoint = checkpoint
        self.run = checkpoint.path
        self.device = "mps"
        self.context = context
        self.temperature = self.model.head.temperature
