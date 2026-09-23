"""Load Kev-format weights into Nerqova's model-specific MLX scorer."""

from kev.checkpoint import Checkpoint, resolve_run
from kev.model import load_tokenizer, pad_id


def load_model(run, *, temperature=None, lora_scale=1.0):
    """Return (checkpoint, tokenizer, scorer) with unchanged Kev weights."""
    from .mlx_model import MLXDecisionModel, merge_lora

    checkpoint = Checkpoint(run)
    meta = checkpoint.meta
    if meta.option_isolation or meta.special_embeddings:
        raise ValueError("the MLX scorer does not support option isolation or trained token embeddings")
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    base_dir = resolve_run(f"{meta.base}@{meta.base_revision or ''}")
    model = MLXDecisionModel(base_dir, pad_id(tok), head_dim=meta.head_dim)
    merge_lora(model.lm, checkpoint.path, lora_scale)
    model.head.load_state_dict(meta.head)
    model.eval()
    model.head.temperature = meta.temperature if temperature is None else temperature
    return checkpoint, tok, model
