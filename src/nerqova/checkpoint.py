"""Load Kev-format weights into Nerqova's model-specific MLX scorer."""

from pathlib import Path

from kev.checkpoint import Checkpoint, resolve_run
from kev.model import load_tokenizer, pad_id


def load_model(run, *, temperature=None, lora_scale=1.0,
               unmasked_branches=False, packed_delta=False,
               exit_head=None, exit_threshold=None,
               verify_head=None, verify_threshold=None):
    """Return (checkpoint, tokenizer, scorer) with unchanged Kev weights."""
    import mlx.core as mx

    from .mlx_model import MLXDecisionModel, merge_lora

    if (exit_head is None) != (exit_threshold is None):
        raise ValueError("exit_head and exit_threshold must be set together")
    if exit_head is not None and not packed_delta:
        raise ValueError("early exit requires packed_delta")
    if exit_head is not None and (temperature is not None or lora_scale != 1.0):
        raise ValueError("trained exit heads require the checkpoint's temperature and LoRA scale")
    if (verify_head is None) != (verify_threshold is None):
        raise ValueError("verify_head and verify_threshold must be set together")
    if verify_head is not None and exit_head is None:
        raise ValueError("verifier requires an early-exit head")
    if exit_head is not None:
        from .early_exit import EarlyExitDecisionModel
        model_class = EarlyExitDecisionModel
    else:
        model_class = MLXDecisionModel

    checkpoint = Checkpoint(run)
    meta = checkpoint.meta
    if meta.option_isolation or meta.special_embeddings:
        raise ValueError("the MLX scorer does not support option isolation or trained token embeddings")
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    base_dir = resolve_run(f"{meta.base}@{meta.base_revision or ''}")
    model = model_class(base_dir, pad_id(tok), head_dim=meta.head_dim,
                        unmasked_branches=unmasked_branches or packed_delta)
    merge_lora(model.lm, checkpoint.path, lora_scale)
    if packed_delta:
        model.enable_packed_delta()
    model.head.load_state_dict(meta.head)
    model.eval()
    model.head.temperature = meta.temperature if temperature is None else temperature
    if exit_head is not None:
        model.load_exit(exit_head, exit_threshold, Path(checkpoint.path).name)
    if verify_head is not None:
        model.load_verifier(verify_head, verify_threshold, Path(checkpoint.path).name)
    # Adapter and exit-head loading buffers are dead once the scorer is ready.
    mx.clear_cache()
    return checkpoint, tok, model
