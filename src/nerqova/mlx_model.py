"""MLX scorer for Qwen3 and Qwen3.5 Kev-format decision checkpoints.

Kev provides the encoder and pointer head. MLX runs the pretrained backbone
and pointer readout on Apple Silicon. Only final logits move to host memory.
This module implements the model contract used by Kev's HTTP server.
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, make_prompt_cache
from mlx_lm.utils import load_model

from kev.model import PointerHead, encode, rows_of, rows_per_pass


def merge_lora(lm, adapter_dir, scale=1.0):
    """Fold a PEFT adapter into the mlx-lm model's weights the way the torch path does: W + (B @ A) * alpha / r in fp32,
    rounded once to the backbone dtype. `scale` is LoadOptions.lora_scale (WiSE-FT interpolation). Returns the tensor count."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("trainable_token_indices"):
        raise ValueError("the MLX backend does not carry trained token embeddings (special_embeddings checkpoints); use backend=torch")
    alpha = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    weights = mx.load(str(adapter_dir / "adapter_model.safetensors"))
    params = dict(tree_flatten(lm.parameters()))
    merged_count = 0
    # Merge one tensor at a time. Keep only a small pool of reusable CPU buffers
    # during loading, then restore the callers allocator policy.
    previous_cache_limit = mx.set_cache_limit(1024 ** 3)
    model_root = "language_model.model." if hasattr(lm, "language_model") else "model."
    try:
        with mx.stream(mx.cpu):   # the GPU's fp32 matmul is a reduced-precision fast path (~1e-3 relative on an M5); the merge is one-time and must be exact
            for name, a in weights.items():
                if not name.endswith(".lora_A.weight"):
                    continue
                stem = name[: -len(".lora_A.weight")]
                # PEFT names the wrapped text model `base_model.model.<layers...>`; mlx-lm's text root differs by family.
                target = stem.replace("base_model.model.", model_root, 1) + ".weight"
                if target not in params:
                    raise ValueError(f"adapter tensor {stem} has no weight in the mlx-lm model (looked for {target})")
                base = params.pop(target)
                delta = (weights[stem + ".lora_B.weight"].astype(mx.float32) @ a.astype(mx.float32)) * (alpha * scale)
                merged = (base.astype(mx.float32) + delta).astype(base.dtype)
                mx.eval(merged)
                lm.load_weights([(target, merged)], strict=False)
                merged_count += 1
                del base, delta, merged
        mx.eval(lm.parameters())
        mx.clear_cache()
    finally:
        mx.set_cache_limit(previous_cache_limit)
    return merged_count


class MLXDecisionModel:
    """Prefill-only scorer: hidden states and pointer logits run in MLX."""
    backend, device, option_isolation = "mlx", "mlx", False
    prefix_min_tokens = 0   # kev.serve caches the state prefix for every request: on Metal the branch-only pass is always the cheaper one

    def __init__(self, base_dir, pad_id, head_dim=256, unmasked_branches=False):
        self.lm, _ = load_model(Path(base_dir))                       # weights as stored (bf16 for the Qwen3.5 bases)
        if self.lm.model_type == "qwen3_5":
            self.hybrid, self.text = True, self.lm.language_model.model
        elif self.lm.model_type == "qwen3":
            self.hybrid, self.text = False, self.lm.model
        else:
            raise ValueError(f"the MLX decision scorer supports Qwen3 and Qwen3.5, not {self.lm.model_type}")
        self.pad_id = pad_id
        self.unmasked_branches = unmasked_branches
        self.packed_delta = False
        self.head = PointerHead(self.text.embed_tokens.weight.shape[1], dp=head_dim).eval()

    def enable_packed_delta(self):
        if (not self.hybrid or len(self.text.layers) != 32
                or self.text.embed_tokens.weight.shape[1] != 2560):
            raise ValueError("packed DeltaNet supports only the Kev-4B Qwen3.5 backbone")
        from .packed_delta import PackedGatedDeltaNet

        self.packed_delta = True
        for layer in self.text.layers:
            if layer.is_linear:
                if (layer.linear_attn.key_dim, layer.linear_attn.value_dim,
                        layer.linear_attn.num_v_heads) != (2048, 4096, 32):
                    raise ValueError("packed DeltaNet head dimensions differ from Kev-4B")
                # LoRA is already merged. Rebind the method without copying or
                # changing any loaded weight array.
                layer.linear_attn.__class__ = PackedGatedDeltaNet

    @property
    def dtype(self):
        return str(self.text.embed_tokens.weight.dtype).removeprefix("mlx.core.")

    def eval(self):
        self.head.eval()
        self._head_weights = {
            name: mx.array(parameter.detach().numpy())
            for name, parameter in self.head.named_parameters()
        }
        mx.eval(self._head_weights)
        return self

    def encode(self, tok, rec, **kw):
        return encode(tok, rec, option_isolation=False, **kw)

    def _hidden(self, rows, cache=None):
        """[N, L, d] hidden states of right-padded token rows. Pads sit after every real token and both layer kinds are
        causal (attention: causal mask; DeltaNet: a left-to-right recurrence), so no real token sees a pad."""
        L = max(len(r) for r in rows)
        ids = mx.array([r + [self.pad_id] * (L - len(r)) for r in rows], dtype=mx.int32)
        h = self.text(ids, cache=cache)
        mx.eval(h)
        return h

    def _batch_logits(self, h, positions):
        """Score a branch batch in MLX, padding option anchors only within this batch."""
        counts = [len(opts) for _, opts in positions]
        decide = mx.array([d for d, _ in positions], dtype=mx.int32)
        options = mx.array([opts + [opts[0]] * (max(counts) - len(opts)) for _, opts in positions], dtype=mx.int32)
        rows = mx.arange(len(positions), dtype=mx.int32)
        h_decide = h[rows, decide].astype(mx.float32)
        h_options = h[rows[:, None], options].astype(mx.float32)
        q = h_decide @ self._head_weights["q.weight"].T + self._head_weights["q.bias"]
        k = h_options @ self._head_weights["k.weight"].T + self._head_weights["k.bias"]
        logits = mx.sum(k * q[:, None, :], axis=-1) * self.head.scale
        return (logits if self.head.temperature == 1.0 else logits / self.head.temperature), counts

    @staticmethod
    def _host_batches(batches):
        mx.eval([logits for logits, _ in batches])
        out = []
        for logits, counts in batches:
            matrix = np.asarray(logits)
            out.extend(torch.from_numpy(matrix[i, :count].copy()) for i, count in enumerate(counts))
        return out

    def forward_rows(self, enc):
        """Row form, as the torch path computes it: every question is one causal row of state + branch tokens, the state
        recomputed per row. The reference the prefix form is checked against (tests/test_mlx.py); serving uses `forward`."""
        S, _, rows = rows_of(enc)
        chunk, out = rows_per_pass([S + r["ids"] for r in rows]), []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            h = self._hidden([S + r["ids"] for r in part])
            out.append(self._batch_logits(h, [(len(S) + r["decide"], [len(S) + o for o in r["opts"]]) for r in part]))
        return self._host_batches(out)

    # --- state prefix: the state runs once into an mlx-lm prompt cache (attention KV and, for Qwen3.5, DeltaNet conv +
    # recurrent state); the branches run as one batch on a replicated copy, so the prefix stays pristine and can be
    # reused by the next request with the same state. On Metal this is also the cheapest way to answer a single request
    # (state once instead of once per question), so it is the only path `forward` / `probs` take.

    def prefix(self, enc):
        Ls = enc["seg"].count(0)
        state_ids = tuple(enc["ids"][:Ls])
        cache = make_prompt_cache(self.lm)
        self._hidden([list(state_ids)], cache)
        return state_ids, cache

    def _branch_logits(self, enc, cache):
        """Branches as rows on a replicated copy of the state cache, rows_per_pass rows (and cache copies) at a time."""
        _, _, rows = rows_of(enc)
        chunk, out = rows_per_pass([r["ids"] for r in rows], enc["seg"].count(0)), []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            batch = [type(c).merge([c] * len(part)) for c in cache]      # merge copies the arrays: `cache` is not mutated
            if self.unmasked_branches:
                # Every branch starts from the same prefix, so the merged recurrent
                # caches have zero left padding. Right padding is after all scored
                # anchors; the branch caches are discarded after this pass.
                for state in batch:
                    if isinstance(state, ArraysCache):
                        state.left_padding = None
            h = self._hidden([r["ids"] for r in part], batch)
            out.append(self._batch_logits(h, [(r["decide"], r["opts"]) for r in part]))
        return self._host_batches(out)

    def _branch_probs(self, enc, cache):
        return [F.softmax(z, -1) for z in self._branch_logits(enc, cache)]

    def forward(self, enc):
        """List of logits tensors, one per question."""
        return self._branch_logits(enc, self.prefix(enc)[1])

    def probs(self, enc):
        return self.probs_and_prefix(enc)[0]

    def probs_and_prefix(self, enc):
        state_ids, _, rows = rows_of(enc)
        if self.packed_delta and len(rows) == 1 and state_ids:
            # One causal pass can score this question and capture its state prefix.
            # Recurrent kernels snapshot at the boundary; attention caches trim it.
            row = rows[0]
            count = len(state_ids)
            cache = make_prompt_cache(self.lm)
            for layer_cache in cache:
                if isinstance(layer_cache, ArraysCache):
                    layer_cache.capture_prefix_tokens = count
            h = self._hidden([state_ids + row["ids"]], cache)
            for layer_cache in cache:
                if isinstance(layer_cache, ArraysCache):
                    del layer_cache.capture_prefix_tokens
                else:
                    layer_cache.trim(len(row["ids"]))
            logits = self._host_batches([self._batch_logits(
                h, [(count + row["decide"], [count + o for o in row["opts"]])]
            )])
            return [F.softmax(z, -1) for z in logits], (tuple(state_ids), cache)
        prefix = self.prefix(enc)
        return self._branch_probs(enc, prefix[1]), prefix

    def probs_with_prefix(self, enc, prefix):
        state_ids, cache = prefix
        if tuple(enc["ids"][:enc["seg"].count(0)]) != state_ids:
            raise ValueError("prefix does not match this record's state")
        return self._branch_probs(enc, cache)
