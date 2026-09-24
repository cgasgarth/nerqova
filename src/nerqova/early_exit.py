"""Partial Qwen3.5 forward pass for trained decision exits.

The prefix and each question keep separate prompt caches. A deferred question
can continue through the remaining layers with the same weights and state.
"""

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, make_prompt_cache

from kev.model import rows_of, rows_per_pass
from .mlx_model import MLXDecisionModel


def run_layers(text, hidden, caches, start, stop):
    for index in range(start, stop):
        layer = text.layers[index]
        mask = (create_ssm_mask(hidden, caches[index]) if layer.is_linear
                else create_attention_mask(hidden, caches[index]))
        hidden = layer(hidden, mask=mask, cache=caches[index])
    mx.eval(hidden)
    return hidden


def fork_prefix(caches, batch_size):
    branches = [type(cache).merge([cache] * batch_size) for cache in caches]
    for cache in branches:
        if isinstance(cache, ArraysCache):
            cache.left_padding = None
    return branches


def encoded_rows(model, enc):
    state_ids, _, rows = rows_of(enc)
    return state_ids, rows, rows_per_pass([row["ids"] for row in rows], len(state_ids))


def prefix_at(model, state_ids, layer):
    cache = make_prompt_cache(model.lm)
    hidden = model.text.embed_tokens(mx.array([state_ids], dtype=mx.int32))
    return run_layers(model.text, hidden, cache, 0, layer), cache


def branch_at(model, rows, cache, layer):
    length = max(len(row["ids"]) for row in rows)
    ids = mx.array([row["ids"] + [model.pad_id] * (length - len(row["ids"]))
                    for row in rows], dtype=mx.int32)
    branches = fork_prefix(cache[:layer], len(rows)) + make_prompt_cache(model.lm)[layer:]
    hidden = run_layers(model.text, model.text.embed_tokens(ids), branches, 0, layer)
    return hidden, branches


def branch_at_with_capture(model, rows, cache, layer, capture_at):
    """Retain an intermediate branch activation without a GPU barrier."""
    length = max(len(row["ids"]) for row in rows)
    ids = mx.array([row["ids"] + [model.pad_id] * (length - len(row["ids"]))
                    for row in rows], dtype=mx.int32)
    branches = fork_prefix(cache[:layer], len(rows)) + make_prompt_cache(model.lm)[layer:]
    hidden = model.text.embed_tokens(ids)
    captured = None
    for index in range(layer):
        current = model.text.layers[index]
        mask = (create_ssm_mask(hidden, branches[index]) if current.is_linear
                else create_attention_mask(hidden, branches[index]))
        hidden = current(hidden, mask=mask, cache=branches[index])
        if index + 1 == capture_at:
            captured = hidden
    mx.eval(hidden, captured)
    return hidden, branches, captured


def anchor_vectors(model, hidden, rows):
    """Copy only the normalized decide/option vectors to host memory."""
    hidden = model.text.norm(hidden)
    positions = [(row["decide"], *row["opts"]) for row in rows]
    vectors = [np.asarray(hidden[index, list(pos)].astype(mx.float16)).copy()
               for index, pos in enumerate(positions)]
    return vectors


def required_exit_gap(option_count, gap, wide_gap):
    return wide_gap if wide_gap is not None and option_count >= 14 else gap


class EarlyExitDecisionModel(MLXDecisionModel):
    """Use a trained partial-layer head; continue uncertain rows exactly."""

    def load_exit(self, path, gap, checkpoint_revision, wide_gap=None):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact["train_manifest"]["checkpoint_revision"] != checkpoint_revision:
            raise ValueError("exit head was trained for another checkpoint revision")
        self.exit_layer = int(artifact["layer"])
        if self.exit_layer not in (8, 16, 24):
            raise ValueError("unsupported exit layer")
        if not 0 <= gap < float("inf"):
            raise ValueError("exit log odds gap must be finite and nonnegative")
        if wide_gap is not None and not 0 <= wide_gap <= gap:
            raise ValueError("wide-choice exit gap must be nonnegative and no larger than the base gap")
        self.exit_gap = gap
        self.exit_gap_wide = wide_gap
        self.max_exit_state_tokens = int(artifact["train_manifest"]["max_state_tokens"])
        self.exit_temperature = float(artifact["temperature"])
        self.exit_suite_sha256 = artifact["train_manifest"].get("suite_sha256")
        if artifact["head_type"] != "pair":
            raise ValueError("unsupported exit head")
        self._exit_weights = {name: mx.array(value.numpy()) for name, value in artifact["head"].items()}
        mx.eval(self._exit_weights)

    def load_verifier(self, path, threshold, checkpoint_revision):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact["train_manifest"]["checkpoint_revision"] != checkpoint_revision:
            raise ValueError("verifier head was trained for another checkpoint revision")
        if artifact["train_manifest"].get("suite_sha256") != self.exit_suite_sha256:
            raise ValueError("verifier head was trained for another suite")
        if int(artifact["layer"]) >= self.exit_layer:
            raise ValueError("verifier must run before the exit layer")
        if not 0 < threshold <= 1:
            raise ValueError("verifier confidence must be in (0, 1]")
        self.verify_layer = int(artifact["layer"])
        self.verify_threshold = threshold
        self.verify_temperature = float(artifact["temperature"])
        if artifact["head_type"] != "pair":
            raise ValueError("unsupported verifier head")
        self._verify_weights = {name: mx.array(value.numpy()) for name, value in artifact["head"].items()}
        mx.eval(self._verify_weights)

    def prefix(self, enc):
        state_ids, _, _ = encoded_rows(self, enc)
        hidden, cache = prefix_at(self, state_ids, self.exit_layer)
        return {"state_ids": tuple(state_ids), "hidden": hidden, "cache": cache, "full": False}

    def _exit_logits(self, hidden, rows, *, weights=None, temperature=None):
        weights = self._exit_weights if weights is None else weights
        temperature = self.exit_temperature if temperature is None else temperature
        positions = [(row["decide"], row["opts"]) for row in rows]
        counts = [len(opts) for _, opts in positions]
        decide = mx.array([d for d, _ in positions], dtype=mx.int32)
        options = mx.array([opts + [opts[0]] * (max(counts) - len(opts)) for _, opts in positions], dtype=mx.int32)
        indices = mx.arange(len(rows), dtype=mx.int32)
        hidden = self.text.norm(hidden)
        query = hidden[indices, decide].astype(mx.float32)
        option = hidden[indices[:, None], options].astype(mx.float32)
        q = query @ weights["query.weight"].T + weights["query.bias"]
        k = option @ weights["option.weight"].T + weights["option.bias"]
        q, k = q * mx.sigmoid(q), k * mx.sigmoid(k)
        q = mx.broadcast_to(q[:, None, :], k.shape)
        pair = mx.concatenate((q, k, q * k, mx.abs(q - k)), axis=-1)
        joint = pair @ weights["joint.0.weight"].T + weights["joint.0.bias"]
        joint = joint * mx.sigmoid(joint)
        logits = (joint @ weights["joint.2.weight"].T
                  + weights["joint.2.bias"]).squeeze(-1)
        logits = logits / temperature
        return self._host_batches([(logits, counts)])

    def _score_with_prefix(self, enc, prefix):
        state_ids, rows, chunk = encoded_rows(self, enc)
        if tuple(state_ids) != prefix["state_ids"]:
            raise ValueError("prefix does not match this record's state")
        force_full = len(state_ids) > self.max_exit_state_tokens
        out = []
        deferred_count = 0
        verifier_vetoes = 0
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            if force_full:
                hidden, branches = branch_at(self, part, prefix["cache"], self.exit_layer)
                verify_logits = None
            elif hasattr(self, "verify_layer"):
                hidden, branches, check_hidden = branch_at_with_capture(
                    self, part, prefix["cache"], self.exit_layer, self.verify_layer,
                )
                verify_logits = self._exit_logits(
                    check_hidden, part, weights=self._verify_weights,
                    temperature=self.verify_temperature,
                )
            else:
                hidden, branches = branch_at(self, part, prefix["cache"], self.exit_layer)
                verify_logits = None
            early_logits = [None] * len(part) if force_full else self._exit_logits(hidden, part)
            deferred = list(range(len(part))) if force_full else []
            if not force_full:
                for index, logits in enumerate(early_logits):
                    top_two = F.softmax(logits, -1).topk(2).values
                    gap = float((top_two[0].clamp_min(1e-9) / top_two[1].clamp_min(1e-9)).log())
                    if gap < required_exit_gap(len(logits), self.exit_gap, self.exit_gap_wide):
                        deferred.append(index)
                    elif verify_logits is not None:
                        check = F.softmax(verify_logits[index], -1)
                        uniform = 1 / len(check)
                        margin = (float(check.max()) - uniform) / (1 - uniform)
                        if margin < self.verify_threshold or int(check.argmax()) != int(logits.argmax()):
                            deferred.append(index)
                            verifier_vetoes += 1
            deferred_count += len(deferred)
            if deferred:
                if not prefix["full"]:
                    run_layers(self.text, prefix["hidden"], prefix["cache"], self.exit_layer, len(self.text.layers))
                    prefix["full"] = True
                selected = mx.array(deferred, dtype=mx.int32)
                for cache in branches[:self.exit_layer]:
                    cache.filter(selected)
                for layer in range(self.exit_layer, len(self.text.layers)):
                    branches[layer] = type(prefix["cache"][layer]).merge([prefix["cache"][layer]] * len(deferred))
                    if isinstance(branches[layer], ArraysCache):
                        branches[layer].left_padding = None
                finished = run_layers(self.text, hidden[selected], branches,
                                      self.exit_layer, len(self.text.layers))
                final_rows = [part[index] for index in deferred]
                scores, counts = self._batch_logits(self.text.norm(finished),
                                                    [(row["decide"], row["opts"]) for row in final_rows])
                final_logits = self._host_batches([(scores, counts)])
                for index, logits in zip(deferred, final_logits, strict=True):
                    early_logits[index] = logits
            out.extend(early_logits)
        self.last_exit_stats = {"questions": len(rows), "exited": len(rows) - deferred_count,
                                "deferred": deferred_count, "verifier_vetoes": verifier_vetoes}
        return out

    def forward(self, enc):
        return self._score_with_prefix(enc, self.prefix(enc))

    def probs(self, enc):
        return [F.softmax(logits, -1) for logits in self.forward(enc)]

    def probs_and_prefix(self, enc):
        prefix = self.prefix(enc)
        return [F.softmax(logits, -1) for logits in self._score_with_prefix(enc, prefix)], prefix

    def probs_with_prefix(self, enc, prefix):
        return [F.softmax(logits, -1) for logits in self._score_with_prefix(enc, prefix)]
