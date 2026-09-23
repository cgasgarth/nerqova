"""Checkpoint backed scoring checks. Set NERQOVA_TEST_RUN to a local checkpoint."""

import os
import platform

import pytest
import torch.nn.functional as F


pytestmark = pytest.mark.model


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64" or not os.getenv("NERQOVA_TEST_RUN"),
    reason="requires Apple Silicon and NERQOVA_TEST_RUN",
)
@pytest.mark.parametrize("mode", ["baseline", "unmasked", "packed"])
def test_prefix_cache_keeps_questions_isolated(mode):
    from nerqova.checkpoint import load_model

    _, tok, model = load_model(
        os.environ["NERQOVA_TEST_RUN"],
        unmasked_branches=mode == "unmasked",
        packed_delta=mode == "packed",
    )
    questions = [
        {"instr": "Which color was ordered?", "options": ["Blue", "Red", "Green"], "label": 0},
        {"instr": "Which color arrived?", "options": ["Blue", "Red"], "label": 1},
    ]
    rec = {"state": "The customer ordered a blue shirt and received a red shirt.", "questions": questions}
    enc = model.encode(tok, rec)
    full = [F.softmax(logits, -1) for logits in model.forward_rows(enc)]
    first, prefix = model.probs_and_prefix(enc)
    hit = model.probs_with_prefix(enc, prefix)
    alone = [model.probs(model.encode(tok, {**rec, "questions": [q]}))[0] for q in questions]

    for group in (first, hit, alone):
        assert max(float((a - b).abs().max()) for a, b in zip(group, full)) < 0.01

    other = model.encode(tok, {**rec, "state": rec["state"].replace("blue", "gold")})
    assert other["seg"].count(0) == enc["seg"].count(0)
    with pytest.raises(ValueError, match="prefix"):
        model.probs_with_prefix(other, prefix)


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires Apple Silicon",
)
@pytest.mark.parametrize("batch,tokens", [(1, 64), (5, 60)])
def test_packed_delta_matches_original_bitwise(batch, tokens):
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_kernel

    from nerqova.packed_delta import packed_kernel

    mx.random.seed(27)
    q = (mx.random.normal((batch, tokens, 16, 128)) * 0.05).astype(mx.bfloat16)
    k = (mx.random.normal(q.shape) * 0.05).astype(mx.bfloat16)
    v = (mx.random.normal((batch, tokens, 32, 128)) * 0.05).astype(mx.bfloat16)
    gamma = mx.full((batch, tokens, 32), 0.99, dtype=mx.float32)
    beta = mx.full((batch, tokens, 32), 0.5, dtype=mx.bfloat16)
    state = mx.zeros((batch, 32, 128, 128), dtype=mx.float32)
    expected = gated_delta_kernel(q, k, v, gamma, beta, state)
    actual = packed_kernel(q, k, v, gamma, beta, state)
    mx.eval(expected, actual)
    assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(expected, actual))
