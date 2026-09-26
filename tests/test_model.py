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

    if mode == "packed":
        single = model.encode(tok, {**rec, "questions": questions[:1]})
        cold, captured = model.probs_and_prefix(single)
        model.probs(other)
        reused = model.probs_with_prefix(enc, captured)
        reordered = model.probs_with_prefix(
            model.encode(tok, {**rec, "questions": questions[::-1]}), captured
        )[::-1]
        for actual, expected in [(cold, first[:1]), (reused, first), (reordered, first)]:
            assert max(float((a - b).abs().max()) for a, b in zip(actual, expected)) < 0.01
            assert all(a.argmax().item() == b.argmax().item() for a, b in zip(actual, expected))


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


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64" or not os.getenv("NERQOVA_TEST_RUN"),
    reason="requires Apple Silicon and NERQOVA_TEST_RUN",
)
def test_early_exit_fallback_preserves_full_logits(tmp_path):
    import torch
    from pathlib import Path

    from nerqova.checkpoint import load_model
    from nerqova.early_head import PairHead
    from nerqova.early_exit import EarlyExitDecisionModel
    from nerqova.mlx_model import MLXDecisionModel

    checkpoint, tok, model = load_model(os.environ["NERQOVA_TEST_RUN"], packed_delta=True)
    rec = {"state": "The customer ordered a blue shirt and received a red shirt.",
           "questions": [
               {"instr": "Which color was ordered?", "options": ["Blue", "Red", "Green"], "label": 0},
               {"instr": "Which color arrived?", "options": ["Blue", "Red"], "label": 1},
           ]}
    enc = model.encode(tok, rec)
    expected = model.forward(enc)
    head = PairHead()
    for weight in head.parameters():
        weight.data.zero_()
    artifact = tmp_path / "head.pt"
    torch.save({"head": head.state_dict(), "head_type": "pair", "layer": 8, "temperature": 1.0,
                "train_manifest": {"checkpoint_revision": Path(checkpoint.path).name,
                                   "max_state_tokens": 384}}, artifact)
    model.__class__ = EarlyExitDecisionModel
    model.load_exit(artifact, 0.1, Path(checkpoint.path).name)
    actual, prefix = model.probs_and_prefix(enc)
    hit = model.probs_with_prefix(enc, prefix)
    for result in (actual, hit):
        for logits, probability in zip(expected, result, strict=True):
            assert float((torch.softmax(logits, -1) - probability).abs().max()) < 1e-5
    assert prefix["full"]

    # A zero gap accepts both uniform synthetic answers in question order.
    model.load_exit(artifact, 0.0, Path(checkpoint.path).name)
    mixed = model.probs(enc)
    assert model.last_exit_stats == {"questions": 2, "exited": 2, "deferred": 0,
                                     "verifier_vetoes": 0}
    for probability in mixed:
        assert float((torch.full_like(probability, 1 / len(probability)) - probability).abs().max()) < 1e-5

    main_artifact = tmp_path / "head16.pt"
    torch.save({"head": head.state_dict(), "head_type": "pair", "layer": 16, "temperature": 1.0,
                "train_manifest": {"checkpoint_revision": Path(checkpoint.path).name,
                                   "max_state_tokens": 384}}, main_artifact)
    model.load_exit(main_artifact, 0.0, Path(checkpoint.path).name)
    model.load_verifier(artifact, 0.6, Path(checkpoint.path).name)
    verified = model.probs(enc)
    assert model.last_exit_stats == {"questions": 2, "exited": 0, "deferred": 2,
                                     "verifier_vetoes": 2}
    for logits, probability in zip(expected, verified, strict=True):
        assert float((torch.softmax(logits, -1) - probability).abs().max()) < 1e-4

    # A captured single-question prefix must support fallback and later questions.
    single = model.encode(tok, {**rec, "questions": [rec["questions"][0]]})
    reference, reference_prefix = MLXDecisionModel.probs_and_prefix(model, single)
    first, captured = model.probs_and_prefix(single)
    hit = model.probs_with_prefix(single, captured)
    reference_hit = MLXDecisionModel.probs_with_prefix(model, single, reference_prefix)
    assert float((first[0] - reference[0]).abs().max()) < 1e-4
    assert float((hit[0] - reference_hit[0]).abs().max()) < 1e-4
    alternate = model.encode(tok, {**rec, "questions": [rec["questions"][1]]})
    reused = model.probs_with_prefix(alternate, captured)
    reference_reused = MLXDecisionModel.probs_with_prefix(model, alternate, reference_prefix)
    assert float((reused[0] - reference_reused[0]).abs().max()) < 1e-4
    singleton = model.encode(tok, {**rec, "questions": [
        {"instr": "Continue", "options": ["Observe the window"], "label": 0},
    ]})
    only, singleton_prefix = model.probs_and_prefix(singleton)
    assert only[0].tolist() == [1.0]
    assert model.probs_with_prefix(singleton, singleton_prefix)[0].tolist() == [1.0]

    # A state longer than the training context must use the full pointer head.
    restricted = tmp_path / "head-short-context.pt"
    torch.save({"head": head.state_dict(), "head_type": "pair", "layer": 16,
                "temperature": 1.0,
                "train_manifest": {"checkpoint_revision": Path(checkpoint.path).name,
                                   "max_state_tokens": 1}}, restricted)
    model.load_exit(restricted, 0.0, Path(checkpoint.path).name)
    guarded = model.probs(enc)
    assert model.last_exit_stats == {"questions": 2, "exited": 0, "deferred": 2,
                                     "verifier_vetoes": 0}
    for logits, probability in zip(expected, guarded, strict=True):
        assert float((torch.softmax(logits, -1) - probability).abs().max()) < 1e-4


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires Apple Silicon",
)
@pytest.mark.parametrize("prefix_tokens", [1, 31, 64])
def test_packed_delta_captures_prefix_without_changing_outputs(prefix_tokens):
    import mlx.core as mx
    from nerqova.packed_delta import packed_kernel

    mx.random.seed(27)
    q = (mx.random.normal((1, 64, 16, 128)) * 0.05).astype(mx.bfloat16)
    k = (mx.random.normal(q.shape) * 0.05).astype(mx.bfloat16)
    v = (mx.random.normal((1, 64, 32, 128)) * 0.05).astype(mx.bfloat16)
    gamma = mx.full((1, 64, 32), 0.99, dtype=mx.float32)
    beta = mx.full((1, 64, 32), 0.5, dtype=mx.bfloat16)
    state = mx.random.normal((1, 32, 128, 128)) * 0.01
    whole = packed_kernel(q, k, v, gamma, beta, state)
    captured = packed_kernel(q, k, v, gamma, beta, state, prefix_tokens)
    prefix = packed_kernel(*(x[:, :prefix_tokens] for x in (q, k, v, gamma, beta)), state)
    mx.eval(whole, captured, prefix)
    assert bool(mx.array_equal(whole[0], captured[0]).item())
    assert bool(mx.array_equal(prefix[1], captured[1]).item())
