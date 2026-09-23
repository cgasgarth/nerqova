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
def test_prefix_cache_keeps_questions_isolated():
    from nerqova.checkpoint import load_student

    _, tok, model = load_student(os.environ["NERQOVA_TEST_RUN"])
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
